"""Fuzzy canonicalization of bibliographic works.

Cleaned-up port of the algorithm originally in ``label_papers_old.py``.
We compute a weighted similarity over title / authors / journal and then
perform a single-pass clustering: each row is compared against the
representatives of existing clusters; the first cluster that scores above
the threshold (and has a year within ``year_window``) wins. Otherwise the
row starts a new cluster.

Usage
-----

>>> from citegraph.dedup import canonicalize_works, DedupConfig
>>> works, edges, stats = canonicalize_works(sources, citations_raw, DedupConfig())

``works`` is a canonical DataFrame indexed by stable ``w-`` work ids with
``ring``/``source_file`` provenance columns; ``edges`` is the deduplicated
``citing_id``/``cited_id`` table; ``stats`` reports merges, dropped
self-loops, and the per-citation-row cluster mapping.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio

from citegraph._progress import iter_with_progress
from citegraph.ids import _first_author_token, make_work_id
from citegraph.io import parse_authors_list, require_columns

logger = logging.getLogger(__name__)


_NON_ALNUM = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_text(text: object) -> str:
    """Lowercase, strip, and remove non-alphanumeric characters.

    Lists of authors are joined with commas before normalisation.
    """
    if isinstance(text, list):
        text = ", ".join(str(t) for t in text)
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _NON_ALNUM.sub(" ", text.casefold())
    return _WS.sub(" ", text).strip()


@dataclass
class DedupConfig:
    """Tunable thresholds for fuzzy deduplication."""

    title_weight: float = 0.65
    authors_weight: float = 0.25
    journal_weight: float = 0.10
    year_window: int = 1
    threshold: float = 85.0


def _read_year(value: object) -> tuple[int | None, bool]:
    """Coerce ``value`` to a year for the dedup year predicate.

    Returns ``(year, parsed_ok)``:

    - ``(None, True)``  — the value is *missing* (None / "" / sentinel ``0`` /
      non-positive). Callers should not use the year window to reject the
      cluster in this case; the title/authors fuzzy score still has to clear
      the threshold, which is what carries the matching decision.
    - ``(int, True)``   — a usable year.
    - ``(None, False)`` — the value was non-empty but couldn't be parsed
      (e.g. ``"abc"``). Callers should fail closed in this case, preserving
      the original conservative behaviour for corrupt data.
    """
    if value is None or pd.isna(value) or value == "":
        return None, True
    try:
        year = int(value)
    except (TypeError, ValueError, OverflowError):
        return None, False
    if year <= 0:
        return None, True
    return year, True


def compare_papers(paper1: dict, paper2: dict, cfg: DedupConfig) -> bool:
    """Return ``True`` if two records are likely the same paper.

    Each record is a dict with at least ``Title``, ``Authors``, ``Journal``
    and ``Year`` keys (the same shape returned by Gemini extraction).

    The year predicate is *tolerant* of missing data (the schema's sentinel
    ``Year=0``, ``None``, or empty string): if either side is missing, we
    decline to reject on year and let title/authors carry the decision.
    Genuinely unparseable values (e.g. a stray non-numeric string) still
    fail closed — see :func:`_read_year`.
    """
    return _assess_work_match(paper1, paper2, cfg)["decision"] == "match"


def _normalize_work_record(record: dict) -> dict:
    """Normalize representations without changing bibliographic display values."""
    out = dict(record)
    authors = parse_authors_list(record.get("Authors_List"))
    if authors and not isinstance(record.get("Authors"), str):
        out["Authors"] = ", ".join(authors)
    elif authors and not record.get("Authors", "").strip():
        out["Authors"] = ", ".join(authors)
    return out


def _assess_work_match(paper1: dict, paper2: dict, cfg: DedupConfig) -> dict:
    """Score a pair, retaining reasons when strong fuzzy evidence is unsafe."""
    paper1, paper2 = _normalize_work_record(paper1), _normalize_work_record(paper2)
    title1 = normalize_text(paper1.get("Title"))
    title2 = normalize_text(paper2.get("Title"))
    raw_title1 = paper1.get("Title") if isinstance(paper1.get("Title"), str) else ""
    raw_title2 = paper2.get("Title") if isinstance(paper2.get("Title"), str) else ""
    subtitle_shortening = False
    for raw_left, raw_right in ((raw_title1, raw_title2), (raw_title2, raw_title1)):
        if ":" in raw_left or "—" in raw_left:
            prefix = raw_left.replace("—", ":").split(":", 1)[0]
            subtitle_shortening |= _title_without_leading_article(normalize_text(prefix)) == _title_without_leading_article(normalize_text(raw_right))
    title_score = token_set_ratio(
        _title_without_leading_article(title1), _title_without_leading_article(title2)
    )
    authors_score = token_set_ratio(
        normalize_text(paper1.get("Authors")), normalize_text(paper2.get("Authors"))
    )
    journal_score = token_set_ratio(
        normalize_text(paper1.get("Journal")), normalize_text(paper2.get("Journal"))
    )

    weighted = (
        title_score * cfg.title_weight
        + authors_score * cfg.authors_weight
        + journal_score * cfg.journal_weight
    )

    y1, ok1 = _read_year(paper1.get("Year"))
    y2, ok2 = _read_year(paper2.get("Year"))
    if not ok1 or not ok2:
        year_ok = False  # fail closed on unparseable data
    elif y1 is None or y2 is None:
        year_ok = True  # at least one side missing -> rely on title/authors
    else:
        year_ok = abs(y1 - y2) <= cfg.year_window

    reasons = _title_conflict_reasons(title1, title2, subtitle_shortening=subtitle_shortening)
    if not year_ok:
        decision, reasons = "reject", ["invalid_year" if not ok1 or not ok2 else "year_window"]
    elif weighted < cfg.threshold:
        decision, reasons = "reject", ["below_threshold"]
    elif reasons:
        decision = "review"
    else:
        decision, reasons = "match", ["compatible_metadata"]
    return dict(decision=decision, reason_codes=reasons, title_score=title_score,
                title_full_score=ratio(title1, title2), authors_score=authors_score,
                journal_score=journal_score, weighted_score=weighted, year_ok=year_ok)


_LEADING_ARTICLES = {"a", "an", "the"}
_NEGATION_WORDS = {"no", "not", "without", "non"}
_PART_MARKER = re.compile(r"^(part|volume|vol|book)$")


def _title_without_leading_article(title: str) -> str:
    words = title.split()
    return " ".join(words[1:] if words and words[0] in _LEADING_ARTICLES else words)


def _title_conflict_reasons(left: str, right: str, *, subtitle_shortening: bool = False) -> list[str]:
    """Guard high-scoring containment matches that change work identity."""
    lw = _title_without_leading_article(left).split()
    rw = _title_without_leading_article(right).split()
    reasons = []
    if (set(lw) & _NEGATION_WORDS) != (set(rw) & _NEGATION_WORDS):
        reasons.append("negation_conflict")
    def marker(words: list[str]) -> tuple[str, str] | None:
        for i, word in enumerate(words[:-1]):
            if _PART_MARKER.match(word) and re.fullmatch(r"[ivxlcdm]+|\d+", words[i + 1]):
                return word, words[i + 1]
        return None
    lm, rm = marker(lw), marker(rw)
    if lm != rm and (lm or rm):
        reasons.append("part_conflict")
    # Explicit subtitle omission may explain containment, but cannot explain
    # away the protected conflicts above. No semantic equivalence is inferred.
    if not subtitle_shortening and (set(lw) < set(rw) or set(rw) < set(lw)):
        reasons.append("title_expansion")
    return reasons


def _row_to_dict(row: pd.Series) -> dict:
    # Authors_List is preserved alongside the comma-joined Authors so that
    # downstream stages (author normalization in particular) can recover
    # individual author strings without having to split the joined form —
    # comma-splitting "Smith, J., García, A." would mistake each initial
    # for its own author.
    authors_list = parse_authors_list(row.get("Authors_List"))
    authors = row.get("Authors")
    if (not isinstance(authors, str) or not authors.strip()) and isinstance(authors_list, list):
        authors = "; ".join(str(a) for a in authors_list)
    return {
        "Authors": authors,
        "Authors_List": row.get("Authors_List"),
        "Journal": row.get("Journal"),
        "Title": row.get("Title"),
        "Year": row.get("Year"),
    }


def _title_block_key(title: object) -> str:
    words = _title_without_leading_article(normalize_text(title)).split()
    return " ".join(words[:6])


def _author_block_key(row: pd.Series) -> str:
    authors_list = parse_authors_list(row.get("Authors_List"))
    value = authors_list or row.get("Authors", "")
    def strip_accents(text: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    if isinstance(value, list):
        value = [strip_accents(a) for a in value]
    elif isinstance(value, str):
        value = strip_accents(value)
    return _first_author_token(value)


def _title_block_keys(title: object) -> set[str]:
    full = _title_without_leading_article(normalize_text(title))
    if not full:
        return set()
    words = full.split()
    return {"full:" + full, "prefix:" + " ".join(words[:6]),
            "prefix4:" + " ".join(words[:4]),
            "tokens:" + " ".join(sorted(set(full.split())))}


def _years_can_match(a: object, b: object, cfg: DedupConfig) -> bool:
    y1, ok1 = _read_year(a)
    y2, ok2 = _read_year(b)
    if not ok1 or not ok2:
        return False
    if y1 is None or y2 is None:
        return True
    return abs(y1 - y2) <= cfg.year_window


def _candidate_index_lookup(df: pd.DataFrame) -> tuple[dict[str, set[int]], dict[str, set[int]], set[int]]:
    author_blocks: dict[str, set[int]] = defaultdict(set)
    title_blocks: dict[str, set[int]] = defaultdict(set)
    unknown_author: set[int] = set()

    for idx, row in df.iterrows():
        author_key = _author_block_key(row)
        author_blocks[author_key].add(idx)
        if author_key == "unknown":
            unknown_author.add(idx)
        for title_key in _title_block_keys(row.get("Title")):
            title_blocks[title_key].add(idx)
    return author_blocks, title_blocks, unknown_author


def _candidate_indices(
    row: pd.Series,
    *,
    author_blocks: dict[str, set[int]],
    title_blocks: dict[str, set[int]],
    unknown_author: set[int],
) -> set[int]:
    author_key = _author_block_key(row)
    if author_key == "unknown":
        # Symmetric with the known-author side's inclusion of unknown rows.
        return set().union(*author_blocks.values()) if author_blocks else set()
    candidates = set(author_blocks.get(author_key, set())) | unknown_author
    for title_key in _title_block_keys(row.get("Title")):
        candidates.update(title_blocks.get(title_key, set()))
    return candidates


def _cluster_rows(
    df: pd.DataFrame,
    cfg: DedupConfig,
    make_cluster_id: Callable[[int, pd.Series], str],
    *,
    show_progress: bool,
    description: str,
    decisions: list[dict] | None = None,
    collisions: list[dict] | None = None,
) -> tuple[list[str], list[tuple[str, dict]]]:
    """Single-pass blocked clustering shared by dedup and canonicalization.

    Returns the per-row cluster id list and ``(cluster_id, representative
    dict)`` pairs in first-seen order. The representative is always the
    *first* member of its cluster.
    """
    cluster_ids: list[str | None] = [None] * len(df)
    representatives: list[tuple[str, dict]] = []
    used_ids: set[str] = set()
    author_blocks, title_blocks, unknown_author = _candidate_index_lookup(df)

    for i in iter_with_progress(
        list(range(len(df))),
        show_progress=show_progress,
        description=description,
        item_label=lambda idx: f"row {idx + 1}/{len(df)}",
    ):
        if cluster_ids[i] is not None:
            continue
        paper_i = _row_to_dict(df.iloc[i])
        cluster_id = make_cluster_id(i, df.iloc[i])
        # Distinct works can slug to the same id (e.g. two untitled citations
        # by the same first author and year); downstream tables join by id,
        # so collisions get a deterministic -2/-3… suffix like author ids do.
        if cluster_id in used_ids:
            n = 2
            while f"{cluster_id}-{n}" in used_ids:
                n += 1
            new_id = f"{cluster_id}-{n}"
            if collisions is not None:
                collisions.append({"row_index": i, "base_id": cluster_id, "assigned_id": new_id})
            cluster_id = new_id
        used_ids.add(cluster_id)
        cluster_ids[i] = cluster_id
        representatives.append((cluster_id, paper_i))

        candidate_js = _candidate_indices(
            df.iloc[i],
            author_blocks=author_blocks,
            title_blocks=title_blocks,
            unknown_author=unknown_author,
        )
        for j in sorted(idx for idx in candidate_js if idx > i):
            if cluster_ids[j] is not None:
                continue
            if not _years_can_match(df.iloc[i].get("Year"), df.iloc[j].get("Year"), cfg):
                continue
            assessment = _assess_work_match(paper_i, _row_to_dict(df.iloc[j]), cfg)
            if decisions is not None and assessment["decision"] != "reject":
                decisions.append({"left_index": i, "right_index": j,
                                  "representative_id": cluster_id, **assessment})
            if assessment["decision"] == "match":
                cluster_ids[j] = cluster_id
    # Every entry is filled: each row either joined a cluster or started one.
    return cluster_ids, representatives  # type: ignore[return-value]


def _compute_rings(ring0_ids: set[str], edges: pd.DataFrame) -> dict[str, int]:
    """BFS discovery depth from the ring-0 seed set over citation edges."""
    rings = {wid: 0 for wid in ring0_ids}
    out_edges: dict[str, set[str]] = defaultdict(set)
    for citing, cited in zip(edges["citing_id"], edges["cited_id"], strict=False):
        out_edges[str(citing)].add(str(cited))
    frontier = set(ring0_ids)
    depth = 0
    while frontier:
        depth += 1
        nxt = {c for w in frontier for c in out_edges.get(w, ()) if c not in rings}
        for c in nxt:
            rings[c] = depth
        frontier = nxt
    return rings


_WORK_COLUMNS = ["ring", "source_file", "Title", "Authors", "Authors_List", "Journal", "Year"]


def canonicalize_works(
    sources: pd.DataFrame,
    citations_raw: pd.DataFrame,
    cfg: DedupConfig | None = None,
    *,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Cluster sources + raw citations into canonical works.

    Sources occupy the head of the combined frame, so the single-pass
    first-match-wins loop guarantees: (a) two matching sources merge
    (duplicate PDFs), (b) a citation matching a source becomes that
    work, and (c) every cluster representative — hence its metadata —
    is the full-text side whenever one exists.

    Returns ``(works, edges, stats)``: ``works`` is indexed by ``id``
    with ``ring``/``source_file`` ahead of the bibliographic columns;
    ``edges`` is the deduplicated ``citing_id``/``cited_id`` table with
    self-loops removed; ``stats`` counts what was dropped or merged.
    """
    cfg = cfg or DedupConfig()
    require_columns(sources, ["id", "source_file", "Title", "Year"], artifact="sources")
    require_columns(citations_raw, ["Title", "Year", "citing_id"], artifact="citations_raw")
    for frame, name in ((sources, "sources"), (citations_raw, "citations_raw")):
        if "Authors" not in frame.columns and "Authors_List" not in frame.columns:
            raise ValueError(f"{name} is missing required column: Authors or Authors_List")

    src = sources.reset_index(drop=True)
    cit = citations_raw.reset_index(drop=True)
    n_src = len(src)

    if not n_src and cit.empty:
        empty_works = pd.DataFrame(columns=_WORK_COLUMNS)
        empty_works.index.name = "id"
        empty_edges = pd.DataFrame(columns=["citing_id", "cited_id"])
        return empty_works, empty_edges, {
            "n_self_loops_dropped": 0,
            "n_source_duplicates_merged": 0,
            "source_cluster_ids": [],
            "citation_cluster_ids": [],
            "decisions": [],
            "id_collisions": [],
        }

    combined = pd.concat([src, cit], ignore_index=True, sort=False)

    def _cluster_id(i: int, row: pd.Series) -> str:
        if i < n_src:
            return str(row["id"])
        return make_work_id(
            row.get("Authors_List") or row.get("Authors", ""),
            row.get("Year"),
            row.get("Title", ""),
        )

    decisions: list[dict] = []
    collisions: list[dict] = []
    cluster_ids, representatives = _cluster_rows(
        combined,
        cfg,
        _cluster_id,
        show_progress=show_progress,
        description="Canonicalizing works",
        decisions=decisions,
        collisions=collisions,
    )

    source_canonical = {str(src.iloc[i]["id"]): cluster_ids[i] for i in range(n_src)}
    ring0_ids = set(cluster_ids[:n_src])

    edges = pd.DataFrame(
        {
            "citing_id": [
                source_canonical.get(str(c), str(c)) for c in cit["citing_id"]
            ],
            "cited_id": cluster_ids[n_src:],
        }
    )
    n_before = len(edges)
    edges = edges[edges["citing_id"] != edges["cited_id"]]
    n_self_loops = n_before - len(edges)
    edges = edges.drop_duplicates().reset_index(drop=True)

    source_file_by_cluster: dict[str, str] = {}
    for i in range(n_src):
        source_file_by_cluster.setdefault(cluster_ids[i], str(src.iloc[i]["source_file"]))

    rings = _compute_rings(ring0_ids, edges)
    records = []
    for cluster_id, rep in representatives:
        record = dict(rep)
        record["id"] = cluster_id
        # A work absent from `rings` can only be an orphan whose sole
        # inbound edge was a dropped self-loop variant; ring 1 is the
        # defensively correct depth for it.
        record["ring"] = rings.get(cluster_id, 1)
        record["source_file"] = source_file_by_cluster.get(cluster_id, "")
        records.append(record)
    works = pd.DataFrame(records).set_index("id")[_WORK_COLUMNS]

    stats = {
        "n_self_loops_dropped": int(n_self_loops),
        "n_source_duplicates_merged": int(n_src - len(ring0_ids)),
        # Per-citation-row canonical id, aligned with citations_raw order.
        # Lets callers (e.g. the report's merge audit) reconstruct which
        # raw citation events merged into which work.
        "citation_cluster_ids": [str(c) for c in cluster_ids[n_src:]],
        "source_cluster_ids": [str(c) for c in cluster_ids[:n_src]],
        "decisions": decisions,
        "id_collisions": collisions,
    }
    logger.info(
        "Canonicalized %d sources + %d citations into %d works (%d edges)",
        n_src,
        len(cit),
        len(works),
        len(edges),
    )
    return works, edges, stats
