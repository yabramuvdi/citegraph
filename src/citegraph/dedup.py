"""Fuzzy deduplication of bibliographic references.

Cleaned-up port of the algorithm originally in ``label_papers_old.py``.
We compute a weighted similarity over title / authors / journal and then
perform a single-pass clustering: each row is compared against the
representatives of existing clusters; the first cluster that scores above
the threshold (and has a year within ``year_window``) wins. Otherwise the
row starts a new cluster.

Usage
-----

>>> from citegraph.dedup import dedup_references, DedupConfig
>>> df_dedup, mapping = dedup_references(df_raw, DedupConfig())

``df_dedup`` is a deduplicated DataFrame indexed by stable reference IDs;
``mapping`` is a Series aligned with the input that gives the cluster ID
for each row.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from rapidfuzz.fuzz import token_set_ratio

from citegraph._progress import iter_with_progress
from citegraph.ids import _first_author_token, make_reference_id, make_work_id
from citegraph.io import require_columns

logger = logging.getLogger(__name__)


_NON_ALNUM = re.compile(r"[^a-zA-Z0-9\s]")


def normalize_text(text: object) -> str:
    """Lowercase, strip, and remove non-alphanumeric characters.

    Lists of authors are joined with commas before normalisation.
    """
    if isinstance(text, list):
        text = ", ".join(str(t) for t in text)
    if not isinstance(text, str):
        return ""
    return _NON_ALNUM.sub("", text.lower().strip())


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
    if value is None or value == "":
        return None, True
    try:
        year = int(value)
    except (TypeError, ValueError):
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
    # token_set_ratio (rather than plain Levenshtein ratio or token_sort_ratio)
    # tolerates the two most common citation-style mismatches: a record listing
    # only the first author ("Bowles, S.") vs the full list ("Samuel Bowles, ..."),
    # and a title with a subtitle ("...: presidential address, APSA") vs the
    # bare title. Both patterns scored 77-85 with the previous scorer mix and
    # slipped under the 85 threshold; token_set_ratio scores them ~100 while
    # still rejecting genuinely different papers that share only a few tokens.
    title_score = token_set_ratio(
        normalize_text(paper1.get("Title")), normalize_text(paper2.get("Title"))
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

    return weighted >= cfg.threshold and year_ok


def _row_to_dict(row: pd.Series) -> dict:
    # Authors_List is preserved alongside the comma-joined Authors so that
    # downstream stages (author normalization in particular) can recover
    # individual author strings without having to split the joined form —
    # comma-splitting "Smith, J., García, A." would mistake each initial
    # for its own author.
    return {
        "Authors": row.get("Authors"),
        "Authors_List": row.get("Authors_List"),
        "Journal": row.get("Journal"),
        "Title": row.get("Title"),
        "Year": row.get("Year"),
    }


def _title_block_key(title: object) -> str:
    words = normalize_text(title).split()
    return " ".join(words[:6])


def _author_block_key(row: pd.Series) -> str:
    return _first_author_token(row.get("Authors_List") or row.get("Authors", ""))


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
        title_key = _title_block_key(row.get("Title"))
        author_blocks[author_key].add(idx)
        if author_key == "unknown":
            unknown_author.add(idx)
        if title_key:
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
    title_key = _title_block_key(row.get("Title"))
    candidates = set(author_blocks.get(author_key, set())) | unknown_author
    if title_key:
        candidates.update(title_blocks.get(title_key, set()))
    return candidates


def _cluster_rows(
    df: pd.DataFrame,
    cfg: DedupConfig,
    make_cluster_id: Callable[[int, pd.Series], str],
    *,
    show_progress: bool,
    description: str,
) -> tuple[list[str], list[tuple[str, dict]]]:
    """Single-pass blocked clustering shared by dedup and canonicalization.

    Returns the per-row cluster id list and ``(cluster_id, representative
    dict)`` pairs in first-seen order. The representative is always the
    *first* member of its cluster.
    """
    cluster_ids: list[str | None] = [None] * len(df)
    representatives: list[tuple[str, dict]] = []
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
            if compare_papers(paper_i, _row_to_dict(df.iloc[j]), cfg):
                cluster_ids[j] = cluster_id
    # Every entry is filled: each row either joined a cluster or started one.
    return cluster_ids, representatives  # type: ignore[return-value]


def dedup_references(
    df: pd.DataFrame,
    cfg: DedupConfig | None = None,
    *,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Cluster duplicate references and return canonical rows + mapping.

    Parameters
    ----------
    df:
        DataFrame with at least ``Title``, ``Authors``, ``Journal`` and
        ``Year`` columns (the output of :mod:`citegraph.extract_references`).
    cfg:
        :class:`DedupConfig` of weights/thresholds. Defaults match the
        original ``label_papers_old.py`` settings.

    Returns
    -------
    canonical_df:
        One row per cluster, indexed by a stable cluster id (``r-...``).
        The representative is the *first* member of each cluster.
    mapping:
        ``pd.Series`` with the same index as ``df`` mapping each input row
        to its cluster id.
    """
    cfg = cfg or DedupConfig()
    require_columns(df, ["Title", "Year"], artifact="dedup input")
    if "Authors" not in df.columns and "Authors_List" not in df.columns:
        raise ValueError("dedup input is missing required column: Authors or Authors_List")
    if df.empty:
        empty = df.copy()
        empty["id"] = pd.Series(dtype=str)
        return empty.set_index("id"), pd.Series([], dtype=str, index=df.index)

    df = df.reset_index(drop=True)

    def _ref_cluster_id(_i: int, row: pd.Series) -> str:
        return make_reference_id(
            row.get("Authors_List") or row.get("Authors", ""),
            row.get("Year"),
            row.get("Title", ""),
        )

    cluster_id_list, representatives = _cluster_rows(
        df,
        cfg,
        _ref_cluster_id,
        show_progress=show_progress,
        description="Deduplicating references",
    )
    mapping = pd.Series(cluster_id_list, index=df.index, name="cited_id", dtype=object)

    canonical_records = []
    for cluster_id, rep in representatives:
        record = dict(rep)
        record["id"] = cluster_id
        canonical_records.append(record)
    canonical_df = pd.DataFrame(canonical_records).set_index("id")

    logger.info(
        "Deduplicated %d references into %d canonical entries",
        len(df),
        len(canonical_df),
    )
    return canonical_df, mapping


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
    combined = pd.concat([src, cit], ignore_index=True, sort=False)

    def _cluster_id(i: int, row: pd.Series) -> str:
        if i < n_src:
            return str(row["id"])
        return make_work_id(
            row.get("Authors_List") or row.get("Authors", ""),
            row.get("Year"),
            row.get("Title", ""),
        )

    cluster_ids, representatives = _cluster_rows(
        combined,
        cfg,
        _cluster_id,
        show_progress=show_progress,
        description="Canonicalizing works",
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
    }
    logger.info(
        "Canonicalized %d sources + %d citations into %d works (%d edges)",
        n_src,
        len(cit),
        len(works),
        len(edges),
    )
    return works, edges, stats
