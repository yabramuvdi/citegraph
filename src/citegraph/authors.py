"""Author parsing and corpus-level normalization.

The pipeline extracts author *strings* per reference, but a citation graph
is much more useful when those strings are clustered into canonical
authors so the user can answer "who is the most cited person across the
corpus?" or "show me every reference that cites Cárdenas".

Two layers live here:

1. :func:`parse_author` turns one raw string ("Cárdenas, J.-C.",
   "Adele Diamond", "García Márquez, Gabriel") into a structured
   :class:`ParsedAuthor` — surname / given-names / initials / suffix —
   with a diacritic-stripped, lowercased ``surname_norm`` for blocking.

2. :func:`normalize_authors` clusters every parsed author across the
   corpus into canonical author records.  The default mode is
   precision-first: it merges initial-only records into a full-name
   anchor *only* when there's exactly one matching full first name in
   the surname block, otherwise the record is held aside as ambiguous.
   ``merge_mode="loose"`` collapses by ``(surname, first_initial)``
   regardless of full-name evidence — useful when you want recall over
   precision.

OpenAlex / ORCID identifiers (collected by the optional enrichment
stage; see ``enrich.py``) are treated as ground truth when present:
records that share an OpenAlex author id are merged regardless of
string evidence, and records with different OpenAlex ids are kept
separate even when their names look identical.

Hand-curated overrides (``aliases``) are the final escape hatch —
two-column CSV mapping arbitrary cluster ids to a canonical id, applied
after clustering so the user can fix anything the algorithm gets wrong.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from slugify import slugify

from citegraph.io import parse_authors_list

logger = logging.getLogger(__name__)


# Surname particles that are conventionally lowercased and form part of
# the surname rather than the given names ("García Márquez", "de la Cruz",
# "von Neumann"). When a name is given comma-first ("Márquez, Gabriel
# García") the parser already trusts the comma; this list only matters
# when there is no comma and we have to guess where the surname ends.
_PARTICLES = frozenset(
    {
        "de", "del", "la", "las", "los", "da", "das", "do", "dos",
        "von", "van", "der", "den", "ten", "ter", "le", "lo",
        "di", "du", "el", "al", "bin", "ibn", "abu",
    }
)

# Honorific suffixes that should be stripped off the surname before
# comparing — "Jr.", "III", etc.  Kept on the parsed record for display
# but not part of surname_norm.
_SUFFIX_PATTERN = re.compile(
    r"\b(Jr\.?|Sr\.?|II|III|IV|PhD\.?|M\.?D\.?|Esq\.?)\b",
    re.IGNORECASE,
)

# A run of letters within a token (used to decide whether a token is
# "essentially an initial": every contiguous letter run is one letter).
_LETTER_RUN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")


def _is_initial_only_token(token: str) -> bool:
    """True when every alphabetic run inside ``token`` is a single letter.

    Catches all the citation-style spellings of initials we see in the
    wild: ``"J."``, ``"J.C."``, ``"J. C."`` (each token), ``"J.-C."``.
    Multi-letter words like ``"Juan"`` correctly come back ``False``.
    """
    runs = _LETTER_RUN.findall(token)
    return bool(runs) and all(len(r) == 1 for r in runs)

# Stuff the LLM occasionally leaves in author strings — footnote markers,
# affiliation glyphs, stray asterisks, leading "and"/"&", trailing
# punctuation. Stripped before parsing.
_NOISE = re.compile(r"[†‡§¶*★]+|^\s*(and|&)\s+", re.IGNORECASE)


def _strip_diacritics(text: str) -> str:
    """NFD-decompose and drop combining marks: 'Cárdenas' -> 'Cardenas'."""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def _norm_surname(surname: str) -> str:
    """Lowercase + diacritic-strip + drop non-alphabetic characters.

    Spaces are kept so multi-word surnames ('garcia marquez') can be
    compared by full normalised form. The blocking step uses this as the
    bucket key, so it MUST be deterministic across runs.
    """
    s = _strip_diacritics(surname).lower().strip()
    # Keep letters and internal spaces; drop apostrophes, hyphens, etc.
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _initials_from_token(token: str) -> str:
    """Extract initial letters from a given-name token.

    Handles every common spelling of multi-initial citations:
    ``'J.'`` -> ``'J'`` ; ``'J.C.'`` -> ``'JC'`` ; ``'J.-C.'`` -> ``'JC'`` ;
    ``'Juan-Camilo'`` -> ``'JC'`` ; ``'Juan'`` -> ``'J'`` ; ``''`` -> ``''``.

    The implementation walks every contiguous letter run and takes its
    first letter — so it doesn't matter whether the segments are joined
    by periods, hyphens, or whitespace.
    """
    if not token:
        return ""
    return "".join(run[0].upper() for run in _LETTER_RUN.findall(token))


@dataclass(frozen=True)
class ParsedAuthor:
    """Structured representation of a single author string.

    Attributes
    ----------
    raw:
        The original string as it appeared in the citation, untouched.
    surname:
        The surname with original casing and diacritics, used for display.
    surname_norm:
        Diacritic-stripped, lowercased, alphabetic-only surname. The
        clustering step blocks on this.
    given_names:
        List of given-name tokens in citation order ("Juan", "Camilo")
        or initials ("J.", "C."). Empty when only a surname was given.
    initials:
        Concatenated first letters of every given-name token, uppercased.
        "Juan-Camilo" -> "JC"; "J. C." -> "JC".
    has_full_first:
        ``True`` if the first given-name token is a real word, not just
        an initial. The clustering step uses this to anchor clusters.
    suffix:
        "Jr.", "III", etc. Preserved for display; ignored by matching.
    """

    raw: str
    surname: str
    surname_norm: str
    given_names: tuple[str, ...]
    initials: str
    has_full_first: bool
    suffix: str | None = None

    @property
    def first_initial(self) -> str:
        return self.initials[:1]

    @property
    def first_given_norm(self) -> str:
        """Normalised form of the first given-name token, '' if only initial."""
        if not self.has_full_first or not self.given_names:
            return ""
        return _strip_diacritics(self.given_names[0]).lower().strip(".-")

    def display(self) -> str:
        """Best human-readable form of this name (surname-first)."""
        if self.given_names:
            given = " ".join(self.given_names)
            return f"{self.surname}, {given}"
        return self.surname


def parse_author(raw: str) -> ParsedAuthor | None:
    """Parse a single raw author string into a :class:`ParsedAuthor`.

    Returns ``None`` for strings that contain no usable surname (empty
    string, pure punctuation, "et al.", "Anonymous", etc.). Callers
    should treat ``None`` as "skip this author".
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None

    # Drop common noise tokens that signal "not an author".
    bare = _strip_diacritics(s).lower()
    if bare in {"et al", "et al.", "anonymous", "anon", "n.a.", "na", "unknown"}:
        return None

    # Strip footnote/affiliation glyphs and a leading 'and'/'&'.
    s = _NOISE.sub(" ", s).strip()
    s = re.sub(r"\s+", " ", s)
    if not s:
        return None

    # Pull off honorific suffix if present.
    suffix: str | None = None
    m = _SUFFIX_PATTERN.search(s)
    if m:
        suffix = m.group(1)
        s = (s[: m.start()] + s[m.end():]).strip(" ,")

    # Two parsing branches: comma-style (Surname, Given) vs space-style.
    if "," in s:
        surname_part, given_part = s.split(",", 1)
        surname = surname_part.strip()
        given_str = given_part.strip()
    else:
        surname, given_str = _split_no_comma(s)

    surname = surname.strip(" .-")
    if not surname or not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", surname):
        return None
    # A "surname" whose every letter run is a single letter is almost
    # certainly a misparsed initial ("J.", "X. Y."). Real surnames have
    # at least one multi-letter run. Rejecting here prevents the mega-
    # clusters of one-letter surnames that arise when an upstream stage
    # has comma-split "Smith, J., García, A." into separate fragments.
    if _is_initial_only_token(surname):
        return None

    given_tokens = _tokenize_given(given_str)
    initials = "".join(_initials_from_token(t) for t in given_tokens)
    has_full_first = bool(given_tokens) and not _is_initial_only_token(given_tokens[0])

    return ParsedAuthor(
        raw=raw,
        surname=surname,
        surname_norm=_norm_surname(surname),
        given_names=tuple(given_tokens),
        initials=initials,
        has_full_first=has_full_first,
        suffix=suffix,
    )


def _split_no_comma(s: str) -> tuple[str, str]:
    """Split a name with no comma into (surname, given-string).

    Heuristic: the last token is the surname, *unless* the second-to-last
    token is a particle ('de', 'van', 'García' before 'Márquez'…), in
    which case the surname extends backwards to include it. Compound
    surnames without particles ('García Márquez' alone, no comma) are
    ambiguous — we can't tell from the string alone, and we conservatively
    take only the last token as the surname.
    """
    tokens = s.split()
    if not tokens:
        return "", ""
    if len(tokens) == 1:
        return tokens[0], ""

    surname_idx = len(tokens) - 1
    # Walk backwards while the preceding token looks like a particle.
    while surname_idx > 0:
        prev = tokens[surname_idx - 1].strip(".-").lower()
        if prev in _PARTICLES:
            surname_idx -= 1
        else:
            break

    surname = " ".join(tokens[surname_idx:])
    given = " ".join(tokens[:surname_idx])
    return surname, given


def _tokenize_given(given_str: str) -> list[str]:
    """Split a given-name string into tokens, preserving 'J.-C.' as one."""
    if not given_str:
        return []
    # Split on whitespace; keep hyphenated initials intact.
    parts = [p for p in given_str.split() if p]
    return parts


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


@dataclass
class AuthorOccurrence:
    """One concrete appearance of an author in the corpus.

    The clustering output is grouped by canonical author id; each
    occurrence retains enough back-pointers to reconstruct the
    citation edge ("author X cited in reference R by paper P").
    """

    parsed: ParsedAuthor
    record_id: str            # 'r-…' for a reference, 'p-…' for a source paper
    record_kind: str          # 'reference' or 'paper'
    position: int             # 0-based index in the author list
    co_author_keys: tuple[str, ...] = ()  # surname_norm of the other authors on the same record
    citing_paper_id: str | None = None    # only meaningful when record_kind=='reference'
    openalex_id: str | None = None
    orcid: str | None = None


@dataclass
class AuthorClusterConfig:
    """Tunable knobs for :func:`normalize_authors`."""

    # 'strict' (default) — precision-first. Merges 'Diamond, A.' into
    # 'Diamond, Adele' only when 'Adele' is the *only* full first name
    # observed for surname 'Diamond' in this corpus. Otherwise the
    # initial-only record is held aside in its own cluster and flagged.
    #
    # 'loose' — recall-first. Collapses every (surname, first_initial)
    # combination into one cluster regardless of full-name disagreement.
    merge_mode: str = "strict"


@dataclass
class AuthorCluster:
    """One canonical author after corpus-wide normalization."""

    id: str
    display_name: str
    surname: str
    surname_norm: str
    canonical_given: str
    initials: str
    openalex_id: str | None
    orcid: str | None
    occurrences: list[AuthorOccurrence] = field(default_factory=list)
    review_reason: str | None = None  # populated for low-confidence clusters

    @property
    def n_occurrences(self) -> int:
        return len(self.occurrences)


def _make_author_id(surname_norm: str, given_token: str, n: int = 0) -> str:
    """Stable slug-style id for an author cluster.

    ``n`` is appended only to disambiguate within a corpus when the same
    (surname, first-initial-or-name) tuple genuinely refers to two
    distinct people that the algorithm split apart.
    """
    base = f"a-{slugify(surname_norm) or 'unknown'}-{slugify(given_token) or 'x'}"
    if n:
        base = f"{base}-{n}"
    return base


def _canonical_given_for_cluster(occs: list[AuthorOccurrence]) -> tuple[str, str]:
    """Pick a (display-friendly given string, slug-friendly token) pair.

    Prefer the longest full first name actually seen in the cluster. If
    the cluster only has initials, use the canonical initial sequence.
    """
    full_firsts = [
        o.parsed.given_names[0]
        for o in occs
        if o.parsed.has_full_first and o.parsed.given_names
    ]
    if full_firsts:
        canonical = max(full_firsts, key=len)
        return canonical, _strip_diacritics(canonical).lower().strip(".-")
    # Fallback to the longest initial sequence we saw.
    inits = max((o.parsed.initials for o in occs), key=len, default="")
    return inits, inits.lower() or "x"


def _has_coauthor_overlap(
    occ: AuthorOccurrence,
    cluster: list[AuthorOccurrence],
) -> bool:
    """True if ``occ`` shares a co-author surname with any cluster member."""
    if not occ.co_author_keys:
        return False
    cluster_keys: set[str] = set()
    for c in cluster:
        cluster_keys.update(c.co_author_keys)
    return bool(set(occ.co_author_keys) & cluster_keys)


def _has_external_id(cluster: list[AuthorOccurrence]) -> bool:
    return any(o.openalex_id or o.orcid for o in cluster)


def _full_first_norms(cluster: list[AuthorOccurrence]) -> set[str]:
    return {
        o.parsed.first_given_norm
        for o in cluster
        if o.parsed.has_full_first and o.parsed.first_given_norm
    }


def _first_initials(cluster: list[AuthorOccurrence]) -> set[str]:
    return {o.parsed.first_initial.lower() for o in cluster if o.parsed.first_initial}


def _merge_unidentified_into_external_clusters(
    clusters: list[list[AuthorOccurrence]],
) -> list[list[AuthorOccurrence]]:
    """Use enriched full-name clusters as anchors for no-id name variants.

    OpenAlex / ORCID-bearing clusters remain authoritative: different external
    ids are never merged together. This pass only moves clusters with no
    external id into exactly one compatible external-id cluster.
    """
    external_clusters = [c for c in clusters if _has_external_id(c)]
    if not external_clusters:
        return clusters

    merged: list[list[AuthorOccurrence]] = list(external_clusters)
    unresolved: list[list[AuthorOccurrence]] = []
    for cluster in clusters:
        if _has_external_id(cluster):
            continue

        full_norms = _full_first_norms(cluster)
        if full_norms:
            candidates = [
                ext for ext in external_clusters
                if full_norms & _full_first_norms(ext)
            ]
        else:
            initials = _first_initials(cluster)
            candidates = [
                ext for ext in external_clusters
                if initials & _first_initials(ext)
            ]

        target: list[AuthorOccurrence] | None = None
        if len(candidates) == 1:
            target = candidates[0]
        elif len(candidates) > 1:
            with_overlap = [
                ext for ext in candidates
                if any(_has_coauthor_overlap(o, ext) for o in cluster)
            ]
            if len(with_overlap) == 1:
                target = with_overlap[0]

        if target is None:
            unresolved.append(cluster)
        else:
            target.extend(cluster)

    merged.extend(unresolved)
    return merged


def _cluster_block(
    occs: list[AuthorOccurrence],
    cfg: AuthorClusterConfig,
) -> list[list[AuthorOccurrence]]:
    """Cluster all occurrences in a single surname block.

    Returns a list of clusters (each a list of occurrences). The
    algorithm is precision-first by default — see the module docstring.
    """
    # OpenAlex / ORCID ids are ground truth and override everything else.
    # Phase 1: pre-merge by external id where present.
    by_external: dict[str, list[AuthorOccurrence]] = defaultdict(list)
    no_external: list[AuthorOccurrence] = []
    for o in occs:
        key = o.openalex_id or o.orcid
        if key:
            by_external[key].append(o)
        else:
            no_external.append(o)

    clusters: list[list[AuthorOccurrence]] = list(by_external.values())

    if cfg.merge_mode == "loose":
        # Bucket purely by (surname_norm, first_initial) — ignore everything
        # else. Even external-id clusters get merged into the bucket.
        buckets: dict[tuple[str, str], list[AuthorOccurrence]] = defaultdict(list)
        for cluster in clusters:
            for o in cluster:
                buckets[(o.parsed.surname_norm, o.parsed.first_initial)].append(o)
        for o in no_external:
            buckets[(o.parsed.surname_norm, o.parsed.first_initial)].append(o)
        return list(buckets.values())

    # Strict mode --------------------------------------------------------
    # Phase 2: anchor clusters from full-first-name records.
    anchors: dict[str, list[AuthorOccurrence]] = defaultdict(list)
    for o in no_external:
        if o.parsed.has_full_first:
            anchors[o.parsed.first_given_norm].append(o)

    # Each anchor key becomes one cluster.
    anchor_clusters: dict[str, list[AuthorOccurrence]] = {
        k: list(v) for k, v in anchors.items()
    }

    # Phase 3: place initial-only records into anchor clusters.
    ambiguous: list[list[AuthorOccurrence]] = []
    for o in no_external:
        if o.parsed.has_full_first:
            continue
        first_init = o.parsed.first_initial.lower()
        if not first_init:
            # Pure surname with no given-name info at all — stand-alone cluster.
            ambiguous.append([o])
            continue
        candidate_keys = [k for k in anchor_clusters if k.startswith(first_init)]
        if len(candidate_keys) == 1:
            anchor_clusters[candidate_keys[0]].append(o)
        elif len(candidate_keys) > 1:
            # Try co-author tiebreak.
            with_overlap = [
                k for k in candidate_keys
                if _has_coauthor_overlap(o, anchor_clusters[k])
            ]
            if len(with_overlap) == 1:
                anchor_clusters[with_overlap[0]].append(o)
            else:
                ambiguous.append([o])
        else:
            # No anchor with that initial; bucket initial-only records together.
            # (They'll be merged with each other and stand as their own cluster.)
            ambiguous.append([o])

    # Merge initial-only ambiguous records that share the same initials sequence.
    init_buckets: dict[str, list[AuthorOccurrence]] = defaultdict(list)
    final_singletons: list[list[AuthorOccurrence]] = []
    for cluster in ambiguous:
        if len(cluster) == 1 and not cluster[0].parsed.has_full_first:
            init_buckets[cluster[0].parsed.initials.lower()].append(cluster[0])
        else:
            final_singletons.append(cluster)

    clusters.extend(anchor_clusters.values())
    clusters.extend(init_buckets.values())
    clusters.extend(final_singletons)
    return _merge_unidentified_into_external_clusters(clusters)


def normalize_authors(
    *,
    references: pd.DataFrame,
    papers: pd.DataFrame | None = None,
    enriched_references: pd.DataFrame | None = None,
    cfg: AuthorClusterConfig | None = None,
    aliases: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Cluster authors across the corpus.

    Parameters
    ----------
    references:
        Deduplicated references DataFrame (the output of
        :func:`citegraph.dedup.dedup_references`). Must be indexed by ``id``
        and contain an ``Authors_List`` column (or ``Authors``).
    papers:
        Optional source-papers DataFrame. When provided, the source-paper
        authors are clustered alongside the reference authors so the
        result names every person in the corpus.
    enriched_references:
        Optional output of :func:`citegraph.enrich.enrich_references`.
        If present and contains the ``OpenAlex_Authors`` column (a list
        of ``{display_name, openalex_id, orcid}`` dicts per row), those
        identifiers are attached to the matching reference authors so
        OpenAlex acts as ground truth.
    cfg:
        Tunable :class:`AuthorClusterConfig`.
    aliases:
        Optional ``{cluster_id: canonical_id}`` mapping. Applied after
        clustering: every cluster whose id is a key in ``aliases`` is
        merged into the cluster keyed by the value. Lets the user fix
        the algorithm's mistakes without re-running.

    Returns
    -------
    authors_df:
        One row per canonical author, indexed by ``id``.
    citations_df:
        Edge table linking each canonical author to every record (paper
        or reference) they appear on, with the citing-paper id when the
        record is a reference.
    review:
        List of dicts describing low-confidence clusters that the user
        may want to inspect.
    """
    cfg = cfg or AuthorClusterConfig()
    aliases = aliases or {}

    occurrences = _collect_occurrences(
        references=references,
        papers=papers,
        enriched_references=enriched_references,
    )

    # Block by surname_norm and cluster within each block.
    blocks: dict[str, list[AuthorOccurrence]] = defaultdict(list)
    for o in occurrences:
        blocks[o.parsed.surname_norm].append(o)

    raw_clusters: list[list[AuthorOccurrence]] = []
    for _surname, block_occs in blocks.items():
        raw_clusters.extend(_cluster_block(block_occs, cfg))

    # Build AuthorCluster records with stable ids.
    used_ids: dict[str, int] = defaultdict(int)
    clusters: list[AuthorCluster] = []
    for occs in raw_clusters:
        if not occs:
            continue
        canonical_given, given_token = _canonical_given_for_cluster(occs)
        surname_norm = occs[0].parsed.surname_norm
        surname_display = occs[0].parsed.surname
        # Pick the prettiest surname casing actually seen.
        for o in occs:
            if any(c.isupper() for c in o.parsed.surname):
                surname_display = o.parsed.surname
                break
        base_id = _make_author_id(surname_norm, given_token)
        cid = base_id if used_ids[base_id] == 0 else _make_author_id(
            surname_norm, given_token, used_ids[base_id]
        )
        used_ids[base_id] += 1

        oa_id = next((o.openalex_id for o in occs if o.openalex_id), None)
        orcid = next((o.orcid for o in occs if o.orcid), None)
        initials = max((o.parsed.initials for o in occs), key=len, default="")

        display = (
            f"{canonical_given} {surname_display}".strip()
            if canonical_given and not canonical_given.isupper()
            else f"{surname_display}, {canonical_given}".strip(", ")
        )

        clusters.append(
            AuthorCluster(
                id=cid,
                display_name=display,
                surname=surname_display,
                surname_norm=surname_norm,
                canonical_given=canonical_given,
                initials=initials,
                openalex_id=oa_id,
                orcid=orcid,
                occurrences=occs,
                review_reason=_review_reason(occs, cfg),
            )
        )

    # Apply user-curated aliases: merge cluster B into cluster A.
    if aliases:
        clusters = _apply_aliases(clusters, aliases)

    authors_df = _clusters_to_authors_df(clusters)
    citations_df = _clusters_to_citations_df(clusters)
    review = [
        {
            "author_id": c.id,
            "display_name": c.display_name,
            "surname": c.surname_norm,
            "n_occurrences": c.n_occurrences,
            "raw_strings": sorted({o.parsed.raw for o in c.occurrences}),
            "reason": c.review_reason,
        }
        for c in clusters
        if c.review_reason
    ]

    logger.info(
        "Author normalization: %d occurrences → %d clusters (%d flagged for review)",
        len(occurrences),
        len(clusters),
        len(review),
    )
    return authors_df, citations_df, review


def _collect_occurrences(
    *,
    references: pd.DataFrame,
    papers: pd.DataFrame | None,
    enriched_references: pd.DataFrame | None,
) -> list[AuthorOccurrence]:
    """Walk references + papers and emit one AuthorOccurrence per author.

    Co-author surname-norm keys are computed per record so the clustering
    step can use them as a tiebreak signal.
    """
    occurrences: list[AuthorOccurrence] = []

    # Build a per-reference enrichment map: ref_id -> [(display, oa_id, orcid), ...]
    enrich_map: dict[str, list[dict]] = {}
    if enriched_references is not None and not enriched_references.empty:
        if "OpenAlex_Authors" in enriched_references.columns:
            for ref_id, row in enriched_references.iterrows():
                authors = row.get("OpenAlex_Authors")
                if isinstance(authors, list):
                    enrich_map[str(ref_id)] = authors

    if not references.empty:
        for ref_id, row in references.iterrows():
            authors_list = _row_authors(row)
            parsed = [parsed_or_none for parsed_or_none in (parse_author(a) for a in authors_list)]
            co_keys = tuple(p.surname_norm for p in parsed if p is not None)
            enrich_authors = enrich_map.get(str(ref_id), [])
            for pos, p in enumerate(parsed):
                if p is None:
                    continue
                # Best-effort match between our parsed author and the
                # enrichment list: same position when lengths match,
                # otherwise surname-fuzzy. Falls back to no enrichment.
                oa_id, orcid = _match_enrichment(p, pos, parsed, enrich_authors)
                occurrences.append(
                    AuthorOccurrence(
                        parsed=p,
                        record_id=str(ref_id),
                        record_kind="reference",
                        position=pos,
                        co_author_keys=tuple(k for i, k in enumerate(co_keys) if i != pos and k),
                        citing_paper_id=None,
                        openalex_id=oa_id,
                        orcid=orcid,
                    )
                )

    if papers is not None and not papers.empty:
        for _idx, row in papers.iterrows():
            paper_id = row.get("id") if "id" in papers.columns else None
            if not paper_id:
                continue
            authors_list = _row_authors(row)
            parsed = [parse_author(a) for a in authors_list]
            co_keys = tuple(p.surname_norm for p in parsed if p is not None)
            for pos, p in enumerate(parsed):
                if p is None:
                    continue
                occurrences.append(
                    AuthorOccurrence(
                        parsed=p,
                        record_id=str(paper_id),
                        record_kind="paper",
                        position=pos,
                        co_author_keys=tuple(k for i, k in enumerate(co_keys) if i != pos and k),
                        citing_paper_id=str(paper_id),
                        openalex_id=None,
                        orcid=None,
                    )
                )

    return occurrences


def _row_authors(row: pd.Series) -> list[str]:
    """Pull a list of author strings from a DataFrame row.

    ``Authors_List`` is preferred (the canonical schema field) and round-
    trips through CSV as a Python repr that we parse back with
    :func:`ast.literal_eval`. When only the comma-joined ``Authors``
    string is available — e.g. a ``references.csv`` written before
    Authors_List was preserved through dedup — we fall back to
    :func:`_split_joined_authors`, which is safe against the
    ``"Smith, J., García, A."`` pattern.
    """
    val = row.get("Authors_List")
    if isinstance(val, list):
        return [str(a) for a in val if a]
    if isinstance(val, str):
        parsed = parse_authors_list(val)
        if len(parsed) > 1:
            return parsed
        if parsed:
            return _split_joined_authors(parsed[0])
    fallback = row.get("Authors")
    if isinstance(fallback, str) and fallback.strip():
        return _split_joined_authors(fallback)
    return []


def _split_joined_authors(s: str) -> list[str]:
    """Split a delimiter-joined author string into individual names.

    Prefers ``;`` (unambiguous) when present. Otherwise splits on ``,``
    and re-glues obvious ``Surname, Given`` pairs so that
    ``"Smith, J., García, A."`` and ``"Smith, John, García, Ana"`` become
    two authors rather than four fragments. Strings without that pattern
    (e.g. ``"John Smith, Mary Jones"``) pass through unchanged.
    """
    s = s.strip()
    if not s:
        return []
    if ";" in s:
        return [p.strip() for p in s.split(";") if p.strip()]

    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: list[str] = []
    i = 0
    while i < len(parts):
        cur = parts[i]
        nxt = parts[i + 1] if i + 1 < len(parts) else None
        if (
            nxt is not None
            and not _is_initial_only_token(cur)
            and (
                _is_initial_only_token(nxt)
                or _looks_like_surname_given_pair(cur, nxt)
            )
        ):
            out.append(f"{cur}, {nxt}")
            i += 2
        else:
            out.append(cur)
            i += 1
    return out


def _looks_like_surname_given_pair(surname_part: str, given_part: str) -> bool:
    """Heuristic for comma-joined fallback strings.

    This intentionally handles only high-confidence cases: one-word
    surnames (``Smith, John``) and particle compounds
    (``de la Cruz, Juan``). Multi-word chunks such as ``Talbot Page`` are
    likely already in first-last order, so pairing them with the next
    comma part would create a false author.
    """
    surname_words = surname_part.split()
    given_words = given_part.split()
    if not surname_words or not given_words or len(given_words) > 2:
        return False
    if len(surname_words) == 1:
        return True
    return any(w.strip(".-").lower() in _PARTICLES for w in surname_words[:-1])


def _match_enrichment(
    parsed: ParsedAuthor,
    pos: int,
    all_parsed: Iterable[ParsedAuthor | None],
    enrich_authors: list[dict],
) -> tuple[str | None, str | None]:
    """Match one parsed author to its OpenAlex/ORCID counterpart, if any."""
    if not enrich_authors:
        return None, None
    parsed_list = [p for p in all_parsed if p is not None]
    # When list lengths match exactly, trust positional alignment.
    if len(parsed_list) == len(enrich_authors):
        # Find this parsed author's position in the filtered list.
        idx = 0
        for i, p in enumerate(all_parsed):
            if p is parsed:
                idx = sum(1 for q in list(all_parsed)[:i] if q is not None)
                break
        item = enrich_authors[idx]
        return item.get("openalex_id"), item.get("orcid")
    # Fallback: surname-fuzzy match. We strip diacritics on both sides.
    target = parsed.surname_norm
    for item in enrich_authors:
        candidate_name = item.get("display_name") or ""
        cand = parse_author(candidate_name)
        if cand is not None and cand.surname_norm == target:
            return item.get("openalex_id"), item.get("orcid")
    return None, None


def _review_reason(
    occs: list[AuthorOccurrence],
    cfg: AuthorClusterConfig,
) -> str | None:
    """Return a short reason string when the cluster looks low-confidence."""
    has_full = any(o.parsed.has_full_first for o in occs)
    has_external = any(o.openalex_id or o.orcid for o in occs)
    if has_external:
        return None
    if not has_full and len(occs) >= 3:
        # Initial-only cluster with several citations — most worth checking.
        return "initial-only cluster with no full-name evidence"
    return None


def _apply_aliases(
    clusters: list[AuthorCluster],
    aliases: dict[str, str],
) -> list[AuthorCluster]:
    """Merge clusters according to ``aliases``: {cluster_id: canonical_id}.

    Aliases are followed transitively (A→B, B→C ⇒ A absorbed into C).
    Unknown ids on either side are silently ignored — the alias file is a
    user-edited surface and we'd rather not blow up on stale entries.
    """
    by_id = {c.id: c for c in clusters}

    def resolve(cid: str, seen: set[str]) -> str:
        if cid in seen or cid not in aliases:
            return cid
        seen.add(cid)
        return resolve(aliases[cid], seen)

    merged: dict[str, AuthorCluster] = {}
    for c in clusters:
        target_id = resolve(c.id, set())
        if target_id == c.id:
            merged.setdefault(c.id, c)
            continue
        target = by_id.get(target_id)
        if target is None:
            merged.setdefault(c.id, c)
            continue
        absorber = merged.setdefault(target.id, target)
        absorber.occurrences.extend(c.occurrences)
        # Pull external ids forward if the absorber didn't have them.
        absorber.openalex_id = absorber.openalex_id or c.openalex_id
        absorber.orcid = absorber.orcid or c.orcid

    return list(merged.values())


def _clusters_to_authors_df(clusters: list[AuthorCluster]) -> pd.DataFrame:
    """Render clusters as the on-disk ``authors.csv`` shape."""
    rows = []
    for c in clusters:
        n_papers = len({o.citing_paper_id for o in c.occurrences if o.citing_paper_id})
        n_refs = sum(1 for o in c.occurrences if o.record_kind == "reference")
        rows.append(
            {
                "id": c.id,
                "display_name": c.display_name,
                "surname": c.surname,
                "surname_norm": c.surname_norm,
                "canonical_given": c.canonical_given,
                "initials": c.initials,
                "openalex_id": c.openalex_id,
                "orcid": c.orcid,
                "n_occurrences": c.n_occurrences,
                "n_reference_citations": n_refs,
                "n_distinct_papers_citing": n_papers,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df.set_index(pd.Index([], name="id"))
    return df.set_index("id").sort_values("n_reference_citations", ascending=False)


def _clusters_to_citations_df(clusters: list[AuthorCluster]) -> pd.DataFrame:
    """Render the per-occurrence edges as ``author_citations.csv``."""
    rows = []
    for c in clusters:
        for o in c.occurrences:
            rows.append(
                {
                    "author_id": c.id,
                    "record_kind": o.record_kind,
                    "record_id": o.record_id,
                    "position": o.position,
                    "citing_paper_id": o.citing_paper_id or "",
                    "raw_author": o.parsed.raw,
                }
            )
    return pd.DataFrame(rows)


def load_aliases(path: Path | str | None) -> dict[str, str]:
    """Load a ``cluster_id,canonical_id`` CSV into a dict.

    Empty / missing path returns ``{}``. Header row optional. Comments
    (lines starting with ``#``) are ignored.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 2:
                continue
            src, dst = parts[0], parts[1]
            if src.lower() == "cluster_id" and dst.lower() == "canonical_id":
                continue  # header
            if src and dst:
                out[src] = dst
    return out
