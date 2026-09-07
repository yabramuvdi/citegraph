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
   corpus into canonical author records.  Parsing is evidence-driven: a
   first pass over every raw string builds a lexicon of attested
   multi-word surnames (comma forms, hyphenated and particle compounds,
   enrichment family names), and a second pass re-parses comma-less
   names against it, so "Jose Alberto Guerra Forero" adopts the
   "Guerra Forero" boundary when — and only when — the corpus attests
   it.  The default clustering mode is precision-first: records are
   bucketed by their full given-name *signature* (words + initials in
   order) and a bucket joins a cluster only when its whole signature is
   compatible with exactly one candidate (or a co-author overlap breaks
   the tie); otherwise it is held aside as ambiguous.  Word-vs-word
   comparison tolerates near-identical OCR typos ("Camillo"/"Camilo")
   but never prefix pairs like Gabriel/Gabriela.  ``merge_mode="loose"``
   collapses by ``(surname, first_initial)`` regardless of full-name
   evidence — useful when you want recall over precision.

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
from rapidfuzz import fuzz
from slugify import slugify

from citegraph.io import parse_authors_list, require_columns

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

# 'et al.' glued onto a real name ("B. Puranen et al.") — the bare forms
# are caught earlier by the exact-string check in parse_author.
_ET_AL_TRAILING = re.compile(r"[,\s]+et\s+al\.?\s*$", re.IGNORECASE)

# OCR-mangled accents: PDF extraction renders 'Bénabou' as "B ' enabou"
# and 'Ibáñez' as "Ib ' a ˜ nez". An apostrophe-like or floating accent
# mark with whitespace on *both* sides is never part of a real name
# (genuine spaced apostrophes — Dutch "van 't Hoff", quoted nicknames —
# touch a letter on one side), so the mark and its spaces are removed to
# rejoin the letter runs.
_OCR_SPACED_MARK = re.compile(r"\s+['’´ʼ`˜~̃]\s+")

# Bare uppercase initials run as one token — the Vancouver / NLM citation
# style writes "Cardenas JC" (surname first, initials last, no comma).
# Accented uppercase included so "Ibáñez ÁC" is recognized too.
_BARE_CAPS_INITIALS = re.compile(r"[A-ZÀ-ÖØ-Þ]{1,3}")

# Words that mark an author string as an institution rather than a person.
# Deliberately conservative: every entry is a word that essentially never
# appears in a personal name. Matched on diacritic-stripped lowercase
# tokens, and only for comma-less multi-token strings (comma forms are
# person-shaped), so "Banks, J." never trips it.
_CORPORATE_WORDS = frozenset(
    {
        "university", "universidad", "universidade", "universite",
        "institute", "instituto", "institution", "ministry", "ministerio",
        "department", "departamento", "association", "asociacion",
        "bank", "banco", "council", "commission", "comision", "committee",
        "organization", "organisation", "organizacion", "agency", "agencia",
        "foundation", "fundacion", "fondo", "center", "centre", "centro",
        "office", "oficina", "bureau", "society", "sociedad", "nations",
        "congreso", "congress", "senado", "senate", "alcaldia",
        "gobernacion", "secretaria", "authority", "administration",
        "administracion", "programme", "observatory", "observatorio",
        "consortium", "assessment", "corporation", "grupo", "world",
    }
)

# "(NRC)", "(USAID)" — a parenthesised all-caps acronym is an
# organization signature regardless of the surrounding shape.
_ACRONYM_PARENS = re.compile(r"\([A-Z]{2,}\)")


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
    # Hyphens are spelling variance, not structure: 'Casas-Casas' and
    # 'Casas Casas' must produce the same blocking key.
    s = s.replace("-", " ")
    # Keep letters and internal spaces; drop apostrophes, periods, etc.
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
    is_corporate:
        ``True`` when the string names an institution ("The World Bank")
        rather than a person. The whole string is kept as the surname and
        such records only ever cluster with identical normalized strings.
    """

    raw: str
    surname: str
    surname_norm: str
    given_names: tuple[str, ...]
    initials: str
    has_full_first: bool
    suffix: str | None = None
    is_corporate: bool = False

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


def parse_author(
    raw: str,
    *,
    known_surnames: Iterable[str] | None = None,
) -> ParsedAuthor | None:
    """Parse a single raw author string into a :class:`ParsedAuthor`.

    Returns ``None`` for strings that contain no usable surname (empty
    string, pure punctuation, "et al.", "Anonymous", etc.). Callers
    should treat ``None`` as "skip this author".

    ``known_surnames`` is an optional set of *normalized* multi-word
    surnames already attested elsewhere (comma forms, hyphenated forms,
    enrichment family names). When given, comma-less names whose token
    suffix matches an attested compound surname are split at that
    boundary instead of the conservative last-token guess — this is how
    "Jose Alberto Guerra Forero" learns its surname is "Guerra Forero"
    from a "Guerra Forero, J.A." citation elsewhere in the corpus.
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
    # Rejoin OCR-mangled accents ("B ' enabou" -> "Benabou") before any
    # tokenization; a spaced mark would otherwise split the surname.
    s = _OCR_SPACED_MARK.sub("", s)
    # 'et al.' glued onto a real name keeps the name, drops the tail.
    s = _ET_AL_TRAILING.sub("", s)
    s = re.sub(r"\s+", " ", s).strip(" ,")
    if not s:
        return None

    # Institutions are not person-parsed: the whole string is the name.
    if _is_corporate_author(s):
        return ParsedAuthor(
            raw=raw,
            surname=s,
            surname_norm=_norm_surname(s),
            given_names=(),
            initials="",
            has_full_first=False,
            suffix=None,
            is_corporate=True,
        )

    # For persons, a parenthetical is an affiliation ("Smith, J. (MIT)"),
    # never part of the name.
    s = re.sub(r"\s*\([^)]*\)", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" ,")
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
        reversed_split = _split_reversed_citation(s.split())
        if reversed_split is not None:
            # Vancouver / NLM style: "Guerra JA" is surname-then-initials.
            surname, given_str = reversed_split
        else:
            surname, given_str = _split_no_comma(s, known_surnames)

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


def _split_no_comma(
    s: str,
    known_surnames: Iterable[str] | None = None,
) -> tuple[str, str]:
    """Split a name with no comma into (surname, given-string).

    Heuristic: the last token is the surname, *unless* the second-to-last
    token is a particle ('de', 'van', 'García' before 'Márquez'…), in
    which case the surname extends backwards to include it. Compound
    surnames without particles ('García Márquez' alone, no comma) are
    ambiguous — we can't tell from the string alone, and we conservatively
    take only the last token as the surname *unless* ``known_surnames``
    attests the compound (see :func:`parse_author`).
    """
    tokens = s.split()
    if not tokens:
        return "", ""
    if len(tokens) == 1:
        return tokens[0], ""

    # Corpus-evidence pass: adopt a multi-token surname the corpus has
    # already attested. Longest suffix wins; a whole-string match means
    # the string is a bare compound surname with no given names.
    if known_surnames:
        for k in range(min(4, len(tokens)), 1, -1):
            cand = _norm_surname(" ".join(tokens[-k:]))
            if cand in known_surnames:
                if k == len(tokens):
                    return s, ""
                return " ".join(tokens[-k:]), " ".join(tokens[:-k])

    surname_idx = len(tokens) - 1
    # Walk backwards while the preceding token looks like a particle.
    while surname_idx > 0:
        prev_raw = tokens[surname_idx - 1]
        prev = prev_raw.strip(".-").lower()
        if prev in _PARTICLES:
            surname_idx -= 1
        elif (
            prev_raw == "y"
            and surname_idx >= 2
            and not _is_initial_only_token(tokens[surname_idx - 2])
        ):
            # Spanish surname connector: 'Ortega y Gasset'. Only the bare
            # lowercase word counts — a 'Y.' initial must not trigger it.
            surname_idx -= 2
        else:
            break

    surname = " ".join(tokens[surname_idx:])
    given = " ".join(tokens[:surname_idx])
    return surname, given


def _split_reversed_citation(tokens: list[str]) -> tuple[str, str] | None:
    """Detect Vancouver-style 'Surname(s) AB' ordering; ``None`` if absent.

    The trailing run of initials-like tokens ("JA", "J.A.", "J") becomes
    the given part and everything before it the surname — but only when
    no leading token is a single-letter initial, so "A. Diamond"
    (given-first) is left alone. All-caps leading tokens are fine ("LEE
    KY", "CAMERON L"). Bare caps runs are expanded ("JA" -> "J. A.") so
    the initials extraction sees one letter per run.
    """

    def is_initials_like(t: str) -> bool:
        return _is_initial_only_token(t) or bool(_BARE_CAPS_INITIALS.fullmatch(t))

    # Consume the trailing initials run, but never the first token — in
    # "LI X" the all-caps "LI" is the surname, not more initials.
    i = len(tokens)
    while i > 1 and is_initials_like(tokens[i - 1]):
        i -= 1
    if i == len(tokens):
        return None
    leading = tokens[:i]
    if any(_is_initial_only_token(t) for t in leading):
        return None
    trailing = tokens[i:]
    trailing_letters = sum(len(run) for t in trailing for run in _LETTER_RUN.findall(t))
    if len(leading) == 1 and len(leading[0]) <= 2 and trailing_letters >= 2:
        # 'Bo XU' vs 'Ma JA' is genuinely ambiguous (caps surname vs
        # Vancouver initials); keep the conservative last-token parse.
        return None
    given_tokens = [
        " ".join(f"{c}." for c in t) if _BARE_CAPS_INITIALS.fullmatch(t) else t
        for t in trailing
    ]
    return " ".join(leading), " ".join(given_tokens)


def _is_corporate_author(s: str) -> bool:
    """True when the string names an institution rather than a person."""
    if "," in s:
        # Comma forms are person-shaped ("Banks, J.", "Smith, J. (MIT)").
        return False
    if _ACRONYM_PARENS.search(s):
        return True
    tokens = re.findall(r"[^\W\d_]+", _strip_diacritics(s).lower())
    if len(tokens) < 2:
        return False
    if not any(t in _CORPORATE_WORDS for t in tokens):
        return False
    # Two-token strings only count when the corporate word leads
    # ("Fundación Natura", "World Bank") — a trailing corporate word may
    # be a real surname ("Steven Bank").
    return len(tokens) >= 3 or tokens[0] in _CORPORATE_WORDS


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
    (surname, given-names) tuple genuinely refers to two distinct people
    that the algorithm split apart. Corporate authors have no given
    token and slug to just the normalized name.
    """
    base = f"a-{slugify(surname_norm) or 'unknown'}"
    given_slug = slugify(given_token) if given_token else ""
    if given_slug:
        base = f"{base}-{given_slug}"
    if n:
        base = f"{base}-{n}"
    return base


def _given_units(parsed: ParsedAuthor) -> tuple[tuple[str, str], ...]:
    """Flatten given names into comparable units.

    Each contiguous letter run becomes one unit: a full word ``("w",
    "juan")`` or an initial ``("i", "j")``. Hyphens and periods dissolve,
    so "Juan-Camilo", "Juan Camilo", and "J.C." become comparable:
    ``[w:juan, w:camilo]`` / ``[w:juan, w:camilo]`` / ``[i:j, i:c]``.
    """
    units: list[tuple[str, str]] = []
    for tok in parsed.given_names:
        for run in _LETTER_RUN.findall(tok):
            kind = "i" if len(run) == 1 else "w"
            units.append((kind, _strip_diacritics(run).lower()))
    return tuple(units)


def _w_equivalent(a: str, b: str) -> bool:
    """True when two full given-name words plausibly name the same person.

    Exact match, or a near-identical spelling that reads as an OCR /
    extraction typo ('camilo' vs 'camillo'): both words at least 4
    letters, rapidfuzz ratio >= 90, and *neither a prefix of the other* —
    the prefix guard is what keeps gendered pairs like Gabriel/Gabriela
    and Daniel/Daniela (ratio > 90) firmly apart.
    """
    if a == b:
        return True
    if len(a) < 4 or len(b) < 4:
        return False
    if a.startswith(b) or b.startswith(a):
        return False
    return fuzz.ratio(a, b) >= 90


def _units_compatible(
    a: tuple[tuple[str, str], ...],
    b: tuple[tuple[str, str], ...],
) -> bool:
    """True when two given-name unit sequences could name the same person.

    Position-by-position over the shared prefix: word vs word must be
    equivalent (exact or typo-close, see :func:`_w_equivalent`), initial
    vs initial must be equal, and an initial matches a word starting
    with it. The *whole* overlapping sequence must agree — this is what
    keeps "J.P." out of a "Juan Camilo" cluster even though the first
    initials match.
    """
    # strict=False is the point: only the shared prefix is compared.
    for (ka, va), (kb, vb) in zip(a, b, strict=False):
        if ka == kb:
            if ka == "w":
                if not _w_equivalent(va, vb):
                    return False
            elif va != vb:
                return False
        else:
            word = va if ka == "w" else vb
            initial = vb if ka == "w" else va
            if not word.startswith(initial):
                return False
    return True


def _units_informativeness(units: tuple[tuple[str, str], ...]) -> tuple[int, int]:
    return (sum(1 for kind, _ in units if kind == "w"), len(units))


def _cluster_best_units(
    cluster: list[AuthorOccurrence],
) -> tuple[tuple[str, str], ...]:
    """The most informative given-name signature observed in a cluster."""
    best: tuple[tuple[str, str], ...] = ()
    best_key = (-1, -1)
    for o in cluster:
        units = _given_units(o.parsed)
        key = _units_informativeness(units)
        if key > best_key:
            best, best_key = units, key
    return best


def _canonical_given_for_cluster(occs: list[AuthorOccurrence]) -> tuple[str, str]:
    """Pick a (display-friendly given string, slug-friendly token) pair.

    Prefer the fullest given-name *sequence* actually seen in the cluster
    ("Jose Alberto", not just "Jose"), so display names never drop
    observed middle names. If the cluster only has initials, use the
    canonical initial sequence. Corporate clusters return empty strings.
    """
    frequency: dict[str, int] = defaultdict(int)
    for o in occs:
        frequency[" ".join(o.parsed.given_names)] += 1

    best_key: tuple[int, int, int, int, str] | None = None
    best_units: tuple[tuple[str, str], ...] = ()
    best_given = ""
    for o in occs:
        if o.parsed.is_corporate:
            continue
        units = _given_units(o.parsed)
        n_words = sum(1 for kind, _ in units if kind == "w")
        if n_words == 0:
            continue
        given_str = " ".join(o.parsed.given_names)
        # Fullest sequence first; among equally full ones, the spelling
        # seen most often wins so a lone typo never names the cluster.
        key = (n_words, len(units), frequency[given_str], len(given_str), given_str)
        if best_key is None or key > best_key:
            best_key, best_units, best_given = key, units, given_str
    if best_given:
        token = " ".join(v for kind, v in best_units if kind == "w")
        return best_given, token
    if any(o.parsed.is_corporate for o in occs):
        return "", ""
    # Fallback to the longest initial sequence we saw.
    inits = max((o.parsed.initials for o in occs), key=len, default="")
    return inits, inits.lower()


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


def _cluster_signatures(
    cluster: list[AuthorOccurrence],
) -> set[tuple[tuple[str, str], ...]]:
    """Every distinct non-empty given-name signature observed in a cluster."""
    return {
        units
        for o in cluster
        if (units := _given_units(o.parsed))
    }


def _merge_unidentified_into_external_clusters(
    clusters: list[list[AuthorOccurrence]],
) -> list[list[AuthorOccurrence]]:
    """Use enriched full-name clusters as anchors for no-id name variants.

    OpenAlex / ORCID-bearing clusters remain authoritative: different external
    ids are never merged together. This pass only moves clusters with no
    external id into exactly one signature-compatible external-id cluster —
    where *any* attested variant in the id cluster can make the match, so an
    id cluster containing both "Juan Camilo" and "Camilo" still anchors a
    no-id "Camilo" cluster.
    """
    external_clusters = [c for c in clusters if _has_external_id(c)]
    if not external_clusters:
        return clusters

    ext_signatures = [(ext, _cluster_signatures(ext)) for ext in external_clusters]
    merged: list[list[AuthorOccurrence]] = list(external_clusters)
    unresolved: list[list[AuthorOccurrence]] = []
    for cluster in clusters:
        if _has_external_id(cluster):
            continue

        signatures = _cluster_signatures(cluster)
        if not signatures or any(o.parsed.is_corporate for o in cluster):
            unresolved.append(cluster)
            continue
        candidates = [
            ext for ext, ext_sigs in ext_signatures
            if any(
                _units_compatible(sig, ext_sig)
                for sig in signatures
                for ext_sig in ext_sigs
            )
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
    # Corporate authors never mix with persons: identical normalized
    # strings (== the block key) form exactly one cluster.
    corporate = [o for o in occs if o.parsed.is_corporate]
    persons = [o for o in occs if not o.parsed.is_corporate]
    corporate_clusters: list[list[AuthorOccurrence]] = (
        [corporate] if corporate else []
    )

    # OpenAlex / ORCID ids are ground truth and override everything else.
    # Phase 1: pre-merge by external id where present.
    by_external: dict[str, list[AuthorOccurrence]] = defaultdict(list)
    no_external: list[AuthorOccurrence] = []
    for o in persons:
        key = o.openalex_id or o.orcid
        if key:
            by_external[key].append(o)
        else:
            no_external.append(o)

    external_clusters: list[list[AuthorOccurrence]] = list(by_external.values())

    if cfg.merge_mode == "loose":
        # Bucket purely by (surname_norm, first_initial) — ignore everything
        # else. Even external-id clusters get merged into the bucket.
        buckets: dict[tuple[str, str], list[AuthorOccurrence]] = defaultdict(list)
        for cluster in external_clusters:
            for o in cluster:
                buckets[(o.parsed.surname_norm, o.parsed.first_initial)].append(o)
        for o in no_external:
            buckets[(o.parsed.surname_norm, o.parsed.first_initial)].append(o)
        return corporate_clusters + list(buckets.values())

    # Strict mode --------------------------------------------------------
    # Phase 2: bucket by exact given-name signature, then greedily place
    # each bucket, most informative first. A bucket joins an accepted
    # cluster only when its whole signature is compatible with exactly
    # one candidate (or a co-author overlap breaks the tie); identical
    # raw evidence therefore always travels together, and "J.P." can
    # never fall into a "Juan Camilo" cluster off the first initial.
    sig_buckets: dict[tuple[tuple[str, str], ...], list[AuthorOccurrence]] = (
        defaultdict(list)
    )
    for o in no_external:
        sig_buckets[_given_units(o.parsed)].append(o)

    def bucket_order(
        item: tuple[tuple[tuple[str, str], ...], list[AuthorOccurrence]],
    ) -> tuple[int, int, str]:
        n_words, n_units = _units_informativeness(item[0])
        return (-n_words, -n_units, " ".join(v for _, v in item[0]))

    accepted: list[tuple[tuple[tuple[str, str], ...], list[AuthorOccurrence]]] = []
    held_aside: list[list[AuthorOccurrence]] = []
    for units, bucket in sorted(sig_buckets.items(), key=bucket_order):
        if not units:
            # Pure surname with no given-name info at all — stand-alone cluster.
            held_aside.append(bucket)
            continue
        # A merge needs full-name evidence on at least one side: "J." may
        # join "John C." but never "J.C." on the shared prefix alone.
        has_word = any(kind == "w" for kind, _ in units)
        candidates = [
            c for c in accepted
            if _units_compatible(units, c[0])
            and (has_word or any(kind == "w" for kind, _ in c[0]))
        ]
        if len(candidates) == 1:
            candidates[0][1].extend(bucket)
        elif len(candidates) > 1:
            with_overlap = [
                c for c in candidates
                if any(_has_coauthor_overlap(o, c[1]) for o in bucket)
            ]
            if len(with_overlap) == 1:
                with_overlap[0][1].extend(bucket)
            else:
                held_aside.append(bucket)
        else:
            accepted.append((units, bucket))

    clusters = corporate_clusters + external_clusters
    clusters.extend(bucket for _units, bucket in accepted)
    clusters.extend(held_aside)
    return _merge_unidentified_into_external_clusters(clusters)


def _cluster_external_ids(cluster: list[AuthorOccurrence]) -> set[str]:
    return {x for o in cluster for x in (o.openalex_id, o.orcid) if x}


def _merge_clusters_by_external_id(
    clusters: list[list[AuthorOccurrence]],
) -> list[list[AuthorOccurrence]]:
    """Merge clusters that share an external id across surname blocks.

    Blocking splits 'Guerra Forero, J.A.' and 'Guerra, Jose Alberto' into
    different blocks, but a shared OpenAlex/ORCID id is person-level
    ground truth and overrides the blocking disagreement. The union is
    transitive: a cluster carrying two ids pulls both id-groups into one.
    """
    by_id: dict[str, list[AuthorOccurrence]] = {}
    absorbed: set[int] = set()
    for cluster in clusters:
        ids = sorted(_cluster_external_ids(cluster))
        targets: list[list[AuthorOccurrence]] = []
        for ext_id in ids:
            prior = by_id.get(ext_id)
            if prior is not None and prior is not cluster \
                    and all(prior is not t for t in targets):
                targets.append(prior)
        if not targets:
            for ext_id in ids:
                by_id[ext_id] = cluster
            continue
        primary = targets[0]
        for other in targets[1:]:
            primary.extend(other)
            absorbed.add(id(other))
        primary.extend(cluster)
        absorbed.add(id(cluster))
        for ext_id, owner in by_id.items():
            if any(owner is t for t in targets):
                by_id[ext_id] = primary
        for ext_id in ids:
            by_id[ext_id] = primary
    return [c for c in clusters if id(c) not in absorbed]


def _cluster_surname(occs: list[AuthorOccurrence]) -> tuple[str, str]:
    """Order-independent (surname_norm, display surname) for a cluster.

    The most specific (most words), then most frequent, normalized
    surname wins — so a cluster merged across blocks by external ids or
    bridging is named by its compound form no matter which occurrence
    happened to land first. Display casing follows the chosen norm,
    preferring properly-cased spellings, then the most frequent one.
    """
    norm_counts: dict[str, int] = defaultdict(int)
    for o in occs:
        norm_counts[o.parsed.surname_norm] += 1
    surname_norm = max(
        norm_counts,
        key=lambda n: (len(n.split()), norm_counts[n], n),
    )
    display_counts: dict[str, int] = defaultdict(int)
    for o in occs:
        if o.parsed.surname_norm == surname_norm:
            display_counts[o.parsed.surname] += 1
    surname_display = max(
        display_counts,
        key=lambda s: (any(c.isupper() for c in s), display_counts[s], s),
    )
    return surname_norm, surname_display


def _bridge_compound_surnames(
    clusters: list[list[AuthorOccurrence]],
) -> list[list[AuthorOccurrence]]:
    """Merge single-surname clusters into compound-surname clusters.

    'Reyes, Sandra' can join 'Polanía Reyes, Sandra' — but only on strong
    evidence: the compound surname starts or ends with the single one,
    the given-name signatures are fully compatible, *and* at least one
    exact full given name is shared. Clusters that both carry external
    ids never bridge (same-id cases were already merged by
    :func:`_merge_clusters_by_external_id`, so remaining id pairs
    disagree). Initials-only records never bridge; ambiguity (two
    possible compound targets) leaves the cluster where it is.
    """

    def block_key(cluster: list[AuthorOccurrence]) -> str:
        return _cluster_surname(cluster)[0]

    by_key: dict[str, list[list[AuthorOccurrence]]] = defaultdict(list)
    for cluster in clusters:
        by_key[block_key(cluster)].append(cluster)
    compound_keys = sorted(k for k in by_key if " " in k)
    if not compound_keys:
        return clusters

    absorbed: set[int] = set()
    for single_key in sorted(k for k in by_key if " " not in k):
        target_keys = [
            ck for ck in compound_keys
            if ck.split()[0] == single_key or ck.split()[-1] == single_key
        ]
        if not target_keys:
            continue
        for single in by_key[single_key]:
            s_units = _cluster_best_units(single)
            s_words = {v for kind, v in s_units if kind == "w"}
            if not s_words or any(o.parsed.is_corporate for o in single):
                continue
            s_ids = _cluster_external_ids(single)
            targets: list[list[AuthorOccurrence]] = []
            for ck in target_keys:
                for compound in by_key[ck]:
                    c_ids = _cluster_external_ids(compound)
                    if s_ids and c_ids:
                        # Both sides carry ids and (post id-merge) they
                        # necessarily disagree — never bridge over that.
                        continue
                    c_units = _cluster_best_units(compound)
                    c_words = {v for kind, v in c_units if kind == "w"}
                    if (
                        c_words
                        and (s_words & c_words)
                        and _units_compatible(s_units, c_units)
                    ):
                        targets.append(compound)
            if len(targets) == 1:
                targets[0].extend(single)
                absorbed.add(id(single))
    return [c for c in clusters if id(c) not in absorbed]


def normalize_authors(
    *,
    references: pd.DataFrame,
    papers: pd.DataFrame | None = None,
    enriched_references: pd.DataFrame | None = None,
    citation_edges: pd.DataFrame | None = None,
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
    citation_edges:
        Optional citation graph with ``citing_id`` and ``cited_id`` columns.
        When provided, ``authors.csv`` counts distinct source papers that cite
        references by each canonical author.
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

    # Cross-block passes: shared external ids override blocking, then
    # single-surname clusters bridge into compound-surname clusters when
    # the evidence is strong enough (see the function docstrings).
    raw_clusters = _merge_clusters_by_external_id(raw_clusters)
    raw_clusters = _bridge_compound_surnames(raw_clusters)
    block_keys = frozenset(blocks)

    # Build AuthorCluster records with stable ids.
    used_ids: dict[str, int] = defaultdict(int)
    clusters: list[AuthorCluster] = []
    for occs in raw_clusters:
        if not occs:
            continue
        canonical_given, given_token = _canonical_given_for_cluster(occs)
        surname_norm, surname_display = _cluster_surname(occs)
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
                review_reason=_review_reason(occs, cfg, block_keys),
            )
        )

    # Post-hoc ambiguity flag: a no-id cluster whose name evidence is
    # compatible with two or more sibling clusters in the same surname
    # block was held apart only because the evidence could not decide —
    # surface that in the review file instead of leaving the split invisible.
    by_block: dict[str, list[AuthorCluster]] = defaultdict(list)
    for c in clusters:
        by_block[c.surname_norm].append(c)
    for block_clusters in by_block.values():
        if len(block_clusters) < 3:
            continue  # fewer than two possible rivals — nothing ambiguous
        for c in block_clusters:
            if c.review_reason or c.openalex_id or c.orcid:
                continue
            units = _cluster_best_units(c.occurrences)
            if not units:
                continue
            n_compatible = sum(
                1
                for other in block_clusters
                if other is not c
                and (other_units := _cluster_best_units(other.occurrences))
                and _units_compatible(units, other_units)
            )
            if n_compatible >= 2:
                c.review_reason = (
                    "ambiguous name evidence: compatible with multiple "
                    "clusters in this surname block"
                )

    # Apply user-curated aliases: merge cluster B into cluster A.
    if aliases:
        clusters = _apply_aliases(clusters, aliases)

    authors_df = _clusters_to_authors_df(clusters, citation_edges=citation_edges)
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

    Runs in two passes: the first parse of every author string builds a
    corpus surname lexicon from trusted forms (comma forms, hyphenated
    and particle compounds, enrichment family names); the second parse
    uses that lexicon so comma-less names adopt attested compound
    surname boundaries. Co-author surname-norm keys are computed per
    record so the clustering step can use them as a tiebreak signal.
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

    ref_rows: list[tuple[str, list[str]]] = []
    if not references.empty:
        for ref_id, row in references.iterrows():
            ref_rows.append((str(ref_id), _row_authors(row)))

    paper_rows: list[tuple[str, list[str]]] = []
    if papers is not None and not papers.empty:
        for _idx, row in papers.iterrows():
            paper_id = row.get("id") if "id" in papers.columns else None
            if not paper_id:
                continue
            paper_rows.append((str(paper_id), _row_authors(row)))

    lexicon = _build_surname_lexicon(
        (a for _rid, authors_list in [*ref_rows, *paper_rows] for a in authors_list),
        enrich_map,
    )

    for ref_id, authors_list in ref_rows:
        parsed = [parse_author(a, known_surnames=lexicon) for a in authors_list]
        co_keys = tuple(p.surname_norm for p in parsed if p is not None)
        enrich_authors = enrich_map.get(ref_id, [])
        for pos, p in enumerate(parsed):
            if p is None:
                continue
            # Best-effort match between our parsed author and the
            # enrichment list: same position when lengths match,
            # otherwise surname-fuzzy. Falls back to no enrichment.
            oa_id, orcid = _match_enrichment(p, pos, parsed, enrich_authors, lexicon)
            occurrences.append(
                AuthorOccurrence(
                    parsed=p,
                    record_id=ref_id,
                    record_kind="reference",
                    position=pos,
                    co_author_keys=tuple(k for i, k in enumerate(co_keys) if i != pos and k),
                    citing_paper_id=None,
                    openalex_id=oa_id,
                    orcid=orcid,
                )
            )

    for paper_id, authors_list in paper_rows:
        parsed = [parse_author(a, known_surnames=lexicon) for a in authors_list]
        co_keys = tuple(p.surname_norm for p in parsed if p is not None)
        for pos, p in enumerate(parsed):
            if p is None:
                continue
            occurrences.append(
                AuthorOccurrence(
                    parsed=p,
                    record_id=paper_id,
                    record_kind="paper",
                    position=pos,
                    co_author_keys=tuple(k for i, k in enumerate(co_keys) if i != pos and k),
                    citing_paper_id=paper_id,
                    openalex_id=None,
                    orcid=None,
                )
            )

    return occurrences


def _build_surname_lexicon(
    raw_strings: Iterable[str],
    enrich_map: dict[str, list[dict]],
) -> frozenset[str]:
    """Collect normalized multi-word surnames attested by trusted forms.

    Trusted sources: comma forms ("Guerra Forero, J.A." states its own
    boundary), hyphenated surnames ("Polania-Reyes" attests "polania
    reyes"), particle compounds ("van der Berg"), the Spanish 'y'
    connector, and enrichment ``family`` fields. Single-word surnames are
    never collected — a bare "Knowles" elsewhere must not re-split
    "Caitlin Knowles Myers".
    """
    lexicon: set[str] = set()
    for raw in raw_strings:
        p = parse_author(raw)
        if p is None or p.is_corporate:
            continue
        norm = p.surname_norm
        if " " not in norm:
            continue
        trusted = (
            "," in p.raw
            or "-" in p.surname
            or norm.split()[0] in _PARTICLES
            or " y " in f" {norm} "
        )
        if trusted:
            lexicon.add(norm)
    for items in enrich_map.values():
        for item in items:
            family = item.get("family")
            if isinstance(family, str):
                family_norm = _norm_surname(family)
                if " " in family_norm:
                    lexicon.add(family_norm)
    return frozenset(lexicon)


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
    known_surnames: frozenset[str] | None = None,
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
    # Fallback: surname-fuzzy match. We strip diacritics on both sides and
    # parse the candidate with the same surname lexicon so a re-split
    # "Guerra Forero" still lines up with its enrichment display name.
    target = parsed.surname_norm
    for item in enrich_authors:
        candidate_name = item.get("display_name") or ""
        cand = parse_author(candidate_name, known_surnames=known_surnames)
        if cand is not None and cand.surname_norm == target:
            return item.get("openalex_id"), item.get("orcid")
    return None, None


def _review_reason(
    occs: list[AuthorOccurrence],
    cfg: AuthorClusterConfig,
    block_keys: frozenset[str] = frozenset(),
) -> str | None:
    """Return a short reason string when the cluster looks low-confidence."""
    if any(o.parsed.is_corporate for o in occs):
        return "detected as corporate author (not a person)"
    has_external = any(o.openalex_id or o.orcid for o in occs)
    if has_external:
        return None
    for o in occs:
        p = o.parsed
        if "," in p.raw:
            continue
        words = [v for kind, v in _given_units(p) if kind == "w"]
        if len(words) >= 3:
            # 'Jose Alberto Guerra Forero' parsed conservatively — the
            # trailing given names may really be a compound surname the
            # corpus could not corroborate.
            return "possible compound surname: multiple full given names in no-comma form"
        if len(words) == 2 and words[-1] in block_keys and words[-1] != p.surname_norm:
            return "possible compound surname: a given name matches another surname block"
    has_full = any(o.parsed.has_full_first for o in occs)
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


def _reference_citers(citation_edges: pd.DataFrame | None) -> dict[str, set[str]]:
    if citation_edges is None or citation_edges.empty:
        return {}
    require_columns(
        citation_edges,
        ["citing_id", "cited_id"],
        artifact="citation graph",
    )
    out: dict[str, set[str]] = defaultdict(set)
    for _, row in citation_edges.iterrows():
        cited_id = row.get("cited_id")
        citing_id = row.get("citing_id")
        if cited_id and citing_id:
            out[str(cited_id)].add(str(citing_id))
    return out


def _clusters_to_authors_df(
    clusters: list[AuthorCluster],
    *,
    citation_edges: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Render clusters as the on-disk ``authors.csv`` shape."""
    ref_citers = _reference_citers(citation_edges)
    rows = []
    for c in clusters:
        citing_papers: set[str] = set()
        for o in c.occurrences:
            if o.record_kind == "reference":
                citing_papers.update(ref_citers.get(o.record_id, set()))
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
                "n_distinct_papers_citing": len(citing_papers),
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
