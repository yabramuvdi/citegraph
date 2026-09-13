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

import ast
import csv
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

from citegraph.io import (
    AUTHOR_CITATION_COLUMNS,
    AUTHOR_COLUMNS,
    parse_authors_list,
    require_columns,
)

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


# Providers spell hyphenated names with typographic dashes: OpenAlex returns
# "Villegas‐Palacio" (U+2010 HYPHEN), not an ASCII "-". They are the same
# character to a reader and must be the same to the normaliser, or one person
# gets two blocking keys and no provider id ever attaches. The soft hyphen is an
# invisible line-break hint, so it is deleted rather than turned into a hyphen.
_UNICODE_DASHES = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2015": "-", "\u2212": "-", "\u00ad": "",
})


def _fold_dashes(text: str) -> str:
    """Map typographic dashes onto the ASCII hyphen the parser understands."""
    return text.translate(_UNICODE_DASHES)


def _norm_surname(surname: str) -> str:
    """Lowercase + diacritic-strip + drop non-alphabetic characters.

    Spaces are kept so multi-word surnames ('garcia marquez') can be
    compared by full normalised form. The blocking step uses this as the
    bucket key, so it MUST be deterministic across runs.
    """
    s = _strip_diacritics(_fold_dashes(surname)).lower().strip()
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
    # Fold dashes before anything inspects the string: the compound-surname
    # checks test for a literal "-", so a typographic one would read as an
    # unhyphenated name. ``raw`` keeps the original spelling for display.
    s = _fold_dashes(raw).strip()
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


# Below this length a token is too short to absorb a typo without pairing
# unrelated people ("Lee"/"Loe"), so it must match exactly.
_AGREEMENT_FUZZ_MIN_LEN = 5

# Providers spell apostrophes typographically ("D’Adda") where
# bibliographies use ASCII ("D'Adda"); both must fold to one token, the same
# way :func:`_fold_dashes` handles the unicode dashes OpenAlex emits. The
# OCR-spaced form ("Ca ' rdenas") folds here too.
_APOSTROPHES = ("'", "’", "‘", "ʼ", "´", "`")


def author_list_agreement(
    extracted: list[str],
    provider: list[str],
    *,
    fuzz_threshold: float = 85.0,
) -> bool | None:
    """Do two author lists name any of the same people?

    ``True`` when at least one name pairs, ``False`` when both sides carry
    usable person names and none pair, ``None`` to abstain.

    Comparison is over order-free name tokens rather than
    :attr:`ParsedAuthor.surname_norm`, because the surname path needs the
    corpus-wide compound-surname lexicon and that does not exist yet when
    enrichment runs. Without it every comma-less provider name collapses to
    its last token, so "Moros, L." and "Lina Moros Canon" stop matching.
    Tokens pair on equality, or on ``fuzz_threshold`` similarity when long
    enough to survive it, which is what lets a bibliography's
    "Hirschmann, A." reach the provider's "Albert O. Hirschman".

    Abstains whenever either side names an institution, since a provider
    record for a corporate-authored report legitimately lists the human
    chapter authors instead ("IFPRI" -> "Fredrick O. Wanyama").
    """
    left = _agreement_tokens(extracted)
    right = _agreement_tokens(provider)
    if left is None or right is None:
        return None
    for a in left:
        for b in right:
            if a == b:
                return True
            if (
                len(a) >= _AGREEMENT_FUZZ_MIN_LEN
                and len(b) >= _AGREEMENT_FUZZ_MIN_LEN
                and fuzz.ratio(a, b) >= fuzz_threshold
            ):
                return True
    return False


def _is_acronym_author(s: str) -> bool:
    """True for a bare all-caps acronym ("IFPRI", "UNFCCC", "IPCC").

    :func:`_is_corporate_author` needs two tokens to fire, so acronyms slip
    past it. They are institutions all the same.
    """
    stripped = s.strip()
    return (
        len(stripped) >= 3
        and stripped.isupper()
        and stripped.isalpha()
        and " " not in stripped
    )


def _agreement_tokens(names: list[str]) -> set[str] | None:
    """Order-free name tokens, or ``None`` when there is nothing to compare."""
    tokens: set[str] = set()
    for raw in names:
        if not isinstance(raw, str) or not raw.strip():
            continue
        if _is_corporate_author(raw) or _is_acronym_author(raw):
            return None
        cleaned = _strip_diacritics(_fold_dashes(raw))
        for apostrophe in _APOSTROPHES:
            cleaned = cleaned.replace(apostrophe, "")
        for token in re.split(r"[^A-Za-z]+", cleaned):
            lowered = token.lower()
            if len(token) > 2 and lowered not in _PARTICLES:
                tokens.add(lowered)
    return tokens or None


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
    """One concrete appearance of an author on a canonical work.

    The clustering output is grouped by canonical author id; each
    occurrence keeps the work id so callers can join back onto
    ``works.csv`` (for rings) and ``citation_graph.csv`` (for citers).
    """

    parsed: ParsedAuthor
    record_id: str            # 'w-…' canonical work id
    position: int             # 0-based index in the author list
    co_author_keys: tuple[str, ...] = ()  # surname_norm of the other authors on the same record
    openalex_id: str | None = None
    orcid: str | None = None
    cannot_link: set[tuple[str, int]] = field(default_factory=set)
    ambiguous_identity_links: set[tuple[str, int]] = field(default_factory=set)


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


def _author_conflict_reasons(left: list[AuthorOccurrence], right: list[AuthorOccurrence]) -> list[str]:
    """Contradictions are independent of positive matching evidence."""
    if any((r.record_id, r.position) in member.cannot_link for member in left for r in right):
        return ["explicit_separate"]
    if any((r.record_id, r.position) in member.ambiguous_identity_links for member in left for r in right):
        return ["ambiguous_external_identity"]
    lo = {o.orcid for o in left if o.orcid}
    ro = {o.orcid for o in right if o.orcid}
    if lo and ro and len(lo | ro) > 1:
        return ["orcid_conflict"]
    if any(a.parsed.is_corporate != b.parsed.is_corporate for a in left for b in right):
        return ["corporate_person_conflict"]
    li, ri = _cluster_external_ids(left), _cluster_external_ids(right)
    if li & ri:
        return []  # Consistent direct identity supports observed name variants.
    reasons = []
    if li and ri:
        reasons.append("distinct_external_ids")
    if {o.record_id for o in left} & {o.record_id for o in right}:
        reasons.append("same_work_distinct_positions")
    for left_occurrence in left:
        for right_occurrence in right:
            lu, ru = _given_units(left_occurrence.parsed), _given_units(right_occurrence.parsed)
            if lu and ru and not (_units_compatible(lu, ru) or _given_subsequence(lu, ru)):
                reasons.append("given_name_conflict")
    return sorted(set(reasons))


def _author_clusters_conflict(left: list[AuthorOccurrence], right: list[AuthorOccurrence]) -> bool:
    return bool(_author_conflict_reasons(left, right))


def _name_supported(left: list[AuthorOccurrence], right: list[AuthorOccurrence]) -> bool:
    for left_occurrence in left:
        for right_occurrence in right:
            if left_occurrence.parsed.surname_norm != right_occurrence.parsed.surname_norm:
                continue
            if left_occurrence.parsed.is_corporate and right_occurrence.parsed.is_corporate:
                return True
            lu, ru = _given_units(left_occurrence.parsed), _given_units(right_occurrence.parsed)
            if lu and ru and _units_compatible(lu, ru) and (
                lu == ru or any(k == "w" for k, _ in lu + ru)
            ):
                return True
    return False


def _merge_unidentified_into_external_clusters(
    clusters: list[list[AuthorOccurrence]],
) -> list[list[AuthorOccurrence]]:
    external = [c for c in clusters if _has_external_id(c)]
    unresolved = []
    for cluster in clusters:
        if _has_external_id(cluster):
            continue
        # Coauthor evidence belongs to each occurrence, never its signature bucket.
        for occurrence in cluster:
            candidates = [c for c in external if _name_supported([occurrence], c)
                          and not _author_clusters_conflict([occurrence], c)]
            if len(candidates) > 1:
                candidates = [c for c in candidates if _has_coauthor_overlap(occurrence, c)]
            if len(candidates) == 1:
                candidates[0].append(occurrence)
            else:
                unresolved.append([occurrence])
    return external + unresolved


def _cluster_block(occs: list[AuthorOccurrence], cfg: AuthorClusterConfig,
                   seeds: list[list[AuthorOccurrence]] | None = None) -> list[list[AuthorOccurrence]]:
    """Resolve external identities globally before occurrence-specific name joins."""
    seeds = seeds if seeds is not None else [[o] for o in occs]
    external = _merge_clusters_by_external_id([c for c in seeds if len(c) > 1 or _has_external_id(c)])
    unidentified = [c[0] for c in seeds if len(c) == 1 and not _has_external_id(c)]
    # Full evidence comes first; stable tie ordering preserves deterministic IDs.
    unidentified.sort(key=lambda o: (
        tuple(-x for x in _units_informativeness(_given_units(o.parsed))),
        " ".join(v for _, v in _given_units(o.parsed)), o.record_id, o.position,
    ))
    clusters = external[:]
    surname_candidates: dict[str, list[list[AuthorOccurrence]]] = defaultdict(list)
    def index_cluster(cluster):
        for surname in {o.parsed.surname_norm for o in cluster}:
            surname_candidates[surname].append(cluster)
    for cluster in clusters:
        index_cluster(cluster)
    held: list[list[AuthorOccurrence]] = []
    for occurrence in unidentified:
        one = [occurrence]
        if cfg.merge_mode == "loose":
            candidates = [c for c in surname_candidates[occurrence.parsed.surname_norm] if c[0].parsed.surname_norm == occurrence.parsed.surname_norm
                          and c[0].parsed.first_initial == occurrence.parsed.first_initial
                          and not _author_clusters_conflict(one, c)]
        else:
            candidates = [c for c in surname_candidates[occurrence.parsed.surname_norm] if _name_supported(one, c)
                          and not _author_clusters_conflict(one, c)]
        if len(candidates) > 1:
            overlap = [c for c in candidates if _has_coauthor_overlap(occurrence, c)]
            if len(overlap) == 1:
                candidates = overlap
        if len(candidates) == 1:
            candidates[0].append(occurrence)
        elif candidates:
            held.append(one)
        else:
            clusters.append(one)
            index_cluster(one)
    # Identical unresolved signatures can combine across works, without making
    # an ambiguous initial cluster an anchor for other abbreviated signatures.
    for one in held:
        matches = [c for c in surname_candidates[one[0].parsed.surname_norm] if not _has_external_id(c)
                   and c[0].parsed.surname_norm == one[0].parsed.surname_norm
                   and _given_units(c[0].parsed) == _given_units(one[0].parsed)
                   and not _author_clusters_conflict(one, c)]
        if len(matches) == 1:
            matches[0].extend(one)
        else:
            clusters.append(one)
            index_cluster(one)
    return clusters


def _cluster_external_ids(cluster: list[AuthorOccurrence]) -> set[str]:
    return {kind + value for o in cluster
            for kind, value in (("oa:", o.openalex_id), ("orcid:", o.orcid)) if value}


def _mark_identity_ambiguities(clusters: list[list[AuthorOccurrence]]) -> None:
    """Find contradictory identity components before any order-dependent unions.

    A seed compatible with both sides of a contradiction cannot choose either.
    Keep that evidence unresolved across every later name/identity merge pass.
    Explicitly forced groups are already seeds, so reviewed choices remain valid.
    """
    parent = list(range(len(clusters)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owner: dict[str, int] = {}
    for i, cluster in enumerate(clusters):
        for key in _cluster_external_ids(cluster):
            if key in owner:
                parent[root(i)] = root(owner[key])
            else:
                owner[key] = i
    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(clusters)):
        components[root(i)].append(i)
    blocked: set[tuple[int, int]] = set()
    for indices in components.values():
        conflicts = [(i, j) for pos, i in enumerate(indices) for j in indices[pos + 1:]
                     if {"orcid_conflict", "explicit_separate"}
                     & set(_author_conflict_reasons(clusters[i], clusters[j]))]
        for left, right in conflicts:
            for i in indices:
                if i in (left, right):
                    continue
                # Membership of this component already supplies transitive ID
                # evidence. Distinct direct IDs or name spellings are not hard
                # contradictions to an anchor elsewhere in the component.
                hard_conflicts = {"orcid_conflict", "explicit_separate", "corporate_person_conflict"}
                if (not hard_conflicts.intersection(_author_conflict_reasons(clusters[i], clusters[left]))
                        and not hard_conflicts.intersection(_author_conflict_reasons(clusters[i], clusters[right]))):
                    blocked.update(((i, left), (i, right)))
    for left, right in blocked:
        for own, other in ((left, right), (right, left)):
            coordinates = {(o.record_id, o.position) for o in clusters[other]}
            for occurrence in clusters[own]:
                occurrence.ambiguous_identity_links.update(coordinates)


def _merge_clusters_by_external_id(clusters: list[list[AuthorOccurrence]]) -> list[list[AuthorOccurrence]]:
    """Union direct identities only while every known ORCID remains consistent."""
    result: list[list[AuthorOccurrence]] = []
    for cluster in clusters:
        targets = [c for c in result if _cluster_external_ids(c) & _cluster_external_ids(cluster)
                   and not _author_clusters_conflict(c, cluster)]
        # An ORCID-free bridge shared by conflicting identities cannot choose one.
        if len({o.orcid for c in targets + [cluster] for o in c if o.orcid}) > 1:
            targets = []
        if any('explicit_separate' in _author_conflict_reasons(a, b)
               for i, a in enumerate(targets) for b in targets[i + 1:]):
            targets = []
        if targets:
            primary = targets[0]
            primary.extend(cluster)
            for other in targets[1:]:
                primary.extend(other)
                result = [c for c in result if c is not other]
        else:
            result.append(cluster)
    return result


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
                        and not _author_clusters_conflict(single, compound)
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
    works: pd.DataFrame,
    enriched_works: pd.DataFrame | None = None,
    citation_edges: pd.DataFrame | None = None,
    cfg: AuthorClusterConfig | None = None,
    aliases: dict[str, str] | None = None,
    external_ids: dict[str, dict[str, str | None]] | None = None,
    constraints: list[dict] | None = None,
    audit: list[dict] | None = None,
    identity_state: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Cluster authors across the corpus.

    Parameters
    ----------
    works:
        Canonical works DataFrame (the output of
        :func:`citegraph.dedup.canonicalize_works`). Must be indexed by
        ``id`` and contain an ``Authors_List`` column (or ``Authors``).
        A ``ring`` column, when present, feeds the ``n_core_works``
        metric; frames without it are accepted (the metric reads 0).
    enriched_works:
        Optional output of :func:`citegraph.enrich.enrich_works`.
        If present and contains the ``OpenAlex_Authors`` column (a list
        of ``{display_name, openalex_id, orcid}`` dicts per row), those
        identifiers are attached to the matching work authors so
        OpenAlex acts as ground truth — for core works too.
    citation_edges:
        Optional citation graph with ``citing_id`` and ``cited_id`` columns.
        When provided, ``authors.csv`` counts citations received by each
        canonical author's works and the distinct works citing them.
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
        Edge table linking each canonical author to every work they
        appear on (``author_id``, ``record_id``, ``position``,
        ``raw_author``).
    review:
        List of dicts describing low-confidence clusters that the user
        may want to inspect.
    """
    cfg = cfg or AuthorClusterConfig()
    aliases = aliases or {}

    attachment_audit: list[dict] = []
    occurrences = _collect_occurrences(
        works=works,
        enriched_works=enriched_works,
        audit=attachment_audit,
    )

    # Block by surname_norm and cluster within each block.
    blocks: dict[str, list[AuthorOccurrence]] = defaultdict(list)
    for o in occurrences:
        blocks[o.parsed.surname_norm].append(o)

    from citegraph.author_overrides import prepare_constraints
    seeds = prepare_constraints(occurrences, constraints or [])
    _mark_identity_ambiguities(seeds)
    raw_clusters = _cluster_block(occurrences, cfg, seeds)

    # Cross-block passes: shared external ids override blocking, then
    # single-surname clusters bridge into compound-surname clusters when
    # the evidence is strong enough (see the function docstrings).
    raw_clusters = _merge_clusters_by_external_id(raw_clusters)
    raw_clusters = _bridge_compound_surnames(raw_clusters)
    block_keys = frozenset(blocks)

    # Build AuthorCluster records with stable ids.
    used_ids: set[str] = set()
    clusters: list[AuthorCluster] = []
    for occs in raw_clusters:
        if not occs:
            continue
        canonical_given, given_token = _canonical_given_for_cluster(occs)
        surname_norm, surname_display = _cluster_surname(occs)
        base_id = _make_author_id(surname_norm, given_token)
        cid = base_id
        suffix = 1
        while cid in used_ids:
            cid = _make_author_id(surname_norm, given_token, suffix)
            suffix += 1
        used_ids.add(cid)

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

    identity_changes = []
    if identity_state is not None:
        from citegraph.identity import occurrence_key, reconcile, resolve_redirect

        def memberships():
            return {c.id: {occurrence_key(o.record_id, o.position, o.parsed.raw)
                           for o in c.occurrences} for c in clusters}

        mapping, algorithm, changes = reconcile(memberships(), identity_state.get('algorithm'))
        for c in clusters:
            c.id = mapping[c.id]
        identity_changes.extend(changes)
        for source in aliases:
            resolve_redirect(source, aliases)  # reject cycles before resolving history
        # A curated alias asserts that two spellings are one person. That
        # outranks both how the algorithm clustered them and how an earlier run
        # published them, so the merge is settled here — against stored
        # occurrences, before any published ID is assigned — and identity
        # assignment simply follows the membership the user chose.
        previous_entries = (identity_state.get('published') or {}).get('entries', {})
        current = memberships()
        parent = {cid: cid for cid in current}

        def _find(cid: str) -> str:
            while parent[cid] != cid:
                parent[cid] = parent[parent[cid]]
                cid = parent[cid]
            return cid

        def _union(left: str, right: str) -> None:
            left, right = _find(left), _find(right)
            if left != right:
                parent[right] = left

        def _holders(public_id: str) -> list[str]:
            """Clusters now holding a published ID's occurrences.

            An ID may have been retired by a split, continued by one cluster, or
            spread over several; following its stored occurrences covers all
            three without the caller needing to know which happened. An ID with
            no stored occurrences is only usable if it names a cluster outright.
            """
            members = set(previous_entries.get(public_id, ()))
            holders = sorted(cid for cid, held in current.items() if members & held)
            if holders:
                return holders
            return [public_id] if public_id in current else []

        named: dict[str, set[str]] = defaultdict(set)
        stale: set[str] = set()
        bindings = {}
        for source, target in aliases.items():
            sources, targets = _holders(source), _holders(target)
            if not sources or not targets:
                stale.add(source if not sources else target)
                continue
            for cid in sources[1:] + targets:
                _union(sources[0], cid)
            for cid in targets:
                named[cid].add(target)
            bindings[source] = {'target': target, 'source_ids': sources, 'target_ids': targets}
        if stale:
            raise ValueError(
                f'Stale author aliases: {sorted(stale)}; review against current authors')

        components: dict[str, list[str]] = defaultdict(list)
        for cid in current:
            components[_find(cid)].append(cid)
        migrated_aliases, pins = {}, {}
        for component in components.values():
            claims = {t for cid in component for t in named.get(cid, ())}
            if not claims:
                continue
            # Absorb into a cluster the user actually named, the largest one
            # first, so the survivor carries that person's display name instead
            # of an initials-only fragment's.
            canonical = max((cid for cid in component if named.get(cid)),
                            key=lambda cid: (len(current[cid]), cid))
            for cid in component:
                if cid != canonical:
                    migrated_aliases[cid] = canonical
            pins[canonical] = max(claims, key=lambda t: (len(previous_entries.get(t, ())), t))
        aliases = migrated_aliases

    # Apply user-curated aliases: merge cluster B into cluster A.
    if aliases:
        clusters = _apply_aliases(clusters, aliases)

    if identity_state is not None:
        # Membership is already what the user asked for; assigning IDs is all
        # that is left. The pins carry their merges through a split that would
        # otherwise retire the very IDs external crosswalks join on.
        mapping, published, changes = reconcile(
            memberships(), identity_state.get('published'), pinned=pins)
        for c in clusters:
            c.id = mapping[c.id]
        identity_changes.extend(changes)
        identity_state.update(schema_version=1, algorithm=algorithm, published=published,
                              alias_bindings=bindings)

    if external_ids:
        _apply_external_ids(clusters, external_ids)

    authors_df = _clusters_to_authors_df(
        clusters, works=works, citation_edges=citation_edges
    )
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
    review.extend({'author_id': entry['old_id'], 'reason': 'identity_split', **entry}
                  for entry in identity_changes if entry['decision'] == 'identity_split')

    author_by_occurrence = {(row.record_id, row.position): row.author_id
                            for row in citations_df.itertuples(index=False)}
    for entry in attachment_audit:
        entry["author_id"] = author_by_occurrence.get((entry["record_id"], entry["position"]))
        if entry["decision"] in {"ambiguous", "conflict"}:
            # A provider author nobody claimed is the mirror image of a
            # conflict, not a cluster to review: it has no position and no
            # author_id, so filing it as one inflates the backlog with rows
            # that cannot be acted on.
            review.append({**entry, "reason": "enrichment_provider_author_unused"
                           if entry.get("detail") == "unassigned_provider_author"
                           else "enrichment_" + entry["decision"]})
    resolution_audit: list[dict] = []
    for cluster in clusters:
        identifiers = {"openalex_ids": sorted({o.openalex_id for o in cluster.occurrences if o.openalex_id}),
                       "orcids": sorted({o.orcid for o in cluster.occurrences if o.orcid})}
        resolution_audit.append({"decision": "cluster_identity", "author_id": cluster.id,
                                 "occurrences": [{"record_id": o.record_id, "position": o.position}
                                                 for o in cluster.occurrences], **identifiers})
        reasons = []
        if len(identifiers["openalex_ids"]) > 1:
            reasons.append("multiple_openalex_ids")
        if len({o.record_id for o in cluster.occurrences}) < len(cluster.occurrences):
            reasons.append("duplicate_authorship")
        for reason in reasons:
            review.append({"author_id": cluster.id, "reason": reason, **identifiers})
    evidence_candidates: dict[str, set[int]] = defaultdict(set)
    for i, cluster in enumerate(clusters):
        for key in ({'surname:' + o.parsed.surname_norm for o in cluster.occurrences}
                    | _cluster_external_ids(cluster.occurrences)):
            evidence_candidates[key].add(i)
    for i, left in enumerate(clusters):
        candidate_indices: set[int] = set()
        for key in ({'surname:' + o.parsed.surname_norm for o in left.occurrences}
                    | _cluster_external_ids(left.occurrences)):
            candidate_indices.update(evidence_candidates[key])
        for j in sorted(index for index in candidate_indices if index > i):
            right = clusters[j]
            if not (_name_supported(left.occurrences, right.occurrences)
                    or _cluster_external_ids(left.occurrences) & _cluster_external_ids(right.occurrences)):
                continue
            reasons = _author_conflict_reasons(left.occurrences, right.occurrences)
            if reasons:
                finding = {"author_id": left.id, "other_author_id": right.id,
                           "reason": ",".join(reasons), "decision": "prevented_merge"}
                review.append(finding)
                resolution_audit.append(finding)
    if audit is not None:
        audit.extend(identity_changes)
        audit.extend(attachment_audit)
        audit.extend(resolution_audit)
        audit.extend({"decision": "manual_override", **row} for row in constraints or [])
        audit.extend({"decision": "legacy_alias", "cluster_id": source, "canonical_id": target}
                     for source, target in aliases.items())

    logger.info(
        "Author normalization: %d occurrences → %d clusters (%d flagged for review)",
        len(occurrences),
        len(clusters),
        len(review),
    )
    return authors_df, citations_df, review


def _collect_occurrences(
    *,
    works: pd.DataFrame,
    enriched_works: pd.DataFrame | None,
    audit: list[dict] | None = None,
) -> list[AuthorOccurrence]:
    """Walk the works table and emit one AuthorOccurrence per author.

    Runs in two passes: the first parse of every author string builds a
    corpus surname lexicon from trusted forms (comma forms, hyphenated
    and particle compounds, enrichment family names); the second parse
    uses that lexicon so comma-less names adopt attested compound
    surname boundaries. Co-author surname-norm keys are computed per
    record so the clustering step can use them as a tiebreak signal.
    """
    occurrences: list[AuthorOccurrence] = []

    # Build a per-work enrichment map: work_id -> [(display, oa_id, orcid), ...]
    enrich_map: dict[str, list[dict]] = {}
    if enriched_works is not None and not enriched_works.empty:
        if "OpenAlex_Authors" in enriched_works.columns:
            for work_id, row in enriched_works.iterrows():
                authors = row.get("OpenAlex_Authors")
                if isinstance(authors, list):
                    enrich_map[str(work_id)] = authors

    work_rows: list[tuple[str, list[str]]] = []
    if not works.empty:
        for work_id, row in works.iterrows():
            work_rows.append((str(work_id), _row_authors(row)))

    lexicon = _build_surname_lexicon(
        (a for _wid, authors_list in work_rows for a in authors_list),
        enrich_map,
    )

    for work_id, authors_list in work_rows:
        parsed = [parse_author(a, known_surnames=lexicon) for a in authors_list]
        co_keys = tuple(p.surname_norm if p is not None else "" for p in parsed)
        enrich_authors = enrich_map.get(work_id, [])
        work_audit: list[dict] = []
        assignments = _match_enrichment_authors(parsed, enrich_authors, lexicon, work_audit)
        if audit is not None:
            audit.extend({"record_id": work_id, **entry} for entry in work_audit)
        for pos, p in enumerate(parsed):
            if p is None:
                continue
            oa_id, orcid = assignments[pos]
            occurrences.append(
                AuthorOccurrence(
                    parsed=p,
                    record_id=work_id,
                    position=pos,
                    co_author_keys=tuple(k for i, k in enumerate(co_keys) if i != pos and k),
                    openalex_id=oa_id,
                    orcid=orcid,
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
    # A multi-word `family` redefines a surname boundary for the whole corpus,
    # so a provider record that *contradicts* another record for the same name
    # must not carry it: CrossRef returned family="Claudia Lopez"/given="Maria"
    # for María Claudia López on one work while others said "Lopez", and
    # adopting the longer boundary shattered her into three clusters (one of
    # them holding her OpenAlex id). When the corpus attests a shorter boundary
    # for the same display name, keep the conservative last-token parse — which
    # `author_review.json` already flags as a possible compound surname —
    # rather than trusting the longer one. An uncontradicted `family` is still
    # trusted on its own, so a lone "Guerra Forero" record still teaches it.
    families_by_display: dict[str, set[str]] = defaultdict(set)
    candidates: list[tuple[str, str]] = []
    for items in enrich_map.values():
        for item in items:
            family = item.get("family")
            if not isinstance(family, str):
                continue
            family_norm = _norm_surname(family)
            display = item.get("display_name")
            display_norm = _norm_surname(display) if isinstance(display, str) else ""
            families_by_display[display_norm].add(family_norm)
            if " " not in family_norm:
                continue
            # CrossRef sometimes stuffs the entire name into `family`
            # (with an empty `given`); a family that swallows the whole
            # display name attests nothing about the surname boundary.
            if display_norm and display_norm == family_norm:
                continue
            candidates.append((display_norm, family_norm))
    for display_norm, family_norm in candidates:
        shorter = {
            other
            for other in families_by_display[display_norm] - {family_norm}
            if other and family_norm.endswith(f" {other}")
        }
        if display_norm and shorter:
            continue
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
        return [str(a) if a is not None else "" for a in val]
    if isinstance(val, str):
        try:
            slots = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            slots = None
        if isinstance(slots, list):
            return [str(a) if a is not None else "" for a in slots]
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


def _given_subsequence(a: tuple, b: tuple) -> bool:
    """Recognize omitted given names without admitting conflicting full words."""
    short, long = sorted((a, b), key=len)
    if not short or len(short) == len(long):
        return False
    remaining = iter(long)
    return all(any(_units_compatible((unit,), (other,)) for other in remaining)
               for unit in short)


def _match_enrichment_authors(
    parsed: list[ParsedAuthor | None],
    enrichment: list[dict],
    known_surnames: frozenset[str],
    audit: list[dict],
) -> list[tuple[str | None, str | None]]:
    """Attach only mutually unique names, consuming each provider slot once."""
    providers = [parse_author(item.get("display_name") or "", known_surnames=known_surnames)
                 for item in enrichment]
    pairs: dict[tuple[int, int], str] = {}
    evidence: list[list[dict]] = [[] for _ in parsed]
    for i, author in enumerate(parsed):
        for j, candidate in enumerate(providers):
            reason = "unusable_name"
            tier = None
            if author is not None and candidate is not None:
                au, cu = _given_units(author), _given_units(candidate)
                aw = {v for k, v in au if k == "w"}
                cw = {v for k, v in cu if k == "w"}
                single, compound = sorted((author.surname_norm, candidate.surname_norm),
                                          key=lambda name: len(name.split()))
                surname_ok = author.surname_norm == candidate.surname_norm or (
                    len(single.split()) == 1 and single not in _PARTICLES
                    and len(compound.split()) > 1
                    and single in (compound.split()[0], compound.split()[-1])
                    and bool(aw & cw)
                )
                if author.is_corporate != candidate.is_corporate:
                    reason = "corporate_person_conflict"
                elif not surname_ok:
                    reason = "surname_conflict"
                elif author.is_corporate:
                    tier, reason = "full", "exact_corporate_name"
                elif not au or not cu:
                    reason = "missing_given_name"
                elif not (_units_compatible(au, cu) or _given_subsequence(au, cu)):
                    reason = "given_name_conflict"
                else:
                    tier = "full" if aw and cw and aw == cw else "initial"
                    reason = "compatible_" + tier
            if tier:
                pairs[i, j] = tier
            evidence[i].append({"provider_position": j, "reason": reason, "tier": tier})
    assigned: dict[int, int] = {}
    used: set[int] = set()
    for tier in ("full", "initial"):
        while True:
            candidates = {(i, j) for (i, j), t in pairs.items()
                          if i not in assigned and j not in used
                          and (tier == "initial" or t == "full")}
            round_pairs = [(i, j) for i, j in sorted(candidates)
                           if sum(a == i for a, _ in candidates) == 1
                           and sum(b == j for _, b in candidates) == 1]
            if not round_pairs:
                break
            assigned.update(round_pairs)
            used.update(j for _, j in round_pairs)
    result = []
    for i, author in enumerate(parsed):
        j = assigned.get(i)
        item = enrichment[j] if j is not None else {}
        ids = tuple(
            value.strip().rstrip("/").rsplit("/", 1)[-1] if isinstance(value, str) and value.strip() else None
            for value in (item.get("openalex_id"), item.get("orcid"))
        )
        result.append(ids)
        decision = ("assigned" if j is not None else "ambiguous"
                    if any(a == i for a, _ in pairs) else "conflict" if enrichment else "unmatched")
        audit.append({"position": i, "raw_author": author.raw if author else None,
                      "normalized_name": {"surname": author.surname_norm,
                                          "given_units": list(_given_units(author))} if author else None,
                      "candidates": evidence[i], "provider_position": j,
                      "openalex_id": ids[0], "orcid": ids[1], "decision": decision})
    for j, candidate in enumerate(providers):
        if j not in used:
            audit.append({"position": None, "raw_author": None,
                          "provider_position": j,
                          "provider_name": candidate.raw if candidate else None,
                          "decision": "conflict", "detail": "unassigned_provider_author"})
    return result


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
    Reject stale identifiers and cycles so corrections cannot silently disappear.
    """
    by_id = {c.id: c for c in clusters}
    stale = (set(aliases) | set(aliases.values())) - set(by_id)
    if stale:
        raise ValueError(f'Stale author aliases: {sorted(stale)}; review against current authors')

    def resolve(cid: str, seen: set[str]) -> str:
        if cid in seen:
            raise ValueError(f'Author alias cycle involving {cid}')
        if cid not in aliases:
            return cid
        seen.add(cid)
        return resolve(aliases[cid], seen)

    groups: dict[str, list[AuthorOccurrence]] = defaultdict(list)
    for c in clusters:
        groups[resolve(c.id, set())].extend(c.occurrences)
    for target, group in groups.items():
        if len({o.orcid for o in group if o.orcid}) > 1:
            raise ValueError(f'Author alias {target} has conflicting ORCIDs')
        if any((r.record_id, r.position) in member.cannot_link for member in group for r in group):
            raise ValueError(f'Author alias {target} contradicts explicit separate constraint')
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


def _clusters_to_authors_df(
    clusters: list[AuthorCluster],
    *,
    works: pd.DataFrame,
    citation_edges: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Render clusters as the on-disk ``authors.csv`` shape.

    Metrics per canonical author:

    - ``n_works`` — distinct canonical works the author appears on,
    - ``n_core_works`` — of those, how many are ring 0 (the user's corpus),
    - ``n_citations_received`` — citation edges into any of their works,
    - ``n_distinct_citing_works`` — distinct works doing that citing.
    """
    ring_by_work = works["ring"].to_dict() if "ring" in works.columns else {}
    in_edges: dict[str, set[str]] = defaultdict(set)
    n_in_edges: dict[str, int] = defaultdict(int)
    if citation_edges is not None and not citation_edges.empty:
        require_columns(
            citation_edges,
            ["citing_id", "cited_id"],
            artifact="citation graph",
        )
        for _, row in citation_edges.iterrows():
            cited_id = row.get("cited_id")
            citing_id = row.get("citing_id")
            if cited_id and citing_id:
                in_edges[str(cited_id)].add(str(citing_id))
                n_in_edges[str(cited_id)] += 1

    rows = []
    for c in clusters:
        work_ids = {o.record_id for o in c.occurrences}
        citing: set[str] = set()
        for w in work_ids:
            citing |= in_edges.get(w, set())
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
                "n_works": len(work_ids),
                "n_core_works": sum(1 for w in work_ids if ring_by_work.get(w) == 0),
                "n_citations_received": sum(n_in_edges.get(w, 0) for w in work_ids),
                "n_distinct_citing_works": len(citing),
            }
        )
    df = pd.DataFrame(rows, columns=AUTHOR_COLUMNS)
    if df.empty:
        return df.set_index("id")
    # Stable two-key sort: without edges, authorship volume still yields a
    # deterministic, meaningful order.
    return df.set_index("id").sort_values(
        ["n_citations_received", "n_works"],
        ascending=[False, False],
        kind="mergesort",
    )


def _clusters_to_citations_df(clusters: list[AuthorCluster]) -> pd.DataFrame:
    """Render the per-occurrence edges as ``author_citations.csv``."""
    rows = []
    for c in clusters:
        for o in c.occurrences:
            rows.append(
                {
                    "author_id": c.id,
                    "record_id": o.record_id,
                    "position": o.position,
                    "raw_author": o.parsed.raw,
                }
            )
    return pd.DataFrame(rows, columns=AUTHOR_CITATION_COLUMNS)


def _apply_external_ids(
    clusters: list[AuthorCluster],
    external_ids: dict[str, dict[str, str | None]],
) -> None:
    """Stamp hand-curated identifiers onto canonical authors, in place.

    Applied *after* identity reconciliation, because the file keys on the
    published ``a-`` id the user actually sees. A curated value wins over
    whatever the provider supplied: someone who opened the provider record
    and checked is better evidence than a name match that failed.

    This deliberately does not merge anything. ``author_aliases.csv`` decides
    who is the same person; letting an identifier also force merges would put
    two curated files in charge of the same question.
    """
    by_id = {c.id: c for c in clusters}
    stale = sorted(set(external_ids) - set(by_id))
    if stale:
        raise ValueError(
            f'Stale author external ids: {stale}; review against current authors.csv'
        )
    claimed: dict[tuple[str, str], str] = {}
    for author_id, ids in external_ids.items():
        for field_name in ("openalex_id", "orcid"):
            value = ids.get(field_name)
            if not value:
                continue
            owner = claimed.get((field_name, value))
            if owner is not None and owner != author_id:
                raise ValueError(
                    f'External {field_name} {value} claimed by both {owner} and '
                    f'{author_id}; one identifier cannot name two authors'
                )
            claimed[field_name, value] = author_id
            setattr(by_id[author_id], field_name, value)


def load_external_ids(path: Path | str | None) -> dict[str, dict[str, str | None]]:
    """Load ``author_id,openalex_id,orcid`` overrides keyed by author id.

    Missing path or file returns ``{}``. Blank identifier cells mean "leave
    this one alone", so a row can set only an ORCID; a row that sets neither
    is a mistake and raises rather than being silently ignored. Any further
    columns (a ``note`` explaining the override) are read and discarded.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, dict[str, str | None]] = {}
    with p.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(_strip_comments(handle)))
    for row in rows:
        author_id = (row.get("author_id") or "").strip()
        if not author_id:
            continue
        openalex_id = (row.get("openalex_id") or "").strip() or None
        orcid = (row.get("orcid") or "").strip() or None
        if not openalex_id and not orcid:
            raise ValueError(
                f'Author external id row for {author_id} sets no identifier; '
                'give an openalex_id or an orcid, or delete the row'
            )
        if author_id in out:
            raise ValueError(f'Duplicate author external id row for {author_id}')
        out[author_id] = {"openalex_id": openalex_id, "orcid": orcid}
    return out


def _strip_comments(lines):
    for line in lines:
        if not line.lstrip().startswith("#"):
            yield line


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
                if src in out and out[src] != dst:
                    raise ValueError(f'Author alias conflict for {src}: {out[src]} versus {dst}')
                out[src] = dst
    return out
