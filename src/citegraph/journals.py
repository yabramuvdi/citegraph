"""Canonical journal names.

The same journal reaches us under many spellings — the AER under six in one
corpus, *Ecological Economics* under three — so any rollup that groups on the
raw string understates concentration. Enrichment already collapses most of the
variation by preferring the provider's container title; this module folds what
is left.

Two deliberate limits:

- Folding is conservative. It normalises case, diacritics, punctuation,
  ``&``/``and`` and a leading ``The``, and nothing else. Abbreviations are
  *not* matched automatically: an initials heuristic over one corpus paired
  "public choice" with "protecting the commons" and "energy policy" with
  "economics and philosophy". Abbreviations belong in the alias file, where a
  human vouches for them; there are only about ten worth writing.
- Preprint repositories are not journals. OpenAlex reports RePEc, SSRN and
  publisher eBook platforms as container titles, which would rank RePEc third
  and SSRN ninth among this corpus's "journals". They fold to
  :data:`WORKING_PAPER` instead, and an alias can override that.
"""

from __future__ import annotations

import csv
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

WORKING_PAPER = "Working paper"

# Matched against the folded key, so punctuation and case are already gone.
_REPOSITORY_PATTERNS = (
    "repec",
    "research papers in economics",
    "ssrn",
    "ebooks",
    "social science research network",
)

_LEADING_ARTICLE = re.compile(r"^the\s+")
_AMPERSAND = re.compile(r"\s*&\s*")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def journal_key(name: object) -> str:
    """Fold one journal name to its grouping key.

    Returns ``""`` for anything missing, so a work with no journal never
    joins a group.
    """
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    text = str(name).strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _AMPERSAND.sub(" and ", text.lower())
    text = _LEADING_ARTICLE.sub("", text)
    return _NON_ALNUM.sub(" ", text).strip()


def _is_repository(key: str) -> bool:
    return any(pattern in key for pattern in _REPOSITORY_PATTERNS)


def canonicalize_journals(
    names: Iterable[object],
    aliases: Mapping[str, str] | None = None,
    *,
    strict: bool = False,
) -> dict[str, str]:
    """Map every observed journal spelling to one canonical display name.

    The display name is the most frequent spelling in ``names``, with the
    longest winning a tie — the same "fullest, most frequent wins" rule the
    author stage uses, so a rare lowercased variant never names a group.

    ``aliases`` is applied last and matched by folded key, so the user writes
    one spelling ("Am. Econ. Rev.") and every variant of it follows. With
    ``strict``, an alias that matches nothing raises: a curated row that
    silently does not apply is worse than no row at all.
    """
    observed = [str(n).strip() for n in names if journal_key(n)]
    counts: dict[str, Counter] = {}
    for raw in observed:
        counts.setdefault(journal_key(raw), Counter())[raw] += 1

    alias_by_key = {journal_key(raw): canonical for raw, canonical in (aliases or {}).items()}
    if strict:
        unused = sorted(
            raw for raw in (aliases or {}) if journal_key(raw) not in counts
        )
        if unused:
            raise ValueError(
                "Journal aliases match no journal in this corpus: "
                + ", ".join(repr(u) for u in unused)
                + ". Fix the spelling or delete the row."
            )

    display: dict[str, str] = {}
    for key, spellings in counts.items():
        if key in alias_by_key:
            display[key] = alias_by_key[key]
        elif _is_repository(key):
            display[key] = WORKING_PAPER
        else:
            display[key] = max(spellings.items(), key=lambda kv: (kv[1], len(kv[0])))[0]

    return {
        (str(n).strip() if journal_key(n) else ""): display.get(journal_key(n), "")
        for n in names
    }


def load_journal_aliases(path: Path | str | None) -> dict[str, str]:
    """Read ``journal_aliases.csv`` (``raw,canonical``); absent file is empty."""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    aliases: dict[str, str] = {}
    for row in rows:
        raw = (row.get("raw") or "").strip()
        canonical = (row.get("canonical") or "").strip()
        if raw and canonical:
            aliases[raw] = canonical
    return aliases


def add_canonical_journals(
    df: pd.DataFrame,
    aliases: Mapping[str, str] | None = None,
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """Return ``df`` with a ``Journal_Canonical`` column beside ``Journal``.

    The observed name is never overwritten — the canonical form is what you
    group on, the observed one is what the record actually said.
    """
    out = df.copy()
    if "Journal" not in out.columns:
        out["Journal_Canonical"] = ""
        return out
    mapping = canonicalize_journals(out["Journal"].tolist(), aliases, strict=strict)
    out["Journal_Canonical"] = [
        mapping.get(str(n).strip() if journal_key(n) else "", "") for n in out["Journal"]
    ]
    return out
