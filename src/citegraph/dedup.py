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
from dataclasses import dataclass

import pandas as pd
from rapidfuzz.fuzz import ratio

from citegraph.ids import make_reference_id

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

    title_weight: float = 0.7
    authors_weight: float = 0.3
    journal_weight: float = 0.0
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
    title_score = ratio(normalize_text(paper1.get("Title")), normalize_text(paper2.get("Title")))
    authors_score = ratio(
        normalize_text(paper1.get("Authors")), normalize_text(paper2.get("Authors"))
    )
    journal_score = ratio(
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
    return {
        "Authors": row.get("Authors"),
        "Journal": row.get("Journal"),
        "Title": row.get("Title"),
        "Year": row.get("Year"),
    }


def dedup_references(
    df: pd.DataFrame,
    cfg: DedupConfig | None = None,
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
    if df.empty:
        empty = df.copy()
        empty["id"] = pd.Series(dtype=str)
        return empty.set_index("id"), pd.Series([], dtype=str, index=df.index)

    df = df.reset_index(drop=True)
    cluster_ids: list[str | None] = [None] * len(df)
    representatives: list[tuple[str, dict]] = []

    for i in range(len(df)):
        if cluster_ids[i] is not None:
            continue
        paper_i = _row_to_dict(df.iloc[i])
        cluster_id = make_reference_id(
            df.iloc[i].get("Authors_List") or df.iloc[i].get("Authors", ""),
            df.iloc[i].get("Year"),
            df.iloc[i].get("Title", ""),
        )
        cluster_ids[i] = cluster_id
        representatives.append((cluster_id, paper_i))

        for j in range(i + 1, len(df)):
            if cluster_ids[j] is not None:
                continue
            if compare_papers(paper_i, _row_to_dict(df.iloc[j]), cfg):
                cluster_ids[j] = cluster_id

    mapping = pd.Series(cluster_ids, index=df.index, name="cited_id", dtype=object)

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
