"""Optional reference enrichment via CrossRef and OpenAlex.

This module is wrapped behind the ``[crossref]`` extra so the base install
stays light. Both APIs are used over plain HTTPS via :mod:`httpx`; CrossRef
is queried first because it is the canonical DOI registry, with OpenAlex
as a fallback.

The function :func:`enrich_references` is **opt-in** and is only called by
the :class:`citegraph.Pipeline` when ``enrich=True``.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from rapidfuzz.fuzz import ratio
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from citegraph.io import OutLayout

logger = logging.getLogger(__name__)


CROSSREF_URL = "https://api.crossref.org/works"
OPENALEX_URL = "https://api.openalex.org/works"


@dataclass
class EnrichConfig:
    """Configuration for the enrichment pass."""

    # Contact email included in the User-Agent for CrossRef's polite pool.
    # Set via Pipeline(enrich_config=EnrichConfig(contact_email="you@example.com"))
    # or the --enrich-contact CLI flag.
    contact_email: str = ""
    title_match_threshold: float = 90.0
    timeout_s: float = 15.0
    rows: int = 3
    max_workers: int = 8
    year_mismatch_penalty: float = 8.0
    retry_attempts: int = 3
    retry_wait_s: float = 0.25

    @property
    def user_agent(self) -> str:
        if self.contact_email:
            return f"citegraph/0.1 (mailto:{self.contact_email})"
        return "citegraph/0.1"


_DIAGNOSTIC_COLUMNS = {
    "enrichment_status": None,
    "enrichment_miss_reason": None,
    "enrichment_title_score": None,
    "enrichment_adjusted_score": None,
    "enrichment_candidate_title": None,
    "enrichment_year_match": None,
    "enrichment_year_delta": None,
}


@dataclass(frozen=True)
class _LookupReport:
    match: dict | None = None
    miss_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class _TransientEnrichmentError(Exception):
    """A temporary provider error that should be retried."""


class _LookupHTTPError(Exception):
    """A provider request failed after retry handling."""


def _try_import_httpx():
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Reference enrichment requires the [crossref] extra. "
            'Install with: pip install "citegraph[crossref]"'
        ) from exc
    return httpx


def _crossref_lookup(
    title: str,
    authors: str,
    year: int | None,
    cfg: EnrichConfig,
    client: Any,
) -> dict | None:
    return _crossref_lookup_report(title, authors, year, cfg, client).match


def _crossref_lookup_report(
    title: str,
    authors: str,
    year: int | None,
    cfg: EnrichConfig,
    client: Any,
) -> _LookupReport:
    if not title:
        return _LookupReport(miss_reason="empty_title")
    params = {
        "query.title": title,
        "rows": cfg.rows,
    }
    if authors:
        params["query.author"] = authors
    try:
        payload = _request_json(
            client,
            CROSSREF_URL,
            params=params,
            cfg=cfg,
        )
    except _LookupHTTPError as exc:
        logger.debug("CrossRef lookup failed for %r: %s", title, exc)
        return _LookupReport(miss_reason="http_error")

    items = payload.get("message", {}).get("items", [])
    return _best_match_report(items, title, year, cfg, source="crossref")


def _openalex_lookup(
    title: str,
    year: int | None,
    cfg: EnrichConfig,
    client: Any,
) -> dict | None:
    return _openalex_lookup_report(title, year, cfg, client).match


def _openalex_lookup_report(
    title: str,
    year: int | None,
    cfg: EnrichConfig,
    client: Any,
) -> _LookupReport:
    if not title:
        return _LookupReport(miss_reason="empty_title")
    params = {
        "search": title,
        "per-page": cfg.rows,
    }
    try:
        payload = _request_json(
            client,
            OPENALEX_URL,
            params=params,
            cfg=cfg,
        )
    except _LookupHTTPError as exc:
        logger.debug("OpenAlex lookup failed for %r: %s", title, exc)
        return _LookupReport(miss_reason="http_error")

    items = payload.get("results", [])
    return _best_match_report(items, title, year, cfg, source="openalex")


def _request_json(
    client: Any,
    url: str,
    *,
    params: dict[str, Any],
    cfg: EnrichConfig,
) -> dict:
    retryer = Retrying(
        stop=stop_after_attempt(cfg.retry_attempts),
        wait=wait_exponential(multiplier=cfg.retry_wait_s),
        retry=retry_if_exception_type(_TransientEnrichmentError),
        reraise=True,
    )
    try:
        for attempt in retryer:
            with attempt:
                try:
                    resp = client.get(
                        url,
                        params=params,
                        headers={"User-Agent": cfg.user_agent},
                        timeout=cfg.timeout_s,
                    )
                except Exception as exc:  # noqa: BLE001
                    if _is_transient_exception(exc):
                        raise _TransientEnrichmentError(str(exc)) from exc
                    raise
                status_code = getattr(resp, "status_code", None)
                if status_code in {429, 503}:
                    raise _TransientEnrichmentError(f"HTTP {status_code}")
                resp.raise_for_status()
                return resp.json()
    except Exception as exc:  # noqa: BLE001
        raise _LookupHTTPError(str(exc)) from exc
    return {}


def _is_transient_exception(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "timeout" in name or "timeout" in text


def _best_match(
    items: list[dict],
    title: str,
    year: int | None,
    cfg: EnrichConfig,
    *,
    source: str,
) -> dict | None:
    return _best_match_report(items, title, year, cfg, source=source).match


def _best_match_report(
    items: list[dict],
    title: str,
    year: int | None,
    cfg: EnrichConfig,
    *,
    source: str,
) -> _LookupReport:
    best: tuple[float, float, dict | None, str, bool | None, int | None] = (
        0.0,
        0.0,
        None,
        "",
        None,
        None,
    )
    for item in items:
        cand_title = _candidate_title(item, source)
        if not cand_title:
            continue
        score = ratio(title.lower(), cand_title.lower())
        cand_year = _candidate_year(item, source)
        year_match, year_delta = _year_comparison(year, cand_year)
        adjusted = score - (cfg.year_mismatch_penalty if year_delta else 0.0)
        if adjusted > best[0]:
            best = (adjusted, score, item, cand_title, year_match, year_delta)
    adjusted, score, item, cand_title, year_match, year_delta = best
    if item is None:
        return _LookupReport(miss_reason=f"no_{source}_candidates")
    diagnostics = {
        "enrichment_title_score": float(score),
        "enrichment_adjusted_score": float(adjusted),
        "enrichment_candidate_title": cand_title,
        "enrichment_year_match": year_match,
        "enrichment_year_delta": year_delta,
    }
    if adjusted < cfg.title_match_threshold:
        reason = (
            "year_mismatch"
            if score >= cfg.title_match_threshold and year_delta
            else "below_title_threshold"
        )
        return _LookupReport(miss_reason=reason, diagnostics=diagnostics)

    match = _normalize_record(item, source)
    match["enrichment_status"] = "matched"
    match["enrichment_miss_reason"] = None
    match.update(diagnostics)
    return _LookupReport(match=match)


def _candidate_title(item: dict, source: str) -> str:
    if source == "crossref":
        titles = item.get("title") or []
        return titles[0] if titles else ""
    if source == "openalex":
        return item.get("title") or item.get("display_name") or ""
    return ""


def _candidate_year(item: dict, source: str) -> int | None:
    if source == "crossref":
        year = (
            item.get("issued", {}).get("date-parts", [[None]])[0][0]
            if item.get("issued")
            else None
        )
    elif source == "openalex":
        year = item.get("publication_year")
    else:
        year = None
    try:
        return int(year) if year else None
    except (TypeError, ValueError):
        return None


def _year_comparison(
    source_year: int | None,
    candidate_year: int | None,
) -> tuple[bool | None, int | None]:
    if source_year is None or candidate_year is None:
        return None, None
    delta = abs(int(source_year) - int(candidate_year))
    return delta == 0, delta


def _normalize_record(item: dict, source: str) -> dict:
    """Flatten a CrossRef or OpenAlex hit into our DataFrame row shape.

    The author lists are returned as parallel arrays:

    - ``Authors_List`` — display names, preserved for backwards compat.
    - ``OpenAlex_Authors`` — a list of ``{display_name, openalex_id,
      orcid}`` dicts in the same positional order. Carries the
      authoritative identifiers that the author-normalization stage
      uses as ground truth. CrossRef populates ``orcid`` only when the
      record explicitly carries one; ``openalex_id`` is always ``None``
      from CrossRef.
    """
    if source == "crossref":
        author_objs = []
        for a in item.get("author", []):
            given = a.get("given")
            family = a.get("family")
            display = " ".join(filter(None, [given, family]))
            orcid = a.get("ORCID")
            if isinstance(orcid, str) and orcid.startswith("http"):
                # CrossRef returns ORCIDs as full URLs; keep just the id.
                orcid = orcid.rstrip("/").rsplit("/", 1)[-1]
            author_objs.append(
                {"display_name": display, "openalex_id": None, "orcid": orcid}
            )
        authors = [a["display_name"] for a in author_objs if a["display_name"]]
        title = (item.get("title") or [""])[0]
        container = item.get("container-title") or [""]
        journal = container[0] if container else ""
        year = (
            item.get("issued", {}).get("date-parts", [[None]])[0][0]
            if item.get("issued")
            else None
        )
        doi = item.get("DOI")
    else:  # openalex
        author_objs = []
        for a in item.get("authorships", []):
            author = a.get("author") or {}
            display = author.get("display_name", "")
            oa_id = author.get("id")
            if isinstance(oa_id, str) and oa_id.startswith("https://openalex.org/"):
                oa_id = oa_id[len("https://openalex.org/"):]
            orcid = author.get("orcid")
            if isinstance(orcid, str) and orcid.startswith("http"):
                orcid = orcid.rstrip("/").rsplit("/", 1)[-1]
            if display:
                author_objs.append(
                    {"display_name": display, "openalex_id": oa_id, "orcid": orcid}
                )
        authors = [a["display_name"] for a in author_objs]
        title = item.get("title") or item.get("display_name") or ""
        primary_location = item.get("primary_location") or {}
        source_obj = primary_location.get("source") or {}
        journal = source_obj.get("display_name") or ""
        year = item.get("publication_year")
        doi = item.get("doi")
        if isinstance(doi, str) and doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]

    return {
        "doi": doi,
        "Title": title,
        "Authors_List": authors,
        "Authors": ", ".join(authors),
        "OpenAlex_Authors": author_objs,
        "Journal": journal,
        "Year": int(year) if year else None,
        "enrichment_source": source,
    }


def _write_cache(cache_path: Path, data: dict) -> None:
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.rename(cache_path)


def _miss_record(reason: str, diagnostics: dict[str, Any] | None = None) -> dict:
    return {
        **_DIAGNOSTIC_COLUMNS,
        **(diagnostics or {}),
        "doi": None,
        "enrichment_source": None,
        "enrichment_status": "miss",
        "enrichment_miss_reason": reason,
    }


def _with_cache_diagnostics(cached: dict) -> dict:
    out = dict(cached)
    if "enrichment_status" not in out:
        out["enrichment_status"] = (
            "matched" if out.get("doi") or out.get("enrichment_source") else "miss"
        )
    if "enrichment_miss_reason" not in out:
        out["enrichment_miss_reason"] = (
            None if out["enrichment_status"] == "matched" else "cached_miss"
        )
    for column, default in _DIAGNOSTIC_COLUMNS.items():
        out.setdefault(column, default)
    return out


def _enrich_one(
    ref_id: str,
    row: pd.Series,
    cfg: EnrichConfig,
    client: Any,
    enrichment_dir: Path | None,
) -> dict:
    """Resolve one reference row, using the per-ref cache when available."""
    row_dict = row.to_dict()

    if enrichment_dir is not None:
        cache_path = enrichment_dir / f"{ref_id}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            row_dict.update(_with_cache_diagnostics(cached))
            return row_dict

    title = str(row.get("Title") or "")
    authors = str(row.get("Authors") or "")
    try:
        year = int(row.get("Year")) if row.get("Year") else None
    except (TypeError, ValueError):
        year = None

    crossref = _crossref_lookup_report(title, authors, year, cfg, client)
    openalex = _LookupReport()
    match = crossref.match
    if match is None:
        openalex = _openalex_lookup_report(title, year, cfg, client)
        match = openalex.match

    if match:
        if enrichment_dir is not None:
            _write_cache(enrichment_dir / f"{ref_id}.json", match)
        row_dict["doi"] = match["doi"]
        row_dict["enrichment_source"] = match["enrichment_source"]
        for column in ("Title", "Authors_List", "Authors", "Journal", "Year"):
            if match.get(column):
                row_dict[column] = match[column]
        # OpenAlex_Authors is a list-of-dicts; it may legitimately be []
        # for entries that have no authorships, so we don't gate on
        # truthiness — copy it through whenever the match carries it.
        if "OpenAlex_Authors" in match:
            row_dict["OpenAlex_Authors"] = match["OpenAlex_Authors"]
        for column in _DIAGNOSTIC_COLUMNS:
            row_dict[column] = match.get(column)
    else:
        report = _select_miss_report(crossref, openalex)
        miss = _miss_record(
            report.miss_reason or "no_openalex_candidates",
            report.diagnostics,
        )
        if enrichment_dir is not None:
            _write_cache(enrichment_dir / f"{ref_id}.json", miss)
        row_dict.update(miss)

    return row_dict


def _select_miss_report(*reports: _LookupReport) -> _LookupReport:
    priority = {
        "http_error": 4,
        "year_mismatch": 3,
        "below_title_threshold": 2,
        "no_crossref_candidates": 1,
        "no_openalex_candidates": 1,
        "empty_title": 0,
        None: -1,
    }
    return max(
        reports,
        key=lambda r: (
            priority.get(r.miss_reason, 0),
            1 if r.miss_reason == "no_openalex_candidates" else 0,
        ),
    )


def _write_enrichment_sidecars(
    enriched: pd.DataFrame,
    cfg: EnrichConfig,
    layout: OutLayout | None,
) -> None:
    if layout is None:
        return

    status = enriched.get("enrichment_status")
    if status is None:
        misses = enriched.iloc[0:0].copy()
    else:
        misses = enriched[status == "miss"].copy()

    miss_columns = [
        "Title",
        "Authors",
        "Year",
        "Journal",
        "enrichment_miss_reason",
        "enrichment_title_score",
        "enrichment_adjusted_score",
        "enrichment_candidate_title",
        "enrichment_year_match",
        "enrichment_year_delta",
    ]
    misses_out = misses[[c for c in miss_columns if c in misses.columns]].copy()
    if enriched.index.name == "id":
        misses_out.insert(0, "id", misses_out.index)
    misses_out.to_csv(layout.enrichment_misses_csv, index=False)

    source_counts = (
        enriched["enrichment_source"].fillna("miss").value_counts().to_dict()
        if "enrichment_source" in enriched.columns
        else {}
    )
    n_matched = int((enriched.get("enrichment_status") == "matched").sum()) if "enrichment_status" in enriched else 0
    n_missed = int((enriched.get("enrichment_status") == "miss").sum()) if "enrichment_status" in enriched else 0
    summary = {
        "n_references": int(len(enriched)),
        "n_matched": n_matched,
        "n_missed": n_missed,
        "match_rate": (n_matched / len(enriched)) if len(enriched) else 0.0,
        "sources": {str(k): int(v) for k, v in source_counts.items()},
        "config": asdict(cfg),
    }
    layout.enrichment_summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def enrich_references(
    df: pd.DataFrame,
    cfg: EnrichConfig | None = None,
    layout: OutLayout | None = None,
) -> pd.DataFrame:
    """Add ``doi`` / canonical metadata columns to ``df`` where possible.

    Rows that don't get a confident match are returned with ``doi = None``
    and the original metadata untouched. Previously resolved rows are loaded
    from the per-reference cache in ``layout.enrichment_dir`` when provided.
    """
    cfg = cfg or EnrichConfig()
    httpx = _try_import_httpx()
    enrichment_dir = layout.enrichment_dir if layout is not None else None

    results: dict[Any, dict] = {}
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
            futures = {
                pool.submit(
                    _enrich_one,
                    str(idx),
                    row,
                    cfg,
                    client,
                    enrichment_dir,
                ): idx
                for idx, row in df.iterrows()
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()

    enriched_rows = [results[idx] for idx in df.index]
    enriched = pd.DataFrame(enriched_rows)
    if df.index.name == "id":
        enriched.index = df.index

    _write_enrichment_sidecars(enriched, cfg, layout)

    logger.info(
        "Enrichment complete: %d/%d references resolved",
        enriched["doi"].notna().sum() if "doi" in enriched.columns else 0,
        len(enriched),
    )
    return enriched
