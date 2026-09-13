"""Optional reference enrichment via CrossRef and OpenAlex.

This module is wrapped behind the ``[crossref]`` extra so the base install
stays light. Both APIs are used over plain HTTPS via :mod:`httpx`; CrossRef
is queried first because it is the canonical DOI registry, with OpenAlex
as a fallback.

The function :func:`enrich_works` is **opt-in** and is only called by
the :class:`citegraph.Pipeline` when ``enrich=True``.
"""

from __future__ import annotations

import html
import json
import logging
import math
import re
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

from citegraph.authors import author_list_agreement
from citegraph.io import (
    fingerprint as content_fingerprint,
)
from citegraph.io import (
    metadata_fingerprint,
    parse_authors_list,
    unverified_collision_ids,
    write_json,
)

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
    # OpenAlex API key (raises the free daily credit budget 10x; prepaid
    # credits go further). Sent only to OpenAlex, never CrossRef, and
    # redacted from enrichment_summary.json.
    openalex_api_key: str = ""
    title_match_threshold: float = 90.0
    timeout_s: float = 15.0
    rows: int = 3
    max_workers: int = 8
    year_mismatch_penalty: float = 8.0
    # Similarity at which two name tokens count as the same person,
    # absorbing extraction typos ("Hirschmann" / "Hirschman").
    author_name_fuzz: float = 85.0
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
    "enrichment_author_agreement": None,
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
    authors: list[str],
    year: int | None,
    cfg: EnrichConfig,
    client: Any,
) -> dict | None:
    return _crossref_lookup_report(title, authors, year, cfg, client).match


def _crossref_lookup_report(
    title: str,
    authors: list[str],
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
        params["query.author"] = ", ".join(authors)
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
    return _best_match_report(items, title, year, cfg, source="crossref", authors=authors)


def _openalex_lookup(
    title: str,
    authors: list[str],
    year: int | None,
    cfg: EnrichConfig,
    client: Any,
) -> dict | None:
    return _openalex_lookup_report(title, authors, year, cfg, client).match


def _strip_openalex_wildcards(title: str) -> str:
    """Remove ``?`` and ``*`` from an OpenAlex ``search`` value.

    OpenAlex reads both as wildcard operators and rejects the entire query
    with a 400 unless ``search.exact=`` is used. Question marks are common
    in academic titles ("Is There a Role for Rural Communities?"), so
    leaving them in silently kills the fallback for those works. Dropping
    them is safe: the search is stemmed anyway and final acceptance is
    decided by rapidfuzz scoring against the returned candidates.
    """
    if not title:
        return ""
    return " ".join(title.replace("?", " ").replace("*", " ").split())


def _openalex_lookup_report(
    title: str,
    authors: list[str],
    year: int | None,
    cfg: EnrichConfig,
    client: Any,
) -> _LookupReport:
    search = _strip_openalex_wildcards(title)
    if not search:
        return _LookupReport(miss_reason="empty_title")
    params = {
        "search": search,
        "per-page": cfg.rows,
    }
    # OpenAlex's polite pool keys on a mailto query param (the User-Agent
    # form only covers CrossRef) — anonymous callers get throttled hard.
    if cfg.contact_email:
        params["mailto"] = cfg.contact_email
    if cfg.openalex_api_key:
        params["api_key"] = cfg.openalex_api_key
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
    return _best_match_report(items, title, year, cfg, source="openalex", authors=authors)


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


_AGREEMENT_LABELS = {True: "match", False: "mismatch", None: "unknown"}


@dataclass(frozen=True)
class _Candidate:
    item: dict
    title: str
    score: float
    adjusted: float
    year_match: bool | None
    year_delta: int | None
    agreement: bool | None

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "enrichment_title_score": float(self.score),
            "enrichment_adjusted_score": float(self.adjusted),
            "enrichment_candidate_title": self.title,
            "enrichment_year_match": self.year_match,
            "enrichment_year_delta": self.year_delta,
            "enrichment_author_agreement": _AGREEMENT_LABELS[self.agreement],
        }


def _best_match_report(
    items: list[dict],
    title: str,
    year: int | None,
    cfg: EnrichConfig,
    *,
    source: str,
    authors: list[str] | None = None,
) -> _LookupReport:
    """Pick the best provider record for one work.

    Author agreement is a *veto* on candidate eligibility, not a term in the
    score: the title still decides acceptance, so author evidence can never
    push a weak title over the threshold. A provider record whose authors are
    a different set of people is almost always a book review or a chapter
    filed under the reviewed work's title, and accepting it overwrites
    ``Journal`` and ``Year`` as well as the author identifiers.
    """
    scored: list[_Candidate] = []
    for item in items:
        cand_title = _candidate_title(item, source)
        if not cand_title:
            continue
        score = ratio(title.lower(), cand_title.lower())
        year_match, year_delta = _year_comparison(year, _candidate_year(item, source))
        scored.append(
            _Candidate(
                item=item,
                title=cand_title,
                score=score,
                adjusted=score - (cfg.year_mismatch_penalty if year_delta else 0.0),
                year_match=year_match,
                year_delta=year_delta,
                agreement=author_list_agreement(
                    list(authors or []),
                    _candidate_authors(item, source),
                    fuzz_threshold=cfg.author_name_fuzz,
                ),
            )
        )
    if not scored:
        return _LookupReport(miss_reason=f"no_{source}_candidates")

    eligible = [c for c in scored if c.agreement is not False]
    best = max(eligible or scored, key=lambda c: c.adjusted)
    diagnostics = best.diagnostics
    if not eligible:
        return _LookupReport(miss_reason="author_mismatch", diagnostics=diagnostics)
    if best.adjusted < cfg.title_match_threshold:
        reason = (
            "year_mismatch"
            if best.score >= cfg.title_match_threshold and best.year_delta
            else "below_title_threshold"
        )
        return _LookupReport(miss_reason=reason, diagnostics=diagnostics)

    match = _normalize_record(best.item, source)
    match["enrichment_status"] = "matched"
    match["enrichment_miss_reason"] = None
    match.update(diagnostics)
    return _LookupReport(match=match)


def _candidate_authors(item: dict, source: str) -> list[str]:
    if source == "crossref":
        return [
            " ".join(filter(None, [a.get("given"), a.get("family")]))
            for a in item.get("author", [])
        ]
    if source == "openalex":
        return [
            (a.get("author") or {}).get("display_name", "")
            for a in item.get("authorships", [])
        ]
    return []


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
    - ``OpenAlex_Authors`` — a list of ``{display_name, family, given,
      openalex_id, orcid}`` dicts in the same positional order. Carries
      the authoritative identifiers that the author-normalization stage
      uses as ground truth, plus CrossRef's structured ``family``/
      ``given`` split, which feeds the author stage's compound-surname
      lexicon. CrossRef populates ``orcid`` only when the record
      explicitly carries one; ``openalex_id`` is always ``None`` from
      CrossRef, and OpenAlex provides no family/given split.
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
            if not display:
                continue
            author_objs.append(
                {
                    "display_name": display,
                    "family": family,
                    "given": given,
                    "openalex_id": None,
                    "orcid": orcid,
                }
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
                    {
                        "display_name": display,
                        "family": None,
                        "given": None,
                        "openalex_id": oa_id,
                        "orcid": orcid,
                    }
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

    # Both providers return raw HTML entities in free-text fields, so the
    # canonical journal for one corpus read "Journal of Economic Behavior
    # &amp; Organization" on 80 works. Unescape at the source rather than in
    # each consumer.
    title = _unescape(title)
    journal = _unescape(journal)
    for obj in author_objs:
        for field_name in ("display_name", "family", "given"):
            obj[field_name] = _unescape(obj[field_name])
    authors = [a["display_name"] for a in author_objs if a["display_name"]]

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


def _unescape(value: str | None) -> str | None:
    return html.unescape(value) if isinstance(value, str) else value


def _scalar_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _write_cache(cache_path: Path, data: dict, input_fingerprint: str) -> None:
    write_json(cache_path, {"schema_version": 2, "input_fingerprint": input_fingerprint,
                           "result_fingerprint": content_fingerprint(data), "result": data})


_FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}")


def _valid_result(result: Any, *, strict: bool) -> bool:
    if not isinstance(result, dict):
        return False
    if not strict:
        if not {'doi', 'enrichment_source'} <= result.keys():
            return False
        result = _with_cache_diagnostics(result)
    status = result.get("enrichment_status")
    if strict and status not in {"matched", "miss"}:
        return False
    source = result.get("enrichment_source")
    reason = result.get("enrichment_miss_reason")
    if strict and status == "matched" and not {
        "doi", "Title", "Authors", "Authors_List", "OpenAlex_Authors",
        "Journal", "Year", "enrichment_source", "enrichment_status",
        "enrichment_miss_reason",
    }.issubset(result):
        return False
    if status == "matched" and (source not in {"crossref", "openalex"} or reason is not None):
        return False
    if status == "miss" and (source is not None or not isinstance(reason, str)):
        return False
    if "doi" in result and result["doi"] is not None and not isinstance(result["doi"], str):
        return False
    for key in ("Title", "Authors", "Journal"):
        if key in result and not isinstance(result[key], str):
            return False
    if "Authors_List" in result and (
        not isinstance(result["Authors_List"], list)
        or any(not isinstance(author, str) for author in result["Authors_List"])
    ):
        return False
    if "Year" in result and result["Year"] is not None and (
        isinstance(result["Year"], bool) or not isinstance(result["Year"], int)
    ):
        return False
    agreement = result.get("enrichment_author_agreement")
    if agreement is not None and (
        not isinstance(agreement, str) or agreement not in _AGREEMENT_LABELS.values()
    ):
        return False
    for key in ("enrichment_title_score", "enrichment_adjusted_score"):
        value = result.get(key)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return False
    candidate = result.get("enrichment_candidate_title")
    if candidate is not None and not isinstance(candidate, str):
        return False
    year_match = result.get("enrichment_year_match")
    if year_match is not None and not isinstance(year_match, bool):
        return False
    year_delta = result.get("enrichment_year_delta")
    if year_delta is not None and (
        isinstance(year_delta, bool) or not isinstance(year_delta, int) or year_delta < 0
    ):
        return False
    authors = result.get("OpenAlex_Authors")
    if authors is not None and (
        not isinstance(authors, list)
        or any(
            not isinstance(author, dict)
            or not isinstance(author.get("display_name"), str)
            or any(
                author.get(key) is not None and not isinstance(author.get(key), str)
                for key in ("family", "given", "openalex_id", "orcid")
            )
            for author in authors
        )
    ):
        return False
    return True


def _read_cache_index(
    layout: OutLayout,
) -> tuple[dict[str, tuple[str, dict]], dict[str, list[dict]], dict[str, dict]]:
    """Read each enrichment cache once and index validated envelopes."""
    by_id: dict[str, tuple[str, dict]] = {}
    by_fingerprint: dict[str, list[dict]] = {}
    legacy_by_id: dict[str, dict] = {}
    if not layout.enrichment_dir.is_dir():
        return by_id, by_fingerprint, legacy_by_id
    for path in layout.enrichment_dir.glob("*.json"):
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            logger.warning("Ignoring malformed enrichment cache %s: %s", path.name, exc)
            continue
        if not isinstance(cached, dict):
            logger.warning("Ignoring malformed enrichment cache %s", path.name)
            continue
        if cached.get("schema_version") == 2:
            fingerprint = cached.get("input_fingerprint")
            result = cached.get("result")
            if (
                not isinstance(fingerprint, str)
                or _FINGERPRINT_RE.fullmatch(fingerprint) is None
                or not _valid_result(result, strict=True)
                or ('result_fingerprint' in cached
                    and cached['result_fingerprint'] != content_fingerprint(result))
            ):
                logger.warning("Ignoring malformed enrichment cache %s", path.name)
                continue
            if result.get("enrichment_miss_reason") == "http_error":
                continue
            by_id[path.stem] = (fingerprint, result)
            by_fingerprint.setdefault(fingerprint, []).append(result)
        elif (
            'schema_version' not in cached and _valid_result(cached, strict=False)
            and cached.get("enrichment_miss_reason") != "http_error"
        ):
            legacy_by_id[path.stem] = cached
        else:
            logger.warning("Ignoring malformed enrichment cache %s", path.name)
    return by_id, by_fingerprint, legacy_by_id


def _same_result(results: list[dict]) -> dict | None:
    if not results:
        return None
    serialized = {json.dumps(result, sort_keys=True, ensure_ascii=False) for result in results}
    return results[0] if len(serialized) == 1 else None


def load_cached_enrichments(
    works: pd.DataFrame,
    layout: OutLayout,
    *,
    verified_only: bool = True,
) -> pd.DataFrame:
    """Load independently validated cache hits for the current works, offline."""
    by_id, by_fingerprint, legacy_by_id = _read_cache_index(layout)
    collisions = set() if verified_only else unverified_collision_ids(layout)
    rows: dict[Any, dict] = {}
    for work_id, row in works.iterrows():
        work_id_str = str(work_id)
        fingerprint = metadata_fingerprint(row.to_dict())
        exact = by_id.get(work_id_str)
        result = exact[1] if exact is not None and exact[0] == fingerprint else None
        if result is None:
            result = _same_result(by_fingerprint.get(fingerprint, []))
        if result is None and not verified_only and work_id_str not in collisions:
            result = legacy_by_id.get(work_id_str)
            if result is None and work_id_str.startswith("w-"):
                result = legacy_by_id.get(f"r-{work_id_str[2:]}")
        if result is not None:
            rows[work_id] = _apply_enrichment_result(row.to_dict(), result)
    cached = pd.DataFrame.from_dict(rows, orient="index")
    cached.index.name = works.index.name
    return cached


def _apply_enrichment_result(row: dict, result: dict) -> dict:
    """Apply the same non-destructive policy on cache hits and fresh matches."""
    out = dict(row)
    bibliographic = {"Title", "Authors", "Authors_List", "Journal", "Year"}
    for key, value in _with_cache_diagnostics(result).items():
        if key not in bibliographic or (value is not None and not (isinstance(value, float) and math.isnan(value)) and bool(value)):
            out[key] = value
    authors = parse_authors_list(out.get("Authors_List"))
    if authors:
        out["Authors_List"] = authors
        out["Authors"] = ", ".join(authors)
    return out


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
    *,
    read_cache: bool = True,
) -> dict:
    """Resolve one reference row, using the per-ref cache when available."""
    row_dict = row.to_dict()
    input_fingerprint = metadata_fingerprint(row_dict)

    if enrichment_dir is not None and read_cache:
        cache_path = enrichment_dir / f"{ref_id}.json"
        if not cache_path.exists() and ref_id.startswith("w-"):
            # Pre-works-model corpora cached under the legacy r- prefix;
            # the slug body is identical, so serve those without re-crawling.
            legacy = enrichment_dir / f"r-{ref_id[2:]}.json"
            if legacy.exists():
                cache_path = legacy
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if not isinstance(cached, dict):
                    raise TypeError("expected a JSON object")
                valid = True
                if cached.get("schema_version") == 2:
                    result = cached.get("result")
                    if not isinstance(result, dict):
                        raise TypeError("expected a result object")
                    valid = cached.get("input_fingerprint") == input_fingerprint
                    if 'result_fingerprint' in cached:
                        valid = valid and cached['result_fingerprint'] == content_fingerprint(result)
                    cached = result
                else:
                    # Path is derived from OutLayout, including for direct cache callers.
                    from citegraph.io import OutLayout, unverified_collision_ids
                    valid = ref_id not in unverified_collision_ids(OutLayout(enrichment_dir.parent))
            except (OSError, UnicodeError, TypeError, ValueError) as exc:
                logger.warning("Refreshing malformed enrichment cache for %s: %s", ref_id, exc)
            else:
                # An http_error miss is a transient outage (rate limit, 5xx,
                # timeout), not a lookup result — retry the providers.
                if valid and cached.get("enrichment_miss_reason") != "http_error":
                    return _apply_enrichment_result(row_dict, cached)
                if not valid:
                    logger.warning("Refreshing incompatible enrichment cache for %s", ref_id)

    # NaN is truthy and str()s to "nan", which would be sent as a real query.
    title = _scalar_str(row.get("Title"))
    authors = parse_authors_list(row.get("Authors_List")) or [
        a.strip() for a in _scalar_str(row.get("Authors")).split(",") if a.strip()
    ]
    try:
        year = int(row.get("Year")) if row.get("Year") else None
    except (TypeError, ValueError):
        year = None

    crossref = _crossref_lookup_report(title, authors, year, cfg, client)
    openalex = _LookupReport()
    match = crossref.match
    if match is None:
        openalex = _openalex_lookup_report(title, authors, year, cfg, client)
        match = openalex.match

    if match:
        result = match
    else:
        report = _select_miss_report(crossref, openalex)
        result = _miss_record(report.miss_reason or "no_openalex_candidates", report.diagnostics)
    if enrichment_dir is not None:
        _write_cache(enrichment_dir / f"{ref_id}.json", result, input_fingerprint)
    return _apply_enrichment_result(row_dict, result)


def _select_miss_report(*reports: _LookupReport) -> _LookupReport:
    priority = {
        "http_error": 5,
        "author_mismatch": 4,
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
        "enrichment_author_agreement",
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
    cfg_out = asdict(cfg)
    if cfg_out.get("openalex_api_key"):
        cfg_out["openalex_api_key"] = "***"
    summary = {
        "n_references": int(len(enriched)),
        "n_matched": n_matched,
        "n_missed": n_missed,
        "match_rate": (n_matched / len(enriched)) if len(enriched) else 0.0,
        "sources": {str(k): int(v) for k, v in source_counts.items()},
        "config": cfg_out,
    }
    if "ring" in enriched.columns:
        summary["by_ring"] = {
            str(ring): {
                "n": int(len(group)),
                "n_matched": int((group.get("enrichment_status") == "matched").sum()),
            }
            for ring, group in enriched.groupby("ring")
        }
    layout.enrichment_summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def enrich_works(
    df: pd.DataFrame,
    cfg: EnrichConfig | None = None,
    layout: OutLayout | None = None,
) -> pd.DataFrame:
    """Add ``doi`` / canonical metadata columns to ``df`` where possible.

    ``df`` is any works-shaped frame (``Title``/``Authors``/``Year``
    columns, indexed by ``id``) — all rings are treated identically. Rows
    that don't get a confident match are returned with ``doi = None`` and
    the original metadata untouched. Previously resolved rows are loaded
    from the per-work cache in ``layout.enrichment_dir`` when provided
    (legacy ``r-``-prefixed cache files are honored for ``w-`` ids).
    """
    cfg = cfg or EnrichConfig()
    enrichment_dir = layout.enrichment_dir if layout is not None else None

    cached = (
        load_cached_enrichments(df, layout, verified_only=False)
        if layout is not None
        else pd.DataFrame()
    )
    results: dict[Any, dict] = cached.to_dict(orient="index")
    pending = [(idx, row) for idx, row in df.iterrows() if idx not in results]
    if pending:
        httpx = _try_import_httpx()
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
                        read_cache=False,
                    ): idx
                    for idx, row in pending
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
        "Enrichment complete: %d/%d works resolved",
        enriched["doi"].notna().sum() if "doi" in enriched.columns else 0,
        len(enriched),
    )
    return enriched
