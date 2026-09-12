"""Self-contained HTML QC dashboard for a citegraph out directory.

``citegraph report --out ./out`` (or :func:`write_report`) reads whatever
artifacts exist under ``out_dir`` and writes a single ``report.html`` next
to them. The page is fully self-contained — no CDN scripts, no webfonts,
no external requests — so it can be opened from disk and archived with the
CSVs.

The module is deliberately split into two halves so the data layer is
testable without parsing HTML:

- :func:`collect_report_data` gathers everything into one JSON-serializable
  dict. Every individual file read is fault-tolerant: a corrupt JSON/CSV is
  recorded as ``{"path": ..., "error": ...}`` in ``data["errors"]`` instead
  of raising. A missing or empty ``out_dir`` yields a valid "nothing yet"
  structure.
- :func:`build_report_html` renders that dict to HTML. All user-derived
  strings are escaped server-side; the embedded JSON blob additionally has
  ``<``/``>``/``&`` escaped as ``\\uXXXX`` so it is inert markup.
"""

from __future__ import annotations

import html
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from citegraph.io import (
    AUTHOR_AUDIT_INPUTS,
    AUTHOR_AUDIT_OUTPUTS,
    WORK_AUDIT_INPUTS,
    WORK_AUDIT_OUTPUTS,
    OutLayout,
    artifact_fingerprint,
    read_json,
)

__all__ = ["collect_report_data", "build_report_html", "write_report"]

# Mirrors ``citegraph.pdf_to_markdown._is_image_only`` but operates on text
# already read (each markdown file is read exactly once during collection).
_IMAGE_ONLY_MIN_CHARS = 200

# Per-paper preview caps (bytes of markdown, number of references).
_MARKDOWN_PREVIEW_CHARS = 2_500
_REFS_PREVIEW_N = 15
# When the rendered page would exceed this, previews are shrunk and a note
# is added to the page.
_MAX_HTML_BYTES = 10 * 1024 * 1024
_SHRUNK_MARKDOWN_PREVIEW_CHARS = 500
_SHRUNK_REFS_PREVIEW_N = 5

_SUSPICIOUSLY_FEW_REFS = 5
_SHORT_TITLE_CHARS = 10
_MERGE_AUDIT_MAX_CLUSTERS = 200


# ---------------------------------------------------------------------------
# Fault-tolerant readers
# ---------------------------------------------------------------------------
def _record_error(errors: list[dict], path: Path, exc: Exception) -> None:
    errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})


def _read_text_safe(path: Path, errors: list[dict]) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - corrupt files are a finding, not a crash
        _record_error(errors, path, exc)
        return None


def _read_json_safe(path: Path, errors: list[dict]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _record_error(errors, path, exc)
        return None


def _read_jsonl_safe(path: Path, errors: list[dict]) -> list[dict] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _record_error(errors, path, exc)
        return None
    rows: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception as exc:  # noqa: BLE001
            _record_error(errors, path, ValueError(f"line {lineno}: {exc}"))
    return rows


def _read_csv_safe(path: Path, errors: list[dict], **kwargs: Any) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:  # noqa: BLE001
        _record_error(errors, path, exc)
        return None


# ---------------------------------------------------------------------------
# Small value helpers
# ---------------------------------------------------------------------------
def _is_image_only_text(text: str, min_text_chars: int = _IMAGE_ONLY_MIN_CHARS) -> bool:
    stripped = text.replace("<!-- image -->", "")
    lines = [ln for ln in stripped.splitlines() if not ln.lstrip().startswith("#")]
    return len("".join(lines).strip()) < min_text_chars


def _as_int(value: object) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _jsonable(obj: Any) -> Any:
    """Coerce numpy scalars / NaN into plain JSON-safe Python values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    item = getattr(obj, "item", None)
    if callable(item):  # numpy scalar
        try:
            return _jsonable(item())
        except Exception:  # noqa: BLE001
            return str(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
def collect_report_data(out_dir: Path, *, include_markdown_previews: bool = True) -> dict:
    """Gather everything the report shows into one JSON-serializable dict.

    Never raises for missing or corrupt artifacts: a nonexistent ``out_dir``
    returns a valid "nothing yet" structure and every unreadable file is
    recorded in ``data["errors"]``.

    ``include_markdown_previews=False`` omits the per-paper markdown preview
    text (``markdown_preview`` is ``None``) while still computing char counts
    and the image-only check — used by ``citegraph ui``, which fetches
    previews lazily per paper instead of shipping them all in one payload.
    """
    out_dir = Path(out_dir)
    layout = OutLayout(out_dir)
    errors: list[dict] = []

    data: dict = {
        "out_dir": str(out_dir),
        "out_dir_exists": out_dir.is_dir(),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "errors": errors,
        "truncated_for_size": False,
    }

    # --- sidecars -----------------------------------------------------
    conversion_warnings = (
        _read_json_safe(layout.conversion_warnings_json, errors)
        if layout.conversion_warnings_json.exists()
        else None
    )
    source_duplicates = (
        _read_json_safe(layout.source_duplicates_json, errors)
        if layout.source_duplicates_json.exists()
        else None
    )
    papers_no_refs = (
        _read_json_safe(layout.papers_no_references_json, errors)
        if layout.papers_no_references_json.exists()
        else None
    )
    metadata_failures = (
        _read_jsonl_safe(layout.metadata_failures_jsonl, errors)
        if layout.metadata_failures_jsonl.exists()
        else None
    )
    references_failures = (
        _read_jsonl_safe(layout.references_failures_jsonl, errors)
        if layout.references_failures_jsonl.exists()
        else None
    )
    author_review = (
        _read_json_safe(layout.author_review_json, errors)
        if layout.author_review_json.exists()
        else None
    )

    duplicate_of: dict[str, str] = {}  # duplicate stem -> canonical source file
    if isinstance(source_duplicates, list):
        for group in source_duplicates:
            if not isinstance(group, dict):
                continue
            canonical = _as_str(group.get("canonical_source_file"))
            for dup in group.get("duplicate_source_files", []):
                duplicate_of[Path(_as_str(dup)).stem] = canonical

    metadata_failed_stems = {
        Path(_as_str(f.get("source_file"))).stem for f in (metadata_failures or [])
    }
    references_failed_stems = {
        Path(_as_str(f.get("source_file"))).stem for f in (references_failures or [])
    }
    no_ref_stems = {
        Path(_as_str(e.get("source_file"))).stem
        for e in (papers_no_refs or [])
        if isinstance(e, dict)
    }

    # --- per-paper rows -----------------------------------------------
    md_stems = (
        {p.stem for p in layout.markdown_dir.glob("*.md")}
        if layout.markdown_dir.is_dir()
        else set()
    )
    meta_stems = (
        {p.stem for p in layout.metadata_dir.glob("*.json")}
        if layout.metadata_dir.is_dir()
        else set()
    )
    ref_stems = (
        {p.stem for p in layout.references_dir.glob("*.json")}
        if layout.references_dir.is_dir()
        else set()
    )

    papers: list[dict] = []
    n_image_only = 0
    n_zero_refs = 0
    n_pending = 0
    for stem in sorted(md_stems | meta_stems | ref_stems):
        row = _collect_paper_row(
            stem,
            layout,
            errors,
            in_markdown=stem in md_stems,
            in_metadata=stem in meta_stems,
            in_references=stem in ref_stems,
            duplicate_of=duplicate_of.get(stem),
            metadata_failed=stem in metadata_failed_stems,
            references_failed=stem in references_failed_stems,
            no_refs_flagged=stem in no_ref_stems,
            include_markdown_preview=include_markdown_previews,
        )
        if row["markdown"]["status"] == "image-only":
            n_image_only += 1
        if row["references"]["status"] == "zero":
            n_zero_refs += 1
        if row["metadata"]["status"] == "pending" or row["references"]["status"] == "pending":
            n_pending += 1
        papers.append(row)
    data["papers"] = papers

    # --- flat CSV artifacts -------------------------------------------
    sources_df = (
        _read_csv_safe(layout.sources_csv, errors) if layout.sources_csv.exists() else None
    )
    raw_refs_df = (
        _read_csv_safe(layout.citations_raw_csv, errors)
        if layout.citations_raw_csv.exists()
        else None
    )
    works_df = (
        _read_csv_safe(layout.works_csv, errors, index_col="id")
        if layout.works_csv.exists()
        else None
    )
    graph_df = (
        _read_csv_safe(layout.graph_csv, errors) if layout.graph_csv.exists() else None
    )
    enriched_df = (
        _read_csv_safe(layout.enriched_works_csv, errors)
        if layout.enriched_works_csv.exists()
        else None
    )

    # Pre-works-model out_dirs: surface as a finding, never a crash.
    legacy_papers_csv = layout.out_dir / "papers.csv"
    if works_df is None and legacy_papers_csv.exists():
        errors.append(
            {
                "path": str(legacy_papers_csv),
                "error": (
                    "legacy pre-works-model artifacts detected — re-run "
                    "`citegraph metadata`, `citegraph references`, and "
                    "`citegraph dedup` (per-paper caches are reused)"
                ),
            }
        )
    misses_df = (
        _read_csv_safe(layout.enrichment_misses_csv, errors)
        if layout.enrichment_misses_csv.exists()
        else None
    )
    enrichment_summary = (
        _read_json_safe(layout.enrichment_summary_json, errors)
        if layout.enrichment_summary_json.exists()
        else None
    )
    authors_df = (
        _read_csv_safe(layout.authors_csv, errors, index_col="id")
        if layout.authors_csv.exists()
        else None
    )
    author_citations_df = (
        _read_csv_safe(layout.author_citations_csv, errors)
        if layout.author_citations_csv.exists()
        else None
    )
    run_summary = (
        _read_json_safe(layout.out_dir / "run_summary.json", errors)
        if (layout.out_dir / "run_summary.json").exists()
        else None
    )
    artifact_manifest = (
        _read_json_safe(layout.artifact_manifest_json, errors)
        if layout.artifact_manifest_json.exists()
        else None
    )

    # --- stage progress strip -----------------------------------------
    n_md = len(md_stems)
    n_meta = len(meta_stems)
    n_ref = len(ref_stems)
    n_ref_expected = n_md
    data["stages"] = _collect_stages(
        layout,
        n_md=n_md,
        n_meta=n_meta,
        n_ref=n_ref,
        n_ref_expected=n_ref_expected,
        works_df=works_df,
        graph_df=graph_df,
        enriched_df=enriched_df,
        authors_df=authors_df,
    )

    # --- works-model summary ------------------------------------------
    ring_counts: dict[int, int] = {}
    core_to_core: list[dict] = []
    if works_df is not None and "ring" in works_df.columns:
        rings = pd.to_numeric(works_df["ring"], errors="coerce")
        ring_counts = {
            int(k): int(v) for k, v in rings.value_counts().items() if pd.notna(k)
        }
        if graph_df is not None and {"citing_id", "cited_id"} <= set(graph_df.columns):
            ring0 = set(works_df.index[rings == 0])
            cc = graph_df[
                graph_df["citing_id"].isin(ring0) & graph_df["cited_id"].isin(ring0)
            ]
            core_to_core = [
                {"citing_id": _as_str(r["citing_id"]), "cited_id": _as_str(r["cited_id"])}
                for _, r in cc.iterrows()
            ]
    data["ring_counts"] = ring_counts
    data["core_to_core_edges"] = core_to_core

    # --- panels -------------------------------------------------------
    data["dedup"] = _collect_dedup_panel(
        raw_refs_df, works_df, graph_df, errors, layout, sources_df=sources_df
    )
    if data["dedup"] is not None:
        data["dedup"]["ring_counts"] = ring_counts
        data["dedup"]["core_to_core"] = core_to_core
    data["enrichment"] = _collect_enrichment_panel(enriched_df, misses_df, enrichment_summary)
    data["authors"] = _collect_authors_panel(authors_df, author_citations_df, author_review)
    if data["authors"] is not None:
        data["authors"].update(_collect_resolution_audit(
            layout.author_resolution_audit_json, errors,
            inputs=AUTHOR_AUDIT_INPUTS, outputs=AUTHOR_AUDIT_OUTPUTS,
        ))
        data["authors"].pop("saved_audit", None)

    # --- problems summary ---------------------------------------------
    n_missing_year = 0
    if data["dedup"] is not None:
        n_missing_year = data["dedup"].get("n_missing_year") or 0
    cards = [
        ("conversion_warnings", "image-only markdown", n_image_only, "papers"),
        ("metadata_failures", "metadata failures", len(metadata_failures or []), "papers"),
        ("references_failures", "references failures", len(references_failures or []), "papers"),
        ("zero_references", "papers with zero references", n_zero_refs, "papers"),
        ("source_duplicates", "duplicate source PDFs", len(duplicate_of), "papers"),
        ("corrupt_files", "corrupt / unreadable files", len(errors), "integrity"),
        ("author_review", "author clusters to review", len(author_review or []), "authors"),
        ("missing_year", "references without a year", n_missing_year, "dedup"),
        ("pending_papers", "papers not fully extracted", n_pending, "papers"),
    ]
    data["problems"] = {
        "cards": [
            {"key": key, "label": label, "count": int(count), "anchor": anchor}
            for key, label, count, anchor in cards
            if count
        ],
        "total": int(sum(count for _, _, count, _ in cards)),
    }

    # --- integrity table ----------------------------------------------
    data["integrity"] = _collect_integrity(
        layout,
        errors,
        rows=[
            ("sources.csv", layout.sources_csv, sources_df is not None, "stage 2"),
            ("citations_raw.csv", layout.citations_raw_csv, raw_refs_df is not None, "stage 3"),
            ("works.csv", layout.works_csv, works_df is not None, "stage 4"),
            ("citation_graph.csv", layout.graph_csv, graph_df is not None, "stage 4"),
            (
                "enriched_works.csv",
                layout.enriched_works_csv,
                enriched_df is not None,
                "stage 5 (optional)",
            ),
            (
                "enrichment_summary.json",
                layout.enrichment_summary_json,
                enrichment_summary is not None,
                "stage 5 (optional)",
            ),
            (
                "enrichment_misses.csv",
                layout.enrichment_misses_csv,
                misses_df is not None,
                "stage 5 (optional)",
            ),
            ("authors.csv", layout.authors_csv, authors_df is not None, "stage 6 (optional)"),
            (
                "author_citations.csv",
                layout.author_citations_csv,
                author_citations_df is not None,
                "stage 6 (optional)",
            ),
            (
                "author_review.json",
                layout.author_review_json,
                author_review is not None,
                "sidecar — absent = clean",
            ),
            (
                "conversion_warnings.json",
                layout.conversion_warnings_json,
                conversion_warnings is not None,
                "sidecar — absent = clean",
            ),
            (
                "source_duplicates.json",
                layout.source_duplicates_json,
                source_duplicates is not None,
                "sidecar — absent = clean",
            ),
            (
                "papers_no_references.json",
                layout.papers_no_references_json,
                papers_no_refs is not None,
                "sidecar — absent = clean",
            ),
            (
                "metadata_failures.jsonl",
                layout.metadata_failures_jsonl,
                metadata_failures is not None,
                "sidecar — absent = clean",
            ),
            (
                "references_failures.jsonl",
                layout.references_failures_jsonl,
                references_failures is not None,
                "sidecar — absent = clean",
            ),
            (
                "run_summary.json",
                layout.out_dir / "run_summary.json",
                run_summary is not None,
                "written by `citegraph run`",
            ),
            (
                "artifact_manifest.json",
                layout.artifact_manifest_json,
                artifact_manifest is not None,
                "written by `citegraph run`",
            ),
        ],
    )

    return _jsonable(data)


def _collect_paper_row(
    stem: str,
    layout: OutLayout,
    errors: list[dict],
    *,
    in_markdown: bool,
    in_metadata: bool,
    in_references: bool,
    duplicate_of: str | None,
    metadata_failed: bool,
    references_failed: bool,
    no_refs_flagged: bool,
    include_markdown_preview: bool = True,
) -> dict:
    notes: list[str] = []

    # -- markdown / conversion
    markdown: dict = {"status": "missing", "chars": None}
    markdown_preview: str | None = None
    if in_markdown:
        text = _read_text_safe(layout.markdown_dir / f"{stem}.md", errors)
        if text is None:
            markdown = {"status": "error", "chars": None}
        else:
            markdown = {
                "status": "image-only" if _is_image_only_text(text) else "ok",
                "chars": len(text),
            }
            if include_markdown_preview:
                markdown_preview = text[:_MARKDOWN_PREVIEW_CHARS]

    # -- metadata
    metadata: dict = {"status": "pending", "title": "", "authors": "", "journal": "", "year": None}
    if in_metadata:
        obj = _read_json_safe(layout.metadata_dir / f"{stem}.json", errors)
        if not isinstance(obj, dict):
            metadata["status"] = "corrupt"
        else:
            metadata = {
                "status": "ok",
                "title": _as_str(obj.get("Title")),
                "authors": ", ".join(str(a) for a in obj.get("Authors_List") or [])
                or _as_str(obj.get("Authors")),
                "journal": _as_str(obj.get("Journal")),
                "year": _as_int(obj.get("Year")),
            }
    elif metadata_failed:
        metadata["status"] = "failed"

    # -- references
    references: dict = {"status": "pending", "count": None}
    refs_preview: list[dict] = []
    refs_more = 0
    if in_references:
        obj = _read_json_safe(layout.references_dir / f"{stem}.json", errors)
        if not isinstance(obj, list):
            references["status"] = "corrupt"
        else:
            count = len(obj)
            references = {"status": "zero" if count == 0 else "ok", "count": count}
            if 0 < count < _SUSPICIOUSLY_FEW_REFS:
                notes.append(f"suspiciously few references ({count})")
            for ref in obj[:_REFS_PREVIEW_N]:
                if not isinstance(ref, dict):
                    continue
                refs_preview.append(
                    {
                        "title": _as_str(ref.get("Title")),
                        "authors": ", ".join(str(a) for a in ref.get("Authors_List") or [])
                        or _as_str(ref.get("Authors")),
                        "year": _as_int(ref.get("Year")),
                    }
                )
            refs_more = max(count - _REFS_PREVIEW_N, 0)
    elif references_failed:
        references["status"] = "failed"
    elif duplicate_of:
        references["status"] = "skipped"

    if duplicate_of:
        notes.append(f"duplicate of {Path(duplicate_of).stem}")
    if no_refs_flagged and references["status"] != "zero":
        notes.append("flagged in papers_no_references.json")

    problem = bool(
        markdown["status"] != "ok"
        or metadata["status"] != "ok"
        or references["status"] not in ("ok", "skipped")
        or notes
    )
    return {
        "stem": stem,
        "markdown": markdown,
        "markdown_preview": markdown_preview,
        "metadata": metadata,
        "references": references,
        "refs_preview": refs_preview,
        "refs_more": refs_more,
        "notes": notes,
        "problem": problem,
    }


def _collect_stages(
    layout: OutLayout,
    *,
    n_md: int,
    n_meta: int,
    n_ref: int,
    n_ref_expected: int,
    works_df: pd.DataFrame | None,
    graph_df: pd.DataFrame | None,
    enriched_df: pd.DataFrame | None,
    authors_df: pd.DataFrame | None,
) -> list[dict]:
    out = layout.out_dir

    def cache_stage(label: str, cached: int, expected: int, csv_exists: bool, cmd: str) -> dict:
        if cached == 0 and not csv_exists:
            return {"label": label, "status": "not_run", "detail": "not run", "hint": cmd}
        if expected and cached < expected:
            return {
                "label": label,
                "status": "partial",
                "detail": f"{cached}/{expected} papers cached",
                "hint": cmd,
            }
        return {"label": label, "status": "done", "detail": f"{cached} papers cached", "hint": ""}

    stages = [
        {
            "label": "convert",
            "status": "done" if n_md else "not_run",
            "detail": f"{n_md} markdown file(s)" if n_md else "not run",
            "hint": f"citegraph convert <pdf_dir> --out {out}" if not n_md else "",
        },
        cache_stage(
            "metadata", n_meta, n_md, layout.sources_csv.exists(), f"citegraph metadata --out {out}"
        ),
        cache_stage(
            "references",
            n_ref,
            n_ref_expected,
            layout.citations_raw_csv.exists(),
            f"citegraph references --out {out}",
        ),
    ]

    have_works = works_df is not None
    have_graph = graph_df is not None
    if have_works and have_graph:
        stages.append(
            {
                "label": "dedup",
                "status": "done",
                "detail": f"{len(works_df)} works / {len(graph_df)} edges",
                "hint": "",
            }
        )
    elif layout.works_csv.exists() or layout.graph_csv.exists():
        stages.append(
            {
                "label": "dedup",
                "status": "partial",
                "detail": "artifacts incomplete",
                "hint": f"citegraph dedup --out {out}",
            }
        )
    else:
        stages.append(
            {
                "label": "dedup",
                "status": "not_run",
                "detail": "not run",
                "hint": f"citegraph dedup --out {out}",
            }
        )

    stages.append(
        {
            "label": "enrich",
            "status": "done" if enriched_df is not None else "optional_not_run",
            "detail": f"{len(enriched_df)} works"
            if enriched_df is not None
            else "optional, not run",
            "hint": "" if enriched_df is not None else f"citegraph enrich --out {out}",
        }
    )
    stages.append(
        {
            "label": "authors",
            "status": "done" if authors_df is not None else "optional_not_run",
            "detail": f"{len(authors_df)} canonical authors"
            if authors_df is not None
            else "optional, not run",
            "hint": "" if authors_df is not None else f"citegraph authors --out {out}",
        }
    )
    return stages


def _collect_resolution_audit(
    path: Path, errors: list[dict], *, inputs: set[str], outputs: set[str],
) -> dict:
    """Read saved evidence only when every bound checkpoint still matches."""
    result = {"audit_status": "unavailable", "audit_config": None, "audit_decisions": []}
    if not path.exists():
        return result
    try:
        saved = read_json(path)
        if not isinstance(saved, dict):
            raise ValueError("Audit must be a JSON object")
        if saved.get("schema_version") != 2:
            return result
        for key, expected in (("input_fingerprints", inputs), ("output_fingerprints", outputs)):
            values = saved.get(key)
            if not isinstance(values, dict) or set(values) != expected:
                raise ValueError(f"Audit has invalid {key}")
            if any(value is not None and (not isinstance(value, str) or len(value) != 64)
                   for value in values.values()):
                raise ValueError(f"Audit has invalid digest in {key}")
        if not isinstance(saved.get("config"), dict) or not isinstance(saved.get("algorithm_version"), str):
            raise ValueError("Audit is missing its algorithm/configuration")
        decisions = saved.get("decisions")
        if not isinstance(decisions, list) or any(not isinstance(row, dict) for row in decisions):
            raise ValueError("Audit decisions must be a list of objects")
        for key in ("input_fingerprints", "output_fingerprints"):
            for name, expected in saved[key].items():
                if artifact_fingerprint(path.parent / name) != expected:
                    result["audit_status"] = "stale"
                    return result
        result.update(
            audit_status="current", audit_config=saved["config"],
            audit_algorithm=saved["algorithm_version"], audit_decisions=decisions[:200],
            audit_decisions_truncated=max(len(decisions) - 200, 0),
            enrichment_status=saved.get("enrichment_status"), saved_audit=saved,
        )
    except Exception as exc:  # noqa: BLE001 - corrupt artifacts are report findings
        result["audit_status"] = "invalid"
        _record_error(errors, path, exc)
    return result


def _collect_dedup_panel(
    raw_refs_df: pd.DataFrame | None,
    works_df: pd.DataFrame | None,
    graph_df: pd.DataFrame | None,
    errors: list[dict],
    layout: OutLayout,
    *,
    sources_df: pd.DataFrame | None = None,
) -> dict | None:
    if raw_refs_df is None or works_df is None:
        return None

    panel: dict = {
        "n_raw": int(len(raw_refs_df)),
        "n_canonical": int(len(works_df)),
        "n_edges": int(len(graph_df)) if graph_df is not None else None,
    }

    # Cheap corpus-level stats over the raw citation events.
    if "Year" in raw_refs_df.columns:
        years = pd.to_numeric(raw_refs_df["Year"], errors="coerce")
        panel["n_missing_year"] = int((years.isna() | (years <= 0)).sum())
    else:
        panel["n_missing_year"] = 0
    if "Authors" in raw_refs_df.columns:
        authors_col = raw_refs_df["Authors"]
    elif "Authors_List" in raw_refs_df.columns:
        authors_col = raw_refs_df["Authors_List"]
    else:
        authors_col = None
    panel["n_empty_authors"] = (
        int(authors_col.fillna("").astype(str).str.strip().isin(["", "[]"]).sum())
        if authors_col is not None
        else 0
    )
    if "Title" in raw_refs_df.columns:
        titles = raw_refs_df["Title"]
        panel["n_short_titles"] = int(
            (titles.fillna("").astype(str).str.strip().str.len() < _SHORT_TITLE_CHARS).sum()
        )
    else:
        panel["n_short_titles"] = 0

    # Top-10 most-cited canonical works (core works rank here too).
    top_cited: list[dict] = []
    if graph_df is not None and "cited_id" in graph_df.columns:
        counts = graph_df.groupby("cited_id").size().sort_values(ascending=False).head(10)
        for cited_id, n in counts.items():
            title = year = None
            if cited_id in works_df.index:
                work_row = works_df.loc[cited_id]
                if isinstance(work_row, pd.DataFrame):  # duplicate ids — take the first
                    work_row = work_row.iloc[0]
                title = _as_str(work_row.get("Title"))
                year = _as_int(work_row.get("Year"))
            top_cited.append(
                {"id": _as_str(cited_id), "title": title, "year": year, "count": int(n)}
            )
    panel["top_cited"] = top_cited

    # Historical membership comes solely from the stage's bound observations.
    panel["merge_audit"] = []
    panel["merge_audit_truncated"] = 0
    panel.update(_collect_resolution_audit(
        layout.canonicalization_audit_json, errors,
        inputs=WORK_AUDIT_INPUTS, outputs=WORK_AUDIT_OUTPUTS,
    ))
    if panel["audit_status"] != "current":
        return panel
    try:
        saved_audit = panel.pop("saved_audit")
        source_mapping = saved_audit.get("source_cluster_ids")
        citation_mapping = saved_audit.get("citation_cluster_ids")
        if (not isinstance(source_mapping, list) or sources_df is None
                or len(source_mapping) != len(sources_df)
                or not isinstance(citation_mapping, list) or len(citation_mapping) != len(raw_refs_df)):
            raise ValueError("Audit observation mappings do not match the checkpoint rows")
        mapping = source_mapping + citation_mapping
        if any(not isinstance(cid, str) or cid not in works_df.index for cid in mapping):
            raise ValueError("Audit refers to an unknown canonical work")
        raw = pd.concat([sources_df, raw_refs_df], ignore_index=True)
        for decision in saved_audit["decisions"]:
            for key in ("left_index", "right_index"):
                idx = decision.get(key)
                if not isinstance(idx, int) or not 0 <= idx < len(raw):
                    raise ValueError("Audit decision refers to an unknown observation")
        panel["audit_decisions"] = [
            dict(decision, left_title=_as_str(raw.iloc[decision["left_index"]].get("Title")),
                 right_title=_as_str(raw.iloc[decision["right_index"]].get("Title")))
            for decision in saved_audit["decisions"] if decision.get("decision") == "review"
        ][:200]
        panel["id_collisions"] = saved_audit.get("id_collisions", [])
        members_by_cluster: dict[str, list[int]] = {}
        for idx, cluster_id in enumerate(mapping):
            members_by_cluster.setdefault(str(cluster_id), []).append(int(idx))
        multi = sorted(
            ((cid, idxs) for cid, idxs in members_by_cluster.items() if len(idxs) > 1),
            key=lambda item: (-len(item[1]), item[0]),
        )
        panel["merge_audit_truncated"] = max(len(multi) - _MERGE_AUDIT_MAX_CLUSTERS, 0)
        for cluster_id, idxs in multi[:_MERGE_AUDIT_MAX_CLUSTERS]:
            members = []
            for idx in idxs:
                row = raw.iloc[idx]
                members.append(
                    {
                        "title": _as_str(row.get("Title")),
                        "year": _as_int(row.get("Year")),
                        "citing_id": _as_str(row.get("citing_id")),
                        "source_file": _as_str(row.get("source_file")),
                    }
                )
            panel["merge_audit"].append({"id": cluster_id, "members": members})
    except Exception as exc:  # noqa: BLE001 - audit is best-effort
        panel.update(audit_status="invalid", merge_audit=[], audit_decisions=[])
        _record_error(errors, layout.canonicalization_audit_json, exc)

    return panel


def _collect_enrichment_panel(
    enriched_df: pd.DataFrame | None,
    misses_df: pd.DataFrame | None,
    enrichment_summary: object,
) -> dict | None:
    if enriched_df is None:
        return None

    status = (
        enriched_df["enrichment_status"].fillna("")
        if "enrichment_status" in enriched_df.columns
        else pd.Series([""] * len(enriched_df))
    )

    if isinstance(enrichment_summary, dict):
        n_total = _as_int(enrichment_summary.get("n_references")) or len(enriched_df)
        n_matched = _as_int(enrichment_summary.get("n_matched")) or 0
        match_rate = enrichment_summary.get("match_rate")
        sources = enrichment_summary.get("sources") or {}
    else:
        n_total = len(enriched_df)
        n_matched = int((status == "matched").sum())
        match_rate = (n_matched / n_total) if n_total else None
        sources = (
            enriched_df["enrichment_source"].fillna("miss").value_counts().to_dict()
            if "enrichment_source" in enriched_df.columns
            else {}
        )

    panel: dict = {
        "n_references": int(n_total),
        "n_matched": int(n_matched),
        "match_rate": float(match_rate) if match_rate is not None else None,
        "sources": {str(k): int(v) for k, v in dict(sources).items()},
    }

    if "enrichment_miss_reason" in enriched_df.columns:
        reasons = enriched_df.loc[status != "matched", "enrichment_miss_reason"]
        panel["miss_reasons"] = {
            str(k): int(v) for k, v in reasons.fillna("(none)").value_counts().items()
        }
    else:
        panel["miss_reasons"] = {}

    weakest: list[dict] = []
    if "enrichment_title_score" in enriched_df.columns:
        matched = enriched_df[status == "matched"].copy()
        matched["_score"] = pd.to_numeric(matched["enrichment_title_score"], errors="coerce")
        for _, row in matched.sort_values("_score", ascending=True).head(10).iterrows():
            weakest.append(
                {
                    "id": _as_str(row.name if matched.index.name else row.get("id")),
                    "title": _as_str(row.get("Title")),
                    "candidate_title": _as_str(row.get("enrichment_candidate_title")),
                    "title_score": row.get("_score"),
                    "adjusted_score": row.get("enrichment_adjusted_score"),
                    "doi": _as_str(row.get("doi")),
                    "year_match": _as_str(row.get("enrichment_year_match")),
                }
            )
    panel["weakest_matches"] = weakest

    misses_preview: list[dict] = []
    if misses_df is not None:
        for _, row in misses_df.head(15).iterrows():
            misses_preview.append(
                {
                    "id": _as_str(row.get("id")),
                    "title": _as_str(row.get("Title")),
                    "year": _as_int(row.get("Year")),
                    "reason": _as_str(row.get("enrichment_miss_reason")),
                    "score": row.get("enrichment_title_score"),
                }
            )
    panel["misses_preview"] = misses_preview
    panel["n_misses"] = int(len(misses_df)) if misses_df is not None else None
    return panel


def _collect_authors_panel(
    authors_df: pd.DataFrame | None,
    author_citations_df: pd.DataFrame | None,
    author_review: object,
) -> dict | None:
    if authors_df is None:
        return None

    panel: dict = {
        "n_authors": int(len(authors_df)),
        "n_occurrences": int(len(author_citations_df))
        if author_citations_df is not None
        else None,
    }

    top: list[dict] = []
    if "n_citations_received" in authors_df.columns:
        ranked = authors_df.sort_values("n_citations_received", ascending=False).head(10)
        for author_id, row in ranked.iterrows():
            top.append(
                {
                    "id": _as_str(author_id),
                    "display_name": _as_str(row.get("display_name")),
                    "n_citations_received": _as_int(row.get("n_citations_received")),
                    "n_works": _as_int(row.get("n_works")),
                    "n_core_works": _as_int(row.get("n_core_works")),
                }
            )
    panel["top_cited"] = top

    review_entries: list[dict] = []
    if isinstance(author_review, list):
        for entry in author_review:
            if isinstance(entry, dict):
                review_entries.append({str(k): _as_str(v) for k, v in entry.items()})
    panel["review"] = review_entries

    variant_audit: list[dict] = []
    if (
        author_citations_df is not None
        and "author_id" in author_citations_df.columns
        and "raw_author" in author_citations_df.columns
    ):
        grouped = author_citations_df.groupby("author_id")["raw_author"].agg(
            lambda s: sorted({str(v) for v in s.dropna()})
        )
        ranked_ids = grouped.map(len).sort_values(ascending=False).head(10).index
        for author_id in ranked_ids:
            forms = grouped.loc[author_id]
            display = ""
            if author_id in authors_df.index:
                display = _as_str(authors_df.loc[author_id].get("display_name"))
            variant_audit.append(
                {
                    "id": _as_str(author_id),
                    "display_name": display,
                    "n_forms": len(forms),
                    "forms": forms,
                }
            )
    panel["variant_audit"] = variant_audit
    return panel


def _collect_integrity(
    layout: OutLayout,
    errors: list[dict],
    *,
    rows: list[tuple[str, Path, bool, str]],
) -> list[dict]:
    error_by_path = {e["path"]: e["error"] for e in errors}
    table: list[dict] = []
    for name, path, parsed_ok, note in rows:
        if not path.exists():
            status = "missing"
        elif parsed_ok:
            status = "ok"
        else:
            status = "corrupt"
        table.append(
            {
                "name": name,
                "path": str(path),
                "status": status,
                "note": note,
                "error": error_by_path.get(str(path), ""),
            }
        )
    listed = {row["path"] for row in table}
    for err in errors:  # per-paper caches and anything else that failed to read
        if err["path"] in listed:
            continue
        try:
            name = str(Path(err["path"]).relative_to(layout.out_dir))
        except ValueError:
            name = err["path"]
        table.append(
            {
                "name": name,
                "path": err["path"],
                "status": "corrupt",
                "note": "per-file cache",
                "error": err["error"],
            }
        )
        listed.add(err["path"])
    return table


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
def _esc(value: object) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _year_str(year: object) -> str:
    y = _as_int(year)
    return "?" if not y else str(y)


def _num(value: object, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


_PILL_KIND = {
    "ok": "ok",
    "done": "ok",
    "zero": "warn",
    "failed": "warn",
    "corrupt": "warn",
    "error": "warn",
    "image-only": "warn",
    "missing": "warn",
    "partial": "warn",
    "pending": "muted",
    "skipped": "muted",
    "not_run": "muted",
    "optional_not_run": "muted",
}


def _pill(label: str, status: str | None = None) -> str:
    kind = _PILL_KIND.get(status if status is not None else label, "muted")
    return f'<span class="pill pill-{kind}">{_esc(label)}</span>'


def _placeholder(stage_label: str, cmd: str) -> str:
    return (
        '<p class="placeholder">'
        f"{_esc(stage_label)} not run yet — run <code>{_esc(cmd)}</code>.</p>"
    )


def _section(anchor: str, eyebrow: str, title: str, body: str) -> str:
    return (
        f'<section id="{_esc(anchor)}">'
        f'<p class="eyebrow">{_esc(eyebrow)}</p>'
        f"<h2>{_esc(title)}</h2>{body}</section>"
    )


def _render_stage_strip(stages: list[dict]) -> str:
    cells = []
    for stage in stages:
        status = stage["status"]
        mark = {"done": "✓ done", "partial": "partial", "not_run": "not run"}.get(
            status, "optional"
        )
        hint = (
            f'<div class="stage-hint"><code>{_esc(stage["hint"])}</code></div>'
            if stage.get("hint")
            else ""
        )
        cells.append(
            f'<div class="stage stage-{_esc(status)}">'
            f'<div class="stage-name">{_esc(stage["label"])}</div>'
            f"<div>{_pill(mark, status)}</div>"
            f'<div class="stage-detail">{_esc(stage["detail"])}</div>{hint}</div>'
        )
    return f'<div class="stage-strip">{"".join(cells)}</div>'


def _render_problem_cards(problems: dict) -> str:
    cards = problems.get("cards", [])
    if not cards:
        return (
            '<p class="all-clear">No problems detected at the stages run so far.</p>'
        )
    parts = []
    for card in cards:
        parts.append(
            f'<a class="problem-card" href="#{_esc(card["anchor"])}">'
            f'<span class="problem-count">{card["count"]}</span>'
            f'<span class="problem-label">{_esc(card["label"])}</span></a>'
        )
    return f'<div class="problem-cards">{"".join(parts)}</div>'


def _render_paper_details(paper: dict) -> str:
    meta = paper["metadata"]
    blocks = []
    if meta["status"] == "ok":
        blocks.append(
            '<dl class="kv">'
            f"<dt>Title</dt><dd>{_esc(meta['title'])}</dd>"
            f"<dt>Authors</dt><dd>{_esc(meta['authors'])}</dd>"
            f"<dt>Journal</dt><dd>{_esc(meta['journal'])}</dd>"
            f"<dt>Year</dt><dd>{_esc(_year_str(meta['year']))}</dd></dl>"
        )
    refs = paper["refs_preview"]
    if refs:
        items = "".join(
            f"<li>{_esc(r['title'])} <span class='soft'>— {_esc(r['authors'])} "
            f"({_esc(_year_str(r['year']))})</span></li>"
            for r in refs
        )
        more = (
            f'<li class="soft">…and {paper["refs_more"]} more</li>'
            if paper["refs_more"]
            else ""
        )
        blocks.append(
            f'<h4>Extracted references (first {len(refs)})</h4><ol class="ref-list">{items}{more}</ol>'
        )
    if paper["markdown_preview"]:
        blocks.append(
            "<h4>Markdown preview</h4>"
            f'<pre class="md-preview">{_esc(paper["markdown_preview"])}</pre>'
        )
    if not blocks:
        blocks.append('<p class="soft">Nothing cached for this paper yet.</p>')
    return "".join(blocks)


def _render_papers_section(papers: list[dict]) -> str:
    if not papers:
        return _placeholder("Conversion", "citegraph convert <pdf_dir> --out <out>")

    rows = []
    for paper in papers:
        md = paper["markdown"]
        meta = paper["metadata"]
        refs = paper["references"]
        md_cell = _pill(md["status"], md["status"])
        if md["chars"] is not None:
            md_cell = f'<span class="mono">{md["chars"]:,}</span> {md_cell}'
        meta_cell = _pill(meta["status"], meta["status"])
        if meta["status"] == "ok":
            meta_cell += (
                f' <span class="cell-title">{_esc(meta["title"])}'
                f' <span class="soft">({_esc(_year_str(meta["year"]))})</span></span>'
            )
        refs_cell = _pill(refs["status"], refs["status"])
        if refs["count"] is not None:
            refs_cell = f'<span class="mono">{refs["count"]}</span> {refs_cell}'
        notes = "; ".join(paper["notes"])
        rows.append(
            f'<details class="paper-row" data-stem="{_esc(paper["stem"])}" '
            f'data-title="{_esc(meta["title"])}" '
            f'data-problem="{"1" if paper["problem"] else "0"}">'
            f'<summary><span class="col col-stem mono">{_esc(paper["stem"])}</span>'
            f'<span class="col col-md">{md_cell}</span>'
            f'<span class="col col-meta">{meta_cell}</span>'
            f'<span class="col col-refs">{refs_cell}</span>'
            f'<span class="col col-notes">{_esc(notes)}</span></summary>'
            f'<div class="paper-body">{_render_paper_details(paper)}</div></details>'
        )

    header = (
        '<div class="paper-row paper-head">'
        '<span class="col col-stem">stem</span>'
        '<span class="col col-md">conversion</span>'
        '<span class="col col-meta">metadata</span>'
        '<span class="col col-refs">references</span>'
        '<span class="col col-notes">notes</span></div>'
    )
    controls = (
        '<div class="controls">'
        '<input id="filter-text" type="search" placeholder="filter by stem or title…">'
        '<label><input id="filter-problems" type="checkbox"> problems only</label>'
        '<span id="paper-count" class="soft"></span></div>'
    )
    return (
        controls
        + f'<div class="table-scroll"><div class="paper-table">{header}{"".join(rows)}</div></div>'
    )


def _render_dedup_section(panel: dict | None) -> str:
    if panel is None:
        return _placeholder("Dedup", "citegraph dedup --out <out>")

    ring_counts = panel.get("ring_counts") or {}
    # The payload is JSON-normalized, so ring keys arrive as strings.
    n_core = int(ring_counts.get("0", ring_counts.get(0, 0)))
    stats = (
        '<div class="stat-row">'
        f'<div class="stat"><span class="stat-n">{panel["n_raw"]}</span>raw citation events</div>'
        f'<div class="stat"><span class="stat-n">{panel["n_canonical"]}</span>canonical works</div>'
        f'<div class="stat"><span class="stat-n">{n_core}</span>core works (ring 0)</div>'
        f'<div class="stat"><span class="stat-n">{_num(panel["n_edges"], 0)}</span>graph edges</div>'
        f'<div class="stat"><span class="stat-n">{panel["n_missing_year"]}</span>missing year (Year=0)</div>'
        f'<div class="stat"><span class="stat-n">{panel["n_empty_authors"]}</span>empty authors</div>'
        f'<div class="stat"><span class="stat-n">{panel["n_short_titles"]}</span>suspicious short titles</div>'
        "</div>"
    )

    core_to_core = panel.get("core_to_core") or []
    core_rows = "".join(
        f"<tr><td class='mono'>{_esc(e['citing_id'])}</td>"
        f"<td class='mono'>{_esc(e['cited_id'])}</td></tr>"
        for e in core_to_core
    )
    core_html = (
        "<h3>Core cites core</h3>"
        '<p class="soft">Citations where both ends are ring-0 works — your own '
        "papers citing each other.</p>"
        '<div class="table-scroll"><table><thead><tr><th>citing</th><th>cited</th>'
        f"</tr></thead><tbody>{core_rows}</tbody></table></div>"
        if core_to_core
        else ""
    )

    top_rows = "".join(
        f"<tr><td class='mono'>{_esc(t['id'])}</td><td>{_esc(t['title'])}</td>"
        f"<td class='num'>{_esc(_year_str(t['year']))}</td><td class='num'>{t['count']}</td></tr>"
        for t in panel["top_cited"]
    )
    top_html = (
        "<h3>Top cited works</h3>"
        '<div class="table-scroll"><table><thead><tr><th>id</th><th>title</th>'
        "<th>year</th><th>citations</th></tr></thead>"
        f"<tbody>{top_rows}</tbody></table></div>"
        if panel["top_cited"]
        else ""
    )

    audit_parts = []
    for cluster in panel["merge_audit"]:
        members = "".join(
            f"<li>{_esc(m['title'])} <span class='soft'>({_esc(_year_str(m['year']))}, "
            f"<span class='mono'>{_esc(m['source_file'] or m['citing_id'])}</span>)</span></li>"
            for m in cluster["members"]
        )
        audit_parts.append(
            f'<details class="cluster"><summary><span class="mono">{_esc(cluster["id"])}</span>'
            f' <span class="soft">contains {len(cluster["members"])} observations</span>'
            f"</summary><ul>{members}</ul></details>"
        )
    if audit_parts:
        note = ""
        if panel["merge_audit_truncated"]:
            note = (
                f'<p class="soft">…and {panel["merge_audit_truncated"]} more clusters '
                "not shown.</p>"
            )
        audit_html = (
            "<h3>Merge audit</h3>"
            '<p class="soft">Saved clusters containing multiple source or citation observations. '
            "Inspect these for false merges.</p>"
            + "".join(audit_parts)
            + note
        )
    else:
        audit_html = (
            "<h3>Merge audit</h3><p class='soft'>No saved merged groups to display.</p>"
        )

    return stats + core_html + top_html + audit_html + _render_resolution_evidence(panel)


def _render_resolution_evidence(panel: dict) -> str:
    status = panel.get("audit_status", "unavailable")
    if status != "current":
        return (f"<p class='soft'>Resolution evidence: {_esc(status)}. "
                "Re-run the corresponding stage to save evidence for the current artifacts.</p>")
    result = (
        "<h3>Resolution evidence</h3><p class='soft'>Saved evidence matches the current artifacts. "
        f"Algorithm: {_esc(panel.get('audit_algorithm'))}. "
        f"Settings: {_esc(json.dumps(panel.get('audit_config'), sort_keys=True))}.</p>"
    )
    if panel.get("enrichment_status"):
        result += f"<p>Enrichment evidence: {_esc(panel['enrichment_status'])}.</p>"
    decisions = panel.get("audit_decisions", [])
    if decisions:
        items = "".join(
            "<li>" + "; ".join(f"<b>{_esc(k)}</b>: {_esc(v)}" for k, v in row.items()) + "</li>"
            for row in decisions
        )
        result += f"<details><summary>Decisions and review evidence</summary><ul>{items}</ul></details>"
    if panel.get("audit_decisions_truncated"):
        result += "<p class='soft'>Additional decisions are available in the audit JSON file.</p>"
    return result


def _render_enrichment_section(panel: dict | None) -> str:
    if panel is None:
        return _placeholder("Enrichment", "citegraph enrich --out <out>")

    rate = (
        f"{panel['match_rate'] * 100:.1f}%" if panel["match_rate"] is not None else "–"
    )
    sources = ", ".join(f"{_esc(k)}: {v}" for k, v in panel["sources"].items()) or "–"
    stats = (
        '<div class="stat-row">'
        f'<div class="stat"><span class="stat-n">{panel["n_references"]}</span>references</div>'
        f'<div class="stat"><span class="stat-n">{panel["n_matched"]}</span>matched</div>'
        f'<div class="stat"><span class="stat-n">{_esc(rate)}</span>match rate</div>'
        "</div>"
        f'<p class="soft">Sources — {sources}</p>'
    )

    reasons = "".join(
        f"<tr><td>{_esc(reason)}</td><td class='num'>{count}</td></tr>"
        for reason, count in panel["miss_reasons"].items()
    )
    reasons_html = (
        "<h3>Miss reasons</h3>"
        '<div class="table-scroll"><table><thead><tr><th>reason</th><th>count</th></tr>'
        f"</thead><tbody>{reasons}</tbody></table></div>"
        if reasons
        else ""
    )

    weak_rows = "".join(
        f"<tr><td>{_esc(w['title'])}</td><td>{_esc(w['candidate_title'])}</td>"
        f"<td class='num'>{_num(w['title_score'])}</td>"
        f"<td class='num'>{_num(w['adjusted_score'])}</td>"
        f"<td class='mono'>{_esc(w['doi'])}</td><td>{_esc(w['year_match'])}</td></tr>"
        for w in panel["weakest_matches"]
    )
    weak_html = (
        "<h3>Weakest accepted matches</h3>"
        '<p class="soft">Lowest title scores among status="matched" — check these '
        "for wrong DOIs.</p>"
        '<div class="table-scroll"><table><thead><tr><th>title</th><th>candidate title</th>'
        "<th>score</th><th>adjusted</th><th>doi</th><th>year match</th></tr></thead>"
        f"<tbody>{weak_rows}</tbody></table></div>"
        if weak_rows
        else ""
    )

    miss_rows = "".join(
        f"<tr><td class='mono'>{_esc(m['id'])}</td><td>{_esc(m['title'])}</td>"
        f"<td class='num'>{_esc(_year_str(m['year']))}</td><td>{_esc(m['reason'])}</td>"
        f"<td class='num'>{_num(m['score'])}</td></tr>"
        for m in panel["misses_preview"]
    )
    n_misses = panel["n_misses"]
    misses_html = ""
    if miss_rows:
        suffix = f" (first {len(panel['misses_preview'])} of {n_misses})" if n_misses else ""
        misses_html = (
            f"<h3>Unmatched references{_esc(suffix)}</h3>"
            '<div class="table-scroll"><table><thead><tr><th>id</th><th>title</th>'
            "<th>year</th><th>reason</th><th>score</th></tr></thead>"
            f"<tbody>{miss_rows}</tbody></table></div>"
        )

    return stats + reasons_html + weak_html + misses_html


def _render_authors_section(panel: dict | None) -> str:
    if panel is None:
        return _placeholder("Authors", "citegraph authors --out <out>")

    stats = (
        '<div class="stat-row">'
        f'<div class="stat"><span class="stat-n">{panel["n_authors"]}</span>canonical authors</div>'
        f'<div class="stat"><span class="stat-n">{_num(panel["n_occurrences"], 0)}</span>'
        "name occurrences</div></div>"
    )

    top_rows = "".join(
        f"<tr><td class='mono'>{_esc(t['id'])}</td><td>{_esc(t['display_name'])}</td>"
        f"<td class='num'>{_num(t['n_citations_received'], 0)}</td>"
        f"<td class='num'>{_num(t['n_works'], 0)}</td>"
        f"<td class='num'>{_num(t['n_core_works'], 0)}</td></tr>"
        for t in panel["top_cited"]
    )
    top_html = (
        "<h3>Most-cited authors</h3>"
        '<div class="table-scroll"><table><thead><tr><th>id</th><th>name</th>'
        "<th>citations received</th><th>works</th><th>core works</th></tr></thead>"
        f"<tbody>{top_rows}</tbody></table></div>"
        if top_rows
        else ""
    )

    review_html = ""
    if panel["review"]:
        entries = "".join(
            "<li>"
            + "; ".join(f"<b>{_esc(k)}</b>: {_esc(v)}" for k, v in entry.items())
            + "</li>"
            for entry in panel["review"]
        )
        review_html = (
            f"<h3>Flagged for review ({len(panel['review'])})</h3>"
            f'<ul class="review-list">{entries}</ul>'
        )

    variant_parts = []
    for cluster in panel["variant_audit"]:
        forms = "".join(f"<li>{_esc(f)}</li>" for f in cluster["forms"])
        variant_parts.append(
            f'<details class="cluster"><summary><span class="mono">{_esc(cluster["id"])}</span> '
            f"{_esc(cluster['display_name'])} "
            f'<span class="soft">— {cluster["n_forms"]} distinct raw forms</span></summary>'
            f"<ul>{forms}</ul></details>"
        )
    variant_html = ""
    if variant_parts:
        variant_html = (
            "<h3>Name-variant audit</h3>"
            '<p class="soft">Clusters that merged the most distinct raw author '
            "strings — the merge audit for people.</p>" + "".join(variant_parts)
        )

    return stats + top_html + review_html + variant_html + _render_resolution_evidence(panel)


def _render_integrity_section(table: list[dict]) -> str:
    rows = []
    for row in table:
        status = row["status"]
        pill_status = {"ok": "ok", "corrupt": "corrupt", "missing": "missing"}[status]
        soft_missing = (
            "optional" in row["note"] or "absent" in row["note"] or "citegraph run" in row["note"]
        )
        if status == "missing" and soft_missing:
            pill = _pill("absent", "pending")
        else:
            pill = _pill(status.upper() if status == "corrupt" else status, pill_status)
        err = f'<div class="soft">{_esc(row["error"])}</div>' if row["error"] else ""
        rows.append(
            f"<tr><td class='mono'>{_esc(row['name'])}</td><td>{pill}</td>"
            f"<td>{_esc(row['note'])}{err}</td></tr>"
        )
    return (
        '<div class="table-scroll"><table><thead><tr><th>artifact</th><th>status</th>'
        f"<th>notes</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


_CSS = """
:root {
  --paper: #f6f7f4; --surface: #fdfdfb; --line: #dde2db;
  --ink: #1b2420; --ink-soft: #55605a;
  --accent: #1f6a4e; --accent-soft: #e3eee8;
  --brass: #a07a2f; --warn: #b4552d; --warn-soft: #f6e9e2;
  --code-bg: #eef1ec;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #131a17; --surface: #1a231f; --line: #2c3833;
    --ink: #e6eae4; --ink-soft: #9aa8a0;
    --accent: #58b78e; --accent-soft: #1e3a2f;
    --brass: #c9a45c; --warn: #d98a63; --warn-soft: #3a2820;
    --code-bg: #101613;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--paper); color: var(--ink);
  font-family: system-ui, sans-serif; font-size: 15px; line-height: 1.5;
}
h1, h2, h3, h4 { font-family: "Iowan Old Style", Georgia, serif; line-height: 1.2; }
h1 { font-size: 1.7rem; margin: 0 0 0.25rem; }
h2 { font-size: 1.3rem; margin: 0 0 0.75rem; }
h3 { font-size: 1.05rem; margin: 1.5rem 0 0.5rem; }
h4 { font-size: 0.95rem; margin: 1rem 0 0.35rem; }
code, .mono, pre {
  font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.86em;
}
code { background: var(--code-bg); padding: 0.1em 0.35em; border-radius: 4px; }
main { max-width: 1180px; margin: 0 auto; padding: 1rem 1.25rem 4rem; }
section {
  background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  padding: 1.1rem 1.25rem 1.35rem; margin-top: 1.25rem;
}
.eyebrow {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.68rem;
  color: var(--brass); margin: 0 0 0.35rem;
}
.soft { color: var(--ink-soft); }
nav.mini {
  position: sticky; top: 0; z-index: 10; background: var(--paper);
  border-bottom: 1px solid var(--line); padding: 0.5rem 1.25rem;
  display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.85rem;
}
nav.mini a { color: var(--ink-soft); text-decoration: none; }
nav.mini a:hover { color: var(--accent); }
header.page { padding: 1.5rem 0 0; }
header.page .meta { color: var(--ink-soft); font-size: 0.85rem; }
.pill {
  display: inline-block; padding: 0.05em 0.55em; border-radius: 999px;
  font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.72rem;
  white-space: nowrap; vertical-align: middle;
}
.pill-ok { background: var(--accent-soft); color: var(--accent); }
.pill-warn { background: var(--warn-soft); color: var(--warn); }
.pill-muted { background: var(--line); color: var(--ink-soft); }
.stage-strip {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.6rem; margin-top: 0.9rem;
}
.stage {
  border: 1px solid var(--line); border-radius: 8px; padding: 0.6rem 0.7rem;
  background: var(--surface);
}
.stage-done { border-color: var(--accent); }
.stage-name {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.68rem;
  color: var(--ink-soft); margin-bottom: 0.25rem;
}
.stage-detail { font-size: 0.8rem; color: var(--ink-soft); margin-top: 0.3rem; }
.stage-hint { margin-top: 0.3rem; font-size: 0.72rem; overflow-x: auto; }
.problem-cards {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  gap: 0.6rem;
}
.problem-card {
  display: flex; align-items: baseline; gap: 0.5rem; text-decoration: none;
  border: 1px solid var(--warn); border-radius: 8px; padding: 0.55rem 0.7rem;
  background: var(--warn-soft); color: var(--ink);
}
.problem-count {
  font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 1.15rem;
  font-variant-numeric: tabular-nums; color: var(--warn); font-weight: 600;
}
.problem-label { font-size: 0.82rem; }
.all-clear { color: var(--accent); margin: 0; }
.placeholder { color: var(--ink-soft); margin: 0; }
.controls { display: flex; gap: 1rem; align-items: center; margin-bottom: 0.7rem; flex-wrap: wrap; }
.controls input[type="search"] {
  background: var(--paper); color: var(--ink); border: 1px solid var(--line);
  border-radius: 6px; padding: 0.35rem 0.6rem; min-width: 240px; font: inherit;
}
.controls label { font-size: 0.85rem; color: var(--ink-soft); }
.table-scroll { overflow-x: auto; }
.paper-table { min-width: 900px; border-top: 1px solid var(--line); }
.paper-row { border-bottom: 1px solid var(--line); }
.paper-row summary, .paper-head {
  display: grid; grid-template-columns: 2.2fr 1.2fr 2.6fr 1.1fr 1.6fr;
  gap: 0.7rem; padding: 0.45rem 0.3rem; align-items: baseline; cursor: pointer;
  list-style: none;
}
.paper-head { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--ink-soft); font-family: ui-monospace, "SF Mono", Menlo, monospace;
  cursor: default; }
.paper-row summary::-webkit-details-marker { display: none; }
.paper-row summary:hover { background: var(--code-bg); }
.col { overflow: hidden; text-overflow: ellipsis; }
.col-md .mono, .col-refs .mono { font-variant-numeric: tabular-nums; }
.col-notes { font-size: 0.8rem; color: var(--warn); }
.cell-title { font-size: 0.88rem; }
.paper-body { padding: 0.5rem 0.75rem 1rem; background: var(--code-bg); border-radius: 6px;
  margin: 0 0.3rem 0.7rem; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 0.15rem 1rem; margin: 0.5rem 0; }
.kv dt { color: var(--ink-soft); font-size: 0.8rem; }
.kv dd { margin: 0; }
.ref-list { margin: 0.25rem 0; padding-left: 1.5rem; font-size: 0.88rem; }
.md-preview {
  max-height: 260px; overflow: auto; background: var(--paper);
  border: 1px solid var(--line); border-radius: 6px; padding: 0.6rem;
  white-space: pre-wrap; word-break: break-word; margin: 0.25rem 0;
}
.stat-row { display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 0.4rem 0 0.8rem; }
.stat { font-size: 0.8rem; color: var(--ink-soft); }
.stat-n {
  display: block; font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 1.25rem; color: var(--ink); font-variant-numeric: tabular-nums;
}
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
th, td { text-align: left; padding: 0.35rem 0.6rem; border-bottom: 1px solid var(--line);
  vertical-align: top; }
th { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--ink-soft); font-family: ui-monospace, "SF Mono", Menlo, monospace; }
td.num { font-variant-numeric: tabular-nums; text-align: right;
  font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.82rem; }
details.cluster { border: 1px solid var(--line); border-radius: 6px;
  padding: 0.35rem 0.7rem; margin: 0.35rem 0; background: var(--paper); }
details.cluster summary { cursor: pointer; }
details.cluster ul { margin: 0.4rem 0; }
.review-list { font-size: 0.85rem; }
.size-note { color: var(--warn); font-size: 0.85rem; }
"""

_JS = """
(function () {
  var blob = document.getElementById('data');
  var DATA = { papers: [] };
  try { DATA = JSON.parse(blob.textContent); } catch (e) { /* blob is advisory */ }
  var q = document.getElementById('filter-text');
  var cb = document.getElementById('filter-problems');
  var count = document.getElementById('paper-count');
  if (!q || !cb) { return; }
  var rows = Array.prototype.slice.call(document.querySelectorAll('details.paper-row'));
  var total = (DATA.papers && DATA.papers.length) || rows.length;
  function apply() {
    var needle = q.value.trim().toLowerCase();
    var shown = 0;
    rows.forEach(function (el) {
      var hay = (el.dataset.stem + ' ' + el.dataset.title).toLowerCase();
      var ok = (!needle || hay.indexOf(needle) !== -1) &&
               (!cb.checked || el.dataset.problem === '1');
      el.hidden = !ok;
      if (ok) { shown += 1; }
    });
    if (count) { count.textContent = 'showing ' + shown + ' of ' + total; }
  }
  q.addEventListener('input', apply);
  cb.addEventListener('change', apply);
  apply();
})();
"""


def build_report_html(data: dict) -> str:
    """Render the collected data dict into one self-contained HTML page."""
    blob = (
        json.dumps(_jsonable(data), ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )

    size_note = ""
    if data.get("truncated_for_size"):
        size_note = (
            '<p class="size-note">Previews were truncated to keep this report under '
            "10&nbsp;MB.</p>"
        )
    empty_note = ""
    if not data.get("papers"):
        empty_note = (
            '<p class="placeholder">Nothing processed yet — run '
            "<code>citegraph convert &lt;pdf_dir&gt; --out "
            f"{_esc(data['out_dir'])}</code> to get started.</p>"
        )

    header = (
        '<header class="page" id="overview">'
        '<p class="eyebrow">citegraph · quality-control report</p>'
        f"<h1>{_esc(Path(data['out_dir']).name) or 'out'}</h1>"
        f'<p class="meta"><span class="mono">{_esc(data["out_dir"])}</span> · '
        f"generated {_esc(data['generated_at'])}</p>"
        f"{size_note}{empty_note}"
        f"{_render_stage_strip(data['stages'])}</header>"
    )

    nav = (
        '<nav class="mini">'
        '<a href="#overview">Overview</a><a href="#problems">Problems</a>'
        '<a href="#papers">Papers</a><a href="#dedup">Dedup</a>'
        '<a href="#enrichment">Enrichment</a><a href="#authors">Authors</a>'
        '<a href="#integrity">Integrity</a></nav>'
    )

    sections = [
        _section("problems", "summary", "Problems", _render_problem_cards(data["problems"])),
        _section(
            "papers", "per-paper", "Papers", _render_papers_section(data["papers"])
        ),
        _section("dedup", "stage 4", "Deduplication", _render_dedup_section(data["dedup"])),
        _section(
            "enrichment",
            "stage 5 · optional",
            "Enrichment",
            _render_enrichment_section(data["enrichment"]),
        ),
        _section(
            "authors", "stage 6 · optional", "Authors", _render_authors_section(data["authors"])
        ),
        _section("integrity", "artifacts", "Integrity", _render_integrity_section(data["integrity"])),
    ]

    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>citegraph report — {_esc(Path(data['out_dir']).name)}</title>"
        f"<style>{_CSS}</style></head><body>"
        f"{nav}<main>{header}{''.join(sections)}</main>"
        f'<script type="application/json" id="data">{blob}</script>'
        f"<script>{_JS}</script></body></html>"
    )


def _shrink_for_size(data: dict) -> dict:
    """Truncate previews when the rendered page would exceed the size cap."""
    shrunk = dict(data)
    shrunk["truncated_for_size"] = True
    papers = []
    for paper in data.get("papers", []):
        p = dict(paper)
        if p.get("markdown_preview"):
            p["markdown_preview"] = p["markdown_preview"][:_SHRUNK_MARKDOWN_PREVIEW_CHARS]
        preview = p.get("refs_preview") or []
        if len(preview) > _SHRUNK_REFS_PREVIEW_N:
            count = p["references"].get("count") or 0
            p["refs_preview"] = preview[:_SHRUNK_REFS_PREVIEW_N]
            p["refs_more"] = max(count - _SHRUNK_REFS_PREVIEW_N, 0)
        papers.append(p)
    shrunk["papers"] = papers
    return shrunk


def write_report(out_dir: Path, *, data: dict | None = None) -> Path:
    """Collect, render, and write ``report.html`` into ``out_dir``.

    Returns the path of the written file. ``out_dir`` is created if it
    doesn't exist so the command is safe to run at any stage.
    """
    out_dir = Path(out_dir)
    if data is None:
        data = collect_report_data(out_dir)
    html_text = build_report_html(data)
    if len(html_text.encode("utf-8")) > _MAX_HTML_BYTES:
        html_text = build_report_html(_shrink_for_size(data))
    layout = OutLayout(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layout.report_html.write_text(html_text, encoding="utf-8")
    return layout.report_html
