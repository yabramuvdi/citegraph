"""Report and sidecar artifact helpers for pipeline runs."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from citegraph.dedup import DedupConfig
from citegraph.io import OutLayout, read_json, write_json
from citegraph.pdf_to_markdown import _is_image_only

logger = logging.getLogger(__name__)

ARTIFACT_SCHEMA_VERSION = 1


def _package_version() -> str:
    try:
        return version("citegraph")
    except PackageNotFoundError:
        return "0.1.0"


def write_artifact_manifest(
    layout: OutLayout,
    *,
    stage: str,
    artifacts: dict[str, Path],
    config: dict[str, Any] | None = None,
) -> None:
    """Write a minimal manifest describing generated artifacts.

    The manifest is intentionally small: it gives future versions enough
    information to explain stale cache behavior without introducing a
    migration system before the package needs one.
    """
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "package_version": _package_version(),
        "generated_at": datetime.now(UTC).isoformat(),
        "stage": stage,
        "artifacts": {
            name: str(path.relative_to(layout.out_dir))
            if path.is_relative_to(layout.out_dir)
            else str(path)
            for name, path in artifacts.items()
        },
        "config": config or {},
    }
    write_json(layout.artifact_manifest_json, manifest)


def write_failures(path: Path, failures: list[Any]) -> None:
    """Persist failures as JSONL, or remove the file when none occurred."""
    if not failures:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for fail in failures:
            payload = asdict(fail) if is_dataclass(fail) else dict(fail)
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def count_failures(path: Path) -> int:
    """Count entries in a failures.jsonl file; 0 if the file doesn't exist."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def detect_source_duplicates(records: list[dict], cfg: DedupConfig) -> list[dict]:
    """Find records that are duplicate conversions of the same underlying paper."""
    from citegraph.dedup import compare_papers

    canonical_indices: list[int] = []
    duplicate_of: dict[int, int] = {}

    for i, rec in enumerate(records):
        matched = False
        for canon_idx in canonical_indices:
            if compare_papers(rec, records[canon_idx], cfg):
                duplicate_of[i] = canon_idx
                matched = True
                break
        if not matched:
            canonical_indices.append(i)

    canon_to_dupes: dict[int, list[int]] = {}
    for dup_idx, canon_idx in duplicate_of.items():
        canon_to_dupes.setdefault(canon_idx, []).append(dup_idx)

    groups = []
    for canon_idx, dup_indices in canon_to_dupes.items():
        canon = records[canon_idx]
        groups.append(
            {
                "canonical_source_file": canon["source_file"],
                "canonical_paper_id": canon["id"],
                "title": canon.get("Title", ""),
                "duplicate_source_files": [records[i]["source_file"] for i in dup_indices],
            }
        )
    return groups


def write_no_references(path: Path, entries: list[dict]) -> None:
    """Persist papers that returned zero references; remove the file when none exist."""
    if not entries:
        if path.exists():
            path.unlink()
        return
    write_json(path, entries)


def write_source_duplicates(path: Path, groups: list[dict]) -> None:
    """Persist duplicate groups as JSON, or remove the file when none exist."""
    if not groups:
        if path.exists():
            path.unlink()
        return
    write_json(path, groups)


def count_source_duplicates(path: Path) -> int:
    """Total number of duplicate markdown files; 0 if the file doesn't exist."""
    if not path.exists():
        return 0
    data = read_json(path)
    return sum(len(g.get("duplicate_source_files", [])) for g in data)


def count_no_references(path: Path) -> int:
    if not path.exists():
        return 0
    return len(read_json(path))


def check_conversion_quality(
    paths: list[Path], layout: OutLayout, *, ocr_attempted: bool = False
) -> None:
    """After conversion, flag markdown files that appear to be image-only."""
    reason = (
        "image-only markdown even after OCR (manual review needed)"
        if ocr_attempted
        else "image-only markdown (possibly a scanned PDF)"
    )
    warnings = [
        {"source_file": p.name, "reason": reason}
        for p in paths
        if _is_image_only(p)
    ]
    write_conversion_warnings(layout.conversion_warnings_json, warnings)
    if warnings:
        hint = (
            "OCR did not help; inspect the source PDFs and consider manual transcription."
            if ocr_attempted
            else "re-run with ocr=True / --ocr (or ocr='auto' / --ocr-auto) for better results."
        )
        logger.warning(
            "%d markdown file(s) appear image-only; %s See %s",
            len(warnings),
            hint,
            layout.conversion_warnings_json,
        )


def write_conversion_warnings(path: Path, warnings: list[dict]) -> None:
    if not warnings:
        if path.exists():
            path.unlink()
        return
    write_json(path, warnings)


def count_conversion_warnings(path: Path) -> int:
    if not path.exists():
        return 0
    return len(read_json(path))


def write_author_review(path: Path, review: list[dict]) -> None:
    """Persist flagged author clusters, or remove the file when there are none."""
    if not review:
        if path.exists():
            path.unlink()
        return
    write_json(path, review)


def count_author_review(path: Path) -> int:
    if not path.exists():
        return 0
    return len(read_json(path))
