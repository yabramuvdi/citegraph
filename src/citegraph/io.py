"""Filesystem helpers for the on-disk checkpoints.

The pipeline persists every expensive intermediate step under ``out_dir``
so that re-runs can resume:

::

    out_dir/
      markdown/                 # docling output, one .md per PDF
      metadata/<stem>.json      # per-source Gemini metadata response
      references/<stem>.json    # per-source Gemini references response
      sources.csv               # per-PDF source-work metadata (stage 2)
      citations_raw.csv         # all extracted citations, before canonicalization
      works.csv                 # canonical works (ring, source_file, metadata)
      citation_graph.csv        # (citing_id, cited_id) work->work edges
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ValidationError


@dataclass(frozen=True)
class OutLayout:
    """Standard subdirectory / filename layout under ``out_dir``."""

    out_dir: Path

    @property
    def conversion_cache_json(self) -> Path:
        return self.out_dir / "conversion_cache.json"

    @property
    def input_manifest_json(self) -> Path:
        return self.out_dir / "input_manifest.json"

    def markdown_inputs(self) -> list[Path]:
        """Select active inputs while retaining removed/renamed caches on disk."""
        if not self.input_manifest_json.exists():
            return sorted(self.markdown_dir.glob('*.md'))
        saved = read_json(self.input_manifest_json)
        names = saved.get('markdown_names') if isinstance(saved, dict) else None
        if (not isinstance(names, list) or saved.get('schema_version') != 1
                or any(not isinstance(name, str) or Path(name).name != name
                       or not name.endswith('.md') for name in names)
                or len(set(names)) != len(names)):
            raise ValueError('Invalid input_manifest.json; rerun `citegraph convert`')
        paths = [self.markdown_dir / name for name in names]
        if any(not path.is_file() for path in paths):
            raise ValueError('An active markdown input is missing; rerun `citegraph convert`')
        return paths

    @property
    def work_identity_json(self) -> Path:
        return self.out_dir / "work_identity.json"

    @property
    def author_identity_json(self) -> Path:
        return self.out_dir / "author_identity.json"

    @property
    def enrichment_provenance_json(self) -> Path:
        return self.out_dir / "enrichment_provenance.json"

    @property
    def source_ids_json(self) -> Path:
        return self.out_dir / "source_ids.json"

    @property
    def work_id_collisions_json(self) -> Path:
        return self.out_dir / "work_id_collisions.json"

    @property
    def markdown_dir(self) -> Path:
        return self.out_dir / "markdown"

    @property
    def metadata_dir(self) -> Path:
        return self.out_dir / "metadata"

    @property
    def references_dir(self) -> Path:
        return self.out_dir / "references"

    @property
    def enrichment_dir(self) -> Path:
        return self.out_dir / "enrichment"

    def enrichment_cache_count(self) -> int:
        """Number of per-reference enrichment cache files on disk.

        Non-zero while ``enriched_works.csv`` is absent means the
        enrich stage was interrupted before its final CSV write — the
        authors stage warns about that state instead of silently
        clustering without external ids.
        """
        if not self.enrichment_dir.is_dir():
            return 0
        return sum(1 for _ in self.enrichment_dir.glob("*.json"))

    @property
    def enrichment_misses_csv(self) -> Path:
        return self.out_dir / "enrichment_misses.csv"

    @property
    def enrichment_summary_json(self) -> Path:
        return self.out_dir / "enrichment_summary.json"

    @property
    def graph_csv(self) -> Path:
        return self.out_dir / "citation_graph.csv"

    @property
    def sources_csv(self) -> Path:
        return self.out_dir / "sources.csv"

    @property
    def citations_raw_csv(self) -> Path:
        return self.out_dir / "citations_raw.csv"

    @property
    def works_csv(self) -> Path:
        return self.out_dir / "works.csv"

    @property
    def enriched_works_csv(self) -> Path:
        return self.out_dir / "enriched_works.csv"

    @property
    def authors_csv(self) -> Path:
        return self.out_dir / "authors.csv"

    @property
    def author_citations_csv(self) -> Path:
        return self.out_dir / "author_citations.csv"

    @property
    def author_review_json(self) -> Path:
        return self.out_dir / "author_review.json"

    @property
    def author_resolution_audit_json(self) -> Path:
        return self.out_dir / "author_resolution_audit.json"

    @property
    def canonicalization_audit_json(self) -> Path:
        return self.out_dir / "canonicalization_audit.json"

    @property
    def author_aliases_csv(self) -> Path:
        return self.out_dir / "author_aliases.csv"

    @property
    def journal_aliases_csv(self) -> Path:
        return self.out_dir / "journal_aliases.csv"

    @property
    def author_external_ids_csv(self) -> Path:
        return self.out_dir / "author_external_ids.csv"

    @property
    def author_overrides_csv(self) -> Path:
        return self.out_dir / "author_overrides.csv"

    @property
    def author_overrides_meta_json(self) -> Path:
        return self.out_dir / "author_overrides_meta.json"

    @property
    def run_summary_json(self) -> Path:
        return self.out_dir / "run_summary.json"

    @property
    def metadata_failures_jsonl(self) -> Path:
        return self.out_dir / "metadata_failures.jsonl"

    @property
    def references_failures_jsonl(self) -> Path:
        return self.out_dir / "references_failures.jsonl"

    @property
    def source_duplicates_json(self) -> Path:
        return self.out_dir / "source_duplicates.json"

    @property
    def papers_no_references_json(self) -> Path:
        return self.out_dir / "papers_no_references.json"

    @property
    def conversion_warnings_json(self) -> Path:
        return self.out_dir / "conversion_warnings.json"

    @property
    def artifact_manifest_json(self) -> Path:
        return self.out_dir / "artifact_manifest.json"

    @property
    def report_html(self) -> Path:
        return self.out_dir / "report.html"

    def ensure(self) -> None:
        for d in (
            self.out_dir,
            self.markdown_dir,
            self.metadata_dir,
            self.references_dir,
            self.enrichment_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict | list) -> None:
    """Atomically replace JSON checkpoints, never leaving a partial document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(path: Path, frame: pd.DataFrame, *, index: bool = False) -> None:
    """Atomically replace a CSV checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        frame.to_csv(temporary, index=index)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def cache_input_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_provenance(path: Path) -> dict[str, Any] | None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        return None
    try:
        data = read_json(sidecar)
    except (json.JSONDecodeError, UnicodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("version", 1) != 1:
        return None
    input_hash = data.get("input_sha256")
    output_hash = data.get("output_sha256")
    if not isinstance(input_hash, str) or not isinstance(output_hash, str):
        return None
    return data


def cache_is_current(
    path: Path, input_path: Path, *, recipe: dict[str, Any] | None = None
) -> bool:
    """Accept legacy caches; validate new caches by input and output hashes."""
    if not path.exists():
        return False
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        return True
    provenance = _cache_provenance(path)
    if provenance is not None:
        return (
            provenance["input_sha256"] == cache_input_fingerprint(input_path)
            and provenance["output_sha256"] == cache_input_fingerprint(path)
            and (
                recipe is None
                or "recipe" not in provenance
                or provenance["recipe"] == recipe
            )
        )
    try:
        return (
            sidecar.read_text(encoding="ascii").strip()
            == cache_input_fingerprint(input_path)
        )
    except (UnicodeError, OSError):
        return False


def write_cache_fingerprint(
    path: Path, input_path: Path, *, recipe: dict[str, Any] | None = None
) -> None:
    provenance: dict[str, Any] = {
        "version": 1,
        "input_sha256": cache_input_fingerprint(input_path),
        "output_sha256": cache_input_fingerprint(path),
    }
    if recipe is not None:
        provenance["recipe"] = recipe
    write_json(
        path.with_suffix(path.suffix + ".sha256"),
        provenance,
    )


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def read_pydantic_cache(
    path: Path,
    input_path: Path,
    schema: type[BaseModel],
    *,
    many: bool = False,
    recipe: dict[str, Any] | None = None,
) -> BaseModel | list[BaseModel] | None:
    """Return a current, valid cache or ``None`` when it must be refreshed."""
    def validated(cache_path: Path) -> tuple[dict | list, BaseModel | list[BaseModel]] | None:
        try:
            data = read_json(cache_path)
            if many:
                value = (
                    [schema.model_validate(item) for item in data]
                    if isinstance(data, list)
                    else None
                )
            else:
                value = schema.model_validate(data)
            return (data, value) if value is not None else None
        except (json.JSONDecodeError, UnicodeError, OSError, TypeError, ValidationError):
            return None

    if path.exists() and cache_is_current(path, input_path, recipe=recipe):
        result = validated(path)
        if result is not None:
            if _cache_provenance(path) is None:
                write_cache_fingerprint(path, input_path)
            return result[1]

    input_hash = cache_input_fingerprint(input_path)
    # ponytail: this is O(n²) when every file misses; add a per-directory index
    # if extraction caches grow beyond the current corpus scale.
    for sibling in sorted(path.parent.glob("*.json")):
        if sibling == path:
            continue
        provenance = _cache_provenance(sibling)
        if (
            provenance is None
            or provenance["input_sha256"] != input_hash
            or (recipe is not None and provenance.get("recipe") != recipe)
        ):
            continue
        try:
            if provenance["output_sha256"] != cache_input_fingerprint(sibling):
                continue
        except OSError:
            continue
        result = validated(sibling)
        if result is None:
            continue
        write_json(path, result[0])
        write_cache_fingerprint(path, input_path, recipe=provenance.get('recipe'))
        return result[1]
    return None


def write_pydantic(path: Path, obj: BaseModel) -> None:
    write_json(path, obj.model_dump())


def write_pydantic_list(path: Path, objs: Iterable[BaseModel]) -> None:
    write_json(path, [o.model_dump() for o in objs])


def serialize_structured(value: Any) -> str:
    """Serialize structured CSV cell values with JSON syntax."""
    return json.dumps(value, ensure_ascii=False)


def _parse_structured(value: object) -> object:
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    if s.startswith("[") or s.startswith("{"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return None
    return None


def parse_authors_list(value: object) -> list[str]:
    """Parse an Authors_List CSV cell or in-memory value into author strings."""
    parsed = _parse_structured(value)
    if isinstance(parsed, list):
        return [str(item) for item in parsed if item]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if ";" in s:
            return [part.strip() for part in s.split(";") if part.strip()]
        return [s]
    return []


def parse_openalex_authors(value: object) -> list[dict] | None:
    """Parse the OpenAlex_Authors structured CSV cell."""
    parsed = _parse_structured(value)
    if not isinstance(parsed, list):
        return None
    out = []
    for item in parsed:
        if isinstance(item, dict):
            out.append(item)
    return out


def require_columns(df: Any, columns: Iterable[str], *, artifact: str) -> None:
    """Raise a clear error when a DataFrame-like artifact lacks required columns."""
    available = set(getattr(df, "columns", []))
    missing = [column for column in columns if column not in available]
    if missing:
        suffix = "s" if len(missing) > 1 else ""
        raise ValueError(
            f"{artifact} is missing required column{suffix}: {', '.join(missing)}"
        )


SOURCE_COLUMNS = ["id", "source_file", "Title", "Authors", "Authors_List", "Journal", "Year"]
CITATION_COLUMNS = ["Title", "Authors", "Authors_List", "Journal", "Year", "citing_id"]
WORK_COLUMNS = ["id", "ring", "source_file", "Title", "Authors", "Authors_List", "Journal", "Year"]
AUTHOR_COLUMNS = ["id", "display_name", "surname", "surname_norm", "canonical_given", "initials",
                  "openalex_id", "orcid", "n_works", "n_core_works", "n_citations_received",
                  "n_distinct_citing_works"]
AUTHOR_CITATION_COLUMNS = ["author_id", "record_id", "position", "raw_author"]


def read_stage_csv(path: Path, *, columns: list[str], index_col: str | None = None) -> pd.DataFrame:
    """Read a checkpoint, accepting only truly blank legacy files as empty."""
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        frame = pd.DataFrame(columns=columns)
    # Bibliographic APIs support either author representation, including legacy CSVs.
    required = [c for c in columns if c not in ("Authors", "Authors_List", "Journal")]
    if "ring" in columns:
        required = [c for c in required if c not in ("ring", "source_file")]
    require_columns(frame, required, artifact=path.stem)
    if "Authors" in columns and not {"Authors", "Authors_List"} & set(frame.columns):
        raise ValueError(f"{path.name} is missing required column: Authors or Authors_List")
    return frame.set_index(index_col) if index_col else frame


def fingerprint(value: Any) -> str:
    """Content digest for provenance only, never for entity ID generation."""
    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, str):
            if not item:
                return None
            structured = _parse_structured(item)
            return clean(structured) if structured is not None else item
        if item is None or pd.isna(item):
            return None
        if hasattr(item, "item"):
            item = item.item()
        if isinstance(item, float) and item.is_integer():
            return int(item)
        return item
    payload = json.dumps(clean(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def frame_fingerprint(frame: pd.DataFrame) -> str:
    rows = frame.reset_index() if frame.index.name and frame.index.name not in frame.columns else frame
    return fingerprint({"columns": list(rows.columns), "records": rows.to_dict("records")})


def artifact_fingerprint(path: Path) -> str | None:
    """Snapshot a checkpoint; distinguish absence from a present empty artifact."""
    if not path.exists():
        return None
    if path.suffix == ".csv":
        try:
            return frame_fingerprint(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            return frame_fingerprint(pd.DataFrame())
        except pd.errors.ParserError:
            # Hand-curated CSVs may contain comment lines with commas. Hash
            # the bytes rather than making those comments invalid to pandas.
            return hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Fixed names also constrain report reads: audit contents never select arbitrary paths.
WORK_AUDIT_INPUTS = {"sources.csv", "citations_raw.csv", "work_identity.json"}
WORK_AUDIT_OUTPUTS = {"works.csv", "citation_graph.csv"}
AUTHOR_AUDIT_INPUTS = {
    "works.csv", "citation_graph.csv", "enriched_works.csv", "enrichment_provenance.json",
    "source_ids.json", "work_id_collisions.json", "author_aliases.csv", "journal_aliases.csv", "author_external_ids.csv", "author_overrides.csv", "author_overrides_meta.json",
    "work_identity.json", "author_identity.json",
}
AUTHOR_AUDIT_OUTPUTS = {"authors.csv", "author_citations.csv", "author_review.json"}


def unverified_collision_ids(layout: OutLayout) -> set[str]:
    """IDs whose legacy identity evidence is unsafe, including historical collisions."""
    ids: set[str] = set()
    for path in (layout.source_ids_json, layout.work_id_collisions_json):
        if path.exists():
            ids.update(read_json(path).get("collision_ids", []))
    return ids


def metadata_fingerprint(record: dict) -> str:
    authors = parse_authors_list(record.get("Authors_List"))
    return fingerprint({"Title": record.get("Title"), "Authors_List": authors,
                        "Authors": ", ".join(authors) if authors else record.get("Authors"),
                        "Journal": record.get("Journal"), "Year": record.get("Year")})
