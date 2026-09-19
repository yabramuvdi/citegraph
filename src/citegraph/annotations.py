"""Hand-curated facts about works, joined onto the pipeline's output.

Some things about a paper are never in the paper: the theme a reviewer
assigned it, whether it counts as the design the manuscript is about. They
are read off the PDF by a human — or by an agent — and recorded by hand.

They cannot live in ``works.csv``, which stage 4 regenerates from scratch;
a column typed in there survives exactly until the next ``citegraph dedup``.
So annotations live beside it in ``work_annotations.csv``, keyed on the work
``id``, and are joined at read time. This is the arrangement
``journal_aliases.csv`` and ``author_aliases.csv`` already use: lenient read,
strict validation, and a loud failure when a curated row no longer matches
anything.

``annotation_schema.csv`` (``column,type,allowed,description``) declares what
may be recorded. Declared columns are validated on load — an agent writing
``Maybe`` into an enum fails immediately, naming the work. Undeclared columns
are carried untouched, so a reviewer can add a question mid-review without a
code change. The schema is as much the annotator's instructions as it is a
constraint, which is why the description column is not decoration.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from citegraph.io import OutLayout, read_json

SCHEMA_COLUMNS = ["column", "type", "allowed", "description"]
COLUMN_TYPES = ("enum", "text")


@dataclass(frozen=True)
class AnnotationColumn:
    """One declared annotation column."""

    name: str
    type: str
    allowed: tuple[str, ...]
    description: str


def load_annotation_schema(path: Path | str) -> dict[str, AnnotationColumn]:
    """Read ``annotation_schema.csv``; an absent file declares nothing."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    schema: dict[str, AnnotationColumn] = {}
    for number, row in enumerate(rows, 2):
        name = (row.get("column") or "").strip()
        kind = (row.get("type") or "").strip()
        allowed = tuple(
            value.strip() for value in (row.get("allowed") or "").split("|") if value.strip()
        )
        if not name:
            raise ValueError(f"Annotation schema row {number}: column name is empty")
        if kind not in COLUMN_TYPES:
            raise ValueError(
                f"Annotation schema row {number}: unknown type {kind!r} for {name!r}; "
                f"use one of {', '.join(COLUMN_TYPES)}"
            )
        if kind == "enum" and not allowed:
            raise ValueError(
                f"Annotation schema row {number}: enum column {name!r} declares no "
                "allowed values; list them separated by '|'"
            )
        if name in schema:
            raise ValueError(f"Annotation schema declares {name!r} twice")
        schema[name] = AnnotationColumn(
            name=name,
            type=kind,
            allowed=allowed,
            description=(row.get("description") or "").strip(),
        )
    return schema


def _redirects(path: Path) -> dict[str, str]:
    """Work-id redirects recorded by the identity registry beside the file."""
    registry = OutLayout(path.parent).work_identity_json
    return read_json(registry).get("redirects", {}) if registry.exists() else {}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(index=pd.Index([], name="id"))


def load_annotations(
    path: Path | str,
    *,
    works: pd.DataFrame,
    schema_path: Path | str | None = None,
) -> pd.DataFrame:
    """Read ``work_annotations.csv`` as a frame ready to join onto ``works``.

    Rows are keyed on the persistent work ``id`` and resolved through the
    identity registry's redirects, so an annotation written before a merge
    still reaches the work it became. An id that names no work raises rather
    than being dropped: a curated row that silently applies to nothing is
    worse than no row at all.

    Columns that ``works`` already has (``Title``, ``Year``, …) are *context*
    — they are written into the file so a human can read it, and are dropped
    here rather than joined back over the pipeline's own values.
    """
    from citegraph.identity import resolve_redirect

    path = Path(path)
    schema = load_annotation_schema(schema_path) if schema_path is not None else {}
    if not path.exists():
        return _empty()

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if "id" not in fieldnames:
        raise ValueError(f"{path} must have an 'id' column; got {fieldnames}")

    redirects = _redirects(path)
    known = set(works.index)
    records: dict[str, dict[str, object]] = {}
    for number, row in enumerate(rows, 2):
        raw_id = (row.get("id") or "").strip()
        if not raw_id:
            raise ValueError(f"{path} row {number}: empty work id")
        work_id = resolve_redirect(raw_id, redirects)
        if work_id not in known:
            raise ValueError(
                f"{path} row {number}: unknown work id {work_id!r}. Re-run the "
                "import, or delete the row if the work is gone."
            )
        if work_id in records:
            raise ValueError(f"{path} row {number}: work {work_id!r} is annotated twice")
        records[work_id] = {
            key: (value.strip() or None)
            for key, value in row.items()
            if key is not None and key != "id" and isinstance(value, str)
        }

    # Context columns are readable padding for the editor, not annotations.
    columns = [
        name
        for name in fieldnames
        if name != "id" and (name in schema or name not in works.columns)
    ]
    for name in schema:
        if name not in columns:
            columns.append(name)

    frame = pd.DataFrame(
        [[records[work_id].get(name) for name in columns] for work_id in records],
        index=pd.Index(list(records), name="id"),
        columns=columns,
        dtype=object,
    )
    _validate_values(frame, schema, path)
    return frame


def _validate_values(
    frame: pd.DataFrame, schema: Mapping[str, AnnotationColumn], path: Path
) -> None:
    """Reject any value a declared enum column does not allow."""
    for name, column in schema.items():
        if column.type != "enum" or name not in frame.columns:
            continue
        for work_id, value in frame[name].items():
            if value is None or pd.isna(value):
                continue
            if value not in column.allowed:
                raise ValueError(
                    f"{path}: work {work_id!r} has {name}={value!r}, which is not "
                    f"one of {', '.join(column.allowed)}. Fix the value, or add it "
                    "to annotation_schema.csv."
                )
