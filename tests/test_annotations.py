"""Tests for the hand-curated work annotation surface."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from citegraph.annotations import load_annotation_schema, load_annotations
from citegraph.graph import CitationGraph


@pytest.fixture()
def works() -> pd.DataFrame:
    """Two core works and one cited stub, shaped like works.csv."""
    return pd.DataFrame(
        [
            {"id": "w-a", "ring": 0, "source_file": "a.pdf", "Title": "Paper A", "Year": 2020},
            {"id": "w-b", "ring": 0, "source_file": "b.pdf", "Title": "Paper B", "Year": 2021},
            {"id": "w-x", "ring": 1, "source_file": "", "Title": "Ref X", "Year": 1990},
        ]
    ).set_index("id")


def write_csv(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


SCHEMA = """
column,type,allowed,description
Tema,enum,Economics|Development,Broad research theme
Publication_Type,enum,paper|book chapter,Kind of publication
"""


# ---------------------------------------------------------------------------
# Schema file
# ---------------------------------------------------------------------------
def test_schema_declares_enum_values(tmp_path: Path) -> None:
    schema = load_annotation_schema(write_csv(tmp_path / "annotation_schema.csv", SCHEMA))

    assert set(schema) == {"Tema", "Publication_Type"}
    assert schema["Tema"].allowed == ("Economics", "Development")
    assert schema["Tema"].description == "Broad research theme"


def test_absent_schema_declares_nothing(tmp_path: Path) -> None:
    assert load_annotation_schema(tmp_path / "annotation_schema.csv") == {}


def test_unknown_column_type_is_rejected(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "annotation_schema.csv",
        "column,type,allowed,description\nTema,categorical,A|B,Theme",
    )

    with pytest.raises(ValueError, match="categorical"):
        load_annotation_schema(path)


def test_enum_without_allowed_values_is_rejected(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "annotation_schema.csv",
        "column,type,allowed,description\nTema,enum,,Theme",
    )

    with pytest.raises(ValueError, match="Tema"):
        load_annotation_schema(path)


# ---------------------------------------------------------------------------
# Loading annotations
# ---------------------------------------------------------------------------
def test_annotations_are_indexed_by_work_id(tmp_path: Path, works: pd.DataFrame) -> None:
    path = write_csv(
        tmp_path / "work_annotations.csv",
        "id,Tema,Publication_Type\nw-a,Economics,paper\nw-b,Development,book chapter",
    )

    annotations = load_annotations(path, works=works)

    assert annotations.index.name == "id"
    assert list(annotations.index) == ["w-a", "w-b"]
    assert annotations.loc["w-a", "Tema"] == "Economics"
    assert annotations.loc["w-b", "Publication_Type"] == "book chapter"


def test_absent_file_loads_as_empty(tmp_path: Path, works: pd.DataFrame) -> None:
    annotations = load_annotations(tmp_path / "work_annotations.csv", works=works)

    assert annotations.empty
    assert annotations.index.name == "id"


def test_context_columns_are_dropped(tmp_path: Path, works: pd.DataFrame) -> None:
    """Title/Year are in the file so a human can read it, not to be joined."""
    path = write_csv(
        tmp_path / "work_annotations.csv",
        "id,source_file,Title,Year,Tema\nw-a,a.pdf,Paper A,2020,Economics",
    )

    annotations = load_annotations(path, works=works)

    assert list(annotations.columns) == ["Tema"]


def test_undeclared_columns_are_carried(tmp_path: Path, works: pd.DataFrame) -> None:
    path = write_csv(
        tmp_path / "work_annotations.csv",
        "id,Tema,Reviewer_note\nw-a,Economics,checked against the PDF",
    )
    schema_path = write_csv(tmp_path / "annotation_schema.csv", SCHEMA)

    annotations = load_annotations(path, works=works, schema_path=schema_path)

    assert annotations.loc["w-a", "Reviewer_note"] == "checked against the PDF"


def test_declared_column_absent_from_file_is_present_and_empty(
    tmp_path: Path, works: pd.DataFrame
) -> None:
    path = write_csv(tmp_path / "work_annotations.csv", "id,Tema\nw-a,Economics")
    schema_path = write_csv(tmp_path / "annotation_schema.csv", SCHEMA)

    annotations = load_annotations(path, works=works, schema_path=schema_path)

    assert "Publication_Type" in annotations.columns
    assert annotations["Publication_Type"].isna().all()


def test_blank_values_are_missing_not_empty_strings(
    tmp_path: Path, works: pd.DataFrame
) -> None:
    path = write_csv(
        tmp_path / "work_annotations.csv",
        "id,Tema\nw-a,Economics\nw-b,",
    )

    annotations = load_annotations(path, works=works)

    assert pd.isna(annotations.loc["w-b", "Tema"])


def test_missing_id_column_is_rejected(tmp_path: Path, works: pd.DataFrame) -> None:
    path = write_csv(tmp_path / "work_annotations.csv", "work,Tema\nw-a,Economics")

    with pytest.raises(ValueError, match="id"):
        load_annotations(path, works=works)


def test_unknown_work_id_is_rejected(tmp_path: Path, works: pd.DataFrame) -> None:
    path = write_csv(
        tmp_path / "work_annotations.csv",
        "id,Tema\nw-a,Economics\nw-nope,Development",
    )

    with pytest.raises(ValueError, match="w-nope"):
        load_annotations(path, works=works)


def test_duplicate_work_id_is_rejected(tmp_path: Path, works: pd.DataFrame) -> None:
    path = write_csv(
        tmp_path / "work_annotations.csv",
        "id,Tema\nw-a,Economics\nw-a,Development",
    )

    with pytest.raises(ValueError, match="w-a"):
        load_annotations(path, works=works)


def test_value_outside_the_declared_enum_is_rejected(
    tmp_path: Path, works: pd.DataFrame
) -> None:
    path = write_csv(
        tmp_path / "work_annotations.csv",
        "id,Tema\nw-a,Astrology",
    )
    schema_path = write_csv(tmp_path / "annotation_schema.csv", SCHEMA)

    with pytest.raises(ValueError, match="Astrology"):
        load_annotations(path, works=works, schema_path=schema_path)


def test_enum_violation_names_the_work(tmp_path: Path, works: pd.DataFrame) -> None:
    path = write_csv(tmp_path / "work_annotations.csv", "id,Tema\nw-b,Astrology")
    schema_path = write_csv(tmp_path / "annotation_schema.csv", SCHEMA)

    with pytest.raises(ValueError, match="w-b"):
        load_annotations(path, works=works, schema_path=schema_path)


def test_undeclared_column_values_are_not_validated(
    tmp_path: Path, works: pd.DataFrame
) -> None:
    path = write_csv(
        tmp_path / "work_annotations.csv",
        "id,Free_text\nw-a,anything at all",
    )
    schema_path = write_csv(tmp_path / "annotation_schema.csv", SCHEMA)

    annotations = load_annotations(path, works=works, schema_path=schema_path)

    assert annotations.loc["w-a", "Free_text"] == "anything at all"


# ---------------------------------------------------------------------------
# Persistent ids
# ---------------------------------------------------------------------------
def test_redirected_id_follows_the_work_it_became(
    tmp_path: Path, works: pd.DataFrame
) -> None:
    """An annotation written before a merge still reaches its work."""
    (tmp_path / "work_identity.json").write_text(
        json.dumps({"schema_version": 1, "redirects": {"w-old": "w-a"}}), encoding="utf-8"
    )
    path = write_csv(tmp_path / "work_annotations.csv", "id,Tema\nw-old,Economics")

    annotations = load_annotations(path, works=works)

    assert annotations.loc["w-a", "Tema"] == "Economics"


def test_redirect_collision_is_rejected(tmp_path: Path, works: pd.DataFrame) -> None:
    """Two rows resolving to one work is a conflict, not a silent overwrite."""
    (tmp_path / "work_identity.json").write_text(
        json.dumps({"schema_version": 1, "redirects": {"w-old": "w-a"}}), encoding="utf-8"
    )
    path = write_csv(
        tmp_path / "work_annotations.csv",
        "id,Tema\nw-old,Economics\nw-a,Development",
    )

    with pytest.raises(ValueError, match="w-a"):
        load_annotations(path, works=works)


# ---------------------------------------------------------------------------
# The graph join
# ---------------------------------------------------------------------------
def build_out_dir(tmp_path: Path, works: pd.DataFrame) -> Path:
    works.to_csv(tmp_path / "works.csv")
    pd.DataFrame([{"citing_id": "w-a", "cited_id": "w-x"}]).to_csv(
        tmp_path / "citation_graph.csv", index=False
    )
    return tmp_path


def test_graph_without_annotations_reports_none(tmp_path: Path, works: pd.DataFrame) -> None:
    graph = CitationGraph.from_out_dir(build_out_dir(tmp_path, works))

    assert graph.has_annotations is False
    assert "Tema" not in graph.core.columns


def test_core_carries_annotation_columns(tmp_path: Path, works: pd.DataFrame) -> None:
    out_dir = build_out_dir(tmp_path, works)
    write_csv(
        out_dir / "work_annotations.csv",
        "id,Tema,Publication_Type\nw-a,Economics,paper\nw-b,Development,book chapter",
    )

    graph = CitationGraph.from_out_dir(out_dir)

    assert graph.has_annotations is True
    assert graph.core.loc["w-a", "Tema"] == "Economics"
    assert graph.core["Publication_Type"].tolist() == ["paper", "book chapter"]


def test_unannotated_works_keep_a_missing_value(tmp_path: Path, works: pd.DataFrame) -> None:
    """A cited stub has no annotation; it must not vanish from the frame."""
    out_dir = build_out_dir(tmp_path, works)
    write_csv(out_dir / "work_annotations.csv", "id,Tema\nw-a,Economics")

    graph = CitationGraph.from_out_dir(out_dir)

    assert graph.n_works == 3
    assert pd.isna(graph.works.loc["w-x", "Tema"])


def test_annotations_are_available_as_their_own_frame(
    tmp_path: Path, works: pd.DataFrame
) -> None:
    out_dir = build_out_dir(tmp_path, works)
    write_csv(out_dir / "work_annotations.csv", "id,Tema\nw-a,Economics")

    graph = CitationGraph.from_out_dir(out_dir)

    assert list(graph.annotations.columns) == ["Tema"]


def test_graph_validates_annotations_against_the_schema(
    tmp_path: Path, works: pd.DataFrame
) -> None:
    out_dir = build_out_dir(tmp_path, works)
    write_csv(out_dir / "work_annotations.csv", "id,Tema\nw-a,Astrology")
    write_csv(out_dir / "annotation_schema.csv", SCHEMA)

    with pytest.raises(ValueError, match="Astrology"):
        CitationGraph.from_out_dir(out_dir)
