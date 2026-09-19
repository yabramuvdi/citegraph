"""The core-author Excel export."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from click import unstyle
from typer.testing import CliRunner

from citegraph.author_export import ExtraSheet, build_core_author_workbook
from citegraph.cli import app

openpyxl = pytest.importorskip("openpyxl")

runner = CliRunner()


def _make_out_dir(tmp_path: Path) -> Path:
    """A tiny out_dir: two ring-0 works, one ring-1 work, four authors.

    ``a-one`` is on a core work *and* a cited one, ``a-three`` only on the
    cited one — which is what separates "author of my papers" from "author
    anywhere in the corpus".
    """
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [
            {"id": "w-a", "ring": 0, "source_file": "a.pdf", "Title": "Alpha",
             "Authors": "One, A.", "Journal": "Ecological Economics",
             "Journal_Canonical": "Ecological Economics", "Year": 2001},
            {"id": "w-b", "ring": 0, "source_file": "b.pdf", "Title": "Beta",
             "Authors": "Two, B.", "Journal": "Science",
             "Journal_Canonical": "Science", "Year": 2002},
            {"id": "w-c", "ring": 1, "source_file": "", "Title": "Gamma",
             "Authors": "Three, C.", "Journal": "Nature",
             "Journal_Canonical": "Nature", "Year": 1999},
        ]
    ).to_csv(out / "works.csv", index=False)
    pd.DataFrame(
        [
            {"author_id": "a-one", "record_id": "w-a", "position": 0, "raw_author": "One, A."},
            {"author_id": "a-one", "record_id": "w-c", "position": 1, "raw_author": "Ana One"},
            {"author_id": "a-two", "record_id": "w-b", "position": 0, "raw_author": "Two, B."},
            {"author_id": "a-three", "record_id": "w-c", "position": 0, "raw_author": "Three, C."},
        ]
    ).to_csv(out / "author_citations.csv", index=False)
    pd.DataFrame(
        [
            {"id": "a-one", "display_name": "Ana One", "surname": "one",
             "openalex_id": "A1", "orcid": "", "n_works": 2, "n_core_works": 1,
             "n_citations_received": 3, "n_distinct_citing_works": 2},
            {"id": "a-two", "display_name": "Bo Two", "surname": "two",
             "openalex_id": "", "orcid": "0000-0002", "n_works": 1, "n_core_works": 1,
             "n_citations_received": 0, "n_distinct_citing_works": 0},
            {"id": "a-three", "display_name": "Cy Three", "surname": "three",
             "openalex_id": "", "orcid": "", "n_works": 1, "n_core_works": 0,
             "n_citations_received": 5, "n_distinct_citing_works": 4},
        ]
    ).to_csv(out / "authors.csv", index=False)
    return out


def test_authors_sheet_holds_only_authors_of_core_works(tmp_path: Path) -> None:
    path = build_core_author_workbook(_make_out_dir(tmp_path))
    sheet = pd.read_excel(path, sheet_name="Authors")
    assert sorted(sheet["id"]) == ["a-one", "a-two"]


def test_works_sheet_keeps_a_core_authors_cited_works_too(tmp_path: Path) -> None:
    """A core author's non-core work is still theirs, and still interesting."""
    path = build_core_author_workbook(_make_out_dir(tmp_path))
    sheet = pd.read_excel(path, sheet_name="Works")
    pairs = sorted(zip(sheet["author_id"], sheet["record_id"], strict=True))
    assert pairs == [("a-one", "w-a"), ("a-one", "w-c"), ("a-two", "w-b")]
    gamma = sheet[sheet["record_id"] == "w-c"].iloc[0]
    assert gamma["ring"] == 1
    assert gamma["Title"] == "Gamma"
    assert gamma["raw_author"] == "Ana One"


def _career() -> ExtraSheet:
    return ExtraSheet(
        name="Career_Positions",
        frame=pd.DataFrame(
            [
                {"author_id": "a-one", "institution": "Uniandes", "country": "CO"},
                {"author_id": "a-three", "institution": "Nowhere", "country": "XX"},
            ]
        ),
        description="Hand-collected positions.",
    )


def test_extra_sheet_is_filtered_to_core_authors(tmp_path: Path) -> None:
    path = build_core_author_workbook(_make_out_dir(tmp_path), extra_sheets=(_career(),))
    sheet = pd.read_excel(path, sheet_name="Career_Positions")
    assert list(sheet["author_id"]) == ["a-one"]


def test_master_flags_which_authors_the_extra_sheet_covers(tmp_path: Path) -> None:
    """The gap is the point: most core authors have no career metadata."""
    path = build_core_author_workbook(_make_out_dir(tmp_path), extra_sheets=(_career(),))
    sheet = pd.read_excel(path, sheet_name="Authors").set_index("id")
    assert bool(sheet.loc["a-one", "in_Career_Positions"]) is True
    assert bool(sheet.loc["a-two", "in_Career_Positions"]) is False


def test_edge_row_survives_when_only_one_end_is_a_core_author(tmp_path: Path) -> None:
    """A supervisor outside the corpus still supervised somebody in it."""
    ties = ExtraSheet(
        name="Supervision",
        frame=pd.DataFrame(
            [
                {"advisor_author_id": "a-outsider", "advisee_author_id": "a-two"},
                {"advisor_author_id": "a-three", "advisee_author_id": "a-nobody"},
            ]
        ),
    )
    path = build_core_author_workbook(_make_out_dir(tmp_path), extra_sheets=(ties,))
    sheet = pd.read_excel(path, sheet_name="Supervision")
    assert list(sheet["advisee_author_id"]) == ["a-two"]


def test_extra_sheet_without_an_author_id_column_is_rejected(tmp_path: Path) -> None:
    orphan = ExtraSheet(name="Mystery", frame=pd.DataFrame([{"name": "Ana One"}]))
    with pytest.raises(ValueError, match="no author-id column"):
        build_core_author_workbook(_make_out_dir(tmp_path), extra_sheets=(orphan,))


@pytest.mark.parametrize("name", ["README", "Authors", "Works", "Review_Flags", "authors"])
def test_reserved_extra_sheet_name_is_rejected_before_output_changes(
    tmp_path: Path, name: str
) -> None:
    out = _make_out_dir(tmp_path)
    dest = tmp_path / "existing.xlsx"
    dest.write_bytes(b"sentinel")
    extra = ExtraSheet(name=name, frame=pd.DataFrame([{"author_id": "a-one"}]))

    with pytest.raises(ValueError, match="conflicts with workbook sheet"):
        build_core_author_workbook(out, dest, extra_sheets=(extra,))

    assert dest.read_bytes() == b"sentinel"


def test_extra_sheet_names_must_be_unique_case_insensitively(tmp_path: Path) -> None:
    out = _make_out_dir(tmp_path)
    extras = (
        ExtraSheet(name="Career", frame=pd.DataFrame([{"author_id": "a-one"}])),
        ExtraSheet(name="career", frame=pd.DataFrame([{"author_id": "a-two"}])),
    )

    with pytest.raises(ValueError, match="conflicts with workbook sheet"):
        build_core_author_workbook(out, extra_sheets=extras)


def test_master_lists_every_spelling_the_corpus_used_for_an_author(tmp_path: Path) -> None:
    """The variants are the evidence behind a cluster — worth seeing at a glance."""
    path = build_core_author_workbook(_make_out_dir(tmp_path))
    sheet = pd.read_excel(path, sheet_name="Authors").set_index("id")
    assert sheet.loc["a-one", "name_variants"] == "Ana One; One, A."
    assert sheet.loc["a-two", "name_variants"] == "Two, B."


def _add_optional_artifacts(out: Path) -> Path:
    """The QC and network artifacts, which a young out_dir may not have yet."""
    (out / "author_review.json").write_text(
        json.dumps(
            [
                {"author_id": "a-one", "display_name": "Ana One", "n_occurrences": 2,
                 "raw_strings": ["Ana One", "One, A."],
                 "reason": "possible compound surname"},
                {"author_id": "a-three", "display_name": "Cy Three", "n_occurrences": 1,
                 "raw_strings": ["Three, C."], "reason": "initial-only cluster"},
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"author_id": "a-one", "Author": "Ana One", "Citing_publications": 4,
             "Degree": 12, "Degree_centrality": 0.5, "Normalised_strength": 3.2,
             "Degree_rank": 1, "Normalised_strength_rank": 1},
            {"author_id": "a-three", "Author": "Cy Three", "Citing_publications": 3,
             "Degree": 5, "Degree_centrality": 0.2, "Normalised_strength": 1.1,
             "Degree_rank": 2, "Normalised_strength_rank": 2},
        ]
    ).to_csv(out / "author_centrality.csv", index=False)
    return out


def test_review_flags_sheet_is_restricted_to_core_authors(tmp_path: Path) -> None:
    out = _add_optional_artifacts(_make_out_dir(tmp_path))
    path = build_core_author_workbook(out)
    sheet = pd.read_excel(path, sheet_name="Review_Flags")
    assert list(sheet["author_id"]) == ["a-one"]
    assert sheet.iloc[0]["reason"] == "possible compound surname"


def test_centrality_joins_onto_the_master_and_is_blank_where_absent(tmp_path: Path) -> None:
    """Only authors cited often enough are in the co-citation network."""
    out = _add_optional_artifacts(_make_out_dir(tmp_path))
    sheet = pd.read_excel(build_core_author_workbook(out), sheet_name="Authors").set_index("id")
    assert sheet.loc["a-one", "Degree"] == 12
    assert pd.isna(sheet.loc["a-two", "Degree"])


def test_optional_artifacts_are_optional(tmp_path: Path) -> None:
    """A corpus that has not run the figures notebook still exports."""
    path = build_core_author_workbook(_make_out_dir(tmp_path))
    names = openpyxl.load_workbook(path).sheetnames
    assert "Review_Flags" not in names
    assert "Authors" in names


def test_extra_sheet_can_lift_headline_columns_into_the_master(tmp_path: Path) -> None:
    """The master should answer "where is this person?" without a sheet hop."""
    base = ExtraSheet(
        name="Institutional",
        frame=pd.DataFrame(
            [
                {"author_id": "a-one", "affiliation_institution": "Uniandes",
                 "phd_year_end": 2004, "notes": "not lifted"},
            ]
        ),
        master_columns=("affiliation_institution", "phd_year_end"),
    )
    path = build_core_author_workbook(_make_out_dir(tmp_path), extra_sheets=(base,))
    sheet = pd.read_excel(path, sheet_name="Authors").set_index("id")
    assert sheet.loc["a-one", "affiliation_institution"] == "Uniandes"
    assert pd.isna(sheet.loc["a-two", "affiliation_institution"])
    assert "notes" not in sheet.columns


def test_lifting_from_a_sheet_with_repeated_authors_is_rejected(tmp_path: Path) -> None:
    """A tidy per-position table would silently multiply the master's rows."""
    positions = ExtraSheet(
        name="Career_Positions",
        frame=pd.DataFrame(
            [
                {"author_id": "a-one", "institution": "Uniandes"},
                {"author_id": "a-one", "institution": "UMass"},
            ]
        ),
        master_columns=("institution",),
    )
    with pytest.raises(ValueError, match="one row per author"):
        build_core_author_workbook(_make_out_dir(tmp_path), extra_sheets=(positions,))


def test_readme_sheet_documents_every_sheet_and_comes_first(tmp_path: Path) -> None:
    """A shared spreadsheet has to explain itself; nobody reads the CLI docs."""
    out = _add_optional_artifacts(_make_out_dir(tmp_path))
    path = build_core_author_workbook(out, extra_sheets=(_career(),))
    assert openpyxl.load_workbook(path).sheetnames[0] == "README"
    readme = pd.read_excel(path, sheet_name="README").set_index("Sheet")
    assert readme.loc["Authors", "Rows"] == 2
    assert readme.loc["Works", "Rows"] == 3
    assert readme.loc["Career_Positions", "Description"] == "Hand-collected positions."
    assert "author_review.json" in readme.loc["Review_Flags", "Source"]


def test_readme_records_how_many_core_authors_the_extra_sheet_misses(tmp_path: Path) -> None:
    """The 88 authors with no career metadata are the finding, not a defect."""
    path = build_core_author_workbook(_make_out_dir(tmp_path), extra_sheets=(_career(),))
    readme = pd.read_excel(path, sheet_name="README").set_index("Sheet")
    assert readme.loc["Career_Positions", "Core_authors_covered"] == "1 of 2"


def test_missing_openpyxl_says_how_to_install_it(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    with pytest.raises(ImportError, match=r"citegraph\[excel\]"):
        build_core_author_workbook(_make_out_dir(tmp_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_writes_the_workbook_into_out_dir(tmp_path: Path) -> None:
    out = _make_out_dir(tmp_path)
    result = runner.invoke(app, ["export-authors", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "core_authors.xlsx").exists()
    assert "core_authors.xlsx" in result.output


def test_cli_adds_an_extra_csv_as_a_named_sheet(tmp_path: Path) -> None:
    out = _make_out_dir(tmp_path)
    extra = tmp_path / "career_positions.csv"
    _career().frame.to_csv(extra, index=False)
    result = runner.invoke(
        app,
        ["export-authors", "--out", str(out), "--extra-csv", f"Career_Positions={extra}"],
    )
    assert result.exit_code == 0, result.output
    sheet = pd.read_excel(out / "core_authors.xlsx", sheet_name="Career_Positions")
    assert list(sheet["author_id"]) == ["a-one"]


def test_cli_lifts_a_named_column_into_the_master(tmp_path: Path) -> None:
    out = _make_out_dir(tmp_path)
    extra = tmp_path / "base.csv"
    pd.DataFrame([{"author_id": "a-one", "affiliation_institution": "Uniandes"}]).to_csv(
        extra, index=False
    )
    result = runner.invoke(
        app,
        [
            "export-authors", "--out", str(out),
            "--extra-csv", f"Institutional={extra}",
            "--master-column", "Institutional=affiliation_institution",
        ],
    )
    assert result.exit_code == 0, result.output
    sheet = pd.read_excel(out / "core_authors.xlsx", sheet_name="Authors").set_index("id")
    assert sheet.loc["a-one", "affiliation_institution"] == "Uniandes"


def test_cli_rejects_a_master_column_for_an_unknown_sheet(tmp_path: Path) -> None:
    out = _make_out_dir(tmp_path)
    result = runner.invoke(
        app,
        ["export-authors", "--out", str(out), "--master-column", "Ghost=affiliation"],
    )
    assert result.exit_code == 1
    assert "Ghost" in unstyle(result.output)


def test_cli_reports_a_missing_out_dir(tmp_path: Path) -> None:
    result = runner.invoke(app, ["export-authors", "--out", str(tmp_path / "nope")])
    assert result.exit_code == 1
    assert "does not exist" in unstyle(result.output)
