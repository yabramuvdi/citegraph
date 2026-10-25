"""Tests for JEL code extraction and the core-paper JEL table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from citegraph.cli import app
from citegraph.jel import JEL_GENERAL_CATEGORIES, core_jel_codes, extract_jel_codes, write_jel_codes

# ---------------------------------------------------------------------------
# extract_jel_codes — every format below was seen in the paper 4 markdown
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("JEL classification: C93; D02; D03; D64\n\n* Author names", ["C93", "D02", "D03", "D64"]),
        ("JEL: C10, C23, C72.\n", ["C10", "C23", "C72"]),
        ("JEL Codes: C93, H41, Q20, Q28", ["C93", "H41", "Q20", "Q28"]),
        ("JEL classification numbers: C92, D7, D85, H41.", ["C92", "D7", "D85", "H41"]),
        (
            "## JEL classification : A13, C72, C90, D91\n\nKeywords : x",
            ["A13", "C72", "C90", "D91"],
        ),
        ("jel : O15, Z18, Z19.", ["O15", "Z18", "Z19"]),
        # codes on the paragraph after the label
        ("JEL classification:\n\nC35\n\nKeywords: fishing", ["C35"]),
        # docling artefacts: non-breaking space, odd separator, fi ligature
        (
            "JEL Classification C93/uni00A0· D72/uni00A0· D74/uni00A0· D91\n",
            ["C93", "D72", "D74", "D91"],
        ),
        ("JEL Classification H4 /C1 C91 /C1 C93\n", ["H4", "C91", "C93"]),
        ("JEL classi /uniFB01 cation:\n\nC90\n\n## 1 Intro", ["C90"]),
        ("JEL classi /uniFB01 ation:\n\nC78\n", ["C78"]),
        ("JEL classi/fication C91/uni00A0· D91/uni00A0· D81", ["C91", "D91", "D81"]),
        # Elsevier "Article info" box: one code per paragraph, a space inside each
        (
            "## A R T I C L E  I N F O\n\nJEL codes:\n\nQ 52\n\nQ 01\n\nQ57\n\nQ 58\n\n"
            "Keywords:\n\nPro-environmental behavior",
            ["Q52", "Q01", "Q57", "Q58"],
        ),
        # OCR reads the letter O as a zero
        ("(JEL C93, 012, 018, Z13)", ["C93", "O12", "O18", "Z13"]),
        # a thesis carries one block per chapter: union them, in order, deduplicated
        (
            "JEL classification : L51, Q55\n\nchapter text\n\nJEL classification : C91, Q55.",
            ["L51", "Q55", "C91"],
        ),
        # not JEL blocks at all
        ("doi:10.1257/jel.50.2.368. Keynes (1936) wrote A1 pages.", []),
        ("Jell, 2000. 'Cooperation and between-group competition.'", []),
        ("published in the JEL\n\nTable A1 reports the results", []),
        ("No classification here.", []),
    ],
)
def test_extract_jel_codes(text: str, expected: list[str]) -> None:
    assert extract_jel_codes(text) == expected


def test_general_categories_cover_the_jel_alphabet() -> None:
    assert "".join(JEL_GENERAL_CATEGORIES) == "ABCDEFGHIJKLMNOPQRYZ"
    assert JEL_GENERAL_CATEGORIES["Q"].startswith("Agricultural and Natural Resource")


# ---------------------------------------------------------------------------
# core_jel_codes — extracted, hand-assigned, or missing, per core work
# ---------------------------------------------------------------------------


@pytest.fixture()
def out_dir(tmp_path: Path) -> Path:
    pd.DataFrame(
        [
            {"id": "w-a", "ring": 0, "source_file": "a.md", "Title": "Paper A", "Year": 2020},
            {"id": "w-b", "ring": 0, "source_file": "b.md", "Title": "Paper B", "Year": 2021},
            {"id": "w-c", "ring": 0, "source_file": "c.md", "Title": "Paper C", "Year": 2022},
            {"id": "w-x", "ring": 1, "source_file": "", "Title": "Ref X", "Year": 1990},
        ]
    ).to_csv(tmp_path / "works.csv", index=False)
    md = tmp_path / "markdown"
    md.mkdir()
    (md / "a.md").write_text("# Paper A\n\nJEL classification: C93; D64\n", encoding="utf-8")
    (md / "b.md").write_text("# Paper B\n\nNo codes in this one.\n", encoding="utf-8")
    (md / "c.md").write_text("# Paper C\n\nJEL: Q20\n", encoding="utf-8")
    return tmp_path


def test_core_jel_codes_reports_source_per_work(out_dir: Path) -> None:
    (out_dir / "work_annotations.csv").write_text(
        "id,Title,JEL_Codes\nw-b,Paper B,H41; Q2\n", encoding="utf-8"
    )

    frame = core_jel_codes(out_dir)

    assert list(frame.index) == ["w-a", "w-b", "w-c"]
    assert frame.loc["w-a", "JEL_Codes"] == ["C93", "D64"]
    assert frame.loc["w-a", "JEL_Source"] == "extracted"
    assert frame.loc["w-b", "JEL_Codes"] == ["H41", "Q2"]
    assert frame.loc["w-b", "JEL_Source"] == "manual"
    assert frame.loc["w-a", "JEL_General"] == ["C", "D"]


def test_missing_codes_are_an_empty_list_marked_missing(out_dir: Path) -> None:
    frame = core_jel_codes(out_dir)

    assert frame.loc["w-b", "JEL_Codes"] == []
    assert frame.loc["w-b", "JEL_Source"] == "missing"
    assert frame.loc["w-b", "JEL_General"] == []


def test_manual_codes_override_extracted_ones(out_dir: Path) -> None:
    (out_dir / "work_annotations.csv").write_text("id,JEL_Codes\nw-a,d71\n", encoding="utf-8")

    frame = core_jel_codes(out_dir)

    assert frame.loc["w-a", "JEL_Codes"] == ["D71"]
    assert frame.loc["w-a", "JEL_Source"] == "manual"


def test_malformed_manual_code_raises_naming_the_work(out_dir: Path) -> None:
    (out_dir / "work_annotations.csv").write_text(
        "id,JEL_Codes\nw-b,Microeconomics\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="w-b.*Microeconomics"):
        core_jel_codes(out_dir)


def test_unconverted_core_work_counts_as_missing(out_dir: Path) -> None:
    (out_dir / "markdown" / "c.md").unlink()

    frame = core_jel_codes(out_dir)

    assert frame.loc["w-c", "JEL_Source"] == "missing"


def test_write_jel_codes_round_trips_through_csv(out_dir: Path) -> None:
    path = write_jel_codes(out_dir)

    written = pd.read_csv(path, index_col="id", keep_default_na=False)
    assert path == out_dir / "jel_codes.csv"
    assert list(written.columns) == [
        "source_file",
        "Title",
        "JEL_Codes",
        "JEL_General",
        "JEL_Source",
    ]
    assert written.loc["w-a", "JEL_Codes"] == "C93;D64"
    assert written.loc["w-a", "JEL_General"] == "C;D"
    assert written.loc["w-b", "JEL_Codes"] == ""
    assert "w-x" not in written.index


def test_cli_reports_how_many_core_papers_carry_codes(out_dir: Path) -> None:
    result = CliRunner().invoke(app, ["jel", "--out", str(out_dir)])

    assert result.exit_code == 0, result.output
    assert "2 of 3 core papers carry JEL codes in their text" in result.output
    assert "Paper B" in result.output
    assert (out_dir / "jel_codes.csv").exists()
