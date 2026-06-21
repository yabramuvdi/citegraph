"""Smoke tests for the per-stage CLI surface.

These exercise the typer wiring (flags, exit codes, error paths, the
``--yes`` confirmation skip) rather than the underlying pipeline logic —
which is covered by ``test_pipeline.py`` and friends. Tests stay
network-free by either pre-populating per-paper caches or by hitting
commands that never call the LLM (``status``, ``estimate``, ``dedup``).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from citegraph.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def test_status_on_empty_out_dir(tmp_path: Path) -> None:
    result = runner.invoke(app, ["status", "--out", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert "markdown/" in result.output
    assert "missing" in result.output


def test_status_after_partial_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "out"
    (out / "markdown").mkdir(parents=True)
    (out / "markdown" / "a.md").write_text("# a")
    (out / "papers.csv").write_text("id,Title\np-x-2020-foo,Foo\n")

    result = runner.invoke(app, ["status", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "1 files" in result.output
    assert "1 rows" in result.output


# ---------------------------------------------------------------------------
# estimate
# ---------------------------------------------------------------------------
def test_estimate_missing_markdown_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["estimate", "--out", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert "No markdown files" in result.output


def test_estimate_reports_token_summary(tmp_path: Path) -> None:
    out = tmp_path / "out"
    (out / "markdown").mkdir(parents=True)
    (out / "markdown" / "a.md").write_text("Body.\n## References\n1. X.\n")
    (out / "markdown" / "b.md").write_text("Body.\n## References\n2. Y.\n")

    result = runner.invoke(app, ["estimate", "--out", str(out), "--model", "gemini-2.0-flash"])
    assert result.exit_code == 0, result.output
    assert "Files found:" in result.output
    assert "Est. input tokens:" in result.output
    assert "Est. cost:" in result.output


# ---------------------------------------------------------------------------
# convert OCR flags
# ---------------------------------------------------------------------------
def test_convert_rejects_mutually_exclusive_ocr_flags(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "convert",
            str(pdf_dir),
            "--out",
            str(tmp_path / "out"),
            "--ocr",
            "--ocr-auto",
        ],
    )
    assert result.exit_code == 2
    assert "--ocr and --ocr-auto are mutually exclusive" in result.output


def test_llm_concurrency_option_is_exposed_on_llm_commands() -> None:
    for command in ("run", "metadata", "references"):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, result.output
        assert "--llm-concurrency" in result.output


def test_llm_concurrency_option_rejects_zero(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "metadata",
            "--llm-concurrency",
            "0",
            "--out",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 2
    assert "--llm-concurrency" in result.output


# ---------------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------------
def test_dedup_missing_input_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["dedup", "--out", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert "Missing" in result.output


def test_dedup_happy_path_writes_outputs(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    raw = out / "references_raw.csv"
    raw.write_text(
        "Title,Authors_List,Authors,Journal,Year,citing_id\n"
        "Governing the Commons,\"['Elinor Ostrom']\",\"Ostrom, E.\",CUP,1990,p-a\n"
        "Governing the commons.,\"['Elinor Ostrom']\",\"Ostrom, E.\",CUP,1990,p-b\n"
        "Tragedy of the Commons,\"['Garrett Hardin']\",\"Hardin, G.\",Science,1968,p-a\n"
    )

    result = runner.invoke(app, ["dedup", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "Deduplicated 3 -> 2 references" in result.output
    assert (out / "references.csv").exists()
    assert (out / "citation_graph.csv").exists()


# ---------------------------------------------------------------------------
# metadata + references with everything cached (no LLM call needed)
# ---------------------------------------------------------------------------
def _prime_cached_pipeline(out: Path) -> None:
    """Populate markdown/, metadata/, and references/ so the loops only hydrate from cache."""
    (out / "markdown").mkdir(parents=True)
    (out / "metadata").mkdir(parents=True)
    (out / "references").mkdir(parents=True)
    (out / "markdown" / "p.md").write_text("# Body\n## References\n1. ref.\n")
    (out / "metadata" / "p.json").write_text(
        json.dumps(
            {
                "Title": "Cached Paper",
                "Authors_List": ["Some One"],
                "Journal": "J",
                "Year": 2020,
            }
        )
    )
    (out / "references" / "p.json").write_text(
        json.dumps(
            [
                {
                    "Title": "Cached Ref",
                    "Authors_List": ["Other Person"],
                    "Journal": "J",
                    "Year": 2019,
                }
            ]
        )
    )


def test_metadata_command_loads_cache_without_llm(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _prime_cached_pipeline(out)

    result = runner.invoke(app, ["metadata", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "1 papers" in result.output
    assert (out / "papers.csv").exists()


def test_references_yes_flag_skips_prompt_and_runs(tmp_path: Path) -> None:
    """--yes must skip the cost-estimate confirmation; no stdin available in CliRunner."""
    out = tmp_path / "out"
    _prime_cached_pipeline(out)
    # Need papers.csv for the references stage to map source_file -> citing_id.
    runner.invoke(app, ["metadata", "--out", str(out)])

    result = runner.invoke(app, ["references", "--out", str(out), "--yes"])
    assert result.exit_code == 0, result.output
    assert "Cost estimate" not in result.output, "should be suppressed by --yes"
    assert (out / "references_raw.csv").exists()


def test_references_command_skips_prompt_when_all_cached(tmp_path: Path) -> None:
    """When every paper is cached, est.n_references_to_process == 0 → no prompt even without --yes."""
    out = tmp_path / "out"
    _prime_cached_pipeline(out)
    runner.invoke(app, ["metadata", "--out", str(out)])

    result = runner.invoke(app, ["references", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "Cost estimate" not in result.output
    assert (out / "references_raw.csv").exists()


# ---------------------------------------------------------------------------
# --help surface
# ---------------------------------------------------------------------------
def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("run", "convert", "metadata", "references", "dedup", "estimate", "status"):
        assert cmd in result.output
