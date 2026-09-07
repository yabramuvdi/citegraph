"""Tests for the ``citegraph report`` QC dashboard.

Everything here is disk-only: the report reads artifacts and never talks
to the network, so no fake LLM client is needed. Corpora are synthesized
by hand into ``tmp_path`` at various levels of completeness.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from citegraph.cli import app
from citegraph.html_report import build_report_html, collect_report_data, write_report

runner = CliRunner()

_LONG_BODY = ("This paragraph carries enough substantive text to clear the "
              "image-only heuristic. " * 5)


def _stage(data: dict, label: str) -> dict:
    return next(s for s in data["stages"] if s["label"] == label)


def _write_partial_corpus(out: Path) -> None:
    """Convert + metadata done for two papers; references not run."""
    (out / "markdown").mkdir(parents=True)
    (out / "metadata").mkdir(parents=True)
    (out / "markdown" / "alpha.md").write_text(f"# Alpha\n{_LONG_BODY}", encoding="utf-8")
    (out / "markdown" / "beta.md").write_text(f"# Beta\n{_LONG_BODY}", encoding="utf-8")
    for stem, title in (("alpha", "Alpha Paper"), ("beta", "Beta Paper")):
        (out / "metadata" / f"{stem}.json").write_text(
            json.dumps(
                {"Title": title, "Authors_List": ["Some One"], "Journal": "J", "Year": 2020}
            ),
            encoding="utf-8",
        )
    (out / "sources.csv").write_text(
        "id,Title,Authors,Journal,Year,source_file\n"
        "w-one-2020-alpha,Alpha Paper,Some One,J,2020,alpha.md\n"
        "w-one-2020-beta,Beta Paper,Some One,J,2020,beta.md\n",
        encoding="utf-8",
    )


def _write_full_corpus(out: Path) -> None:
    """A corpus exercising every problem category the report knows about."""
    _write_partial_corpus(out)
    (out / "references").mkdir(parents=True)

    # Corrupt metadata cache for a third paper.
    (out / "markdown" / "badmeta.md").write_text(f"# Bad\n{_LONG_BODY}", encoding="utf-8")
    (out / "metadata" / "badmeta.json").write_text("{not json", encoding="utf-8")

    # A paper whose references extraction failed.
    (out / "markdown" / "failref.md").write_text(f"# Fail\n{_LONG_BODY}", encoding="utf-8")
    (out / "references_failures.jsonl").write_text(
        json.dumps(
            {
                "source_file": "failref.md",
                "stage": "references",
                "error_class": "RuntimeError",
                "error_message": "boom",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # A paper that returned zero references.
    (out / "markdown" / "zeroref.md").write_text(f"# Zero\n{_LONG_BODY}", encoding="utf-8")
    (out / "references" / "zeroref.json").write_text("[]", encoding="utf-8")
    (out / "papers_no_references.json").write_text(
        json.dumps(
            [{"paper_id": "p-zero", "source_file": "zeroref.md", "title": "Zero Paper"}]
        ),
        encoding="utf-8",
    )

    # Reference caches for the healthy papers.
    refs = [
        {"Title": f"Reference number {i}", "Authors_List": ["Ref Author"],
         "Journal": "J", "Year": 2000 + i}
        for i in range(20)
    ]
    (out / "references" / "alpha.json").write_text(json.dumps(refs), encoding="utf-8")
    (out / "references" / "beta.json").write_text(json.dumps(refs[:6]), encoding="utf-8")

    # Dedup artifacts: a mergeable duplicate pair plus a Year=0 row.
    (out / "citations_raw.csv").write_text(
        "Title,Authors_List,Authors,Journal,Year,citing_id\n"
        "Governing the Commons,\"['Elinor Ostrom']\",\"Ostrom, E.\",CUP,1990,w-one-2020-alpha\n"
        "Governing the commons.,\"['Elinor Ostrom']\",\"Ostrom, E.\",CUP,1990,w-one-2020-beta\n"
        "Tragedy of the Commons,\"['Garrett Hardin']\",\"Hardin, G.\",Science,0,w-one-2020-alpha\n",
        encoding="utf-8",
    )
    (out / "works.csv").write_text(
        "id,ring,source_file,Title,Authors,Journal,Year\n"
        "w-one-2020-alpha,0,alpha.md,Alpha Paper,Some One,J,2020\n"
        "w-one-2020-beta,0,beta.md,Beta Paper,Some One,J,2020\n"
        "w-ostrom-1990-governing-the-commons,1,,Governing the Commons,\"Ostrom, E.\",CUP,1990\n"
        "w-hardin-0-tragedy-of-the-commons,1,,Tragedy of the Commons,\"Hardin, G.\",Science,0\n",
        encoding="utf-8",
    )
    (out / "citation_graph.csv").write_text(
        "citing_id,cited_id\n"
        "w-one-2020-alpha,w-ostrom-1990-governing-the-commons\n"
        "w-one-2020-beta,w-ostrom-1990-governing-the-commons\n"
        "w-one-2020-alpha,w-hardin-0-tragedy-of-the-commons\n"
        "w-one-2020-alpha,w-one-2020-beta\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 1. Empty / missing out dir
# ---------------------------------------------------------------------------
def test_collect_on_missing_out_dir_returns_valid_structure(tmp_path: Path) -> None:
    data = collect_report_data(tmp_path / "nope")

    assert data["out_dir_exists"] is False
    assert data["papers"] == []
    assert data["errors"] == []
    assert data["dedup"] is None
    assert data["enrichment"] is None
    assert data["authors"] is None
    assert {s["label"] for s in data["stages"]} == {
        "convert", "metadata", "references", "dedup", "enrich", "authors",
    }
    assert _stage(data, "convert")["status"] == "not_run"
    assert _stage(data, "enrich")["status"] == "optional_not_run"
    # The whole structure must survive a JSON round-trip (it becomes the blob).
    json.dumps(data)


def test_write_report_on_missing_out_dir_writes_hint(tmp_path: Path) -> None:
    out = tmp_path / "nope"
    path = write_report(out)

    assert path == out / "report.html"
    html = path.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "citegraph convert" in html


# ---------------------------------------------------------------------------
# 2. Partial corpus
# ---------------------------------------------------------------------------
def test_partial_corpus_stage_strip_and_rows(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _write_partial_corpus(out)

    data = collect_report_data(out)
    assert _stage(data, "convert")["status"] == "done"
    assert _stage(data, "metadata")["status"] == "done"
    assert _stage(data, "references")["status"] == "not_run"
    assert _stage(data, "dedup")["status"] == "not_run"

    stems = [p["stem"] for p in data["papers"]]
    assert stems == ["alpha", "beta"]
    alpha = data["papers"][0]
    assert alpha["markdown"]["status"] == "ok"
    assert alpha["metadata"]["status"] == "ok"
    assert alpha["metadata"]["title"] == "Alpha Paper"
    assert alpha["references"]["status"] == "pending"

    html = build_report_html(data)
    assert "alpha" in html and "beta" in html
    assert "Alpha Paper" in html
    assert "citegraph references" in html  # placeholder hint for the next stage


# ---------------------------------------------------------------------------
# 3. Full synthetic corpus with problems
# ---------------------------------------------------------------------------
def test_full_corpus_flags_all_problem_categories(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _write_full_corpus(out)

    data = collect_report_data(out)

    # Corrupt metadata cache is a recorded finding, not a crash.
    assert any("badmeta.json" in e["path"] for e in data["errors"])
    badmeta = next(p for p in data["papers"] if p["stem"] == "badmeta")
    assert badmeta["metadata"]["status"] == "corrupt"

    failref = next(p for p in data["papers"] if p["stem"] == "failref")
    assert failref["references"]["status"] == "failed"

    zeroref = next(p for p in data["papers"] if p["stem"] == "zeroref")
    assert zeroref["references"]["status"] == "zero"

    # Dedup panel: the Ostrom pair must appear as a >1-member cluster.
    dedup = data["dedup"]
    assert dedup["n_raw"] == 3
    assert dedup["n_missing_year"] == 1
    audit = dedup["merge_audit"]
    assert len(audit) == 1
    assert len(audit[0]["members"]) == 2
    assert {m["citing_id"] for m in audit[0]["members"]} == {
        "w-one-2020-alpha", "w-one-2020-beta",
    }

    # A healthy paper with <5 references gets flagged as suspicious... but 6 is fine.
    beta = next(p for p in data["papers"] if p["stem"] == "beta")
    assert beta["notes"] == []
    assert beta["refs_more"] == 0

    alpha = next(p for p in data["papers"] if p["stem"] == "alpha")
    assert len(alpha["refs_preview"]) == 15
    assert alpha["refs_more"] == 5

    card_keys = {c["key"] for c in data["problems"]["cards"]}
    assert {"references_failures", "zero_references", "corrupt_files",
            "missing_year", "pending_papers"} <= card_keys

    html = build_report_html(data)
    assert "corrupt" in html
    assert "badmeta" in html
    assert "failref" in html
    assert "zeroref" in html
    assert "Governing the Commons" in html
    assert "absorbed 2 citation events" in html
    assert "recomputed with default DedupConfig" in html


def test_suspiciously_few_references_note(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _write_partial_corpus(out)
    (out / "references").mkdir()
    (out / "references" / "alpha.json").write_text(
        json.dumps(
            [{"Title": "Only Ref", "Authors_List": ["A"], "Journal": "J", "Year": 2001}]
        ),
        encoding="utf-8",
    )

    data = collect_report_data(out)
    alpha = next(p for p in data["papers"] if p["stem"] == "alpha")
    assert alpha["references"]["status"] == "ok"
    assert any("suspiciously few" in note for note in alpha["notes"])
    assert alpha["problem"] is True


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------
def test_report_command_writes_html(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _write_partial_corpus(out)

    result = runner.invoke(app, ["report", "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert (out / "report.html").exists()
    assert "report.html" in result.output


def test_report_command_missing_out_dir_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["report", "--out", str(tmp_path / "nope")])

    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_report_command_mentions_problem_count(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _write_full_corpus(out)

    result = runner.invoke(app, ["report", "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert "problem(s) flagged" in result.output


# ---------------------------------------------------------------------------
# 5. HTML safety
# ---------------------------------------------------------------------------
def test_html_escapes_hostile_titles(tmp_path: Path) -> None:
    out = tmp_path / "out"
    (out / "markdown").mkdir(parents=True)
    (out / "metadata").mkdir(parents=True)
    hostile = "<script>alert(1)</script>"
    (out / "markdown" / "evil.md").write_text(
        f"# Evil\n{hostile}\n{_LONG_BODY}", encoding="utf-8"
    )
    (out / "metadata" / "evil.json").write_text(
        json.dumps(
            {"Title": hostile, "Authors_List": [hostile], "Journal": hostile, "Year": 2020}
        ),
        encoding="utf-8",
    )

    path = write_report(out)
    html = path.read_text(encoding="utf-8")

    assert "<script>alert" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


# ---------------------------------------------------------------------------
# 6. Works model: ring summary, core-to-core edges, legacy detection
# ---------------------------------------------------------------------------
def test_report_data_includes_ring_summary(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _write_full_corpus(out)

    data = collect_report_data(out)

    # collect_report_data returns a JSON-normalized payload: keys are strings.
    assert data["ring_counts"] == {"0": 2, "1": 2}
    assert data["core_to_core_edges"] == [
        {"citing_id": "w-one-2020-alpha", "cited_id": "w-one-2020-beta"}
    ]
    html = build_report_html(data)
    assert "core" in html.lower()


def test_report_flags_legacy_out_dir(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "papers.csv").write_text("id,Title\np-x,Foo\n", encoding="utf-8")

    data = collect_report_data(out)

    assert any("legacy" in str(e.get("error", "")).lower() for e in data["errors"])
