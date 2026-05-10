"""One bad paper must not kill the rest of the run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from citegraph.io import read_json
from citegraph.llm import GeminiClient
from citegraph.pipeline import Pipeline
from citegraph.schemas import PaperMetadata, Reference

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _FailFirstClient(GeminiClient):
    """Fake client that raises for the first paper it sees, succeeds for the rest.

    The "raises on which paper" decision is keyed by a substring of the prompt so
    we can scope the failure to either metadata or references stages. Any prompt
    containing ``fail_marker`` will raise; everything else returns a canned good
    response. Caches `seen_prompts` for assertions.
    """

    def __init__(self, fail_marker: str):
        self._client = object()
        self.model = "fake-model"
        from citegraph.config import Settings

        self._settings = Settings(GOOGLE_API_KEY="fake")  # type: ignore[arg-type]
        self._api_key = "fake"
        self._fail_marker = fail_marker
        self.seen_prompts: list[str] = []

    @property
    def default_max_output_tokens(self) -> int:
        return 40_000

    def generate_structured(
        self, *, prompt, response_schema, system_instruction=None, max_output_tokens=None
    ):
        self.seen_prompts.append(prompt)
        if self._fail_marker in prompt:
            raise RuntimeError("simulated Gemini failure")

        if "references or bibliography section" in prompt:
            refs = [
                Reference(
                    Title="A Cited Work",
                    Authors_List=["X. Author"],
                    Journal="J",
                    Year=2020,
                )
            ]
            return SimpleNamespace(parsed=refs, text="", candidates=[])

        meta = PaperMetadata(
            Title="A Successful Paper",
            Authors_List=["Y. Author"],
            Journal="J",
            Year=2021,
        )
        return SimpleNamespace(parsed=meta, text="", candidates=[])


def _two_paper_setup(tmp_path: Path) -> Path:
    """Lay out two markdown files under ``out/markdown/``: a 'good' and a 'doomed' one."""
    md_dir = tmp_path / "out" / "markdown"
    md_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "sample_paper.md", md_dir / "good.md")
    # The "doomed" paper has the failure marker in BOTH body and references
    # section, so the metadata stage (sees first 12k chars of body) and the
    # references stage (sees only the bibliography section) both trip on it.
    (md_dir / "doomed.md").write_text(
        "## Body\nThis paper contains the marker DOOM_TOKEN.\n"
        "## References\n1. DOOM_TOKEN reference.\n"
    )
    return md_dir


# ---------------------------------------------------------------------------
# Metadata stage
# ---------------------------------------------------------------------------
def test_metadata_failure_does_not_block_other_papers(tmp_path: Path) -> None:
    _two_paper_setup(tmp_path)
    pipeline = Pipeline(
        pdf_dir=None, out_dir=tmp_path / "out", client=_FailFirstClient("DOOM_TOKEN")
    )

    df = pipeline.extract_paper_metadata()

    assert len(df) == 1, "the good paper should be in papers.csv"
    assert df.iloc[0]["Title"] == "A Successful Paper"

    failures_path = pipeline.layout.metadata_failures_jsonl
    assert failures_path.exists()
    failures = [json.loads(line) for line in failures_path.read_text().splitlines() if line]
    assert len(failures) == 1
    assert failures[0]["source_file"] == "doomed.md"
    assert failures[0]["stage"] == "metadata"
    assert failures[0]["error_class"] == "RuntimeError"
    assert "simulated" in failures[0]["error_message"]


def test_failed_metadata_is_not_cached_so_rerun_retries(tmp_path: Path) -> None:
    _two_paper_setup(tmp_path)
    pipeline = Pipeline(
        pdf_dir=None, out_dir=tmp_path / "out", client=_FailFirstClient("DOOM_TOKEN")
    )
    pipeline.extract_paper_metadata()

    cache_dir = tmp_path / "out" / "metadata"
    assert (cache_dir / "good.json").exists()
    assert not (cache_dir / "doomed.json").exists(), \
        "failed paper must not have a cache file (otherwise reruns would skip it)"


def test_metadata_failures_file_removed_when_no_failures(tmp_path: Path) -> None:
    """A successful re-run should clear a stale failures file."""
    _two_paper_setup(tmp_path)
    out_dir = tmp_path / "out"

    # First run: doomed paper fails.
    Pipeline(pdf_dir=None, out_dir=out_dir, client=_FailFirstClient("DOOM_TOKEN")).extract_paper_metadata()
    failures_path = out_dir / "metadata_failures.jsonl"
    assert failures_path.exists()

    # Second run with a client that fails nothing: the failures file should be removed.
    Pipeline(pdf_dir=None, out_dir=out_dir, client=_FailFirstClient("UNUSED_MARKER")).extract_paper_metadata()
    assert not failures_path.exists(), "stale failures file must be cleared on a clean run"


# ---------------------------------------------------------------------------
# References stage
# ---------------------------------------------------------------------------
def test_references_failure_does_not_block_other_papers(tmp_path: Path) -> None:
    _two_paper_setup(tmp_path)
    out_dir = tmp_path / "out"

    # Pre-populate metadata for both papers so we can exercise the references
    # stage in isolation without the failure first hitting metadata extraction.
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("good", "doomed"):
        (meta_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "Title": f"{stem.title()} Paper",
                    "Authors_List": [f"{stem} author"],
                    "Journal": "J",
                    "Year": 2020,
                }
            )
        )

    p = Pipeline(
        pdf_dir=None, out_dir=out_dir, client=_FailFirstClient("DOOM_TOKEN")
    )
    papers = p.extract_paper_metadata()  # both succeed (no LLM call — cached)
    assert len(papers) == 2

    raw_refs = p.extract_paper_references(papers_df=papers)
    assert len(raw_refs) == 1, "only the good paper's refs should appear"

    failures_path = p.layout.references_failures_jsonl
    assert failures_path.exists()
    failures = [json.loads(line) for line in failures_path.read_text().splitlines() if line]
    assert len(failures) == 1
    assert failures[0]["stage"] == "references"
    assert failures[0]["source_file"] == "doomed.md"

    # And: the failed references cache must NOT exist.
    assert not (out_dir / "references" / "doomed.json").exists()


# ---------------------------------------------------------------------------
# run_summary surfaces failure counts
# ---------------------------------------------------------------------------
def test_run_summary_includes_failure_counts(tmp_path: Path) -> None:
    _two_paper_setup(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = Pipeline(
        pdf_dir=tmp_path / "no-pdfs-needed",
        out_dir=out_dir,
        client=_FailFirstClient("DOOM_TOKEN"),
    )
    (tmp_path / "no-pdfs-needed").mkdir()

    # Skip stage 1 by pre-populating markdown (already done by _two_paper_setup).
    papers = pipeline.extract_paper_metadata()
    raw_refs = pipeline.extract_paper_references(papers_df=papers)
    pipeline.deduplicate(raw_refs)

    # run() also writes the summary; we mimic the same call here.
    # We don't call run() directly because it would re-attempt convert_pdfs.
    # Instead, write the summary the same way run() does.
    from citegraph.io import write_json
    from citegraph.pipeline import _count_failures

    write_json(
        out_dir / "run_summary.json",
        {
            "n_metadata_failures": _count_failures(pipeline.layout.metadata_failures_jsonl),
            "n_references_failures": _count_failures(pipeline.layout.references_failures_jsonl),
        },
    )

    summary = read_json(out_dir / "run_summary.json")
    assert summary["n_metadata_failures"] == 1
    # The doomed paper failed metadata so its references stage was never tried —
    # source_to_id has no entry for it and the loop `continue`s without raising.
    assert summary["n_references_failures"] == 0


@pytest.mark.parametrize("path_attr", ["metadata_failures_jsonl", "references_failures_jsonl"])
def test_count_failures_handles_missing_file(tmp_path: Path, path_attr: str) -> None:
    from citegraph.pipeline import _count_failures

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    p = Pipeline(pdf_dir=None, out_dir=out_dir, client=_FailFirstClient("X"))
    assert _count_failures(getattr(p.layout, path_attr)) == 0
