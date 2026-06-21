"""End-to-end Pipeline test using a fake Gemini client (no network)."""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

from citegraph.llm import GeminiClient
from citegraph.pipeline import Pipeline, StageNotReadyError
from citegraph.schemas import PaperMetadata, Reference

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeResponse:
    def __init__(self, parsed):
        self.parsed = parsed
        self.text = ""


class _FakeClient(GeminiClient):
    """A GeminiClient that never hits the network.

    Returns a canned ``PaperMetadata`` for the metadata prompt and a canned
    list of references for the references prompt, distinguishing between
    the two by inspecting the prompt text.
    """

    def __init__(self):  # noqa: D401 - intentional override
        self._client = object()
        self.model = "fake-model"

        from citegraph.config import Settings

        self._settings = Settings(GOOGLE_API_KEY="fake")  # type: ignore[arg-type]
        self._api_key = "fake"

    def generate_structured(self, *, prompt, response_schema, system_instruction=None, max_output_tokens=None):
        if "references or bibliography section" in prompt:
            refs = [
                Reference(
                    Title="Governing the Commons",
                    Authors_List=["Elinor Ostrom"],
                    Authors="Ostrom, E.",
                    Journal="Cambridge University Press",
                    Year=1990,
                ),
                Reference(
                    Title="The tragedy of the commons",
                    Authors_List=["Garrett Hardin"],
                    Authors="Hardin, G.",
                    Journal="Science",
                    Year=1968,
                ),
            ]
            return _FakeResponse(parsed=refs)

        meta = PaperMetadata(
            Title="Governing Common-Pool Resources: A Reproducibility Note",
            Authors_List=["Jane Q. Doe", "John A. Smith"],
            Authors="Doe, J. Q.; Smith, J. A.",
            Journal="Journal of Reproducibility Studies",
            Year=2021,
        )
        return _FakeResponse(parsed=meta)


def test_pipeline_end_to_end_no_network(tmp_path: Path) -> None:
    md_dir = tmp_path / "out" / "markdown"
    md_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "sample_paper.md", md_dir / "sample_paper.md")

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    pipeline = Pipeline(
        pdf_dir=pdf_dir,
        out_dir=tmp_path / "out",
        client=_FakeClient(),
    )

    markdown_paths = list(md_dir.glob("*.md"))
    papers = pipeline.extract_paper_metadata(markdown_paths)
    raw_refs = pipeline.extract_paper_references(markdown_paths, papers)
    references, graph = pipeline.deduplicate(raw_refs)

    assert len(papers) == 1
    assert len(raw_refs) == 2
    assert len(references) == 2
    assert len(graph) == 2
    assert set(graph.columns) == {"citing_id", "cited_id"}

    assert (tmp_path / "out" / "papers.csv").exists()
    assert (tmp_path / "out" / "references_raw.csv").exists()
    assert (tmp_path / "out" / "references.csv").exists()
    assert (tmp_path / "out" / "citation_graph.csv").exists()


def test_pipeline_caches_metadata(tmp_path: Path) -> None:
    md_dir = tmp_path / "out" / "markdown"
    md_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "sample_paper.md", md_dir / "sample_paper.md")

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    pipeline = Pipeline(
        pdf_dir=pdf_dir,
        out_dir=tmp_path / "out",
        client=_FakeClient(),
    )

    markdown_paths = list(md_dir.glob("*.md"))
    pipeline.extract_paper_metadata(markdown_paths)
    cache = tmp_path / "out" / "metadata" / "sample_paper.json"
    assert cache.exists()

    reloaded = pd.read_csv(tmp_path / "out" / "papers.csv")
    assert reloaded.iloc[0]["Title"].startswith("Governing Common-Pool Resources")


def test_metadata_extraction_runs_concurrently_and_preserves_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    md_dir = tmp_path / "out" / "markdown"
    md_dir.mkdir(parents=True)
    markdown_paths = []
    for i in range(4):
        path = md_dir / f"paper_{i}.md"
        path.write_text(f"# Paper {i}\n")
        markdown_paths.append(path)

    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_extract(md_path: Path, *, client):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        idx = int(md_path.stem.rsplit("_", 1)[1])
        return PaperMetadata(
            Title=f"Paper {idx}",
            Authors_List=[f"Author {idx}"],
            Journal="J",
            Year=2020 + idx,
        )

    monkeypatch.setattr("citegraph.pipeline.extract_metadata_from_markdown", fake_extract)

    pipeline = Pipeline(
        pdf_dir=None,
        out_dir=tmp_path / "out",
        client=_FakeClient(),
        llm_concurrency=3,
        show_progress=False,
    )
    papers = pipeline.extract_paper_metadata(markdown_paths)

    assert max_active > 1
    assert papers["source_file"].tolist() == [p.name for p in markdown_paths]


def test_reference_extraction_runs_concurrently_and_preserves_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    md_dir = tmp_path / "out" / "markdown"
    md_dir.mkdir(parents=True)
    markdown_paths = []
    paper_rows = []
    for i in range(4):
        path = md_dir / f"paper_{i}.md"
        path.write_text(f"# Paper {i}\n## References\n1. Ref {i}\n")
        markdown_paths.append(path)
        paper_rows.append(
            {
                "id": f"p-paper-{i}",
                "Title": f"Paper {i}",
                "Authors_List": [f"Author {i}"],
                "Journal": "J",
                "Year": 2020 + i,
                "source_file": path.name,
            }
        )

    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_extract(md_path: Path, *, client):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        idx = int(md_path.stem.rsplit("_", 1)[1])
        return [
            Reference(
                Title=f"Ref {idx}",
                Authors_List=[f"Cited {idx}"],
                Journal="J",
                Year=2000 + idx,
            )
        ]

    monkeypatch.setattr("citegraph.pipeline.extract_references_from_markdown", fake_extract)

    pipeline = Pipeline(
        pdf_dir=None,
        out_dir=tmp_path / "out",
        client=_FakeClient(),
        llm_concurrency=3,
        show_progress=False,
    )
    refs = pipeline.extract_paper_references(
        markdown_paths, papers_df=pd.DataFrame(paper_rows)
    )

    assert max_active > 1
    assert refs["citing_id"].tolist() == [row["id"] for row in paper_rows]


def test_pipeline_rejects_non_positive_llm_concurrency(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="llm_concurrency"):
        Pipeline(
            pdf_dir=None,
            out_dir=tmp_path / "out",
            client=_FakeClient(),
            llm_concurrency=0,
        )


def test_progressive_stages_resume_from_disk(tmp_path: Path) -> None:
    """Each stage method, when called with no args, reads its inputs from out_dir."""
    md_dir = tmp_path / "out" / "markdown"
    md_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "sample_paper.md", md_dir / "sample_paper.md")

    # First Pipeline instance: stages 1-3, passing inputs explicitly.
    p1 = Pipeline(pdf_dir=None, out_dir=tmp_path / "out", client=_FakeClient())
    p1.extract_paper_metadata()
    p1.extract_paper_references()

    # Second Pipeline instance to prove dedup picks up references_raw.csv from disk.
    p2 = Pipeline(pdf_dir=None, out_dir=tmp_path / "out", client=_FakeClient())
    refs, graph = p2.deduplicate()

    assert len(refs) == 2
    assert len(graph) == 2
    assert (tmp_path / "out" / "papers.csv").exists()
    assert (tmp_path / "out" / "references_raw.csv").exists()


def test_normalize_authors_writes_csvs_and_review(tmp_path: Path) -> None:
    """End-to-end: pipeline.normalize_authors() reads dedup output and writes author tables."""
    md_dir = tmp_path / "out" / "markdown"
    md_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "sample_paper.md", md_dir / "sample_paper.md")

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    pipeline = Pipeline(pdf_dir=pdf_dir, out_dir=tmp_path / "out", client=_FakeClient())
    markdown_paths = list(md_dir.glob("*.md"))
    papers = pipeline.extract_paper_metadata(markdown_paths)
    raw_refs = pipeline.extract_paper_references(markdown_paths, papers)
    pipeline.deduplicate(raw_refs)

    authors_df, citations_df = pipeline.normalize_authors()
    # The fake client returns 1 paper (2 authors) + 2 references (1 author each).
    assert (tmp_path / "out" / "authors.csv").exists()
    assert (tmp_path / "out" / "author_citations.csv").exists()
    assert len(authors_df) >= 1
    assert {"reference", "paper"}.issubset(set(citations_df["record_kind"]))


def test_stage_not_ready_when_upstream_missing(tmp_path: Path) -> None:
    p = Pipeline(pdf_dir=None, out_dir=tmp_path / "out", client=_FakeClient())

    # Nothing on disk: dedup needs references_raw.csv, metadata needs markdown.
    with pytest.raises(StageNotReadyError, match="references"):
        p.deduplicate()
    with pytest.raises(StageNotReadyError, match="markdown"):
        p.extract_paper_metadata()
    with pytest.raises(StageNotReadyError, match="pdf_dir"):
        p.convert_pdfs()

    # With markdown but no papers.csv, the references stage trips on metadata.
    shutil.copy(FIXTURES / "sample_paper.md", p.layout.markdown_dir / "sample_paper.md")
    with pytest.raises(StageNotReadyError, match="metadata"):
        p.extract_paper_references()
