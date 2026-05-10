"""Tests for the CitationGraph queryable view."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from citegraph.graph import CitationGraph
from citegraph.schemas import PipelineResult


@pytest.fixture()
def small_graph() -> CitationGraph:
    """A toy 3-papers / 4-references / 7-edges graph.

    Citation pattern (papers cite references):
      p-a -> r-x, r-y, r-z
      p-b -> r-x, r-y
      p-c -> r-x, r-w
    So r-x is most cited (3), r-y second (2), r-z and r-w tied (1).
    """
    papers = pd.DataFrame(
        [
            {"id": "p-a", "Title": "Paper A", "Year": 2020},
            {"id": "p-b", "Title": "Paper B", "Year": 2021},
            {"id": "p-c", "Title": "Paper C", "Year": 2022},
        ]
    )
    references = pd.DataFrame(
        [
            {"id": "r-x", "Title": "Ref X", "Year": 1990},
            {"id": "r-y", "Title": "Ref Y", "Year": 1995},
            {"id": "r-z", "Title": "Ref Z", "Year": 2000},
            {"id": "r-w", "Title": "Ref W", "Year": 2005},
        ]
    ).set_index("id")
    edges = pd.DataFrame(
        [
            {"citing_id": "p-a", "cited_id": "r-x"},
            {"citing_id": "p-a", "cited_id": "r-y"},
            {"citing_id": "p-a", "cited_id": "r-z"},
            {"citing_id": "p-b", "cited_id": "r-x"},
            {"citing_id": "p-b", "cited_id": "r-y"},
            {"citing_id": "p-c", "cited_id": "r-x"},
            {"citing_id": "p-c", "cited_id": "r-w"},
        ]
    )
    return CitationGraph(papers=papers, references=references, edges=edges)


# ---------------------------------------------------------------------------
# Counts and repr
# ---------------------------------------------------------------------------
def test_counts(small_graph: CitationGraph) -> None:
    assert small_graph.n_papers == 3
    assert small_graph.n_references == 4
    assert small_graph.n_edges == 7


def test_repr_summarises_shape(small_graph: CitationGraph) -> None:
    r = repr(small_graph)
    assert "3 papers" in r
    assert "4 references" in r
    assert "7 edges" in r


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
def test_cited_by_returns_correct_references(small_graph: CitationGraph) -> None:
    refs = small_graph.cited_by("p-a")
    assert set(refs.index) == {"r-x", "r-y", "r-z"}


def test_cited_by_unknown_paper_returns_empty(small_graph: CitationGraph) -> None:
    refs = small_graph.cited_by("p-does-not-exist")
    assert len(refs) == 0
    # Same column shape as the references table — safe to chain.
    assert list(refs.columns) == list(small_graph.references.columns)


def test_citers_of_returns_correct_papers(small_graph: CitationGraph) -> None:
    papers = small_graph.citers_of("r-x")
    assert set(papers["id"]) == {"p-a", "p-b", "p-c"}


def test_citers_of_unknown_reference_returns_empty(small_graph: CitationGraph) -> None:
    assert len(small_graph.citers_of("r-does-not-exist")) == 0


def test_top_cited_orders_by_count(small_graph: CitationGraph) -> None:
    top = small_graph.top_cited(n=3)
    assert list(top.index)[:1] == ["r-x"]  # 3 citations
    assert top.loc["r-x", "citation_count"] == 3
    assert top.loc["r-y", "citation_count"] == 2
    assert len(top) == 3


def test_top_cited_with_n_larger_than_corpus(small_graph: CitationGraph) -> None:
    top = small_graph.top_cited(n=100)
    assert len(top) == small_graph.n_references


def test_top_cited_on_empty_edges_returns_empty() -> None:
    g = CitationGraph(
        papers=pd.DataFrame(columns=["id", "Title"]),
        references=pd.DataFrame(columns=["Title"]).rename_axis("id"),
        edges=pd.DataFrame(columns=["citing_id", "cited_id"]),
    )
    top = g.top_cited(n=10)
    assert len(top) == 0
    assert "citation_count" in top.columns


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------
def test_from_pipeline_result_round_trip(small_graph: CitationGraph) -> None:
    result = PipelineResult(
        papers=small_graph.papers,
        references=small_graph.references,
        graph=small_graph.edges,
    )
    g = CitationGraph.from_pipeline_result(result)
    assert g.n_papers == small_graph.n_papers
    assert g.n_edges == small_graph.n_edges
    assert g.top_cited(n=1).index.tolist() == ["r-x"]


def test_from_out_dir_loads_csvs(tmp_path: Path, small_graph: CitationGraph) -> None:
    out = tmp_path / "out"
    out.mkdir()
    small_graph.papers.to_csv(out / "papers.csv", index=False)
    small_graph.references.to_csv(out / "references.csv")  # writes id as index column
    small_graph.edges.to_csv(out / "citation_graph.csv", index=False)

    g = CitationGraph.from_out_dir(out)
    assert g.n_papers == 3
    assert g.n_references == 4
    assert g.n_edges == 7
    # Round-trip top_cited produces the same ranking.
    assert g.top_cited(n=1).index.tolist() == ["r-x"]


def test_from_out_dir_missing_files_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing output"):
        CitationGraph.from_out_dir(tmp_path)


# ---------------------------------------------------------------------------
# NetworkX export
# ---------------------------------------------------------------------------
def test_to_networkx_builds_directed_graph(small_graph: CitationGraph) -> None:
    nx = pytest.importorskip("networkx")
    g = small_graph.to_networkx()
    assert isinstance(g, nx.DiGraph)
    assert g.number_of_nodes() == small_graph.n_papers + small_graph.n_references
    assert g.number_of_edges() == small_graph.n_edges
    # Node attributes preserve the kind tag.
    assert g.nodes["p-a"]["kind"] == "paper"
    assert g.nodes["r-x"]["kind"] == "reference"
    # Edge direction: citing -> cited.
    assert g.has_edge("p-a", "r-x")
    assert not g.has_edge("r-x", "p-a")
