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
# Author queries
# ---------------------------------------------------------------------------
def _graph_with_authors() -> CitationGraph:
    """Toy graph with two canonical authors and three reference citations."""
    papers = pd.DataFrame(
        [
            {"id": "p-a", "Title": "Paper A", "Journal": "Ecological Economics", "Year": 2020},
            {"id": "p-b", "Title": "Paper B", "Journal": "Ecological Economics", "Year": 2021},
            {"id": "p-c", "Title": "Paper C", "Journal": "World Development", "Year": 2022},
        ]
    )
    references = pd.DataFrame(
        [
            {"id": "r-1", "Title": "Ref 1", "Journal": "World Development", "Year": 1990},
            {"id": "r-2", "Title": "Ref 2", "Journal": "Ecological Economics", "Year": 1995},
            {"id": "r-3", "Title": "Ref 3", "Journal": "Child Development", "Year": 2000},
        ]
    ).set_index("id")
    edges = pd.DataFrame(
        [
            {"citing_id": "p-a", "cited_id": "r-1"},
            {"citing_id": "p-a", "cited_id": "r-2"},
            {"citing_id": "p-b", "cited_id": "r-1"},
            {"citing_id": "p-c", "cited_id": "r-1"},
            {"citing_id": "p-c", "cited_id": "r-3"},
        ]
    )
    authors = pd.DataFrame(
        [
            {
                "id": "a-cardenas-juan-camilo",
                "display_name": "Juan-Camilo Cárdenas",
                "surname": "Cárdenas",
                "surname_norm": "cardenas",
                "canonical_given": "Juan-Camilo",
                "initials": "JC",
                "openalex_id": None,
                "orcid": None,
                "n_occurrences": 2,
                "n_reference_citations": 2,
                "n_distinct_papers_citing": 1,
            },
            {
                "id": "a-diamond-adele",
                "display_name": "Adele Diamond",
                "surname": "Diamond",
                "surname_norm": "diamond",
                "canonical_given": "Adele",
                "initials": "A",
                "openalex_id": None,
                "orcid": None,
                "n_occurrences": 1,
                "n_reference_citations": 1,
                "n_distinct_papers_citing": 1,
            },
        ]
    ).set_index("id")
    citations = pd.DataFrame(
        [
            {"author_id": "a-cardenas-juan-camilo", "record_kind": "reference",
             "record_id": "r-1", "position": 0, "citing_paper_id": "",
             "raw_author": "Cárdenas, J.-C."},
            {"author_id": "a-cardenas-juan-camilo", "record_kind": "reference",
             "record_id": "r-2", "position": 0, "citing_paper_id": "",
             "raw_author": "Cárdenas, Juan-Camilo"},
            {"author_id": "a-diamond-adele", "record_kind": "reference",
             "record_id": "r-3", "position": 0, "citing_paper_id": "",
             "raw_author": "Diamond, Adele"},
        ]
    )
    return CitationGraph(
        papers=papers, references=references, edges=edges,
        authors=authors, author_citations=citations,
    )


def test_has_authors_flag() -> None:
    g_with = _graph_with_authors()
    assert g_with.has_authors is True


def test_top_cited_authors_orders_by_count() -> None:
    g = _graph_with_authors()
    top = g.top_cited_authors(n=10)
    assert list(top.index)[0] == "a-cardenas-juan-camilo"
    assert int(top.iloc[0]["n_reference_citations"]) == 2


def test_find_author_diacritic_insensitive() -> None:
    g = _graph_with_authors()
    hits_with_accent = g.find_author("Cárdenas")
    hits_without = g.find_author("cardenas")
    hits_partial = g.find_author("card")
    assert list(hits_with_accent.index) == ["a-cardenas-juan-camilo"]
    assert list(hits_without.index) == ["a-cardenas-juan-camilo"]
    assert list(hits_partial.index) == ["a-cardenas-juan-camilo"]


def test_citations_of_returns_referenced_works() -> None:
    g = _graph_with_authors()
    refs = g.citations_of("a-cardenas-juan-camilo")
    assert set(refs.index) == {"r-1", "r-2"}
    assert refs.loc["r-1", "citing_paper_ids"] == ["p-a", "p-b", "p-c"]
    assert refs.loc["r-2", "citing_paper_ids"] == ["p-a"]


def test_papers_citing_author_returns_source_papers() -> None:
    g = _graph_with_authors()
    papers = g.papers_citing_author("a-diamond-adele")
    assert set(papers["id"]) == {"p-c"}


def test_citation_context_for_author_joins_source_papers_to_cited_references() -> None:
    g = _graph_with_authors()

    context = g.citation_context_for_author("a-cardenas-juan-camilo")

    assert len(context) == 4
    assert set(context["citing_paper_id"]) == {"p-a", "p-b", "p-c"}
    assert set(context["cited_reference_id"]) == {"r-1", "r-2"}
    row = context[
        (context["citing_paper_id"] == "p-a")
        & (context["cited_reference_id"] == "r-2")
    ].iloc[0]
    assert row["source_paper_journal"] == "Ecological Economics"
    assert row["cited_reference_title"] == "Ref 2"
    assert row["raw_author"] == "Cárdenas, Juan-Camilo"


def test_citing_papers_by_author_counts_distinct_source_papers() -> None:
    g = _graph_with_authors()

    papers = g.citing_papers_by_author("a-cardenas-juan-camilo")

    assert list(papers["paper_id"]) == ["p-a", "p-b", "p-c"]
    assert papers.loc[papers["paper_id"] == "p-a", "n_cited_references_by_author"].iloc[0] == 2
    assert papers.loc[papers["paper_id"] == "p-a", "cited_reference_ids"].iloc[0] == ["r-1", "r-2"]
    assert papers.loc[papers["paper_id"] == "p-b", "n_cited_references_by_author"].iloc[0] == 1


def test_source_journals_citing_author_counts_source_paper_journals() -> None:
    g = _graph_with_authors()

    journals = g.source_journals_citing_author("a-cardenas-juan-camilo")

    assert list(journals["source_paper_journal"]) == ["Ecological Economics", "World Development"]
    assert list(journals["n_papers"]) == [2, 1]
    assert list(journals["share_of_papers"].round(3)) == [0.667, 0.333]


def test_top_cited_authors_raises_without_authors_loaded(small_graph: CitationGraph) -> None:
    with pytest.raises(RuntimeError, match="Author tables"):
        small_graph.top_cited_authors()


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
