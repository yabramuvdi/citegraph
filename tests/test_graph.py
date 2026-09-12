"""Tests for the CitationGraph queryable view."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from citegraph.graph import CitationGraph
from citegraph.schemas import PipelineResult


@pytest.fixture()
def small_graph() -> CitationGraph:
    """A toy 3-core-works / 4-stubs / 7-edges graph.

    Citation pattern (core works cite stubs):
      w-a -> w-x, w-y, w-z
      w-b -> w-x, w-y
      w-c -> w-x, w-w
    So w-x is most cited (3), w-y second (2), w-z and w-w tied (1).
    """
    works = pd.DataFrame(
        [
            {"id": "w-a", "ring": 0, "source_file": "a.pdf", "Title": "Paper A", "Year": 2020},
            {"id": "w-b", "ring": 0, "source_file": "b.pdf", "Title": "Paper B", "Year": 2021},
            {"id": "w-c", "ring": 0, "source_file": "c.pdf", "Title": "Paper C", "Year": 2022},
            {"id": "w-x", "ring": 1, "source_file": "", "Title": "Ref X", "Year": 1990},
            {"id": "w-y", "ring": 1, "source_file": "", "Title": "Ref Y", "Year": 1995},
            {"id": "w-z", "ring": 1, "source_file": "", "Title": "Ref Z", "Year": 2000},
            {"id": "w-w", "ring": 1, "source_file": "", "Title": "Ref W", "Year": 2005},
        ]
    ).set_index("id")
    edges = pd.DataFrame(
        [
            {"citing_id": "w-a", "cited_id": "w-x"},
            {"citing_id": "w-a", "cited_id": "w-y"},
            {"citing_id": "w-a", "cited_id": "w-z"},
            {"citing_id": "w-b", "cited_id": "w-x"},
            {"citing_id": "w-b", "cited_id": "w-y"},
            {"citing_id": "w-c", "cited_id": "w-x"},
            {"citing_id": "w-c", "cited_id": "w-w"},
        ]
    )
    return CitationGraph(works=works, edges=edges)


# ---------------------------------------------------------------------------
# Counts and repr
# ---------------------------------------------------------------------------
def test_counts(small_graph: CitationGraph) -> None:
    assert small_graph.n_works == 7
    assert small_graph.n_core_works == 3
    assert small_graph.n_edges == 7


def test_repr_summarises_shape(small_graph: CitationGraph) -> None:
    r = repr(small_graph)
    assert "3 core works" in r
    assert "7 works" in r
    assert "7 edges" in r


# ---------------------------------------------------------------------------
# Ring views
# ---------------------------------------------------------------------------
def test_core_and_ring_views(small_graph: CitationGraph) -> None:
    assert set(small_graph.core.index) == {"w-a", "w-b", "w-c"}
    assert set(small_graph.ring(1).index) == {"w-x", "w-y", "w-z", "w-w"}
    assert len(small_graph.ring(2)) == 0


def test_core_citations_returns_core_to_core_edges_only(small_graph: CitationGraph) -> None:
    # No core work cites another core work in this fixture.
    assert small_graph.core_citations().empty

    # Add one core->core edge and it shows up alone.
    edges = pd.concat(
        [small_graph.edges, pd.DataFrame([{"citing_id": "w-a", "cited_id": "w-b"}])],
        ignore_index=True,
    )
    g = CitationGraph(works=small_graph.works, edges=edges)
    cc = g.core_citations()
    assert list(cc.itertuples(index=False)) == [("w-a", "w-b")]


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
def test_cited_by_returns_correct_works(small_graph: CitationGraph) -> None:
    refs = small_graph.cited_by("w-a")
    assert set(refs.index) == {"w-x", "w-y", "w-z"}


def test_cited_by_unknown_work_returns_empty(small_graph: CitationGraph) -> None:
    refs = small_graph.cited_by("w-does-not-exist")
    assert len(refs) == 0
    # Same column shape as the works table — safe to chain.
    assert list(refs.columns) == list(small_graph.works.columns)


def test_citers_of_returns_correct_works(small_graph: CitationGraph) -> None:
    citers = small_graph.citers_of("w-x")
    assert set(citers.index) == {"w-a", "w-b", "w-c"}


def test_citers_of_unknown_work_returns_empty(small_graph: CitationGraph) -> None:
    assert len(small_graph.citers_of("w-does-not-exist")) == 0


def test_top_cited_orders_by_count(small_graph: CitationGraph) -> None:
    top = small_graph.top_cited(n=3)
    assert list(top.index)[:1] == ["w-x"]  # 3 citations
    assert top.loc["w-x", "citation_count"] == 3
    assert top.loc["w-y", "citation_count"] == 2
    assert len(top) == 3


def test_top_cited_includes_core_works() -> None:
    works = pd.DataFrame(
        [
            {"id": "w-a", "ring": 0, "source_file": "a.pdf", "Title": "A", "Year": 2020},
            {"id": "w-b", "ring": 0, "source_file": "b.pdf", "Title": "B", "Year": 2021},
            {"id": "w-c", "ring": 1, "source_file": "", "Title": "C", "Year": 1990},
        ]
    ).set_index("id")
    edges = pd.DataFrame(
        [
            {"citing_id": "w-a", "cited_id": "w-c"},
            {"citing_id": "w-b", "cited_id": "w-c"},
            {"citing_id": "w-a", "cited_id": "w-b"},  # core paper cited in-corpus
        ]
    )
    g = CitationGraph(works=works, edges=edges)
    top = g.top_cited(5)
    assert "w-b" in top.index
    assert top.loc["w-c", "citation_count"] == 2
    assert top.loc["w-b", "citation_count"] == 1


def test_top_cited_with_n_larger_than_corpus(small_graph: CitationGraph) -> None:
    top = small_graph.top_cited(n=100)
    # Only works with at least one citation appear.
    assert set(top.index) == {"w-x", "w-y", "w-z", "w-w"}


def test_top_cited_on_empty_edges_returns_empty() -> None:
    g = CitationGraph(
        works=pd.DataFrame(columns=["ring", "source_file", "Title"]).rename_axis("id"),
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
        works=small_graph.works,
        graph=small_graph.edges,
    )
    g = CitationGraph.from_pipeline_result(result)
    assert g.n_works == small_graph.n_works
    assert g.n_core_works == small_graph.n_core_works
    assert g.n_edges == small_graph.n_edges
    assert g.top_cited(n=1).index.tolist() == ["w-x"]


def test_from_pipeline_result_carries_author_tables() -> None:
    source = _graph_with_authors()
    result = PipelineResult(
        works=source.works,
        graph=source.edges,
        authors=source.authors,
        author_citations=source.author_citations,
    )

    g = CitationGraph.from_pipeline_result(result)

    assert g.has_authors is True
    assert g.top_cited_authors(n=1).index.tolist() == ["a-cardenas-juan-camilo"]


def test_empty_author_tables_are_still_loaded(small_graph: CitationGraph) -> None:
    from citegraph.io import AUTHOR_CITATION_COLUMNS, AUTHOR_COLUMNS

    graph = CitationGraph(
        works=small_graph.works,
        edges=small_graph.edges,
        authors=pd.DataFrame(columns=AUTHOR_COLUMNS).set_index("id"),
        author_citations=pd.DataFrame(columns=AUTHOR_CITATION_COLUMNS),
    )

    assert graph.has_authors is True
    assert graph.top_authors().empty
    assert graph.top_cited_authors().empty
    assert graph.find_author("smith").empty


def test_from_out_dir_loads_csvs(tmp_path: Path, small_graph: CitationGraph) -> None:
    out = tmp_path / "out"
    out.mkdir()
    small_graph.works.to_csv(out / "works.csv")  # writes id as index column
    small_graph.edges.to_csv(out / "citation_graph.csv", index=False)

    g = CitationGraph.from_out_dir(out)
    assert g.n_works == 7
    assert g.n_core_works == 3
    assert g.n_edges == 7
    # Round-trip top_cited produces the same ranking.
    assert g.top_cited(n=1).index.tolist() == ["w-x"]


def test_from_out_dir_missing_files_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing output"):
        CitationGraph.from_out_dir(tmp_path)


def test_from_out_dir_legacy_layout_raises_with_migration_hint(tmp_path: Path) -> None:
    pd.DataFrame({"id": ["p-x"]}).to_csv(tmp_path / "papers.csv", index=False)
    with pytest.raises(FileNotFoundError, match="legacy"):
        CitationGraph.from_out_dir(tmp_path)


# ---------------------------------------------------------------------------
# Author queries
# ---------------------------------------------------------------------------
def _graph_with_authors() -> CitationGraph:
    """Toy graph with two canonical authors across core works and stubs."""
    works = pd.DataFrame(
        [
            {"id": "w-a", "ring": 0, "source_file": "a.pdf",
             "Title": "Paper A", "Journal": "Ecological Economics", "Year": 2020},
            {"id": "w-b", "ring": 0, "source_file": "b.pdf",
             "Title": "Paper B", "Journal": "Ecological Economics", "Year": 2021},
            {"id": "w-c", "ring": 0, "source_file": "c.pdf",
             "Title": "Paper C", "Journal": "World Development", "Year": 2022},
            {"id": "w-1", "ring": 1, "source_file": "",
             "Title": "Ref 1", "Journal": "World Development", "Year": 1990},
            {"id": "w-2", "ring": 1, "source_file": "",
             "Title": "Ref 2", "Journal": "Ecological Economics", "Year": 1995},
            {"id": "w-3", "ring": 1, "source_file": "",
             "Title": "Ref 3", "Journal": "Child Development", "Year": 2000},
        ]
    ).set_index("id")
    edges = pd.DataFrame(
        [
            {"citing_id": "w-a", "cited_id": "w-1"},
            {"citing_id": "w-a", "cited_id": "w-2"},
            {"citing_id": "w-b", "cited_id": "w-1"},
            {"citing_id": "w-c", "cited_id": "w-1"},
            {"citing_id": "w-c", "cited_id": "w-3"},
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
                "n_works": 3,
                "n_core_works": 1,
                "n_citations_received": 4,
                "n_distinct_citing_works": 3,
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
                "n_works": 1,
                "n_core_works": 0,
                "n_citations_received": 1,
                "n_distinct_citing_works": 1,
            },
        ]
    ).set_index("id")
    citations = pd.DataFrame(
        [
            # Cárdenas authored core work w-a and is cited via w-1 and w-2.
            {"author_id": "a-cardenas-juan-camilo",
             "record_id": "w-a", "position": 0, "raw_author": "Cárdenas, Juan-Camilo"},
            {"author_id": "a-cardenas-juan-camilo",
             "record_id": "w-1", "position": 0, "raw_author": "Cárdenas, J.-C."},
            {"author_id": "a-cardenas-juan-camilo",
             "record_id": "w-2", "position": 0, "raw_author": "Cárdenas, Juan-Camilo"},
            {"author_id": "a-diamond-adele",
             "record_id": "w-3", "position": 0, "raw_author": "Diamond, Adele"},
        ]
    )
    return CitationGraph(
        works=works, edges=edges,
        authors=authors, author_citations=citations,
    )


def test_has_authors_flag() -> None:
    g_with = _graph_with_authors()
    assert g_with.has_authors is True


def test_top_cited_authors_orders_by_citations_received() -> None:
    g = _graph_with_authors()
    top = g.top_cited_authors(n=10)
    assert list(top.index)[0] == "a-cardenas-juan-camilo"
    assert int(top.iloc[0]["n_citations_received"]) == 4


def test_top_authors_ranks_by_works_and_filters_by_ring() -> None:
    g = _graph_with_authors()

    top_all = g.top_authors(n=10)
    assert list(top_all.index)[0] == "a-cardenas-juan-camilo"
    assert int(top_all.iloc[0]["n_works_in_selection"]) == 3

    top_core = g.top_authors(n=10, ring=0)
    # Only Cárdenas authored a core work (w-a).
    assert list(top_core.index) == ["a-cardenas-juan-camilo"]
    assert int(top_core.iloc[0]["n_works_in_selection"]) == 1


def test_top_authors_raises_without_author_tables(small_graph: CitationGraph) -> None:
    with pytest.raises(RuntimeError, match="Author tables"):
        small_graph.top_authors(ring=0)


def test_find_author_diacritic_insensitive() -> None:
    g = _graph_with_authors()
    hits_with_accent = g.find_author("Cárdenas")
    hits_without = g.find_author("cardenas")
    hits_partial = g.find_author("card")
    assert list(hits_with_accent.index) == ["a-cardenas-juan-camilo"]
    assert list(hits_without.index) == ["a-cardenas-juan-camilo"]
    assert list(hits_partial.index) == ["a-cardenas-juan-camilo"]


def test_citations_of_returns_cited_works() -> None:
    g = _graph_with_authors()
    refs = g.citations_of("a-cardenas-juan-camilo")
    assert set(refs.index) == {"w-1", "w-2"}
    assert refs.loc["w-1", "citing_paper_ids"] == ["w-a", "w-b", "w-c"]
    assert refs.loc["w-2", "citing_paper_ids"] == ["w-a"]


def test_papers_citing_author_returns_citing_works() -> None:
    g = _graph_with_authors()
    papers = g.papers_citing_author("a-diamond-adele")
    assert set(papers.index) == {"w-c"}


def test_citation_context_for_author_joins_citing_works_to_cited_works() -> None:
    g = _graph_with_authors()

    context = g.citation_context_for_author("a-cardenas-juan-camilo")

    assert len(context) == 4
    assert set(context["citing_paper_id"]) == {"w-a", "w-b", "w-c"}
    assert set(context["cited_reference_id"]) == {"w-1", "w-2"}
    row = context[
        (context["citing_paper_id"] == "w-a")
        & (context["cited_reference_id"] == "w-2")
    ].iloc[0]
    assert row["source_paper_journal"] == "Ecological Economics"
    assert row["cited_reference_title"] == "Ref 2"
    assert row["raw_author"] == "Cárdenas, Juan-Camilo"


def test_citing_papers_by_author_counts_distinct_citing_works() -> None:
    g = _graph_with_authors()

    papers = g.citing_papers_by_author("a-cardenas-juan-camilo")

    assert list(papers["paper_id"]) == ["w-a", "w-b", "w-c"]
    assert papers.loc[papers["paper_id"] == "w-a", "n_cited_references_by_author"].iloc[0] == 2
    assert papers.loc[papers["paper_id"] == "w-a", "cited_reference_ids"].iloc[0] == ["w-1", "w-2"]
    assert papers.loc[papers["paper_id"] == "w-b", "n_cited_references_by_author"].iloc[0] == 1


def test_source_journals_citing_author_counts_citing_work_journals() -> None:
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
    assert g.number_of_nodes() == small_graph.n_works
    assert g.number_of_edges() == small_graph.n_edges
    # Node attributes carry the works-model facts.
    assert g.nodes["w-a"]["ring"] == 0
    assert g.nodes["w-x"]["ring"] == 1
    assert g.nodes["w-a"]["source_file"] == "a.pdf"
    # Edge direction: citing -> cited.
    assert g.has_edge("w-a", "w-x")
    assert not g.has_edge("w-x", "w-a")


# ---------------------------------------------------------------------------
# Author co-citation network
# ---------------------------------------------------------------------------
def _cocitation_graph() -> CitationGraph:
    """Toy corpus whose co-citation weights, pruning and self-pairs differ.

    Three core papers cite four stubs written by three canonical authors::

      p1 -> r1 (A), r2 (B), r4 (A & B)
      p2 -> r1 (A), r2 (B)
      p3 -> r1 (A), r3 (C)

    So A and B are co-cited by two papers, A and C by one, B and C by
    none. Papers citing each author: A three, B two, C one. Stub ``r4``
    is co-authored, so p1 cites two A-works and must still yield one
    A-B edge and no A-A self-loop.
    """
    works = pd.DataFrame(
        [
            {"id": "p1", "ring": 0, "source_file": "p1.pdf", "Title": "Core 1", "Year": 2020},
            {"id": "p2", "ring": 0, "source_file": "p2.pdf", "Title": "Core 2", "Year": 2021},
            {"id": "p3", "ring": 0, "source_file": "p3.pdf", "Title": "Core 3", "Year": 2022},
            {"id": "r1", "ring": 1, "source_file": "", "Title": "Ref 1", "Year": 1990},
            {"id": "r2", "ring": 1, "source_file": "", "Title": "Ref 2", "Year": 1995},
            {"id": "r3", "ring": 1, "source_file": "", "Title": "Ref 3", "Year": 2000},
            {"id": "r4", "ring": 1, "source_file": "", "Title": "Ref 4", "Year": 2005},
        ]
    ).set_index("id")
    edges = pd.DataFrame(
        [
            {"citing_id": "p1", "cited_id": "r1"},
            {"citing_id": "p1", "cited_id": "r2"},
            {"citing_id": "p1", "cited_id": "r4"},
            {"citing_id": "p2", "cited_id": "r1"},
            {"citing_id": "p2", "cited_id": "r2"},
            {"citing_id": "p3", "cited_id": "r1"},
            {"citing_id": "p3", "cited_id": "r3"},
        ]
    )
    authors = pd.DataFrame(
        [
            {"id": "a-alpha", "display_name": "Ada Alpha", "surname": "Alpha",
             "surname_norm": "alpha", "canonical_given": "Ada", "initials": "A",
             "openalex_id": None, "orcid": None, "n_works": 2,
             "n_core_works": 0, "n_citations_received": 4, "n_distinct_citing_works": 3},
            {"id": "a-beta", "display_name": "Ben Beta", "surname": "Beta",
             "surname_norm": "beta", "canonical_given": "Ben", "initials": "B",
             "openalex_id": None, "orcid": None, "n_works": 2,
             "n_core_works": 0, "n_citations_received": 3, "n_distinct_citing_works": 2},
            {"id": "a-gamma", "display_name": "Cleo Gamma", "surname": "Gamma",
             "surname_norm": "gamma", "canonical_given": "Cleo", "initials": "C",
             "openalex_id": None, "orcid": None, "n_works": 1,
             "n_core_works": 0, "n_citations_received": 1, "n_distinct_citing_works": 1},
        ]
    ).set_index("id")
    citations = pd.DataFrame(
        [
            {"author_id": "a-alpha", "record_id": "r1", "position": 0, "raw_author": "Alpha, A"},
            {"author_id": "a-beta", "record_id": "r2", "position": 0, "raw_author": "Beta, B"},
            {"author_id": "a-gamma", "record_id": "r3", "position": 0, "raw_author": "Gamma, C"},
            {"author_id": "a-alpha", "record_id": "r4", "position": 0, "raw_author": "Alpha, A"},
            {"author_id": "a-beta", "record_id": "r4", "position": 1, "raw_author": "Beta, B"},
        ]
    )
    return CitationGraph(
        works=works, edges=edges, authors=authors, author_citations=citations
    )


def test_author_cocitation_network_weights_edges_by_shared_citing_papers() -> None:
    nx = pytest.importorskip("networkx")
    g = _cocitation_graph().author_cocitation_network(min_papers=1)

    assert isinstance(g, nx.Graph)
    assert g["a-alpha"]["a-beta"]["weight"] == 2
    assert g["a-alpha"]["a-gamma"]["weight"] == 1
    assert not g.has_edge("a-beta", "a-gamma")


def test_author_cocitation_network_has_no_self_loops() -> None:
    pytest.importorskip("networkx")
    g = _cocitation_graph().author_cocitation_network(min_papers=1)

    # p1 cites two works by Alpha (r1 and r4); that is not a co-citation.
    assert not g.has_edge("a-alpha", "a-alpha")


def test_author_cocitation_network_prunes_authors_below_min_papers() -> None:
    pytest.importorskip("networkx")
    g = _cocitation_graph().author_cocitation_network(min_papers=2)

    # Gamma is cited by one paper only, so both the node and its edge go.
    assert set(g.nodes) == {"a-alpha", "a-beta"}
    assert g.number_of_edges() == 1


def test_author_cocitation_network_carries_display_name_and_paper_count() -> None:
    pytest.importorskip("networkx")
    g = _cocitation_graph().author_cocitation_network(min_papers=1)

    assert g.nodes["a-alpha"]["display_name"] == "Ada Alpha"
    assert g.nodes["a-alpha"]["n_citing_papers"] == 3
    assert g.nodes["a-beta"]["n_citing_papers"] == 2


def test_author_cocitation_network_restricts_to_given_citing_works() -> None:
    pytest.importorskip("networkx")
    # Only p3 remains, which cites Alpha and Gamma but not Beta.
    g = _cocitation_graph().author_cocitation_network(min_papers=1, citing_ids=["p3"])

    assert set(g.nodes) == {"a-alpha", "a-gamma"}
    assert g["a-alpha"]["a-gamma"]["weight"] == 1
    assert g.nodes["a-alpha"]["n_citing_papers"] == 1


def test_author_cocitation_network_drops_authors_absent_from_selection() -> None:
    pytest.importorskip("networkx")
    # Restricting to p2 leaves Alpha and Beta co-cited; Gamma disappears
    # entirely rather than lingering as a zero-degree node.
    g = _cocitation_graph().author_cocitation_network(min_papers=1, citing_ids=["p2"])

    assert set(g.nodes) == {"a-alpha", "a-beta"}


def test_author_cocitation_network_raises_without_author_tables(
    small_graph: CitationGraph,
) -> None:
    pytest.importorskip("networkx")
    with pytest.raises(RuntimeError, match="Author tables"):
        small_graph.author_cocitation_network()
