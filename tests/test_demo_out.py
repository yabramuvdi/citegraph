"""The shipped demo output directory stays loadable and non-trivial.

It is the one integration fixture built from a real corpus, so a schema change
that breaks it would break every user resuming an existing run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from citegraph import CitationGraph

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo_out"


@pytest.fixture(scope="module")
def demo() -> CitationGraph:
    if not DEMO.exists():
        pytest.skip("examples/demo_out is not present in this checkout")
    return CitationGraph.from_out_dir(DEMO)


def test_demo_loads_a_real_corpus(demo: CitationGraph) -> None:
    assert demo.n_core_works == 93
    assert demo.n_works > 3000
    assert demo.n_edges > 4000
    assert demo.has_authors
    assert not demo.has_annotations
    assert set(demo.core["ring"]) == {0}
    assert (demo.core["source_file"] != "").all()


def test_demo_answers_the_readme_queries(demo: CitationGraph) -> None:
    assert demo.find_author("ostrom").index[0] == "a-ostrom-elinor"
    top = demo.top_cited(5)
    assert "w-ostrom-1994-rules-games-and-common-pool-resources" in top.index
    assert demo.top_authors(1, ring=0).index[0] == "a-cardenas-juan-camilo"


def test_demo_cocitation_network(demo: CitationGraph) -> None:
    pytest.importorskip("networkx")
    graph = demo.author_cocitation_network(min_papers=3)
    assert graph.number_of_nodes() > 500
    assert graph["a-ostrom-elinor"]["a-cardenas-juan-camilo"]["weight"] > 40
