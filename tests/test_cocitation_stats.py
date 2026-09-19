"""Tests for the supervision-tie co-citation test.

The fixtures here build a co-citation corpus from an explicit
``{citing work: [authors it mentions]}`` map, because every assertion below is
about *which bibliographies cite whom* and nothing else. Spelling that out
directly keeps the arithmetic of each test visible.
"""

from __future__ import annotations

import pandas as pd
import pytest

import citegraph.cocitation_stats as cocitation_stats
from citegraph.cocitation_stats import (
    fisher_combined_p,
    tie_cocitation_test,
)
from citegraph.graph import CitationGraph


def _graph_from_mentions(
    mentions: dict[str, list[str]],
    authored: dict[str, list[str]] | None = None,
) -> CitationGraph:
    """Build a CitationGraph whose co-citation structure is exactly ``mentions``.

    Each author owns one cited work, so "citing work c mentions author a" is
    one edge ``c -> w-a``. ``authored`` records who wrote each *citing* work,
    which is what the self-citation exclusion keys on.
    """
    authored = authored or {}
    author_ids = sorted({a for names in mentions.values() for a in names})

    works = [
        {"id": c, "ring": 0, "source_file": f"{c}.pdf", "Title": c, "Year": 2020}
        for c in sorted(mentions)
    ] + [
        {"id": f"w-{a}", "ring": 1, "source_file": "", "Title": f"work of {a}", "Year": 1990}
        for a in author_ids
    ]
    edges = [
        {"citing_id": c, "cited_id": f"w-{a}"}
        for c in sorted(mentions)
        for a in sorted(mentions[c])
    ]
    citations = [
        {"author_id": a, "record_id": f"w-{a}", "position": 0, "raw_author": a}
        for a in author_ids
    ] + [
        {"author_id": a, "record_id": c, "position": 0, "raw_author": a}
        for c in sorted(authored)
        for a in sorted(authored[c])
    ]
    authors = [
        {
            "id": a,
            "display_name": a.replace("a-", "").title(),
            "surname": a.replace("a-", "").title(),
            "surname_norm": a.replace("a-", ""),
            "canonical_given": "",
            "initials": "",
            "openalex_id": None,
            "orcid": None,
            "n_works": 1,
            "n_core_works": 0,
            "n_citations_received": 0,
            "n_distinct_citing_works": 0,
        }
        for a in author_ids
    ]
    return CitationGraph(
        works=pd.DataFrame(works).set_index("id"),
        edges=pd.DataFrame(edges),
        authors=pd.DataFrame(authors).set_index("id"),
        author_citations=pd.DataFrame(citations),
    )


# Twenty-four bibliographies. Every author is cited by exactly six of them, so
# a degree-matched null pools all eight together and the *only* thing that can
# separate a pair from the background is how often the two travel together.
# ``a-adv`` and ``a-stu`` share all six; the background authors share at most
# three with any one partner.
_PAIRED_CORPUS: dict[str, list[int]] = {
    "a-adv": [0, 1, 2, 3, 4, 5],
    "a-stu": [0, 1, 2, 3, 4, 5],
    "a-b1": [6, 7, 8, 9, 10, 11],
    "a-b2": [9, 10, 11, 12, 13, 14],
    "a-b3": [12, 13, 14, 15, 16, 17],
    "a-b4": [15, 16, 17, 18, 19, 20],
    "a-b5": [18, 19, 20, 21, 22, 23],
    "a-b6": [21, 22, 23, 6, 7, 8],
}


@pytest.fixture()
def paired_corpus() -> CitationGraph:
    mentions: dict[str, list[str]] = {f"c{i:02d}": [] for i in range(24)}
    for author, cited_by in _PAIRED_CORPUS.items():
        for i in cited_by:
            mentions[f"c{i:02d}"].append(author)
    return _graph_from_mentions(mentions)


def test_the_fixture_holds_prominence_constant() -> None:
    """Guards the arithmetic every assertion below leans on: equal prominence,
    a pair sharing everything, and a background sharing at most half."""
    assert {len(works) for works in _PAIRED_CORPUS.values()} == {6}
    shared = {
        (a, b): len(set(_PAIRED_CORPUS[a]) & set(_PAIRED_CORPUS[b]))
        for a in _PAIRED_CORPUS
        for b in _PAIRED_CORPUS
        if a < b
    }
    assert shared[("a-adv", "a-stu")] == 6
    background = {pair: n for pair, n in shared.items() if "a-adv" not in pair and "a-stu" not in pair}
    assert max(background.values()) == 3


# ---------------------------------------------------------------------------
# The test statistic
# ---------------------------------------------------------------------------
def test_a_pair_that_always_travels_together_beats_its_degree_matched_null(
    paired_corpus: CitationGraph,
) -> None:
    pytest.importorskip("networkx")
    result = tie_cocitation_test(
        paired_corpus, [("a-adv", "a-stu")], min_papers=1, n_draws=500, seed=0
    )

    (tie,) = result.ties
    assert tie.observed == 6
    assert tie.null_median < tie.observed
    assert tie.p_value < 0.05


def test_an_ordinary_pair_does_not_beat_its_null(paired_corpus: CitationGraph) -> None:
    pytest.importorskip("networkx")
    result = tie_cocitation_test(
        paired_corpus, [("a-b1", "a-b2")], min_papers=1, n_draws=500, seed=0
    )

    (tie,) = result.ties
    assert tie.p_value > 0.05


def test_p_value_is_never_zero(paired_corpus: CitationGraph) -> None:
    """The permutation convention counts the observed value in its own null, so
    a p-value is bounded below by 1/(n_draws + 1) rather than collapsing to 0."""
    pytest.importorskip("networkx")
    result = tie_cocitation_test(
        paired_corpus, [("a-adv", "a-stu")], min_papers=1, n_draws=99, seed=0
    )

    assert result.ties[0].p_value >= 1 / 100


def test_observed_counts_shared_bibliographies_not_shared_citations() -> None:
    """One bibliography citing several works by the same author is one mention."""
    pytest.importorskip("networkx")
    # The filler authors are cited exactly as often as the pair, so the pair
    # has a prominence stratum to be scored against at all.
    graph = _graph_from_mentions(
        {
            "c1": ["a-x", "a-y", "a-f1"],
            "c2": ["a-x", "a-y", "a-f2"],
            "c3": ["a-x", "a-f1", "a-f3"],
            "c4": ["a-y", "a-f2", "a-f3"],
            "c5": ["a-f1", "a-f2", "a-f3"],
        }
    )
    result = tie_cocitation_test(graph, [("a-x", "a-y")], min_papers=1, n_draws=50, seed=0)

    assert result.ties[0].observed == 2


def test_observed_agrees_with_the_network_a_companion_figure_draws(
    paired_corpus: CitationGraph,
) -> None:
    """The tie's count and the co-citation edge it is drawn as must be one
    number. Both go through ``author_mentions``; this pins that they stay there."""
    pytest.importorskip("networkx")
    network = paired_corpus.author_cocitation_network(min_papers=1)
    result = tie_cocitation_test(
        paired_corpus,
        [("a-adv", "a-stu"), ("a-b1", "a-b2"), ("a-b1", "a-b4")],
        min_papers=1,
        n_draws=50,
        seed=0,
    )

    for tie in result.ties:
        drawn = (
            network[tie.source][tie.target]["weight"]
            if network.has_edge(tie.source, tie.target)
            else 0
        )
        assert tie.observed == drawn
    assert result.n_network_authors == network.number_of_nodes()


# ---------------------------------------------------------------------------
# The self-citation control
# ---------------------------------------------------------------------------
def test_excluding_self_citations_drops_the_pairs_own_bibliographies() -> None:
    pytest.importorskip("networkx")
    graph = _graph_from_mentions(
        mentions={
            # ``a-g*`` match the pair's prominence over the whole corpus, ``a-f*``
            # match it once the pair's own bibliographies are gone. Both strata
            # have to exist, because the two calls below match on different counts.
            "c1": ["a-adv", "a-stu", "a-g1", "a-g2"],
            "c2": ["a-adv", "a-stu", "a-g1", "a-g2"],
            "c3": ["a-adv", "a-stu", "a-g1", "a-g2"],
            "c4": ["a-f1", "a-f2"],
        },
        # One bibliography from each end, so the exclusion is shown to cover both.
        authored={"c1": ["a-stu"], "c2": ["a-adv"]},
    )

    everything = tie_cocitation_test(graph, [("a-adv", "a-stu")], min_papers=1, n_draws=50, seed=0)
    third_party = tie_cocitation_test(
        graph,
        [("a-adv", "a-stu")],
        min_papers=1,
        n_draws=50,
        seed=0,
        exclude_self_citations=True,
    )

    assert everything.ties[0].observed == 3
    assert everything.ties[0].n_eligible == 4
    # c1 and c2 were written by the pair; only c3 is a third party's statement.
    assert third_party.ties[0].observed == 1
    assert third_party.ties[0].n_eligible == 2
    assert third_party.exclude_self_citations is True


def test_the_null_is_counted_inside_the_ties_eligible_bibliographies() -> None:
    """The exclusion sets a universe; it does not filter one side of a contest.

    ``a-f1``/``a-f2`` are co-cited only by the two bibliographies ``a-adv``
    wrote. Those are excluded for this tie, so as a comparison pair the fillers
    must score 0 — not the 2 they score over the whole corpus. Counting the tie
    inside the eligible set and its null outside it would hand the fillers
    chances the tie had been denied, and bury a real result.
    """
    pytest.importorskip("networkx")
    graph = _graph_from_mentions(
        mentions={
            "c1": ["a-adv", "a-stu"],
            "c2": ["a-adv", "a-stu", "a-f1", "a-f2"],
            "c3": ["a-adv", "a-stu", "a-f1", "a-f2"],
        },
        authored={"c2": ["a-adv"], "c3": ["a-adv"]},
    )
    result = tie_cocitation_test(
        graph,
        [("a-adv", "a-stu")],
        min_papers=1,
        n_draws=500,
        seed=0,
        exclude_self_citations=True,
    )

    (tie,) = result.ties
    assert tie.n_eligible == 1
    assert tie.observed == 1
    assert tie.null_mean == 0.0
    assert tie.p_value < 0.05


def test_eligible_count_is_the_whole_corpus_when_nothing_is_excluded(
    paired_corpus: CitationGraph,
) -> None:
    pytest.importorskip("networkx")
    result = tie_cocitation_test(
        paired_corpus, [("a-adv", "a-stu")], min_papers=1, n_draws=50, seed=0
    )

    assert result.ties[0].n_eligible == result.n_citing_works == 24


# ---------------------------------------------------------------------------
# Determinism — a manuscript figure has to redraw identically
# ---------------------------------------------------------------------------
def test_same_seed_gives_identical_results(paired_corpus: CitationGraph) -> None:
    pytest.importorskip("networkx")
    ties = [("a-adv", "a-stu"), ("a-b1", "a-b2")]
    first = tie_cocitation_test(paired_corpus, ties, min_papers=1, n_draws=200, seed=7)
    second = tie_cocitation_test(paired_corpus, ties, min_papers=1, n_draws=200, seed=7)

    pd.testing.assert_frame_equal(first.to_frame(), second.to_frame())


def test_result_does_not_depend_on_the_order_ties_arrive_in(
    paired_corpus: CitationGraph,
) -> None:
    """Each tie seeds its own generator from its own identity, so a caller who
    sorts its ties differently still gets the same numbers for each tie."""
    pytest.importorskip("networkx")
    forward = tie_cocitation_test(
        paired_corpus, [("a-adv", "a-stu"), ("a-b1", "a-b2")], min_papers=1, n_draws=200, seed=7
    )
    reversed_ = tie_cocitation_test(
        paired_corpus, [("a-b1", "a-b2"), ("a-adv", "a-stu")], min_papers=1, n_draws=200, seed=7
    )

    forward_by_pair = {(t.source, t.target): t.p_value for t in forward.ties}
    reversed_by_pair = {(t.source, t.target): t.p_value for t in reversed_.ties}
    assert forward_by_pair == reversed_by_pair


def test_a_tie_is_keyed_the_same_whichever_way_round_it_is_given(
    paired_corpus: CitationGraph,
) -> None:
    pytest.importorskip("networkx")
    one_way = tie_cocitation_test(
        paired_corpus, [("a-adv", "a-stu")], min_papers=1, n_draws=200, seed=3
    )
    other_way = tie_cocitation_test(
        paired_corpus, [("a-stu", "a-adv")], min_papers=1, n_draws=200, seed=3
    )

    assert one_way.ties[0].p_value == other_way.ties[0].p_value
    # The direction the caller supplied is preserved for display.
    assert (other_way.ties[0].source, other_way.ties[0].target) == ("a-stu", "a-adv")


# ---------------------------------------------------------------------------
# Ties the data cannot carry
# ---------------------------------------------------------------------------
def test_a_tie_touching_an_author_outside_the_network_is_skipped_with_a_reason(
    paired_corpus: CitationGraph,
) -> None:
    pytest.importorskip("networkx")
    result = tie_cocitation_test(
        paired_corpus, [("a-adv", "a-ghost")], min_papers=1, n_draws=50, seed=0
    )

    assert result.ties == ()
    assert len(result.skipped) == 1
    skipped = result.skipped[0]
    assert skipped.target == "a-ghost"
    assert "a-ghost" in skipped.reason


def test_a_self_pair_is_skipped_rather_than_scored(paired_corpus: CitationGraph) -> None:
    pytest.importorskip("networkx")
    result = tie_cocitation_test(
        paired_corpus, [("a-adv", "a-adv")], min_papers=1, n_draws=50, seed=0
    )

    assert result.ties == ()
    assert "same author" in result.skipped[0].reason


def test_tie_is_skipped_when_either_endpoint_has_no_degree_matched_substitute() -> None:
    mentions = {f"c{i}": ["a-u"] for i in range(20)}
    for author, cited_by in {
        "a-v": range(3),
        "a-x": range(3, 6),
        "a-y": range(6, 9),
    }.items():
        for i in cited_by:
            mentions[f"c{i}"].append(author)
    graph = _graph_from_mentions(mentions)

    result = tie_cocitation_test(graph, [("a-u", "a-v")])

    assert result.ties == ()
    assert len(result.skipped) == 1
    assert "no degree-matched substitute" in result.skipped[0].reason
    assert "a-u" in result.skipped[0].reason


def test_a_repeated_tie_is_scored_once(paired_corpus: CitationGraph) -> None:
    pytest.importorskip("networkx")
    result = tie_cocitation_test(
        paired_corpus,
        [("a-adv", "a-stu"), ("a-stu", "a-adv")],
        min_papers=1,
        n_draws=50,
        seed=0,
    )

    assert len(result.ties) == 1


def test_pruning_by_min_papers_removes_a_tie_from_the_test(
    paired_corpus: CitationGraph,
) -> None:
    pytest.importorskip("networkx")
    result = tie_cocitation_test(
        paired_corpus, [("a-adv", "a-stu")], min_papers=99, n_draws=50, seed=0
    )

    assert result.ties == ()
    assert result.n_network_authors == 0


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def test_frame_carries_one_row_per_tie_with_display_names(
    paired_corpus: CitationGraph,
) -> None:
    pytest.importorskip("networkx")
    result = tie_cocitation_test(
        paired_corpus, [("a-adv", "a-stu"), ("a-b1", "a-b2")], min_papers=1, n_draws=100, seed=0
    )
    frame = result.to_frame()

    assert len(frame) == 2
    assert {"source", "target", "source_name", "target_name", "observed", "p_value"} <= set(
        frame.columns
    )
    assert frame.loc[0, "source_name"] == "Adv"


def test_counts_of_significant_ties_and_what_chance_would_give(
    paired_corpus: CitationGraph,
) -> None:
    pytest.importorskip("networkx")
    result = tie_cocitation_test(
        paired_corpus, [("a-adv", "a-stu"), ("a-b1", "a-b2")], min_papers=1, n_draws=200, seed=0
    )

    assert result.n_ties == 2
    assert result.n_significant == 1
    assert result.expected_significant == pytest.approx(2 * 0.05)


def test_dependent_binomial_aggregate_is_not_exposed(
    paired_corpus: CitationGraph,
) -> None:
    pytest.importorskip("networkx")
    result = tie_cocitation_test(
        paired_corpus, [("a-adv", "a-stu")], min_papers=1, n_draws=50, seed=0
    )

    assert not hasattr(result, "binomial_p_value")
    assert not hasattr(cocitation_stats, "binomial_tail_p")
    assert "binomial_tail_p" not in cocitation_stats.__all__


def test_empty_tie_list_yields_an_empty_result(paired_corpus: CitationGraph) -> None:
    pytest.importorskip("networkx")
    result = tie_cocitation_test(paired_corpus, [], min_papers=1, n_draws=50, seed=0)

    assert result.n_ties == 0
    assert result.to_frame().empty
    assert result.fisher_p_value is None


def test_requires_the_author_tables(small_corpus_without_authors: CitationGraph) -> None:
    pytest.importorskip("networkx")
    with pytest.raises(RuntimeError, match="author"):
        tie_cocitation_test(small_corpus_without_authors, [("a", "b")])


@pytest.fixture()
def small_corpus_without_authors() -> CitationGraph:
    works = pd.DataFrame(
        [{"id": "w-a", "ring": 0, "source_file": "a.pdf", "Title": "A", "Year": 2020}]
    ).set_index("id")
    return CitationGraph(works=works, edges=pd.DataFrame(columns=["citing_id", "cited_id"]))


# ---------------------------------------------------------------------------
# The assumption-bound combination statistic, implemented without SciPy
# ---------------------------------------------------------------------------
def test_fisher_combines_independent_p_values() -> None:
    # chi2 = -2 * (ln .01 + ln .01) = 18.42 on 4 df -> 0.00101
    assert fisher_combined_p([0.01, 0.01]) == pytest.approx(0.0010210, abs=1e-6)


def test_fisher_of_a_single_p_value_returns_it_unchanged() -> None:
    assert fisher_combined_p([0.3]) == pytest.approx(0.3, abs=1e-9)


def test_fisher_of_uninformative_p_values_is_uninformative() -> None:
    assert fisher_combined_p([0.5, 0.5, 0.5]) > 0.3


def test_fisher_rejects_a_p_value_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        fisher_combined_p([0.5, 1.5])


def test_fisher_of_nothing_is_none() -> None:
    assert fisher_combined_p([]) is None
