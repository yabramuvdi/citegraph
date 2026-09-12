"""Tests for the fuzzy deduplication module."""

from __future__ import annotations

import pandas as pd

from citegraph.dedup import (
    DedupConfig,
    _assess_work_match,
    _compute_rings,
    canonicalize_works,
    compare_papers,
    normalize_text,
)


def _empty_sources() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["id", "source_file", "Title", "Authors", "Authors_List", "Journal", "Year"]
    )


def _canonicalize_citations(df: pd.DataFrame, cfg: DedupConfig | None = None):
    """Cluster citation rows with no sources; returns (works, per-row cluster ids)."""
    works, _edges, stats = canonicalize_works(
        _empty_sources(), df, cfg or DedupConfig(), show_progress=False
    )
    return works, stats["citation_cluster_ids"]


def test_normalize_text_strips_punctuation_and_case():
    assert normalize_text("Governing the Commons.") == "governing the commons"
    assert normalize_text(["Ostrom, E.", "Hardin, G."]) == "ostrom e hardin g"
    assert normalize_text(None) == ""
    assert normalize_text(123) == ""


def test_compare_papers_identifies_duplicates():
    cfg = DedupConfig()
    a = {
        "Title": "Governing the Commons",
        "Authors": "Ostrom, E.",
        "Journal": "Cambridge University Press",
        "Year": 1990,
    }
    b = {
        "Title": "Governing the commons.",
        "Authors": "Elinor Ostrom",
        "Journal": "Cambridge Univ Press",
        "Year": 1990,
    }
    assert compare_papers(a, b, cfg) is True


def test_canonicalize_considers_fuzzy_match_when_block_keys_differ():
    sources = pd.DataFrame([{
        "id": "w-source", "source_file": "paper.md", "Title": "Understanding collective action in rural communities",
        "Authors": "John Smith", "Authors_List": ["John Smith"], "Journal": "Science", "Year": 2020,
    }])
    citations = pd.DataFrame([{
        "Title": "Understanding collective action in rural comunities", "Authors": "John Smyth",
        "Authors_List": ["John Smyth"], "Journal": "Science", "Year": 2020, "citing_id": "w-source",
    }])
    works, edges, _ = canonicalize_works(sources, citations, show_progress=False)
    assert len(works) == 1
    assert edges.empty


def test_compare_papers_rejects_different_papers():
    cfg = DedupConfig()
    a = {
        "Title": "Governing the Commons",
        "Authors": "Ostrom, E.",
        "Journal": "Cambridge University Press",
        "Year": 1990,
    }
    b = {
        "Title": "Tragedy of the Commons",
        "Authors": "Hardin, G.",
        "Journal": "Science",
        "Year": 1968,
    }
    assert compare_papers(a, b, cfg) is False


def test_compare_papers_year_window():
    cfg = DedupConfig(year_window=0)
    a = {"Title": "Same Title", "Authors": "X", "Journal": "J", "Year": 2000}
    b = {"Title": "Same Title", "Authors": "X", "Journal": "J", "Year": 2001}
    assert compare_papers(a, b, cfg) is False
    cfg = DedupConfig(year_window=1)
    assert compare_papers(a, b, cfg) is True


def test_canonicalize_clusters_duplicate_citations(sample_references):
    canonical, mapping = _canonicalize_citations(sample_references)
    assert len(canonical) == 3, "expected 3 unique works from 5 citations"
    assert mapping[0] == mapping[1], "Ostrom rows should cluster"
    assert mapping[2] == mapping[3], "Hardin rows should cluster"
    assert mapping[4] != mapping[0], "Putnam should be its own cluster"


def test_canonicalize_handles_empty_citations():
    empty = pd.DataFrame(columns=["Title", "Authors", "Journal", "Year", "citing_id"])
    works, edges, stats = canonicalize_works(
        _empty_sources(), empty, DedupConfig(), show_progress=False
    )
    assert works.empty
    assert edges.empty
    assert stats["citation_cluster_ids"] == []


def test_compare_papers_missing_year_one_side_still_matches():
    """The headline fix: year=0 on one side should not block a clear title/authors match."""
    cfg = DedupConfig()
    a = {
        "Title": "Governing the Commons",
        "Authors": "Ostrom, E.",
        "Journal": "Cambridge University Press",
        "Year": 1990,
    }
    b = dict(a, Year=0)  # same paper, year unknown to the LLM
    assert compare_papers(a, b, cfg) is True

    b_none = dict(a, Year=None)
    assert compare_papers(a, b_none, cfg) is True

    b_empty = dict(a, Year="")
    assert compare_papers(a, b_empty, cfg) is True


def test_compare_papers_missing_year_does_not_rescue_weak_title():
    """A missing year doesn't loosen the title/authors threshold — different papers stay separate."""
    cfg = DedupConfig()
    a = {
        "Title": "Governing the Commons",
        "Authors": "Ostrom, E.",
        "Journal": "Cambridge University Press",
        "Year": 1990,
    }
    b = {
        "Title": "Tragedy of the Commons",
        "Authors": "Hardin, G.",
        "Journal": "Science",
        "Year": 0,
    }
    assert compare_papers(a, b, cfg) is False


def test_compare_papers_both_missing_year_still_matches():
    cfg = DedupConfig()
    a = {"Title": "A Paper", "Authors": "X", "Journal": "J", "Year": 0}
    b = {"Title": "A Paper", "Authors": "X", "Journal": "J", "Year": 0}
    assert compare_papers(a, b, cfg) is True


def test_compare_papers_unparseable_year_still_fails_closed():
    """Per CLAUDE.md, parsing failures must fail closed — only missing data is forgiven."""
    cfg = DedupConfig()
    a = {"Title": "A Paper", "Authors": "X", "Journal": "J", "Year": 2000}
    b = {"Title": "A Paper", "Authors": "X", "Journal": "J", "Year": "not-a-year"}
    assert compare_papers(a, b, cfg) is False


def test_dedup_collapses_missing_year_duplicates():
    """End-to-end: two raw refs of the same paper, one missing its year, dedup to one canonical entry."""
    df = pd.DataFrame(
        [
            {
                "Title": "Governing the Commons",
                "Authors_List": ["Elinor Ostrom"],
                "Authors": "Ostrom, E.",
                "Journal": "Cambridge University Press",
                "Year": 1990,
                "citing_id": "p-1",
            },
            {
                "Title": "Governing the Commons",
                "Authors_List": ["Elinor Ostrom"],
                "Authors": "Ostrom, E.",
                "Journal": "Cambridge University Press",
                "Year": 0,  # LLM missed the year on the second citing paper
                "citing_id": "p-2",
            },
        ]
    )
    canonical, mapping = _canonicalize_citations(df)
    assert len(canonical) == 1, "expected the missing-year duplicate to merge"
    assert mapping[0] == mapping[1]


def test_compare_papers_merges_first_author_vs_full_author_list():
    """One record lists only the first author, the other lists all of them."""
    cfg = DedupConfig()
    a = {
        "Title": "Fairness in Simple Bargaining Experiments",
        "Authors": "Forsythe, R.",
        "Journal": "Games and Economic Behavior",
        "Year": 1994,
    }
    b = {
        "Title": "Fairness in Simple Bargaining Experiments",
        "Authors": "Forsythe, Robert, Horowitz, Joel L., Savin, N. E.",
        "Journal": "Games and Economic Behavior",
        "Year": 1994,
    }
    assert compare_papers(a, b, cfg) is True


def test_compare_papers_merges_title_with_subtitle():
    """One record has a subtitle, the other doesn't — token_set should still match."""
    cfg = DedupConfig()
    a = {
        "Title": "A Behavioral Approach to the Rational Choice Theory of Collective Action",
        "Authors": "Ostrom, Elinor",
        "Journal": "American Political Science Review",
        "Year": 1998,
    }
    b = {
        "Title": (
            "A Behavioral approach to the rational choice theory of collective action: "
            "presidential address, American Political Science Association"
        ),
        "Authors": "OSTROM, ELINOR",
        "Journal": "The American Political Science Review",
        "Year": 1998,
    }
    assert compare_papers(a, b, cfg) is True


def test_compare_papers_merges_abbreviated_journal_and_partial_authors():
    """Abbreviated journal + first-author-only — the classic 'looks like a dup' pattern."""
    cfg = DedupConfig()
    a = {
        "Title": "Public Goods Provision in an Experimental Environment",
        "Authors": "Isaac, R. Mark, Kenneth F. McCue, Charles R. Plott",
        "Journal": "Journal of Public Economics",
        "Year": 1985,
    }
    b = {
        "Title": "Public Goods Provision in an Experimental Environment",
        "Authors": "R. Mark Isaac",
        "Journal": "J. Pub. Econ.",
        "Year": 1985,
    }
    assert compare_papers(a, b, cfg) is True


def test_compare_papers_rejects_different_papers_by_same_author_same_year():
    """Guard against the looser scorer accidentally merging distinct works."""
    cfg = DedupConfig()
    a = {
        "Title": "A Behavioral Approach to the Rational Choice Theory of Collective Action",
        "Authors": "Ostrom, Elinor",
        "Journal": "American Political Science Review",
        "Year": 1998,
    }
    b = {
        "Title": "Scaling Up: The Challenge of Self-Governance for Collective Action",
        "Authors": "Ostrom, Elinor",
        "Journal": "World Development",
        "Year": 1998,
    }
    assert compare_papers(a, b, cfg) is False


def test_canonicalize_returns_stable_work_ids(sample_references):
    _canonical, mapping = _canonicalize_citations(sample_references)
    for cluster_id in set(mapping):
        assert isinstance(cluster_id, str)
        assert cluster_id.startswith("w-")


def test_canonicalize_disambiguates_colliding_work_ids():
    """Two distinct works that slug to the same id must get unique ids.

    Real-corpus case: two different Fehr/Fischbacher 2003 papers cited
    without titles (one in Nature, one in Evol. Hum. Behav.) both slug
    to w-fehr-2003-untitled but must not share a row in works.csv.
    """
    cit = pd.DataFrame(
        [
            {
                "citing_id": "w-x",
                "Title": "",
                "Authors": "E. Fehr, U. Fischbacher",
                "Authors_List": ["E. Fehr", "U. Fischbacher"],
                "Journal": "Nature",
                "Year": 2003,
            },
            {
                "citing_id": "w-x",
                "Title": "",
                "Authors": "E. Fehr, U. Fischbacher",
                "Authors_List": ["E. Fehr", "U. Fischbacher"],
                "Journal": "Evol. Hum. Behav.",
                "Year": 2003,
            },
        ]
    )
    works, mapping = _canonicalize_citations(cit)
    assert works.index.is_unique, f"duplicate work ids: {list(works.index)}"
    assert len(works) == 2
    assert mapping[0] != mapping[1]
    assert sorted(works["Journal"]) == ["Evol. Hum. Behav.", "Nature"]


def test_id_collision_with_source_does_not_steal_ring0():
    """A citation colliding with a source id must not inherit ring 0 / source_file.

    Ring and source_file are joined back by id, so an unsuffixed collision
    would make the citation row masquerade as a core work.
    """
    sources = pd.DataFrame(
        [
            {
                "id": "w-fehr-2003-untitled",
                "source_file": "fehr.md",
                "Title": "A completely different treatise",
                "Authors": "Q. Fehr",
                "Authors_List": ["Q. Fehr"],
                "Journal": "Econometrica",
                "Year": 2003,
            }
        ]
    )
    cit = pd.DataFrame(
        [
            {
                "citing_id": "w-fehr-2003-untitled",
                "Title": "",
                "Authors": "E. Fehr, U. Fischbacher",
                "Authors_List": ["E. Fehr", "U. Fischbacher"],
                "Journal": "Nature",
                "Year": 2003,
            }
        ]
    )
    works, edges, _stats = canonicalize_works(
        sources, cit, DedupConfig(), show_progress=False
    )
    assert works.index.is_unique
    assert len(works) == 2
    core = works[works["ring"] == 0]
    assert len(core) == 1
    assert core.iloc[0]["source_file"] == "fehr.md"
    cited = works[works["ring"] != 0]
    assert cited.iloc[0]["source_file"] == ""
    # The edge must point at the disambiguated citation work, not the source.
    assert edges.iloc[0]["cited_id"] == cited.index[0]
    assert edges.iloc[0]["citing_id"] == "w-fehr-2003-untitled"


def test_dedup_uses_candidate_blocking_for_unrelated_rows(monkeypatch):
    rows = []
    for i in range(30):
        rows.append(
            {
                "Title": f"Distinct Paper {i}",
                "Authors_List": [f"Author{i} Surname{i}"],
                "Authors": f"Author{i} Surname{i}",
                "Journal": "J",
                "Year": 1900 + i,
                "citing_id": f"p-{i}",
            }
        )
    df = pd.DataFrame(rows)

    calls = 0

    def counted_compare(a, b, cfg):
        nonlocal calls
        calls += 1
        return _assess_work_match(a, b, cfg)

    monkeypatch.setattr("citegraph.dedup._assess_work_match", counted_compare)

    canonical, mapping = _canonicalize_citations(df, DedupConfig(year_window=0))

    assert len(canonical) == len(df)
    assert len(mapping) == len(df)
    assert calls < 40


# ---------------------------------------------------------------------------
# canonicalize_works
# ---------------------------------------------------------------------------


def _sources_df():
    return pd.DataFrame(
        [
            {
                "id": "w-cardenas-2000-real-wealth",
                "source_file": "real wealth.pdf",
                "Title": "Real wealth and experimental cooperation",
                "Authors": "Cardenas, Juan Camilo",
                "Authors_List": ["Cardenas, Juan Camilo"],
                "Journal": "Journal of Development Economics",
                "Year": 2000,
            },
            {
                "id": "w-ostrom-1990-governing-the-commons",
                "source_file": "governing.pdf",
                "Title": "Governing the Commons",
                "Authors": "Ostrom, Elinor",
                "Authors_List": ["Ostrom, Elinor"],
                "Journal": "CUP",
                "Year": 1990,
            },
        ]
    )


def test_citation_of_a_source_resolves_to_the_source_work():
    citations = pd.DataFrame(
        [
            {
                "Title": "Real wealth and experimental cooperation",
                "Authors": "Cardenas, J.C.",
                "Authors_List": ["Cardenas, J.C."],
                "Journal": "J Dev Econ",
                "Year": 2000,
                "citing_id": "w-ostrom-1990-governing-the-commons",
            }
        ]
    )
    works, edges, stats = canonicalize_works(
        _sources_df(), citations, DedupConfig(), show_progress=False
    )
    # No new work was minted for the citation — it merged into the source.
    assert len(works) == 2
    assert list(edges.itertuples(index=False)) == [
        ("w-ostrom-1990-governing-the-commons", "w-cardenas-2000-real-wealth")
    ]
    # Cluster metadata kept the full-text side, not the citation string.
    assert works.loc["w-cardenas-2000-real-wealth", "Title"] == (
        "Real wealth and experimental cooperation"
    )
    assert works.loc["w-cardenas-2000-real-wealth", "Authors"] == "Cardenas, Juan Camilo"
    assert works.loc["w-cardenas-2000-real-wealth", "ring"] == 0
    assert stats["n_self_loops_dropped"] == 0


def test_unmatched_citation_becomes_ring1_stub():
    citations = pd.DataFrame(
        [
            {
                "Title": "A completely different treatise on fisheries",
                "Authors": "Schlager, E.",
                "Authors_List": ["Schlager, E."],
                "Journal": "Land Economics",
                "Year": 1994,
                "citing_id": "w-cardenas-2000-real-wealth",
            }
        ]
    )
    works, edges, _stats = canonicalize_works(
        _sources_df(), citations, DedupConfig(), show_progress=False
    )
    stub = works[works["ring"] == 1]
    assert len(stub) == 1
    assert stub.index[0].startswith("w-schlager-1994-")
    assert stub.iloc[0]["source_file"] == ""
    assert len(edges) == 1


def test_duplicate_source_pdfs_merge_and_edges_remap():
    sources = _sources_df()
    dup = sources.iloc[[0]].copy()
    dup["id"] = "w-cardenas-2000-real-wealth-and-exp"  # different slug, same paper
    dup["source_file"] = "real wealth (copy).pdf"
    dup["Title"] = "Real wealth and experimental cooperation "
    sources = pd.concat([sources, dup], ignore_index=True)
    citations = pd.DataFrame(
        [
            {
                "Title": "Governing the Commons",
                "Authors": "Ostrom, E.",
                "Authors_List": ["Ostrom, E."],
                "Journal": "CUP",
                "Year": 1990,
                "citing_id": "w-cardenas-2000-real-wealth-and-exp",  # cites via the dup id
            }
        ]
    )
    works, edges, stats = canonicalize_works(
        sources, citations, DedupConfig(), show_progress=False
    )
    assert stats["n_source_duplicates_merged"] == 1
    assert "w-cardenas-2000-real-wealth-and-exp" not in works.index
    # The edge's citing side remapped onto the canonical source id.
    assert list(edges.itertuples(index=False)) == [
        ("w-cardenas-2000-real-wealth", "w-ostrom-1990-governing-the-commons")
    ]
    # First member keeps the cluster's source_file.
    assert works.loc["w-cardenas-2000-real-wealth", "source_file"] == "real wealth.pdf"


def test_self_citation_is_dropped_and_counted():
    citations = pd.DataFrame(
        [
            {
                "Title": "Real wealth and experimental cooperation",
                "Authors": "Cardenas, Juan Camilo",
                "Authors_List": ["Cardenas, Juan Camilo"],
                "Journal": "working paper",  # preprint variant of itself
                "Year": 2000,
                "citing_id": "w-cardenas-2000-real-wealth",
            }
        ]
    )
    _works, edges, stats = canonicalize_works(
        _sources_df(), citations, DedupConfig(), show_progress=False
    )
    assert edges.empty
    assert stats["n_self_loops_dropped"] == 1


def test_compute_rings_general_bfs():
    edges = pd.DataFrame(
        [
            {"citing_id": "w-a", "cited_id": "w-b"},
            {"citing_id": "w-b", "cited_id": "w-c"},  # ring-1 work citing (snowball case)
            {"citing_id": "w-a", "cited_id": "w-d"},
        ]
    )
    rings = _compute_rings({"w-a"}, edges)
    assert rings == {"w-a": 0, "w-b": 1, "w-d": 1, "w-c": 2}
