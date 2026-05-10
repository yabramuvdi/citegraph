"""Tests for the fuzzy deduplication module."""

from __future__ import annotations

import pandas as pd

from citegraph.dedup import (
    DedupConfig,
    compare_papers,
    dedup_references,
    normalize_text,
)


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


def test_dedup_references_clusters_duplicates(sample_references):
    canonical, mapping = dedup_references(sample_references, DedupConfig())
    assert len(canonical) == 3, "expected 3 unique papers from 5 refs"
    assert mapping.iloc[0] == mapping.iloc[1], "Ostrom rows should cluster"
    assert mapping.iloc[2] == mapping.iloc[3], "Hardin rows should cluster"
    assert mapping.iloc[4] != mapping.iloc[0], "Putnam should be its own cluster"


def test_dedup_references_handles_empty():
    empty = pd.DataFrame(columns=["Title", "Authors", "Journal", "Year", "citing_id"])
    canonical, mapping = dedup_references(empty, DedupConfig())
    assert canonical.empty
    assert mapping.empty


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
    canonical, mapping = dedup_references(df, DedupConfig())
    assert len(canonical) == 1, "expected the missing-year duplicate to merge"
    assert mapping.iloc[0] == mapping.iloc[1]


def test_dedup_references_returns_stable_ids(sample_references):
    canonical, mapping = dedup_references(sample_references, DedupConfig())
    for cluster_id in mapping.unique():
        assert isinstance(cluster_id, str)
        assert cluster_id.startswith("r-")
