"""Work resolution regressions with explicit membership expectations."""
from io import StringIO

import pandas as pd
import pytest

from citegraph.dedup import DedupConfig, canonicalize_works, compare_papers, normalize_text


def record(title="Effects of monetary policy on employment", author="John Smith"):
    return dict(Title=title, Authors=author, Authors_List=[author], Journal="Economics", Year=2020,
                citing_id="w-core")


def cluster(rows):
    sources = pd.DataFrame(columns=["id", "source_file", "Title", "Authors", "Journal", "Year"])
    return canonicalize_works(sources, pd.DataFrame(rows), show_progress=False)


def test_accent_and_leading_article_do_not_hide_match():
    a = record(author="José García")
    b = record(title="The effects of monetary policy on employment", author="Jose Garcia")
    assert compare_papers(a, b, DedupConfig())
    assert len(cluster([a, b])[0]) == 1


def test_authors_list_only_matches_after_csv_roundtrip():
    a = record()
    del a["Authors"]
    raw = pd.DataFrame([a, dict(a)])
    for data in [raw, pd.read_csv(StringIO(raw.to_csv(index=False)))]:
        assert len(cluster(data.to_dict("records"))[0]) == 1


def test_unicode_letters_survive_normalization():
    assert normalize_text("共同资源管理") == "共同资源管理"
    assert normalize_text("García") == normalize_text("Garcia")


@pytest.mark.parametrize("title", [
    "No effects of monetary policy on employment",
    "Effects of monetary policy on employment without inflation",
    "Effects of monetary policy on employment and agricultural production",
])
def test_substantive_title_expansion_is_not_automatic_match(title):
    assert not compare_papers(record(), record(title=title), DedupConfig())


def test_different_parts_are_not_same_work():
    assert not compare_papers(record(title="Economic policy part I"),
                              record(title="Economic policy part II"), DedupConfig())


def test_explicit_subtitle_and_typo_positive_cases():
    assert compare_papers(record(title="Economic policy: an experimental study"),
                          record(title="Economic policy"), DedupConfig())
    assert compare_papers(record(), record(title="Effects of monetary pollicy on employment"), DedupConfig())


def test_subtitle_cannot_override_negation_or_part_conflicts():
    assert not compare_papers(record(title="Economic policy"),
                              record(title="Economic policy: without growth"), DedupConfig())
    assert not compare_papers(record(title="Economic policy"),
                              record(title="Economic policy: part II"), DedupConfig())


def test_public_matcher_accepts_list_only_and_missing_csv_year():
    a = record()
    del a["Authors"]
    assert compare_papers(a, a, DedupConfig())
    assert compare_papers(record(), dict(record(), Year=float("nan")), DedupConfig())


def test_review_decisions_have_scores_and_actual_observation_coordinates():
    _, _, stats = cluster([record(), record(title="No effects of monetary policy on employment")])
    decision = stats["decisions"][0]
    assert decision["left_index"] == 0 and decision["right_index"] == 1
    assert decision["decision"] == "review"
    assert "negation_conflict" in decision["reason_codes"]
    assert decision["weighted_score"] > 85


def test_candidate_oracle_for_accepted_variant_pairs():
    from citegraph.dedup import _candidate_index_lookup, _candidate_indices
    rows = [record(author="José García"),
            record(title="Effects of monetary pollicy on employment", author="Jose Garcia"),
            dict(record(title="The effects of monetary policy on employment"), Authors_List=[], Authors=""),
            record(title="Policy monetary effects of employment on", author="Jose Garcia")]
    df = pd.DataFrame(rows)
    authors, titles, unknown = _candidate_index_lookup(df)
    cfg = DedupConfig(title_weight=1, authors_weight=0, journal_weight=0)
    for i, left in enumerate(rows):
        candidates = _candidate_indices(df.iloc[i], author_blocks=authors, title_blocks=titles, unknown_author=unknown)
        for j, right in enumerate(rows):
            if compare_papers(left, right, cfg):
                assert j in candidates


def test_candidate_count_for_one_thousand_observations(monkeypatch):
    import citegraph.dedup as dedup
    original = dedup._assess_work_match
    calls = 0
    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(dedup, "_assess_work_match", counted)
    rows = [record(title=f"Topic {i} empirical evidence", author=f"Author{chr(65 + i // 26)}{chr(65 + i % 26)}, John")
            for i in range(500)]
    works, _, stats = cluster(rows + rows)
    assert len(works) == 500
    assert len(stats["citation_cluster_ids"]) == 1000
    assert 500 <= calls < 2000  # Real scoring versus ~500,000 all-pairs comparisons.
