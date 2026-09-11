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
