"""Tests for journal-name canonicalisation.

The AER appears under six spellings in one corpus and Ecological Economics
under three, so journal rollups understate concentration until the names
are folded.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from citegraph.journals import (
    WORKING_PAPER,
    add_canonical_journals,
    canonicalize_journals,
    journal_key,
    load_journal_aliases,
    unused_aliases,
)
from citegraph.reports import count_journal_alias_warnings

# ---------------------------------------------------------------------------
# journal_key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "left,right",
    [
        ("American Economic Review", "The American economic review"),
        ("Journal of Economic Behavior & Organization",
         "Journal of Economic Behavior and Organization"),
        ("PLOS ONE", "PLoS ONE"),
        ("Revista Colombiana de Antropología", "Revista Colombiana de Antropologia"),
        ("Am. Econ. Rev.", "Am Econ Rev"),
    ],
)
def test_spelling_variants_share_a_key(left, right):
    assert journal_key(left) == journal_key(right)


def test_distinct_journals_keep_distinct_keys():
    assert journal_key("Ecological Economics") != journal_key("Ecological Economics Review")


@pytest.mark.parametrize("value", ["", "   ", None, float("nan")])
def test_missing_journal_has_no_key(value):
    assert journal_key(value) == ""


# ---------------------------------------------------------------------------
# canonicalize_journals
# ---------------------------------------------------------------------------


def test_display_name_is_the_most_frequent_spelling():
    names = (
        ["American Economic Review"] * 3
        + ["The American Economic Review"]
        + ["The American economic review"]
    )
    mapping = canonicalize_journals(names)
    assert set(mapping.values()) == {"American Economic Review"}


def test_longest_spelling_breaks_a_frequency_tie():
    mapping = canonicalize_journals(["Econometrica", "The Econometrica"])
    assert set(mapping.values()) == {"The Econometrica"}


def test_repositories_become_working_papers():
    mapping = canonicalize_journals([
        "RePEc: Research Papers in Economics",
        "SSRN Electronic Journal",
        "Psychology Press eBooks",
    ])
    assert set(mapping.values()) == {WORKING_PAPER}


def test_a_real_journal_is_not_mistaken_for_a_repository():
    mapping = canonicalize_journals(["Journal of Economic Psychology"])
    assert mapping["Journal of Economic Psychology"] == "Journal of Economic Psychology"


def test_missing_names_map_to_empty():
    mapping = canonicalize_journals(["", None, "Science"])
    assert mapping["Science"] == "Science"
    assert mapping[""] == ""


# ---------------------------------------------------------------------------
# aliases
# ---------------------------------------------------------------------------


def test_alias_folds_an_abbreviation_into_its_journal():
    mapping = canonicalize_journals(
        ["Am. Econ. Rev.", "American Economic Review"],
        aliases={"Am Econ Rev": "American Economic Review"},
    )
    assert mapping["Am. Econ. Rev."] == "American Economic Review"


def test_alias_matches_by_key_not_by_exact_spelling():
    # The user writes one spelling; every variant of it should follow.
    mapping = canonicalize_journals(
        ["AM ECON REV"], aliases={"Am. Econ. Rev.": "American Economic Review"}
    )
    assert mapping["AM ECON REV"] == "American Economic Review"


def test_alias_overrides_the_repository_rule():
    mapping = canonicalize_journals(
        ["SSRN Electronic Journal"], aliases={"SSRN Electronic Journal": "SSRN"}
    )
    assert mapping["SSRN Electronic Journal"] == "SSRN"


def test_unused_alias_is_reported_loudly(tmp_path):
    path = tmp_path / "journal_aliases.csv"
    path.write_text("raw,canonical\nNonexistent Journal,Something\n", encoding="utf-8")
    aliases = load_journal_aliases(path)
    with pytest.raises(ValueError, match="Nonexistent Journal"):
        canonicalize_journals(["Science"], aliases=aliases, strict=True)


def test_load_journal_aliases_returns_empty_for_a_missing_file(tmp_path):
    assert load_journal_aliases(tmp_path / "absent.csv") == {}


# ---------------------------------------------------------------------------
# Frame integration
# ---------------------------------------------------------------------------


def test_add_canonical_journals_adds_the_column():
    df = pd.DataFrame({"Journal": [
        "American Economic Review",
        "American Economic Review",
        "The American Economic Review",
        "RePEc: Research Papers in Economics",
        None,
    ]})
    out = add_canonical_journals(df)
    assert list(out["Journal_Canonical"]) == [
        "American Economic Review", "American Economic Review",
        "American Economic Review", WORKING_PAPER, "",
    ]
    # The observed name is never overwritten.
    assert out.loc[2, "Journal"] == "The American Economic Review"


def test_add_canonical_journals_on_a_frame_without_the_column():
    out = add_canonical_journals(pd.DataFrame({"Title": ["x"]}))
    assert list(out["Journal_Canonical"]) == [""]


def test_add_canonical_journals_preserves_the_index():
    df = pd.DataFrame({"Journal": ["Science"]}, index=pd.Index(["w-a"], name="id"))
    out = add_canonical_journals(df)
    assert out.index.name == "id"
    assert list(out.index) == ["w-a"]


# ---------------------------------------------------------------------------
# Pipeline: which fold validates the curated rows
# ---------------------------------------------------------------------------


def _pipeline_with_aliases(tmp_path, rows, *, enrich=False, enriched_journals=None):
    """A Pipeline whose out_dir carries a ``journal_aliases.csv`` of ``rows``."""
    from citegraph.pipeline import Pipeline

    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "journal_aliases.csv").write_text(
        "raw,canonical\n" + "".join(f"{raw},{canon}\n" for raw, canon in rows),
        encoding="utf-8",
    )
    if enriched_journals is not None:
        pd.DataFrame({"id": range(len(enriched_journals)), "Journal": enriched_journals}).to_csv(
            out_dir / "enriched_works.csv", index=False
        )
    return Pipeline(pdf_dir=None, out_dir=out_dir, enrich=enrich)


def test_an_alias_matching_nothing_anywhere_still_raises(tmp_path):
    """The guard that matters: a typo'd row is dead in every vocabulary."""
    pipeline = _pipeline_with_aliases(tmp_path, [("Nonexistent Journal", "Something")])
    with pytest.raises(ValueError, match="Nonexistent Journal"):
        pipeline._with_canonical_journals(pd.DataFrame({"Journal": ["Science"]}))


def test_provider_fold_accepts_an_alias_the_extracted_names_carry(tmp_path):
    """An abbreviation alias is doing its job when it folds nothing here.

    ``journal_aliases.csv`` mostly expands extraction abbreviations. A work the
    provider matched already reports the full container title, so the row has
    nothing to fold on the enriched side — that is success, not a stale row.
    """
    pipeline = _pipeline_with_aliases(tmp_path, [("Ecol. Econ.", "Ecological Economics")])
    enriched = pd.DataFrame({"Journal": ["Ecological Economics", "Science"]})

    out = pipeline._with_canonical_journals(enriched, also_seen=["Ecol. Econ.", "Science"])

    assert out["Journal_Canonical"].tolist() == ["Ecological Economics", "Science"]
    assert not pipeline.layout.journal_alias_warnings_json.exists()


def test_extracted_fold_accepts_an_alias_only_the_provider_uses(tmp_path):
    """The mirror case: overriding the repository rule for a provider-only name.

    "SSRN Electronic Journal" is an OpenAlex container title that no extracted
    bibliography ever writes, so stage 4 must reach for the enriched artifact
    rather than declare the row dead.
    """
    pipeline = _pipeline_with_aliases(
        tmp_path,
        [("SSRN Electronic Journal", "SSRN")],
        enriched_journals=["SSRN Electronic Journal", "Science"],
    )

    out = pipeline._with_canonical_journals(pd.DataFrame({"Journal": ["Science"]}))

    assert out["Journal_Canonical"].tolist() == ["Science"]
    assert not pipeline.layout.journal_alias_warnings_json.exists()


def test_a_pending_provider_vocabulary_warns_instead_of_raising(tmp_path):
    """Enrichment is configured but has not run, so the row may yet apply."""
    pipeline = _pipeline_with_aliases(
        tmp_path, [("SSRN Electronic Journal", "SSRN")], enrich=True
    )

    out = pipeline._with_canonical_journals(pd.DataFrame({"Journal": ["Science"]}))

    assert out["Journal_Canonical"].tolist() == ["Science"]
    warnings = json.loads(pipeline.layout.journal_alias_warnings_json.read_text())
    assert warnings["unused_aliases"] == ["SSRN Electronic Journal"]
    assert count_journal_alias_warnings(pipeline.layout.journal_alias_warnings_json) == 1


def test_the_warning_file_is_removed_once_every_row_applies(tmp_path):
    """Absent = clean, the same contract the other warning sidecars keep."""
    pipeline = _pipeline_with_aliases(
        tmp_path, [("SSRN Electronic Journal", "SSRN")], enrich=True
    )
    pipeline._with_canonical_journals(pd.DataFrame({"Journal": ["Science"]}))
    assert pipeline.layout.journal_alias_warnings_json.exists()

    pipeline._with_canonical_journals(pd.DataFrame({"Journal": ["SSRN Electronic Journal"]}))

    assert not pipeline.layout.journal_alias_warnings_json.exists()


def test_unused_aliases_spans_every_vocabulary_given(tmp_path):
    aliases = {"Ecol. Econ.": "Ecological Economics", "Typo Journal": "Nothing"}

    unused = unused_aliases(aliases, ["Science"], ["Ecol. Econ."])

    assert unused == ["Typo Journal"]
