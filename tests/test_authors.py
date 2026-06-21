"""Tests for the author parsing and normalization module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from citegraph.authors import (
    AuthorClusterConfig,
    load_aliases,
    normalize_authors,
    parse_author,
)

# ---------------------------------------------------------------------------
# parse_author
# ---------------------------------------------------------------------------


def test_parse_surname_first_initial():
    p = parse_author("Diamond, A.")
    assert p is not None
    assert p.surname == "Diamond"
    assert p.surname_norm == "diamond"
    assert p.initials == "A"
    assert p.has_full_first is False
    assert p.given_names == ("A.",)


def test_parse_surname_first_full_name():
    p = parse_author("Diamond, Adele")
    assert p is not None
    assert p.surname == "Diamond"
    assert p.initials == "A"
    assert p.has_full_first is True


def test_parse_first_last_full_name():
    p = parse_author("Adele Diamond")
    assert p is not None
    assert p.surname == "Diamond"
    assert p.given_names == ("Adele",)
    assert p.has_full_first is True


def test_parse_first_last_with_initial():
    p = parse_author("A. Diamond")
    assert p is not None
    assert p.surname == "Diamond"
    assert p.initials == "A"
    assert p.has_full_first is False


def test_parse_diacritics_stripped_in_norm():
    p = parse_author("Cárdenas, J.-C.")
    assert p is not None
    assert p.surname == "Cárdenas"
    assert p.surname_norm == "cardenas"
    assert p.initials == "JC"


def test_parse_hyphenated_full_name():
    p = parse_author("Cárdenas, Juan-Camilo")
    assert p is not None
    assert p.surname == "Cárdenas"
    assert p.initials == "JC"
    assert p.has_full_first is True


def test_parse_compound_surname_with_particle():
    """'García Márquez, Gabriel' — comma resolves ambiguity."""
    p = parse_author("García Márquez, Gabriel")
    assert p is not None
    assert p.surname == "García Márquez"
    assert p.surname_norm == "garcia marquez"
    assert p.has_full_first is True


def test_parse_particle_compound_no_comma():
    """'Juan de la Cruz' — particle drags the surname back to include 'de la'."""
    p = parse_author("Juan de la Cruz")
    assert p is not None
    assert p.surname.lower().startswith("de la cruz")
    assert p.given_names == ("Juan",)


def test_parse_with_suffix():
    p = parse_author("Smith, John Jr.")
    assert p is not None
    assert p.surname == "Smith"
    assert p.suffix is not None and p.suffix.lower().startswith("jr")
    assert p.has_full_first is True


def test_parse_dropped_for_unusable_input():
    assert parse_author("") is None
    assert parse_author("   ") is None
    assert parse_author("et al.") is None
    assert parse_author("Anonymous") is None
    assert parse_author(None) is None  # type: ignore[arg-type]
    assert parse_author("***") is None


def test_parse_strips_footnote_glyphs():
    p = parse_author("Dean, M.*")
    assert p is not None
    assert p.surname == "Dean"
    assert p.initials == "M"


def test_parse_multiple_initials_with_periods_and_spaces():
    """All three common spellings of 'JC' initials parse to the same initials."""
    a = parse_author("Cárdenas, J. C.")
    b = parse_author("Cárdenas, J.C.")
    c = parse_author("Cárdenas, J.-C.")
    assert a is not None and b is not None and c is not None
    assert a.initials == b.initials == c.initials == "JC"
    assert a.surname_norm == b.surname_norm == c.surname_norm == "cardenas"


# ---------------------------------------------------------------------------
# normalize_authors — single-block clustering
# ---------------------------------------------------------------------------


def _refs(rows: list[dict]) -> pd.DataFrame:
    """Build a deduplicated-references DataFrame indexed by id."""
    df = pd.DataFrame(rows)
    return df.set_index("id")


def test_normalize_merges_initial_into_full_name_when_unambiguous():
    """Strict mode merges 'Diamond, A.' into 'Diamond, Adele' when she's the only Diamond full-name."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Diamond, Adele"], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Diamond, A."],    "Year": 2012},
        {"id": "r-3", "Title": "T3", "Authors_List": ["A. Diamond"],     "Year": 2014},
    ])
    authors_df, citations_df, _ = normalize_authors(references=refs)
    assert len(authors_df) == 1, "all three records describe the same person"
    only = authors_df.iloc[0]
    assert only["surname"].lower() == "diamond"
    assert "adele" in only["display_name"].lower()
    assert int(only["n_reference_citations"]) == 3


def test_normalize_keeps_distinct_full_first_names_apart():
    """'Diamond, Adele' and 'Diamond, Andrew' are different people; precision-first keeps them apart."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Diamond, Adele"],  "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Diamond, Andrew"], "Year": 2012},
    ])
    authors_df, _, _ = normalize_authors(references=refs)
    assert len(authors_df) == 2


def test_normalize_initial_only_with_ambiguous_full_names_is_held_aside():
    """'Diamond, A.' with both Adele and Andrew present is ambiguous and gets its own cluster."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Diamond, Adele"],  "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Diamond, Andrew"], "Year": 2012},
        {"id": "r-3", "Title": "T3", "Authors_List": ["Diamond, A."],     "Year": 2014},
    ])
    authors_df, _, review = normalize_authors(references=refs)
    # Adele, Andrew, and the unresolved A. → 3 clusters in strict mode.
    assert len(authors_df) == 3


def test_normalize_coauthor_signal_resolves_ambiguity():
    """When two full-name candidates exist, a shared co-author disambiguates."""
    refs = _refs([
        # Adele Diamond co-authoring with Posner.
        {"id": "r-1", "Title": "T1",
         "Authors_List": ["Diamond, Adele", "Posner, M."], "Year": 2010},
        # Andrew Diamond with a totally different co-author.
        {"id": "r-2", "Title": "T2",
         "Authors_List": ["Diamond, Andrew", "Zhou, K."], "Year": 2012},
        # 'Diamond, A.' on a paper that also cites Posner — should go to Adele.
        {"id": "r-3", "Title": "T3",
         "Authors_List": ["Diamond, A.", "Posner, M."], "Year": 2014},
    ])
    authors_df, citations_df, _ = normalize_authors(references=refs)
    # The Adele cluster should have absorbed the ambiguous 'A.' record.
    adele = authors_df[authors_df["display_name"].str.contains("Adele", case=False)]
    assert len(adele) == 1
    aid = adele.index[0]
    adele_refs = citations_df[citations_df["author_id"] == aid]
    assert set(adele_refs["record_id"]) == {"r-1", "r-3"}


def test_normalize_keeps_conflicting_middle_initials_apart():
    """Different middle initials means different people in strict mode."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Smith, J. E."], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Smith, J. F."], "Year": 2012},
    ])
    authors_df, _, _ = normalize_authors(references=refs)
    assert len(authors_df) == 2


def test_normalize_diacritic_insensitive():
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Cárdenas, Juan-Camilo"], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Cardenas, J. C."],       "Year": 2012},
    ])
    authors_df, _, _ = normalize_authors(references=refs)
    assert len(authors_df) == 1


def test_normalize_loose_mode_collapses_aggressively():
    """'Diamond, Adele' and 'Diamond, Andrew' merge under loose mode."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Diamond, Adele"],  "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Diamond, Andrew"], "Year": 2012},
    ])
    authors_df, _, _ = normalize_authors(
        references=refs, cfg=AuthorClusterConfig(merge_mode="loose"),
    )
    assert len(authors_df) == 1


# ---------------------------------------------------------------------------
# OpenAlex / ORCID ground truth
# ---------------------------------------------------------------------------


def test_openalex_id_merges_records_that_look_different():
    """Two records with different name forms still merge when OpenAlex says they're the same."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Diamond, A."],    "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Diamond, Adele"], "Year": 2012},
    ])
    enriched = pd.DataFrame(
        [
            {"id": "r-1",
             "OpenAlex_Authors": [{"display_name": "Adele Diamond",
                                   "openalex_id": "A5012345678",
                                   "orcid": None}]},
            {"id": "r-2",
             "OpenAlex_Authors": [{"display_name": "Adele Diamond",
                                   "openalex_id": "A5012345678",
                                   "orcid": None}]},
        ]
    ).set_index("id")
    authors_df, _, _ = normalize_authors(references=refs, enriched_references=enriched)
    assert len(authors_df) == 1
    assert authors_df.iloc[0]["openalex_id"] == "A5012345678"


def test_openalex_full_name_anchor_absorbs_unidentified_initials():
    refs = _refs([
        {
            "id": "r-1",
            "Title": "T1",
            "Authors_List": ["Cárdenas, Juan-Camilo"],
            "Year": 2010,
        },
        {
            "id": "r-2",
            "Title": "T2",
            "Authors_List": ["Cardenas, J.C."],
            "Year": 2012,
        },
        {
            "id": "r-3",
            "Title": "T3",
            "Authors_List": ["Juan-Camilo Cardenas"],
            "Year": 2014,
        },
    ])
    enriched = pd.DataFrame(
        [
            {
                "id": "r-1",
                "OpenAlex_Authors": [
                    {
                        "display_name": "Juan-Camilo Cardenas",
                        "openalex_id": "A5042502300",
                        "orcid": "0000-0003-0005-7595",
                    }
                ],
            }
        ]
    ).set_index("id")

    authors_df, citations_df, _ = normalize_authors(
        references=refs,
        enriched_references=enriched,
    )

    assert len(authors_df) == 1
    assert authors_df.iloc[0]["openalex_id"] == "A5042502300"
    assert authors_df.iloc[0]["orcid"] == "0000-0003-0005-7595"
    assert int(authors_df.iloc[0]["n_reference_citations"]) == 3
    assert set(citations_df["record_id"]) == {"r-1", "r-2", "r-3"}


def test_different_openalex_ids_split_identical_names():
    """Two 'J. Smith' records with different OpenAlex ids stay separate."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Smith, J."], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Smith, J."], "Year": 2012},
    ])
    enriched = pd.DataFrame(
        [
            {"id": "r-1",
             "OpenAlex_Authors": [{"display_name": "John Smith",
                                   "openalex_id": "A1", "orcid": None}]},
            {"id": "r-2",
             "OpenAlex_Authors": [{"display_name": "Jane Smith",
                                   "openalex_id": "A2", "orcid": None}]},
        ]
    ).set_index("id")
    authors_df, _, _ = normalize_authors(references=refs, enriched_references=enriched)
    assert len(authors_df) == 2
    assert set(authors_df["openalex_id"]) == {"A1", "A2"}


# ---------------------------------------------------------------------------
# Aliases (hand-curated overrides)
# ---------------------------------------------------------------------------


def test_aliases_force_merge_two_clusters():
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Smith, John"], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Smith, Jane"], "Year": 2012},
    ])
    # First run: two separate clusters.
    pre, _, _ = normalize_authors(references=refs)
    assert len(pre) == 2
    ids = list(pre.index)
    # Force merge with an alias: collapse the second cluster into the first.
    aliases = {ids[1]: ids[0]}
    post, _, _ = normalize_authors(references=refs, aliases=aliases)
    assert len(post) == 1


def test_load_aliases_round_trip(tmp_path: Path):
    p = tmp_path / "aliases.csv"
    p.write_text("cluster_id,canonical_id\n# comment line\na-foo,a-bar\na-baz,a-bar\n")
    loaded = load_aliases(p)
    assert loaded == {"a-foo": "a-bar", "a-baz": "a-bar"}


def test_load_aliases_missing_path_returns_empty(tmp_path: Path):
    assert load_aliases(tmp_path / "nope.csv") == {}
    assert load_aliases(None) == {}


# ---------------------------------------------------------------------------
# Papers + references together
# ---------------------------------------------------------------------------


def test_normalize_includes_source_paper_authors():
    """When ``papers`` is passed, source-paper authors are clustered alongside refs."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Diamond, Adele"], "Year": 2010},
    ])
    papers = pd.DataFrame([
        {"id": "p-1", "Title": "P1", "Authors_List": ["Diamond, A."],
         "Year": 2020, "Journal": "J", "Authors": "Diamond, A.", "source_file": "p1.md"},
    ])
    authors_df, citations_df, _ = normalize_authors(references=refs, papers=papers)
    # Same person — should merge across papers and references.
    assert len(authors_df) == 1
    kinds = set(citations_df["record_kind"])
    assert kinds == {"reference", "paper"}


# ---------------------------------------------------------------------------
# Stable cluster ids
# ---------------------------------------------------------------------------


def test_cluster_ids_are_stable_across_runs():
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Diamond, Adele"], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Diamond, A."],    "Year": 2012},
    ])
    a1, _, _ = normalize_authors(references=refs)
    a2, _, _ = normalize_authors(references=refs)
    assert list(a1.index) == list(a2.index)
    assert all(cid.startswith("a-") for cid in a1.index)


# ---------------------------------------------------------------------------
# Regression: single-letter pseudo-surnames from joined Authors string
# ---------------------------------------------------------------------------


def test_parse_rejects_single_letter_surname():
    """'J.' and 'X. Y.' must not be accepted as surnames — they are misparsed initials."""
    assert parse_author("J.") is None
    assert parse_author("X. Y.") is None
    assert parse_author("M") is None


def test_normalize_authors_no_authors_list_no_single_letter_clusters():
    """A references frame with only the comma-joined ``Authors`` column must
    not yield single-letter pseudo-surnames.

    Before the fix, ``Authors_List`` was dropped during dedup; the author
    stage then split ``"Smith, J., García, A."`` on ``,`` and ended up with
    ``"J"`` and ``"A"`` as surnames, producing mega-clusters keyed on one
    letter.
    """
    refs = pd.DataFrame([
        {"id": "r-1", "Title": "T1", "Year": 2010,
         "Authors": "Smith, J., García, A."},
        {"id": "r-2", "Title": "T2", "Year": 2011,
         "Authors": "Smith, John, García, Ana"},
    ]).set_index("id")
    authors_df, _, _ = normalize_authors(references=refs)
    assert not authors_df.empty
    assert (authors_df["surname_norm"].str.len() > 1).all(), authors_df
    assert set(authors_df["surname_norm"]) == {"smith", "garcia"}


def test_normalize_authors_handles_semicolon_joined_authors():
    """Semicolon-joined author strings split cleanly without re-glue heuristics."""
    refs = pd.DataFrame([
        {"id": "r-1", "Title": "T1", "Year": 2010,
         "Authors": "Smith, J.; García, A."},
    ]).set_index("id")
    authors_df, _, _ = normalize_authors(references=refs)
    assert set(authors_df["surname_norm"]) == {"smith", "garcia"}


def test_normalize_authors_handles_comma_joined_full_given_names():
    """Fallback comma splitting should re-glue obvious Surname, Given pairs."""
    refs = pd.DataFrame([
        {"id": "r-1", "Title": "T1", "Year": 2010,
         "Authors": "Smith, John, García, Ana"},
    ]).set_index("id")
    authors_df, _, _ = normalize_authors(references=refs)
    assert set(authors_df["surname_norm"]) == {"smith", "garcia"}


def test_normalize_authors_does_not_pair_first_last_chunks():
    """Avoid gluing ``Talbot Page, Louis Putterman`` into one false author."""
    refs = pd.DataFrame([
        {"id": "r-1", "Title": "T1", "Year": 2010,
         "Authors": "Bochet, Oliver, Talbot Page, Louis Putterman"},
    ]).set_index("id")
    authors_df, _, _ = normalize_authors(references=refs)
    assert set(authors_df["surname_norm"]) == {"bochet", "page", "putterman"}


def test_dedup_to_authors_round_trip_via_csv(tmp_path: Path):
    """End-to-end: raw refs → dedup → CSV round-trip → normalize_authors.

    Guards against any future stage that drops ``Authors_List`` between
    dedup and author normalization.
    """
    from citegraph.dedup import dedup_references

    raw = pd.DataFrame([
        {"Title": "Paper One",   "Authors": "Smith, J., García, A.",
         "Authors_List": ["Smith, J.", "García, A."],
         "Journal": "J1", "Year": 2010, "citing_id": "p-1"},
        {"Title": "Paper Two",   "Authors": "Smith, John, García, Ana",
         "Authors_List": ["Smith, John", "García, Ana"],
         "Journal": "J2", "Year": 2011, "citing_id": "p-1"},
    ])
    canonical, _ = dedup_references(raw, show_progress=False)
    # Authors_List must survive dedup.
    assert "Authors_List" in canonical.columns

    csv_path = tmp_path / "references.csv"
    canonical.to_csv(csv_path)
    reloaded = pd.read_csv(csv_path, index_col="id")

    authors_df, _, _ = normalize_authors(references=reloaded)
    assert (authors_df["surname_norm"].str.len() > 1).all()
    assert {"smith", "garcia"} <= set(authors_df["surname_norm"])
