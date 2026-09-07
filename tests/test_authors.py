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
    authors_df, citations_df, _ = normalize_authors(works=refs)
    assert len(authors_df) == 1, "all three records describe the same person"
    only = authors_df.iloc[0]
    assert only["surname"].lower() == "diamond"
    assert "adele" in only["display_name"].lower()
    assert int(only["n_works"]) == 3


def test_normalize_keeps_distinct_full_first_names_apart():
    """'Diamond, Adele' and 'Diamond, Andrew' are different people; precision-first keeps them apart."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Diamond, Adele"],  "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Diamond, Andrew"], "Year": 2012},
    ])
    authors_df, _, _ = normalize_authors(works=refs)
    assert len(authors_df) == 2


def test_normalize_initial_only_with_ambiguous_full_names_is_held_aside():
    """'Diamond, A.' with both Adele and Andrew present is ambiguous and gets its own cluster."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Diamond, Adele"],  "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Diamond, Andrew"], "Year": 2012},
        {"id": "r-3", "Title": "T3", "Authors_List": ["Diamond, A."],     "Year": 2014},
    ])
    authors_df, _, review = normalize_authors(works=refs)
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
    authors_df, citations_df, _ = normalize_authors(works=refs)
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
    authors_df, _, _ = normalize_authors(works=refs)
    assert len(authors_df) == 2


def test_normalize_diacritic_insensitive():
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Cárdenas, Juan-Camilo"], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Cardenas, J. C."],       "Year": 2012},
    ])
    authors_df, _, _ = normalize_authors(works=refs)
    assert len(authors_df) == 1


def test_normalize_loose_mode_collapses_aggressively():
    """'Diamond, Adele' and 'Diamond, Andrew' merge under loose mode."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Diamond, Adele"],  "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Diamond, Andrew"], "Year": 2012},
    ])
    authors_df, _, _ = normalize_authors(
        works=refs, cfg=AuthorClusterConfig(merge_mode="loose"),
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
    authors_df, _, _ = normalize_authors(works=refs, enriched_works=enriched)
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
        works=refs,
        enriched_works=enriched,
    )

    assert len(authors_df) == 1
    assert authors_df.iloc[0]["openalex_id"] == "A5042502300"
    assert authors_df.iloc[0]["orcid"] == "0000-0003-0005-7595"
    assert int(authors_df.iloc[0]["n_works"]) == 3
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
    authors_df, _, _ = normalize_authors(works=refs, enriched_works=enriched)
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
    pre, _, _ = normalize_authors(works=refs)
    assert len(pre) == 2
    ids = list(pre.index)
    # Force merge with an alias: collapse the second cluster into the first.
    aliases = {ids[1]: ids[0]}
    post, _, _ = normalize_authors(works=refs, aliases=aliases)
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


def test_normalize_spans_core_and_cited_works():
    """Core (ring-0) and cited (ring-1) appearances cluster into one person."""
    works = pd.DataFrame([
        {"id": "w-1", "ring": 1, "source_file": "", "Title": "T1",
         "Authors_List": ["Diamond, Adele"], "Year": 2010},
        {"id": "w-2", "ring": 0, "source_file": "p1.pdf", "Title": "P1",
         "Authors_List": ["Diamond, A."], "Year": 2020, "Journal": "J"},
    ]).set_index("id")
    authors_df, citations_df, _ = normalize_authors(works=works)
    # Same person — should merge across the corpus and its citations.
    assert len(authors_df) == 1
    assert set(citations_df["record_id"]) == {"w-1", "w-2"}
    assert int(authors_df.iloc[0]["n_core_works"]) == 1


def test_normalize_counts_distinct_citing_papers_from_edges():
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Ostrom, Elinor"], "Year": 1990},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Ostrom, E."], "Year": 1998},
    ])
    edges = pd.DataFrame(
        [
            {"citing_id": "p-a", "cited_id": "r-1"},
            {"citing_id": "p-b", "cited_id": "r-1"},
            {"citing_id": "p-a", "cited_id": "r-2"},
        ]
    )

    authors_df, _, _ = normalize_authors(works=refs, citation_edges=edges)

    ostrom = authors_df.iloc[0]
    assert int(ostrom["n_works"]) == 2
    assert int(ostrom["n_distinct_citing_works"]) == 2


# ---------------------------------------------------------------------------
# Stable cluster ids
# ---------------------------------------------------------------------------


def test_cluster_ids_are_stable_across_runs():
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Diamond, Adele"], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Diamond, A."],    "Year": 2012},
    ])
    a1, _, _ = normalize_authors(works=refs)
    a2, _, _ = normalize_authors(works=refs)
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
    authors_df, _, _ = normalize_authors(works=refs)
    assert not authors_df.empty
    assert (authors_df["surname_norm"].str.len() > 1).all(), authors_df
    assert set(authors_df["surname_norm"]) == {"smith", "garcia"}


def test_normalize_authors_handles_semicolon_joined_authors():
    """Semicolon-joined author strings split cleanly without re-glue heuristics."""
    refs = pd.DataFrame([
        {"id": "r-1", "Title": "T1", "Year": 2010,
         "Authors": "Smith, J.; García, A."},
    ]).set_index("id")
    authors_df, _, _ = normalize_authors(works=refs)
    assert set(authors_df["surname_norm"]) == {"smith", "garcia"}


def test_normalize_authors_handles_comma_joined_full_given_names():
    """Fallback comma splitting should re-glue obvious Surname, Given pairs."""
    refs = pd.DataFrame([
        {"id": "r-1", "Title": "T1", "Year": 2010,
         "Authors": "Smith, John, García, Ana"},
    ]).set_index("id")
    authors_df, _, _ = normalize_authors(works=refs)
    assert set(authors_df["surname_norm"]) == {"smith", "garcia"}


def test_normalize_authors_does_not_pair_first_last_chunks():
    """Avoid gluing ``Talbot Page, Louis Putterman`` into one false author."""
    refs = pd.DataFrame([
        {"id": "r-1", "Title": "T1", "Year": 2010,
         "Authors": "Bochet, Oliver, Talbot Page, Louis Putterman"},
    ]).set_index("id")
    authors_df, _, _ = normalize_authors(works=refs)
    assert set(authors_df["surname_norm"]) == {"bochet", "page", "putterman"}


# ---------------------------------------------------------------------------
# Spanish / multi-part name handling (evidence-driven re-parse)
# ---------------------------------------------------------------------------


def test_parse_vancouver_surname_then_initials():
    """'Guerra JA' is Vancouver style: surname first, bare initials last."""
    for raw, surname, initials in [
        ("Guerra JA", "Guerra", "JA"),
        ("Cardenas JC", "Cardenas", "JC"),
        ("Rand DG", "Rand", "DG"),
    ]:
        p = parse_author(raw)
        assert p is not None, raw
        assert p.surname == surname
        assert p.initials == initials
        assert p.has_full_first is False


def test_parse_vancouver_with_periods_rescued():
    """'Guerra J.A.' / 'Diamond A.' must parse, not be dropped as initial-only surnames."""
    p = parse_author("Guerra J.A.")
    assert p is not None and p.surname == "Guerra" and p.initials == "JA"
    p = parse_author("Diamond A.")
    assert p is not None and p.surname == "Diamond" and p.initials == "A"
    p = parse_author("Smith J")
    assert p is not None and p.surname == "Smith" and p.initials == "J"


def test_parse_trailing_et_al_stripped():
    p = parse_author("B. Puranen et al.")
    assert p is not None and p.surname == "Puranen" and p.initials == "B"
    p = parse_author("Barreteau, et al.")
    assert p is not None and p.surname == "Barreteau"


def test_parse_corporate_author_kept_whole():
    for raw in [
        "Centro Nacional de Memoria Histórica",
        "The World Bank",
        "American Psychiatric Association",
        "NRC (National Research Council)",
    ]:
        p = parse_author(raw)
        assert p is not None, raw
        assert p.is_corporate is True
        assert p.given_names == ()
    p = parse_author("Centro Nacional de Memoria Histórica")
    assert p.surname_norm == "centro nacional de memoria historica"


def test_parse_person_not_flagged_corporate():
    for raw in ["Diamond, Adele", "Juan de la Cruz", "Banks, J.", "Adele Diamond"]:
        p = parse_author(raw)
        assert p is not None and p.is_corporate is False, raw


def test_parse_y_connector_extends_surname():
    p = parse_author("José Ortega y Gasset")
    assert p is not None
    assert p.surname == "Ortega y Gasset"
    assert p.given_names == ("José",)


def test_parse_y_initial_not_treated_as_connector():
    p = parse_author("Che, Y.-K.")
    assert p is not None and p.surname == "Che" and p.initials == "YK"
    p = parse_author("John Y. Smith")
    assert p is not None and p.surname == "Smith"


def test_norm_surname_hyphen_equals_space():
    a = parse_author("Casas-Casas, Andrés")
    b = parse_author("Casas Casas, Andrés")
    assert a is not None and b is not None
    assert a.surname_norm == b.surname_norm == "casas casas"


def test_normalize_merges_hyphenated_and_spaced_surnames():
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Casas-Casas, Andrés"], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Casas Casas, Andrés"], "Year": 2012},
    ])
    authors_df, _, _ = normalize_authors(works=refs)
    assert len(authors_df) == 1


def test_parse_ocr_spaced_apostrophe_cleaned():
    """OCR-mangled accents ('B ' enabou') must normalize to the real surname."""
    p = parse_author("B ' enabou, R")
    assert p is not None and p.surname_norm == "benabou"
    p = parse_author("Ib ' a ˜ nez, A.M.")
    assert p is not None and p.surname_norm == "ibanez"
    # Genuine apostrophe surnames (no surrounding spaces) are untouched.
    p = parse_author("O'Neill, J.")
    assert p is not None and p.surname == "O'Neill" and p.surname_norm == "oneill"


def test_hyphenated_and_spaced_given_names_share_one_cluster():
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Cárdenas, Juan-Camilo"], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Cardenas, Juan Camilo"], "Year": 2012},
        {"id": "r-3", "Title": "T3", "Authors_List": ["Juan-Camilo Cardenas"], "Year": 2014},
        {"id": "r-4", "Title": "T4", "Authors_List": ["Cardenas, J.C."], "Year": 2016},
    ])
    authors_df, _, _ = normalize_authors(works=refs)
    assert len(authors_df) == 1
    assert int(authors_df.iloc[0]["n_works"]) == 4


def test_known_surname_resplits_no_comma_name():
    p = parse_author("Jose Alberto Guerra Forero", known_surnames={"guerra forero"})
    assert p is not None
    assert p.surname == "Guerra Forero"
    assert p.given_names == ("Jose", "Alberto")


def test_comma_form_corroborates_compound_surname_end_to_end():
    """'Guerra Forero, J.A.' in the corpus teaches the no-comma form its boundary."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Guerra Forero, J.A."], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Jose Alberto Guerra Forero"], "Year": 2012},
    ])
    authors_df, _, _ = normalize_authors(works=refs)
    assert len(authors_df) == 1
    only = authors_df.iloc[0]
    assert only["surname_norm"] == "guerra forero"
    assert only["display_name"] == "Jose Alberto Guerra Forero"


def test_hyphenated_surname_attests_spaced_compound():
    """'Polania-Reyes' (one token) proves 'Polanía Reyes' (two tokens) is a surname."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Sandra Polania-Reyes"], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Sandra Polanía Reyes"], "Year": 2012},
    ])
    authors_df, _, _ = normalize_authors(works=refs)
    assert len(authors_df) == 1
    assert authors_df.iloc[0]["surname_norm"] == "polania reyes"


def test_unattested_compound_stays_conservative_but_flagged():
    """No corroboration -> keep the last-token parse, show the full name, flag for review."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Jose Alberto Guerra Forero"], "Year": 2010},
    ])
    authors_df, _, review = normalize_authors(works=refs)
    assert len(authors_df) == 1
    only = authors_df.iloc[0]
    assert only["surname_norm"] == "forero"  # conservative: no evidence invented
    assert only["display_name"] == "Jose Alberto Guerra Forero"  # but nothing is dropped
    assert any("compound" in (r["reason"] or "") for r in review)


def test_bare_lexicon_word_does_not_resplit_middle_names():
    """A single-word surname elsewhere must never trigger a compound re-split."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Knowles, B."], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Caitlin Knowles Myers"], "Year": 2012},
    ])
    authors_df, _, _ = normalize_authors(works=refs)
    assert set(authors_df["surname_norm"]) == {"knowles", "myers"}


def test_maynard_smith_comma_form_corroborates():
    """Evidence-driven splitting works for non-Spanish compounds too."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Maynard Smith, J."], "Year": 1982},
        {"id": "r-2", "Title": "T2", "Authors_List": ["John Maynard Smith"], "Year": 1974},
    ])
    authors_df, _, _ = normalize_authors(works=refs)
    assert len(authors_df) == 1
    assert authors_df.iloc[0]["surname_norm"] == "maynard smith"


def test_incompatible_second_initial_not_absorbed():
    """'Cardenas, J.P.' must not join Juan-Camilo even with co-author overlap."""
    refs = _refs([
        {"id": "r-1", "Title": "T1",
         "Authors_List": ["Cardenas, Juan-Camilo", "Ostrom, E."], "Year": 2010},
        {"id": "r-2", "Title": "T2",
         "Authors_List": ["Cardenas, J.P.", "Ostrom, E."], "Year": 2012},
    ])
    authors_df, _, _ = normalize_authors(works=refs)
    cardenas = authors_df[authors_df["surname_norm"] == "cardenas"]
    assert len(cardenas) == 2


def test_identical_raw_strings_cluster_together():
    """Two 'Diamond, A.' records move as one unit instead of splitting by context."""
    refs = _refs([
        {"id": "r-1", "Title": "T1",
         "Authors_List": ["Diamond, Adele", "Posner, M."], "Year": 2010},
        {"id": "r-2", "Title": "T2",
         "Authors_List": ["Diamond, Andrew", "Zhou, K."], "Year": 2011},
        {"id": "r-3", "Title": "T3",
         "Authors_List": ["Diamond, A.", "Posner, M."], "Year": 2012},
        {"id": "r-4", "Title": "T4",
         "Authors_List": ["Diamond, A."], "Year": 2013},
    ])
    authors_df, citations_df, _ = normalize_authors(works=refs)
    adele = authors_df[authors_df["display_name"].str.contains("Adele", case=False)]
    aid = adele.index[0]
    adele_records = set(citations_df[citations_df["author_id"] == aid]["record_id"])
    assert {"r-1", "r-3", "r-4"} <= adele_records


def test_display_keeps_full_given_sequence():
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Ceballos, Jorge Luis"], "Year": 2010},
    ])
    authors_df, _, _ = normalize_authors(works=refs)
    assert authors_df.iloc[0]["display_name"] == "Jorge Luis Ceballos"


def test_corporate_cluster_flagged_in_review():
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["The World Bank"], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["The World Bank"], "Year": 2012},
    ])
    authors_df, _, review = normalize_authors(works=refs)
    assert len(authors_df) == 1
    assert authors_df.iloc[0]["display_name"] == "The World Bank"
    assert any("corporate" in (r["reason"] or "") for r in review)


def test_bridge_single_surname_into_compound_with_exact_given():
    """'Reyes, Sandra' joins 'Polanía Reyes, Sandra' on the exact-given match."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Polanía Reyes, Sandra"], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Reyes, Sandra"], "Year": 2012},
    ])
    authors_df, citations_df, _ = normalize_authors(works=refs)
    assert len(authors_df) == 1
    only = authors_df.iloc[0]
    assert only["surname_norm"] == "polania reyes"
    assert set(citations_df["record_id"]) == {"r-1", "r-2"}


def test_initials_only_never_bridge_into_compound():
    """'Reyes, S.' has no exact given-name evidence and must stay apart."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Polanía Reyes, Sandra"], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Reyes, S."], "Year": 2014},
    ])
    authors_df, _, _ = normalize_authors(works=refs)
    assert len(authors_df) == 2


def test_same_external_id_merges_across_surname_blocks():
    """External ids are ground truth even when blocking disagrees."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Guerra Forero, J.A."], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Guerra, Jose Alberto"], "Year": 2012},
    ])
    enriched = pd.DataFrame(
        [
            {"id": "r-1",
             "OpenAlex_Authors": [{"display_name": "Jose Alberto Guerra Forero",
                                   "openalex_id": "A77", "orcid": None}]},
            {"id": "r-2",
             "OpenAlex_Authors": [{"display_name": "Jose Alberto Guerra Forero",
                                   "openalex_id": "A77", "orcid": None}]},
        ]
    ).set_index("id")
    authors_df, _, _ = normalize_authors(works=refs, enriched_works=enriched)
    assert len(authors_df) == 1


def test_enrichment_family_feeds_surname_lexicon():
    """An enrichment 'family' field corroborates a compound surname split."""
    refs = _refs([
        {"id": "r-1", "Title": "T1",
         "Authors_List": ["Jose Alberto Guerra Forero"], "Year": 2010},
    ])
    enriched = pd.DataFrame(
        [
            {"id": "r-1",
             "OpenAlex_Authors": [{"display_name": "Jose Alberto Guerra Forero",
                                   "family": "Guerra Forero",
                                   "openalex_id": None, "orcid": None}]},
        ]
    ).set_index("id")
    authors_df, _, _ = normalize_authors(works=refs, enriched_works=enriched)
    assert len(authors_df) == 1
    assert authors_df.iloc[0]["surname_norm"] == "guerra forero"


def test_reversed_enrichment_author_order_does_not_swap_ids():
    """Citation order and publication order can disagree — ids must follow names.

    Real-corpus case: a citation lists 'Ibañez, Moya' but OpenAlex has
    'Moya, Ibáñez'; blind positional pairing gave Ibáñez Moya's ORCID,
    and the same-id rule then pulled Moya's solo core paper into her
    cluster.
    """
    refs = _refs([
        {"id": "r-1", "Title": "Joint paper",
         "Authors_List": ["Ibañez, A.M.", "Moya, A."], "Year": 2010},
        {"id": "r-2", "Title": "Solo paper",
         "Authors_List": ["Andres Moya"], "Year": 2018},
    ])
    moya_ids = {"openalex_id": "A5102739955", "orcid": "0000-0003-0640-5802"}
    enriched = pd.DataFrame(
        [
            {"id": "r-1",
             "OpenAlex_Authors": [
                 {"display_name": "Andres Moya", "family": None, "given": None,
                  **moya_ids},
                 {"display_name": "Ana María Ibáñez", "family": None, "given": None,
                  "openalex_id": None, "orcid": None},
             ]},
            {"id": "r-2",
             "OpenAlex_Authors": [
                 {"display_name": "Andres Moya", "family": None, "given": None,
                  **moya_ids},
             ]},
        ]
    ).set_index("id")
    authors_df, citations_df, _ = normalize_authors(works=refs, enriched_works=enriched)
    moya = authors_df[authors_df["surname_norm"] == "moya"]
    ibanez = authors_df[authors_df["surname_norm"] == "ibanez"]
    assert len(moya) == 1
    assert len(ibanez) == 1
    assert moya.iloc[0]["orcid"] == moya_ids["orcid"]
    assert not (isinstance(ibanez.iloc[0]["orcid"], str) and ibanez.iloc[0]["orcid"])
    solo = citations_df[citations_df["record_id"] == "r-2"]
    assert solo.iloc[0]["author_id"] == moya.index[0]


def test_full_name_stuffed_family_does_not_poison_lexicon():
    """CrossRef sometimes stuffs the entire name into 'family' (given empty).

    A family equal to the whole display name attests nothing about where
    the surname boundary sits, so it must not teach the lexicon a fake
    compound surname (real-corpus case: 'Juan Camilo Cárdenas').
    """
    refs = _refs([
        {"id": "r-1", "Title": "T1",
         "Authors_List": ["Juan Camilo Cárdenas"], "Year": 2010},
    ])
    enriched = pd.DataFrame(
        [
            {"id": "r-1",
             "OpenAlex_Authors": [{"display_name": "Juan Camilo Cárdenas",
                                   "family": "Juan Camilo Cárdenas",
                                   "given": None,
                                   "openalex_id": None, "orcid": None}]},
        ]
    ).set_index("id")
    authors_df, _, _ = normalize_authors(works=refs, enriched_works=enriched)
    assert len(authors_df) == 1
    row = authors_df.iloc[0]
    assert row["surname_norm"] == "cardenas"
    assert row["display_name"] == "Juan Camilo Cárdenas"


def test_typo_variant_full_names_merge():
    """A one-letter typo ('Camillo') must not fork an anchor and strand initials."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Cardenas, Juan Camilo"], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Cardenas, Juan Camilo"], "Year": 2011},
        {"id": "r-3", "Title": "T3", "Authors_List": ["Cardenas, Juan Camillo"], "Year": 2012},
        {"id": "r-4", "Title": "T4", "Authors_List": ["Cardenas, J.C."], "Year": 2014},
    ])
    authors_df, _, _ = normalize_authors(works=refs)
    assert len(authors_df) == 1
    # The most frequent spelling wins the display, never the typo.
    assert authors_df.iloc[0]["display_name"] == "Juan Camilo Cardenas"


def test_gendered_name_pairs_stay_apart():
    """Gabriel/Gabriela and Daniel/Daniela are different people, not typos."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["García, Gabriel"], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["García, Gabriela"], "Year": 2012},
        {"id": "r-3", "Title": "T3", "Authors_List": ["Rodríguez, Daniel"], "Year": 2010},
        {"id": "r-4", "Title": "T4", "Authors_List": ["Rodríguez, Daniela"], "Year": 2012},
    ])
    authors_df, _, _ = normalize_authors(works=refs)
    assert len(authors_df) == 4


# ---------------------------------------------------------------------------
# Review follow-ups: precision guards and cross-block invariants
# ---------------------------------------------------------------------------


def test_initials_prefix_alone_does_not_merge():
    """'Smith, J.' and 'Smith, J.C.' have zero full-name evidence — keep apart."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Smith, J."], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Smith, J.C."], "Year": 2012},
    ])
    authors_df, _, _ = normalize_authors(works=refs)
    assert len(authors_df) == 2


def test_external_id_union_is_transitive():
    """A cluster carrying two external ids must pull both id-groups together."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["García, Gabriel"], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Márquez, Gabriel"], "Year": 2012},
        {"id": "r-3", "Title": "T3", "Authors_List": ["García Márquez, Gabriel"], "Year": 2014},
    ])
    enriched = pd.DataFrame(
        [
            {"id": "r-1",
             "OpenAlex_Authors": [{"display_name": "Gabriel García Márquez",
                                   "openalex_id": "A1", "orcid": None}]},
            {"id": "r-2",
             "OpenAlex_Authors": [{"display_name": "Gabriel García Márquez",
                                   "openalex_id": None, "orcid": "0000-1"}]},
            {"id": "r-3",
             "OpenAlex_Authors": [{"display_name": "Gabriel García Márquez",
                                   "openalex_id": "A1", "orcid": "0000-1"}]},
        ]
    ).set_index("id")
    authors_df, _, _ = normalize_authors(works=refs, enriched_works=enriched)
    assert len(authors_df) == 1


def test_external_cluster_variant_anchors_no_id_records():
    """Any attested variant in an id-cluster can anchor a no-id cluster."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Cardenas, Juan Camilo"], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Cardenas, Camilo"], "Year": 2012},
        {"id": "r-3", "Title": "T3", "Authors_List": ["Cardenas, Camilo"], "Year": 2014},
    ])
    enriched = pd.DataFrame(
        [
            {"id": "r-1",
             "OpenAlex_Authors": [{"display_name": "Juan Camilo Cardenas",
                                   "openalex_id": "A9", "orcid": None}]},
            {"id": "r-2",
             "OpenAlex_Authors": [{"display_name": "Juan Camilo Cardenas",
                                   "openalex_id": "A9", "orcid": None}]},
        ]
    ).set_index("id")
    authors_df, _, _ = normalize_authors(works=refs, enriched_works=enriched)
    assert len(authors_df) == 1


def test_cross_block_merge_id_independent_of_row_order():
    """A cluster spanning surname blocks must not name itself by row order."""
    rows = [
        {"id": "r-1", "Title": "T1", "Authors_List": ["Guerra, Jose Alberto"], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Guerra Forero, Jose Alberto"], "Year": 2012},
    ]
    enriched_rows = [
        {"id": "r-1",
         "OpenAlex_Authors": [{"display_name": "Jose Alberto Guerra Forero",
                               "openalex_id": "A77", "orcid": None}]},
        {"id": "r-2",
         "OpenAlex_Authors": [{"display_name": "Jose Alberto Guerra Forero",
                               "openalex_id": "A77", "orcid": None}]},
    ]
    fwd, _, _ = normalize_authors(
        works=_refs(rows),
        enriched_works=pd.DataFrame(enriched_rows).set_index("id"),
    )
    rev, _, _ = normalize_authors(
        works=_refs(rows[::-1]),
        enriched_works=pd.DataFrame(enriched_rows[::-1]).set_index("id"),
    )
    assert len(fwd) == len(rev) == 1
    assert list(fwd.index) == list(rev.index)
    # The more specific (compound) surname names the merged cluster.
    assert fwd.iloc[0]["surname_norm"] == "guerra forero"


def test_dutch_t_and_quoted_nicknames_preserved():
    """OCR cleanup must not fuse legitimate spaced apostrophes."""
    p = parse_author("van 't Hoff, J.")
    assert p is not None
    assert p.surname_norm == "van t hoff"
    p = parse_author("Sandra 'Sandy' Smith")
    assert p is not None and p.surname == "Smith"


def test_short_allcaps_names_not_garbled():
    p = parse_author("LI X")
    assert p is not None and p.surname == "LI" and p.initials == "X"
    p = parse_author("LEE KY")
    assert p is not None and p.surname == "LEE" and p.initials == "KY"
    # Two short tokens are genuinely ambiguous ('Bo XU' is a caps surname,
    # 'Ma JA' could be Vancouver); fall back to the conservative last-token
    # parse rather than inventing an initials reading.
    p = parse_author("Bo XU")
    assert p is not None and p.surname == "XU"
    p = parse_author("Ibáñez ÁC")
    assert p is not None and p.surname_norm == "ibanez" and p.has_full_first is False


def test_person_with_affiliation_parens_not_corporate():
    p = parse_author("Smith, J. (MIT)")
    assert p is not None
    assert p.is_corporate is False
    assert p.surname == "Smith"


def test_two_token_person_with_corporate_word_surname():
    p = parse_author("Steven Bank")
    assert p is not None and p.is_corporate is False and p.surname == "Bank"
    # Corporate-word-first two-token names are still institutions.
    p = parse_author("Fundación Natura")
    assert p is not None and p.is_corporate is True


def test_ambiguous_full_name_cluster_flagged_for_review():
    """Bare 'Adele' compatible with two middle-initial variants: split + flagged."""
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Diamond, Adele"], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Diamond, Adele B."], "Year": 2012},
        {"id": "r-3", "Title": "T3", "Authors_List": ["Diamond, Adele C."], "Year": 2014},
    ])
    authors_df, _, review = normalize_authors(works=refs)
    assert len(authors_df) == 3  # B. and C. conflict; bare Adele is ambiguous
    assert any("ambiguous" in (r["reason"] or "") for r in review)


def test_ids_stable_with_lexicon_reparse():
    refs = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Guerra Forero, J.A."], "Year": 2010},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Jose Alberto Guerra Forero"], "Year": 2012},
    ])
    a1, _, _ = normalize_authors(works=refs)
    a2, _, _ = normalize_authors(works=refs)
    assert list(a1.index) == list(a2.index)


def test_dedup_to_authors_round_trip_via_csv(tmp_path: Path):
    """End-to-end: raw refs → dedup → CSV round-trip → normalize_authors.

    Guards against any future stage that drops ``Authors_List`` between
    dedup and author normalization.
    """
    from citegraph.dedup import canonicalize_works

    sources = pd.DataFrame(
        columns=["id", "source_file", "Title", "Authors", "Authors_List", "Journal", "Year"]
    )
    raw = pd.DataFrame([
        {"Title": "Paper One",   "Authors": "Smith, J., García, A.",
         "Authors_List": ["Smith, J.", "García, A."],
         "Journal": "J1", "Year": 2010, "citing_id": "w-1"},
        {"Title": "Paper Two",   "Authors": "Smith, John, García, Ana",
         "Authors_List": ["Smith, John", "García, Ana"],
         "Journal": "J2", "Year": 2011, "citing_id": "w-1"},
    ])
    canonical, _edges, _stats = canonicalize_works(sources, raw, show_progress=False)
    # Authors_List must survive canonicalization.
    assert "Authors_List" in canonical.columns

    csv_path = tmp_path / "works.csv"
    canonical.to_csv(csv_path)
    reloaded = pd.read_csv(csv_path, index_col="id")

    authors_df, _, _ = normalize_authors(works=reloaded)
    assert (authors_df["surname_norm"].str.len() > 1).all()
    assert {"smith", "garcia"} <= set(authors_df["surname_norm"])


# ---------------------------------------------------------------------------
# works-model metrics
# ---------------------------------------------------------------------------


def _works_fixture():
    works = pd.DataFrame(
        {
            "ring": [0, 0, 1],
            "source_file": ["a.pdf", "b.pdf", ""],
            "Title": ["Core paper A", "Core paper B", "External classic"],
            "Authors": [
                "Cardenas, Juan Camilo",
                "Cardenas, Juan Camilo, Ostrom, Elinor",
                "Ostrom, Elinor",
            ],
            "Authors_List": [
                ["Cardenas, Juan Camilo"],
                ["Cardenas, Juan Camilo", "Ostrom, Elinor"],
                ["Ostrom, Elinor"],
            ],
            "Journal": ["JDE", "WD", "CUP"],
            "Year": [2000, 2004, 1990],
        },
        index=pd.Index(["w-a", "w-b", "w-c"], name="id"),
    )
    edges = pd.DataFrame(
        [
            {"citing_id": "w-a", "cited_id": "w-c"},
            {"citing_id": "w-b", "cited_id": "w-c"},
            {"citing_id": "w-a", "cited_id": "w-b"},  # core cites core
        ]
    )
    return works, edges


def test_author_metrics_span_core_and_cited_roles():
    works, edges = _works_fixture()
    authors_df, citations_df, _review = normalize_authors(works=works, citation_edges=edges)

    cardenas = authors_df[authors_df["surname_norm"] == "cardenas"].iloc[0]
    assert int(cardenas["n_works"]) == 2  # authored w-a and w-b
    assert int(cardenas["n_core_works"]) == 2
    assert int(cardenas["n_citations_received"]) == 1  # w-b is cited once (by w-a)
    assert int(cardenas["n_distinct_citing_works"]) == 1

    ostrom = authors_df[authors_df["surname_norm"] == "ostrom"].iloc[0]
    assert int(ostrom["n_works"]) == 2  # w-b (co-author) and w-c
    assert int(ostrom["n_core_works"]) == 1  # only w-b is ring 0
    assert int(ostrom["n_citations_received"]) == 3  # 2 into w-c + 1 into w-b

    assert set(citations_df.columns) == {"author_id", "record_id", "position", "raw_author"}
