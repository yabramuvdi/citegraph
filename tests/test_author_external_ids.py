"""Hand-curated external identifiers for canonical authors.

Provider data quality sometimes defeats name matching even when the right
identifier is unambiguous to a human. OpenAlex's canonical display name for
Elinor Ostrom is "Элинор Остром", so her id A5003087402 — present on 12
works of one corpus — never attaches to the cluster built from "Ostrom, E.".

This file answers "what is this person's identifier"; author_aliases.csv
answers "who is the same person". Keeping them apart means neither has to
adjudicate the other.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from citegraph.authors import load_external_ids, normalize_authors


def _refs(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("id")


def _one_author() -> pd.DataFrame:
    return _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Ostrom, E."], "Year": 1990},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Ostrom, Elinor"], "Year": 1992},
    ])


# ---------------------------------------------------------------------------
# load_external_ids
# ---------------------------------------------------------------------------


def test_load_external_ids_reads_both_columns(tmp_path: Path):
    p = tmp_path / "author_external_ids.csv"
    p.write_text(
        "author_id,openalex_id,orcid,note\n"
        "# curated 2026-09-13\n"
        "a-ostrom-elinor,A5003087402,,display name is Cyrillic\n"
        "a-someone,,0000-0002-1825-0097,\n",
        encoding="utf-8",
    )
    assert load_external_ids(p) == {
        "a-ostrom-elinor": {"openalex_id": "A5003087402", "orcid": None},
        "a-someone": {"openalex_id": None, "orcid": "0000-0002-1825-0097"},
    }


def test_load_external_ids_missing_file_is_empty(tmp_path: Path):
    assert load_external_ids(tmp_path / "absent.csv") == {}
    assert load_external_ids(None) == {}


def test_load_external_ids_rejects_a_row_with_no_identifier(tmp_path: Path):
    p = tmp_path / "x.csv"
    p.write_text("author_id,openalex_id,orcid\na-nobody,,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="a-nobody"):
        load_external_ids(p)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def test_curated_openalex_id_lands_on_the_author(tmp_path: Path):
    authors, _, _ = normalize_authors(works=_one_author())
    author_id = authors.index[0]
    assert pd.isna(authors.loc[author_id, "openalex_id"])

    curated, _, _ = normalize_authors(
        works=_one_author(),
        external_ids={author_id: {"openalex_id": "A5003087402", "orcid": None}},
    )
    assert curated.loc[author_id, "openalex_id"] == "A5003087402"


def test_curated_orcid_lands_on_the_author():
    authors, _, _ = normalize_authors(works=_one_author())
    author_id = authors.index[0]
    curated, _, _ = normalize_authors(
        works=_one_author(),
        external_ids={author_id: {"openalex_id": None, "orcid": "0000-0002-1825-0097"}},
    )
    assert curated.loc[author_id, "orcid"] == "0000-0002-1825-0097"


def test_unknown_author_id_is_rejected():
    with pytest.raises(ValueError, match="a-not-a-real-author"):
        normalize_authors(
            works=_one_author(),
            external_ids={"a-not-a-real-author": {"openalex_id": "A1", "orcid": None}},
        )


def test_curated_id_outranks_the_provider():
    """A human who checked the provider record is the stronger signal."""
    works = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Ostrom, E."], "Year": 1990},
    ])
    enriched = works.copy()
    enriched["OpenAlex_Authors"] = [
        [{"display_name": "Ostrom, E.", "family": None, "given": None,
          "openalex_id": "A-WRONG", "orcid": None}]
    ]
    plain, _, _ = normalize_authors(works=works, enriched_works=enriched)
    author_id = plain.index[0]
    assert plain.loc[author_id, "openalex_id"] == "A-WRONG"

    curated, _, _ = normalize_authors(
        works=works, enriched_works=enriched,
        external_ids={author_id: {"openalex_id": "A5003087402", "orcid": None}},
    )
    assert curated.loc[author_id, "openalex_id"] == "A5003087402"


def test_one_openalex_id_cannot_be_claimed_by_two_authors():
    works = _refs([
        {"id": "r-1", "Title": "T1", "Authors_List": ["Ostrom, Elinor"], "Year": 1990},
        {"id": "r-2", "Title": "T2", "Authors_List": ["Hardin, Garrett"], "Year": 1968},
    ])
    authors, _, _ = normalize_authors(works=works)
    a, b = list(authors.index)
    with pytest.raises(ValueError, match="A5003087402"):
        normalize_authors(
            works=works,
            external_ids={
                a: {"openalex_id": "A5003087402", "orcid": None},
                b: {"openalex_id": "A5003087402", "orcid": None},
            },
        )
