"""Schema validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from citegraph.schemas import PaperMetadata, Reference


def test_paper_metadata_valid():
    meta = PaperMetadata(
        Title="Governing the Commons",
        Authors_List=["Elinor Ostrom"],
        Authors="Ostrom, E.",
        Journal="Cambridge University Press",
        Year=1990,
    )
    assert meta.Year == 1990
    assert meta.Authors_List == ["Elinor Ostrom"]


def test_paper_metadata_rejects_missing_field():
    with pytest.raises(ValidationError):
        PaperMetadata(Title="A", Authors_List=["B"], Authors="B", Journal="J")  # missing Year


def test_reference_coerces_year_from_string():
    ref = Reference(
        Title="A",
        Authors_List=["B"],
        Authors="B",
        Journal="J",
        Year="2020",  # type: ignore[arg-type]
    )
    assert ref.Year == 2020


def test_reference_extra_fields_ignored():
    ref = Reference.model_validate(
        {
            "Title": "A",
            "Authors_List": ["B"],
            "Authors": "B",
            "Journal": "J",
            "Year": 2020,
            "doi": "10.1/abc",
            "noise": "...",
        }
    )
    assert not hasattr(ref, "doi")


def test_authors_is_derived_from_list():
    """`Authors` is no longer in the LLM contract — it's a derived property."""
    meta = PaperMetadata(
        Title="A",
        Authors_List=["Jane Q. Doe", "John A. Smith"],
        Journal="J",
        Year=2020,
    )
    assert meta.Authors == "Jane Q. Doe, John A. Smith"


def test_authors_kwarg_silently_ignored():
    """A legacy caller passing Authors=... should not break — the kwarg is dropped and the value re-derived."""
    meta = PaperMetadata(
        Title="A",
        Authors_List=["Jane Doe"],
        Authors="this should be ignored",  # type: ignore[call-arg]
        Journal="J",
        Year=2020,
    )
    assert meta.Authors == "Jane Doe"


def test_model_dump_includes_derived_authors():
    """CSV writers serialize via model_dump and need the Authors column."""
    meta = PaperMetadata(
        Title="A",
        Authors_List=["Jane Doe", "John Smith"],
        Journal="J",
        Year=2020,
    )
    dumped = meta.model_dump()
    assert dumped["Authors"] == "Jane Doe, John Smith"
    assert dumped["Authors_List"] == ["Jane Doe", "John Smith"]


def test_old_cache_with_authors_field_still_loads():
    """An existing cache file written under the old schema should round-trip cleanly."""
    legacy = {
        "Title": "A",
        "Authors_List": ["Jane Doe"],
        "Authors": "Doe, J.",  # original LLM-formatted string from the old cache
        "Journal": "J",
        "Year": 2020,
    }
    meta = PaperMetadata.model_validate(legacy)
    # Authors is recomputed from the list, not taken from the legacy string.
    assert meta.Authors == "Jane Doe"
    # And re-dumping uses the derived value.
    assert meta.model_dump()["Authors"] == "Jane Doe"
