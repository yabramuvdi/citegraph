"""Schema validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from citegraph.schemas import PaperMetadata, Reference


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


def test_metadata_derives_and_serializes_authors():
    """The public property and CSV/cache serialization use the same author list."""
    meta = PaperMetadata(
        Title="A",
        Authors_List=["Jane Q. Doe", "John A. Smith"],
        Journal="J",
        Year=2020,
    )
    dumped = meta.model_dump()
    assert meta.Authors == dumped["Authors"] == "Jane Q. Doe, John A. Smith"
    assert dumped["Authors_List"] == ["Jane Q. Doe", "John A. Smith"]


def test_old_cache_with_authors_field_still_loads():
    """An existing cache file written under the old schema should round-trip cleanly."""
    legacy = {
        "Title": "A",
        "Authors_List": ["Jane Doe"],
        "Authors": "Doe, J.",  # original LLM-formatted string from the old cache
        "Journal": "J",
        "Year": 2020,
    }
    for meta in (PaperMetadata(**legacy), PaperMetadata.model_validate(legacy)):
        assert meta.Authors == "Jane Doe"
        assert meta.model_dump()["Authors"] == "Jane Doe"
