"""Tests for stable id generation."""

from __future__ import annotations

from citegraph.ids import make_work_id


def test_work_id_is_stable_across_calls():
    a = make_work_id("Ostrom, E.", 1990, "Governing the Commons")
    b = make_work_id("Ostrom, E.", 1990, "Governing the Commons")
    assert a == "w-ostrom-1990-governing-the-commons"
    assert a == b


def test_work_id_uses_first_author_surname():
    wid = make_work_id(["Elinor Ostrom", "Garrett Hardin"], 1990, "Some Title")
    assert wid.startswith("w-ostrom-1990-")


def test_work_id_handles_missing_year():
    wid = make_work_id("Smith, J.", None, "Some Paper")
    assert "nd" in wid


def test_work_id_handles_empty_authors():
    wid = make_work_id("", 2020, "Anonymous Work")
    assert "unknown" in wid


def test_work_id_same_slug_body_as_legacy_ids():
    # The enrichment-cache fallback renames r-<slug> -> w-<slug>, which only
    # works if the slug body is identical to what the legacy prefixed id
    # builders produced. The expected string below is the historical output
    # of make_reference_id(...) with its r- prefix swapped for w-.
    work = make_work_id("Cárdenas, J.C.", 2000, "Real wealth and experimental cooperation")
    assert work == "w-cárdenas-2000-real-wealth-and-experimental-cooperation"
