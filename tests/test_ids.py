"""Tests for stable id generation."""

from __future__ import annotations

from citegraph.ids import make_paper_id, make_reference_id, make_work_id


def test_paper_id_is_stable_across_call():
    a = make_paper_id("Ostrom, E.", 1990, "Governing the Commons")
    b = make_paper_id("Ostrom, E.", 1990, "Governing the Commons")
    assert a == b


def test_paper_id_uses_first_author_surname():
    pid = make_paper_id(["Elinor Ostrom", "Garrett Hardin"], 1990, "Some Title")
    assert pid.startswith("p-ostrom-1990-")


def test_paper_id_handles_missing_year():
    pid = make_paper_id("Smith, J.", None, "Some Paper")
    assert "nd" in pid


def test_reference_id_uses_r_prefix():
    rid = make_reference_id("Smith, J.", 2010, "Some Paper")
    assert rid.startswith("r-smith-2010-")


def test_paper_id_handles_empty_authors():
    pid = make_paper_id("", 2020, "Anonymous Work")
    assert "unknown" in pid


def test_make_work_id_prefix_and_determinism():
    a = make_work_id("Ostrom, E.", 1990, "Governing the Commons")
    b = make_work_id("Ostrom, E.", 1990, "Governing the Commons")
    assert a == "w-ostrom-1990-governing-the-commons"
    assert a == b


def test_make_work_id_same_slug_body_as_legacy_ids():
    # The enrichment-cache fallback renames r-<slug> -> w-<slug>, which only
    # works if the slug body is identical across prefixes.
    legacy = make_paper_id("Cárdenas, J.C.", 2000, "Real wealth and experimental cooperation")
    work = make_work_id("Cárdenas, J.C.", 2000, "Real wealth and experimental cooperation")
    assert work == "w-" + legacy.removeprefix("p-")
