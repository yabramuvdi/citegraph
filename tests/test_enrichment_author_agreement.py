"""Tests for the author-agreement veto that gates enrichment matches.

Provider records for book reviews and edited-volume chapters carry the
reviewed work's title verbatim with the reviewer's name, so a title score
of 100 is not by itself evidence that two records are the same work.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click import unstyle
from typer.testing import CliRunner

from citegraph.authors import author_list_agreement
from citegraph.cli import _enrich_config, app
from citegraph.enrich import (
    EnrichConfig,
    _best_match_report,
    _valid_result,
    enrich_works,
)
from citegraph.io import OutLayout, fingerprint, metadata_fingerprint, write_json


def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# author_list_agreement
# ---------------------------------------------------------------------------


def test_shared_surname_agrees():
    assert author_list_agreement(["Bowles, S."], ["Samuel Bowles"]) is True


def test_disjoint_names_disagree():
    assert author_list_agreement(["Bowles, S."], ["david kandel"]) is False


def test_extraction_typo_still_agrees():
    # "Hirschmann, A." is how the bibliography spelled Albert O. Hirschman.
    assert author_list_agreement(["Hirschmann, A."], ["Albert O. Hirschman"]) is True


def test_short_tokens_need_exact_match():
    # Fuzz on 3-4 character tokens would pair unrelated people.
    assert author_list_agreement(["Lee, D."], ["Anna Loe"]) is False


def test_corporate_extracted_author_abstains():
    assert author_list_agreement(["The World Bank"], ["Heather Berkman"]) is None


def test_acronym_author_abstains():
    # IFPRI's provider record correctly lists the human chapter authors.
    assert author_list_agreement(["IFPRI"], ["Fredrick O. Wanyama"]) is None


def test_empty_side_abstains():
    assert author_list_agreement([], ["Samuel Bowles"]) is None
    assert author_list_agreement(["Bowles, S."], []) is None


def test_et_al_tail_is_not_a_name():
    # Two unrelated bibliographies both end in "et al."; that is not a shared
    # author. Guards against a laxer token filter pairing on "et"/"al".
    assert author_list_agreement(["Sampson et.al."], ["Kollock et al."]) is False
    assert author_list_agreement(["Sampson et.al."], ["Robert J. Sampson"]) is True


def test_compound_surname_agrees_across_orderings():
    assert author_list_agreement(
        ["Sánchez de la Sierra, Raúl"], ["Raúl Sánchez de la Sierra"]
    ) is True


# ---------------------------------------------------------------------------
# The veto inside candidate selection
# ---------------------------------------------------------------------------


def _openalex_item(title: str, authors: list[str], year: int, doi: str) -> dict:
    return {
        "title": title,
        "display_name": title,
        "authorships": [{"author": {"display_name": a}} for a in authors],
        "primary_location": {"source": {"display_name": "Some Journal"}},
        "publication_year": year,
        "doi": f"https://doi.org/{doi}",
    }


_CFG = EnrichConfig(title_match_threshold=90.0)


def test_exact_title_with_disjoint_authors_is_rejected():
    # OpenAlex holds a Croatian review of Governing the Commons under the
    # book's own title; accepting it overwrites Journal and Year too.
    review = _openalex_item(
        "Governing the Commons: The Evolution of Institutions for Collective Action",
        ["Marijana Grbeša", "Anamarija Musa"],
        1990,
        "10.0/review",
    )
    report = _best_match_report(
        [review],
        "Governing the Commons: The Evolution of Institutions for Collective Action",
        1990,
        _CFG,
        source="openalex",
        authors=["Ostrom, Elinor"],
    )
    assert report.match is None
    assert report.miss_reason == "author_mismatch"


def test_agreeing_authors_still_match():
    book = _openalex_item(
        "Governing the Commons: The Evolution of Institutions for Collective Action",
        ["Elinor Ostrom"],
        1990,
        "10.0/book",
    )
    report = _best_match_report(
        [book],
        "Governing the Commons: The Evolution of Institutions for Collective Action",
        1990,
        _CFG,
        source="openalex",
        authors=["Ostrom, Elinor"],
    )
    assert report.match is not None
    assert report.match["doi"] == "10.0/book"


def test_vetoed_candidate_falls_through_to_an_eligible_one():
    # Both candidates clear the title gate and the review scores higher; the
    # real book wins only because the veto rules the review out.
    title = "Power and Prosperity: Outgrowing Communist and Capitalist Dictatorships"
    review = _openalex_item(title, ["R. E. Elson"], 2000, "10.0/review")
    real = _openalex_item(title.replace(":", " :"), ["Mancur Olson"], 2000, "10.0/real")
    report = _best_match_report(
        [review, real], title, 2000, _CFG, source="openalex", authors=["Olson, M."],
    )
    assert report.match is not None
    assert report.match["doi"] == "10.0/real"


def test_no_candidates_is_not_reported_as_an_author_mismatch():
    report = _best_match_report(
        [], "Any Title", 2000, _CFG, source="openalex", authors=["Olson, M."],
    )
    assert report.miss_reason == "no_openalex_candidates"


def test_corporate_author_match_survives_the_veto():
    chapter = _openalex_item(
        "Collective action among African smallholders",
        ["Fredrick O. Wanyama"],
        2014,
        "10.0/chapter",
    )
    report = _best_match_report(
        [chapter], "Collective action among African smallholders", 2014, _CFG,
        source="openalex", authors=["IFPRI"],
    )
    assert report.match is not None


def test_author_agreement_is_recorded_as_a_diagnostic():
    book = _openalex_item("A Title", ["Samuel Bowles"], 2004, "10.0/x")
    report = _best_match_report(
        [book], "A Title", 2004, _CFG, source="openalex", authors=["Bowles, S."],
    )
    assert report.match["enrichment_author_agreement"] == "match"


# ---------------------------------------------------------------------------
# End to end through enrich_works
# ---------------------------------------------------------------------------


def _works_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "Title": "Governing the Commons",
            "Authors": "Ostrom, Elinor",
            "Authors_List": ["Ostrom, Elinor"],
            "Year": 1990,
        }],
        index=pd.Index(["w-ostrom-1990-governing-the-commons"], name="id"),
    )


def test_enrich_works_rejects_a_review_filed_under_the_book_title():
    review = _openalex_item("Governing the Commons", ["Marijana Grbeša"], 1990, "10.0/rev")
    client = MagicMock()
    client.get.side_effect = [
        _mock_response({"message": {"items": []}}),
        _mock_response({"results": [review]}),
    ]
    with patch("citegraph.enrich._try_import_httpx") as httpx:
        httpx.return_value.Client.return_value.__enter__.return_value = client
        out = enrich_works(_works_frame(), EnrichConfig())
    assert out.iloc[0]["enrichment_status"] == "miss"
    assert out.iloc[0]["enrichment_miss_reason"] == "author_mismatch"
    assert out.iloc[0]["doi"] is None
    # The original metadata must survive a rejected match.
    assert out.iloc[0]["Year"] == 1990


def test_enrich_works_keeps_a_match_whose_authors_agree():
    book = _openalex_item("Governing the Commons", ["Elinor Ostrom"], 1990, "10.0/book")
    client = MagicMock()
    client.get.side_effect = [
        _mock_response({"message": {"items": []}}),
        _mock_response({"results": [book]}),
    ]
    with patch("citegraph.enrich._try_import_httpx") as httpx:
        httpx.return_value.Client.return_value.__enter__.return_value = client
        out = enrich_works(_works_frame(), EnrichConfig())
    assert out.iloc[0]["enrichment_status"] == "matched"
    assert out.iloc[0]["enrichment_author_agreement"] == "match"


def test_legacy_cache_without_the_agreement_column_still_loads(tmp_path):
    layout = OutLayout(tmp_path)
    layout.enrichment_dir.mkdir(parents=True, exist_ok=True)
    frame = _works_frame()
    legacy = {
        "doi": "10.0/book", "Title": "Governing the Commons",
        "Authors": "Elinor Ostrom", "Authors_List": ["Elinor Ostrom"],
        "OpenAlex_Authors": [], "Journal": "X", "Year": 1990,
        "enrichment_source": "openalex", "enrichment_status": "matched",
        "enrichment_miss_reason": None,
    }
    write_json(
        layout.enrichment_dir / "w-ostrom-1990-governing-the-commons.json",
        {"schema_version": 2,
         "input_fingerprint": metadata_fingerprint(frame.iloc[0].to_dict()),
         "result_fingerprint": fingerprint(legacy),
         "result": legacy},
    )
    out = enrich_works(frame, EnrichConfig(), layout)
    assert out.iloc[0]["doi"] == "10.0/book"
    assert out.iloc[0]["enrichment_author_agreement"] is None


def test_misses_csv_carries_the_agreement_column(tmp_path):
    layout = OutLayout(tmp_path)
    layout.out_dir.mkdir(parents=True, exist_ok=True)
    review = _openalex_item("Governing the Commons", ["Marijana Grbeša"], 1990, "10.0/rev")
    client = MagicMock()
    client.get.side_effect = [
        _mock_response({"message": {"items": []}}),
        _mock_response({"results": [review]}),
    ]
    with patch("citegraph.enrich._try_import_httpx") as httpx:
        httpx.return_value.Client.return_value.__enter__.return_value = client
        enrich_works(_works_frame(), EnrichConfig(), layout)
    misses = pd.read_csv(layout.enrichment_misses_csv)
    assert misses.loc[0, "enrichment_miss_reason"] == "author_mismatch"
    assert misses.loc[0, "enrichment_author_agreement"] == "mismatch"


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_enrich_config_receives_the_author_fuzz_option():
    cfg = _enrich_config(
        contact="me@example.com",
        threshold=90.0,
        timeout=15.0,
        year_penalty=8.0,
        retry_attempts=3,
        retry_wait=0.25,
        max_workers=8,
        author_name_fuzz=92.0,
    )
    assert cfg.author_name_fuzz == 92.0


@pytest.mark.parametrize("command", ["run", "enrich"])
def test_author_fuzz_flag_is_documented(command):
    result = CliRunner().invoke(app, [command, "--help"], color=False, terminal_width=240)
    assert result.exit_code == 0, result.output
    assert "--enrich-author-fuzz" in unstyle(result.output)


def test_cache_with_a_corrupt_agreement_value_is_refreshed(tmp_path):
    layout = OutLayout(tmp_path)
    layout.enrichment_dir.mkdir(parents=True, exist_ok=True)
    frame = _works_frame()
    corrupt = {
        "doi": "10.0/book", "Title": "Governing the Commons",
        "Authors": "Elinor Ostrom", "Authors_List": ["Elinor Ostrom"],
        "OpenAlex_Authors": [], "Journal": "X", "Year": 1990,
        "enrichment_source": "openalex", "enrichment_status": "matched",
        "enrichment_miss_reason": None,
        "enrichment_author_agreement": {"not": "a label"},
    }
    write_json(
        layout.enrichment_dir / "w-ostrom-1990-governing-the-commons.json",
        {"schema_version": 2,
         "input_fingerprint": metadata_fingerprint(frame.iloc[0].to_dict()),
         "result_fingerprint": fingerprint(corrupt),
         "result": corrupt},
    )
    assert _valid_result(corrupt, strict=True) is False


def test_typographic_apostrophe_agrees_with_the_ascii_one():
    # OpenAlex spells apostrophes as U+2019, bibliographies as U+0027.
    assert author_list_agreement(["D'Adda, G."], ["Giovanna D’Adda"]) is True
    assert author_list_agreement(["O'Brien, M."], ["Mary O‘Brien"]) is True
