"""Tests for citegraph.enrich — all HTTP calls are mocked."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from citegraph.enrich import (
    EnrichConfig,
    _best_match,
    _crossref_lookup,
    _normalize_record,
    _openalex_lookup,
    _openalex_lookup_report,
    enrich_works,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CROSSREF_ITEM = {
    "title": ["Attention Is All You Need"],
    "author": [{"given": "Ashish", "family": "Vaswani"}],
    "container-title": ["Advances in Neural Information Processing Systems"],
    "issued": {"date-parts": [[2017]]},
    "DOI": "10.48550/arxiv.1706.03762",
}

_OPENALEX_ITEM = {
    "title": "Attention Is All You Need",
    "display_name": "Attention Is All You Need",
    "authorships": [{"author": {"display_name": "Ashish Vaswani"}}],
    "primary_location": {"source": {"display_name": "NeurIPS"}},
    "publication_year": 2017,
    "doi": "https://doi.org/10.48550/arxiv.1706.03762",
}

_CFG = EnrichConfig(title_match_threshold=90.0)


def _mock_crossref_response(items: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"message": {"items": items}}
    resp.raise_for_status.return_value = None
    return resp


def _mock_openalex_response(results: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"results": results}
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# _normalize_record
# ---------------------------------------------------------------------------

def test_normalize_record_crossref():
    record = _normalize_record(_CROSSREF_ITEM, "crossref")
    assert record["doi"] == "10.48550/arxiv.1706.03762"
    assert record["Title"] == "Attention Is All You Need"
    assert record["Authors"] == "Ashish Vaswani"
    assert record["Journal"] == "Advances in Neural Information Processing Systems"
    assert record["Year"] == 2017
    assert record["enrichment_source"] == "crossref"


def test_normalize_record_openalex():
    record = _normalize_record(_OPENALEX_ITEM, "openalex")
    assert record["doi"] == "10.48550/arxiv.1706.03762"
    assert record["Title"] == "Attention Is All You Need"
    assert record["Authors"] == "Ashish Vaswani"
    assert record["Journal"] == "NeurIPS"
    assert record["Year"] == 2017
    assert record["enrichment_source"] == "openalex"


def test_normalize_record_crossref_keeps_family_and_given():
    """CrossRef's structured family/given split is an authoritative surname
    boundary — the author-normalization stage feeds it into its surname
    lexicon, so it must survive flattening."""
    record = _normalize_record(_CROSSREF_ITEM, "crossref")
    author = record["OpenAlex_Authors"][0]
    assert author["family"] == "Vaswani"
    assert author["given"] == "Ashish"


def test_normalize_record_openalex_allows_missing_source():
    item = {
        **_OPENALEX_ITEM,
        "primary_location": {"source": None},
    }

    record = _normalize_record(item, "openalex")

    assert record["Journal"] == ""
    assert record["enrichment_source"] == "openalex"


# ---------------------------------------------------------------------------
# _best_match
# ---------------------------------------------------------------------------

def test_best_match_above_threshold():
    result = _best_match([_CROSSREF_ITEM], "Attention Is All You Need", 2017, _CFG, source="crossref")
    assert result is not None
    assert result["doi"] == "10.48550/arxiv.1706.03762"


def test_best_match_below_threshold():
    result = _best_match([_CROSSREF_ITEM], "A Completely Different Paper Title", 2017, _CFG, source="crossref")
    assert result is None


def test_best_match_empty_items():
    assert _best_match([], "Any Title", 2017, _CFG, source="crossref") is None


def test_best_match_rejects_exact_title_when_year_penalty_drops_score():
    cfg = EnrichConfig(title_match_threshold=90.0, year_mismatch_penalty=15.0)
    wrong_year = {
        **_CROSSREF_ITEM,
        "issued": {"date-parts": [[2020]]},
    }

    result = _best_match(
        [wrong_year],
        "Attention Is All You Need",
        2017,
        cfg,
        source="crossref",
    )

    assert result is None


# ---------------------------------------------------------------------------
# _crossref_lookup
# ---------------------------------------------------------------------------

def test_crossref_happy_path():
    client = MagicMock()
    client.get.return_value = _mock_crossref_response([_CROSSREF_ITEM])
    result = _crossref_lookup("Attention Is All You Need", ["Vaswani"], 2017, _CFG, client)
    assert result is not None
    assert result["doi"] == "10.48550/arxiv.1706.03762"


def test_crossref_score_below_threshold():
    client = MagicMock()
    client.get.return_value = _mock_crossref_response([_CROSSREF_ITEM])
    result = _crossref_lookup("Totally Unrelated Work on Bananas", [], None, _CFG, client)
    assert result is None


def test_crossref_http_error():
    client = MagicMock()
    client.get.side_effect = Exception("connection refused")
    result = _crossref_lookup("Attention Is All You Need", [], None, _CFG, client)
    assert result is None


def test_crossref_empty_title_returns_none():
    client = MagicMock()
    result = _crossref_lookup("", ["Author"], 2020, _CFG, client)
    assert result is None
    client.get.assert_not_called()


def test_crossref_retries_transient_503_before_matching():
    cfg = EnrichConfig(retry_wait_s=0)
    transient = MagicMock()
    transient.status_code = 503
    success = _mock_crossref_response([_CROSSREF_ITEM])
    client = MagicMock()
    client.get.side_effect = [transient, success]

    result = _crossref_lookup("Attention Is All You Need", ["Vaswani"], 2017, cfg, client)

    assert result is not None
    assert result["doi"] == "10.48550/arxiv.1706.03762"
    assert client.get.call_count == 2


# ---------------------------------------------------------------------------
# _openalex_lookup
# ---------------------------------------------------------------------------

def test_openalex_happy_path():
    client = MagicMock()
    client.get.return_value = _mock_openalex_response([_OPENALEX_ITEM])
    result = _openalex_lookup("Attention Is All You Need", [], 2017, _CFG, client)
    assert result is not None
    assert result["doi"] == "10.48550/arxiv.1706.03762"
    assert result["enrichment_source"] == "openalex"


def test_openalex_http_error():
    client = MagicMock()
    client.get.side_effect = Exception("timeout")
    result = _openalex_lookup("Some Title", [], None, _CFG, client)
    assert result is None


def test_openalex_empty_title_returns_none():
    client = MagicMock()
    result = _openalex_lookup("", [], None, _CFG, client)
    assert result is None
    client.get.assert_not_called()


def test_openalex_sends_mailto_param_for_polite_pool():
    """OpenAlex's polite pool keys on a mailto query param, not the User-Agent."""
    client = MagicMock()
    client.get.return_value = _mock_openalex_response([_OPENALEX_ITEM])
    cfg = EnrichConfig(title_match_threshold=90.0, contact_email="who@example.org")
    _openalex_lookup("Attention Is All You Need", [], 2017, cfg, client)
    params = client.get.call_args.kwargs["params"]
    assert params["mailto"] == "who@example.org"


def test_openalex_omits_mailto_when_no_contact_email():
    client = MagicMock()
    client.get.return_value = _mock_openalex_response([_OPENALEX_ITEM])
    _openalex_lookup("Attention Is All You Need", [], 2017, _CFG, client)
    params = client.get.call_args.kwargs["params"]
    assert "mailto" not in params


def test_openalex_sends_api_key_param():
    """A configured OpenAlex API key rides along as the api_key query param."""
    client = MagicMock()
    client.get.return_value = _mock_openalex_response([_OPENALEX_ITEM])
    cfg = EnrichConfig(title_match_threshold=90.0, openalex_api_key="sk-test-123")
    _openalex_lookup("Attention Is All You Need", [], 2017, cfg, client)
    params = client.get.call_args.kwargs["params"]
    assert params["api_key"] == "sk-test-123"


def test_openalex_omits_api_key_when_unset():
    client = MagicMock()
    client.get.return_value = _mock_openalex_response([_OPENALEX_ITEM])
    _openalex_lookup("Attention Is All You Need", [], 2017, _CFG, client)
    params = client.get.call_args.kwargs["params"]
    assert "api_key" not in params


def test_openalex_strips_wildcard_chars_from_search():
    """OpenAlex reads ? and * as wildcards and 400s the whole query.

    Academic titles are full of question marks ("Is There a Role for Rural
    Communities?"), so leaving them in silently kills the OpenAlex fallback
    for those works.
    """
    client = MagicMock()
    client.get.return_value = _mock_openalex_response([_OPENALEX_ITEM])
    _openalex_lookup(
        "Halting Degradation: Is There a Role for Rural Communities?",
        [],
        1993,
        _CFG,
        client,
    )
    search = client.get.call_args.kwargs["params"]["search"]
    assert "?" not in search
    assert "*" not in search
    assert "Role for Rural Communities" in search


def test_openalex_wildcard_only_title_is_not_sent_as_empty_search():
    """A title that is nothing but wildcards must miss, not query for ''."""
    client = MagicMock()
    report = _openalex_lookup_report("???", [], None, _CFG, client)
    assert report.match is None
    assert report.miss_reason == "empty_title"
    client.get.assert_not_called()


def test_crossref_keeps_question_marks_in_title_query():
    """Only OpenAlex needs the stripping; CrossRef handles ? fine."""
    client = MagicMock()
    client.get.return_value = _mock_crossref_response([_CROSSREF_ITEM])
    _crossref_lookup("Is There a Role?", ["Ostrom"], 1993, _CFG, client)
    assert client.get.call_args.kwargs["params"]["query.title"] == "Is There a Role?"


def test_crossref_never_sends_openalex_api_key():
    """The OpenAlex key must not leak into CrossRef requests."""
    client = MagicMock()
    client.get.return_value = _mock_crossref_response([_CROSSREF_ITEM])
    cfg = EnrichConfig(title_match_threshold=90.0, openalex_api_key="sk-test-123")
    _crossref_lookup("Attention Is All You Need", ["Ashish Vaswani"], 2017, cfg, client)
    params = client.get.call_args.kwargs["params"]
    assert "api_key" not in params


# ---------------------------------------------------------------------------
# enrich_works — integration over a DataFrame
# ---------------------------------------------------------------------------

def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "r-vaswani-2017-attention",
                "Title": "Attention Is All You Need",
                "Authors": "Ashish Vaswani",
                "Authors_List": ["Ashish Vaswani"],
                "Journal": "",
                "Year": 2017,
            }
        ]
    ).set_index("id")


def test_enrich_works_nan_title_short_circuits_as_empty():
    """A NaN title must not stringify to 'nan' and get sent as a search query."""
    df = pd.DataFrame(
        [
            {
                "id": "w-fehr-2003-untitled",
                "Title": float("nan"),
                "Authors": float("nan"),
                "Authors_List": None,
                "Journal": "Nature",
                "Year": 2003,
            }
        ]
    ).set_index("id")

    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client))
        result = enrich_works(df, cfg=_CFG)

    mock_client.get.assert_not_called()
    row = result.iloc[0]
    assert row["enrichment_status"] == "miss"
    assert row["enrichment_miss_reason"] == "empty_title"


def test_enrich_works_crossref_match():
    df = _make_df()
    mock_client = MagicMock()
    mock_client.get.return_value = _mock_crossref_response([_CROSSREF_ITEM])
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client))
        result = enrich_works(df, cfg=_CFG)

    assert "doi" in result.columns
    assert result.iloc[0]["doi"] == "10.48550/arxiv.1706.03762"
    assert result.iloc[0]["enrichment_source"] == "crossref"
    assert result.iloc[0]["enrichment_title_score"] == 100.0
    assert result.iloc[0]["enrichment_candidate_title"] == "Attention Is All You Need"
    assert bool(result.iloc[0]["enrichment_year_match"]) is True


def test_enrich_works_openalex_fallback():
    """CrossRef returns no match; OpenAlex fallback succeeds."""
    df = _make_df()
    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)

    crossref_resp = _mock_crossref_response([])  # no items → no match
    openalex_resp = _mock_openalex_response([_OPENALEX_ITEM])
    mock_client.get.side_effect = [crossref_resp, openalex_resp]

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client))
        result = enrich_works(df, cfg=_CFG)

    assert result.iloc[0]["enrichment_source"] == "openalex"


def test_enrich_works_no_match():
    """Both APIs return nothing; original row preserved with doi=None."""
    df = _make_df()
    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = _mock_crossref_response([])

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client))
        result = enrich_works(df, cfg=_CFG)

    assert result.iloc[0]["doi"] is None
    assert result.iloc[0]["Title"] == "Attention Is All You Need"
    assert result.iloc[0]["enrichment_status"] == "miss"
    assert result.iloc[0]["enrichment_miss_reason"] == "no_openalex_candidates"


def test_enrich_works_reports_year_mismatch_miss():
    df = _make_df()
    wrong_year = {
        **_CROSSREF_ITEM,
        "issued": {"date-parts": [[2020]]},
    }
    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.side_effect = [
        _mock_crossref_response([wrong_year]),
        _mock_openalex_response([]),
    ]
    cfg = EnrichConfig(title_match_threshold=90.0, year_mismatch_penalty=15.0)

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client))
        result = enrich_works(df, cfg=cfg)

    row = result.iloc[0]
    assert row["enrichment_status"] == "miss"
    assert row["enrichment_miss_reason"] == "year_mismatch"
    assert row["enrichment_title_score"] == 100.0
    assert row["enrichment_adjusted_score"] == 85.0
    assert row["enrichment_year_delta"] == 3
    assert row["enrichment_candidate_title"] == "Attention Is All You Need"


def test_enrich_works_reports_below_threshold_miss():
    df = _make_df()
    unrelated = {
        **_CROSSREF_ITEM,
        "title": ["Totally Unrelated Work on Bananas"],
    }
    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.side_effect = [
        _mock_crossref_response([unrelated]),
        _mock_openalex_response([]),
    ]

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client))
        result = enrich_works(df, cfg=_CFG)

    row = result.iloc[0]
    assert row["enrichment_status"] == "miss"
    assert row["enrichment_miss_reason"] == "below_title_threshold"
    assert row["enrichment_candidate_title"] == "Totally Unrelated Work on Bananas"
    assert row["enrichment_title_score"] < _CFG.title_match_threshold


def test_enrich_works_reports_http_error_miss():
    df = _make_df()
    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.side_effect = Exception("connection refused")

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client))
        result = enrich_works(df, cfg=_CFG)

    row = result.iloc[0]
    assert row["enrichment_status"] == "miss"
    assert row["enrichment_miss_reason"] == "http_error"


def test_enrich_works_uses_cache(tmp_path):
    """A cached result is returned without hitting the API."""
    df = _make_df()
    enrichment_dir = tmp_path / "enrichment"
    enrichment_dir.mkdir()

    cached = {
        "doi": "cached-doi",
        "Title": "Attention Is All You Need",
        "Authors_List": ["Ashish Vaswani"],
        "Authors": "Ashish Vaswani",
        "Journal": "NeurIPS",
        "Year": 2017,
        "enrichment_source": "crossref",
    }
    (enrichment_dir / "r-vaswani-2017-attention.json").write_text(
        json.dumps(cached), encoding="utf-8"
    )

    from citegraph.io import OutLayout
    layout = OutLayout(tmp_path)

    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client))
        result = enrich_works(df, cfg=_CFG, layout=layout)

    mock_client.get.assert_not_called()
    assert result.iloc[0]["doi"] == "cached-doi"


@pytest.mark.parametrize(
    "payload",
    ["not json", "[]", '{"schema_version": 2, "result": []}'],
)
def test_enrich_works_refreshes_malformed_cache(tmp_path, payload):
    df = _make_df()

    from citegraph.io import OutLayout
    layout = OutLayout(tmp_path)
    layout.ensure()
    cache_file = layout.enrichment_dir / "r-vaswani-2017-attention.json"
    cache_file.write_text(payload, encoding="utf-8")

    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = _mock_crossref_response([_CROSSREF_ITEM])

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client))
        result = enrich_works(df, cfg=_CFG, layout=layout)

    mock_client.get.assert_called()
    assert result.iloc[0]["doi"] == "10.48550/arxiv.1706.03762"
    assert json.loads(cache_file.read_text())["result"]["enrichment_status"] == "matched"


def test_enrich_works_backfills_diagnostics_for_legacy_cache(tmp_path):
    """Old cache files without diagnostic columns still produce matched rows."""
    df = _make_df()
    enrichment_dir = tmp_path / "enrichment"
    enrichment_dir.mkdir()

    cached = {
        "doi": "cached-doi",
        "Title": "Attention Is All You Need",
        "Authors_List": ["Ashish Vaswani"],
        "Authors": "Ashish Vaswani",
        "Journal": "NeurIPS",
        "Year": 2017,
        "enrichment_source": "crossref",
    }
    (enrichment_dir / "r-vaswani-2017-attention.json").write_text(
        json.dumps(cached), encoding="utf-8"
    )

    from citegraph.io import OutLayout
    layout = OutLayout(tmp_path)

    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client))
        result = enrich_works(df, cfg=_CFG, layout=layout)

    assert result.iloc[0]["enrichment_status"] == "matched"
    assert result.iloc[0]["enrichment_miss_reason"] is None
    summary = json.loads(layout.enrichment_summary_json.read_text())
    assert summary["n_matched"] == 1


def test_summary_redacts_openalex_api_key(tmp_path):
    """The API key must never be written to enrichment_summary.json."""
    df = _make_df()
    enrichment_dir = tmp_path / "enrichment"
    enrichment_dir.mkdir()

    cached = {
        "doi": "cached-doi",
        "Title": "Attention Is All You Need",
        "Authors_List": ["Ashish Vaswani"],
        "Authors": "Ashish Vaswani",
        "Journal": "NeurIPS",
        "Year": 2017,
        "enrichment_source": "crossref",
    }
    (enrichment_dir / "r-vaswani-2017-attention.json").write_text(
        json.dumps(cached), encoding="utf-8"
    )

    from citegraph.io import OutLayout
    layout = OutLayout(tmp_path)

    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)

    cfg = EnrichConfig(title_match_threshold=90.0, openalex_api_key="sk-secret-456")
    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client))
        enrich_works(df, cfg=cfg, layout=layout)

    raw = layout.enrichment_summary_json.read_text()
    assert "sk-secret-456" not in raw
    summary = json.loads(raw)
    assert summary["config"]["openalex_api_key"] == "***"


def test_summary_shows_empty_api_key_when_unset(tmp_path):
    """No key configured -> the summary records an empty string, not a mask."""
    df = _make_df()
    enrichment_dir = tmp_path / "enrichment"
    enrichment_dir.mkdir()

    cached = {
        "doi": "cached-doi",
        "Title": "Attention Is All You Need",
        "Authors_List": ["Ashish Vaswani"],
        "Authors": "Ashish Vaswani",
        "Journal": "NeurIPS",
        "Year": 2017,
        "enrichment_source": "crossref",
    }
    (enrichment_dir / "r-vaswani-2017-attention.json").write_text(
        json.dumps(cached), encoding="utf-8"
    )

    from citegraph.io import OutLayout
    layout = OutLayout(tmp_path)

    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client))
        enrich_works(df, cfg=_CFG, layout=layout)

    summary = json.loads(layout.enrichment_summary_json.read_text())
    assert summary["config"]["openalex_api_key"] == ""


def test_enrich_works_writes_cache(tmp_path):
    """A successful API result is persisted to the cache directory."""
    df = _make_df()

    from citegraph.io import OutLayout
    layout = OutLayout(tmp_path)
    layout.ensure()

    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = _mock_crossref_response([_CROSSREF_ITEM])

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client))
        enrich_works(df, cfg=_CFG, layout=layout)

    cache_file = layout.enrichment_dir / "r-vaswani-2017-attention.json"
    assert cache_file.exists()
    cached = json.loads(cache_file.read_text())["result"]
    assert cached["doi"] == "10.48550/arxiv.1706.03762"
    assert cached["enrichment_status"] == "matched"
    assert cached["enrichment_title_score"] == 100.0


def test_enrich_works_caches_misses(tmp_path):
    """No-match rows are cached so reruns do not repeatedly hit the APIs."""
    df = _make_df()

    from citegraph.io import OutLayout
    layout = OutLayout(tmp_path)
    layout.ensure()

    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = _mock_crossref_response([])

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client))
        first = enrich_works(df, cfg=_CFG, layout=layout)

    cache_file = layout.enrichment_dir / "r-vaswani-2017-attention.json"
    assert cache_file.exists()
    cached = json.loads(cache_file.read_text())["result"]
    assert cached["enrichment_status"] == "miss"
    assert cached["enrichment_miss_reason"] == "no_openalex_candidates"
    assert first.iloc[0]["enrichment_status"] == "miss"

    mock_client_2 = MagicMock()
    mock_client_2.__enter__ = lambda s: mock_client_2
    mock_client_2.__exit__ = MagicMock(return_value=False)

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client_2))
        second = enrich_works(df, cfg=_CFG, layout=layout)

    mock_client_2.get.assert_not_called()
    assert second.iloc[0]["enrichment_status"] == "miss"


def test_enrich_works_retries_cached_http_error_miss(tmp_path):
    """A cached http_error miss is a transient outage, not a result — re-runs retry it.

    Real-corpus case: an anonymous, heavily-throttled run cached thousands
    of 429s as permanent misses that a plain re-run would never retry.
    """
    df = _make_df()

    from citegraph.io import OutLayout
    layout = OutLayout(tmp_path)
    layout.ensure()

    poisoned = {
        "doi": None,
        "enrichment_source": None,
        "enrichment_status": "miss",
        "enrichment_miss_reason": "http_error",
    }
    cache_file = layout.enrichment_dir / "r-vaswani-2017-attention.json"
    cache_file.write_text(json.dumps(poisoned), encoding="utf-8")

    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = _mock_crossref_response([_CROSSREF_ITEM])

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client))
        result = enrich_works(df, cfg=_CFG, layout=layout)

    mock_client.get.assert_called()
    row = result.iloc[0]
    assert row["enrichment_status"] == "matched"
    assert row["doi"] == "10.48550/arxiv.1706.03762"
    cached = json.loads(cache_file.read_text())["result"]
    assert cached["enrichment_status"] == "matched"


def test_enrich_works_writes_misses_and_summary(tmp_path):
    df = _make_df()

    from citegraph.io import OutLayout
    layout = OutLayout(tmp_path)
    layout.ensure()

    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = _mock_crossref_response([])

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client))
        enrich_works(df, cfg=_CFG, layout=layout)

    assert layout.enrichment_misses_csv.exists()
    misses = pd.read_csv(layout.enrichment_misses_csv)
    assert len(misses) == 1
    assert misses.iloc[0]["id"] == "r-vaswani-2017-attention"
    assert misses.iloc[0]["enrichment_miss_reason"] == "no_openalex_candidates"

    assert layout.enrichment_summary_json.exists()
    summary = json.loads(layout.enrichment_summary_json.read_text())
    assert summary["n_references"] == 1
    assert summary["n_matched"] == 0
    assert summary["n_missed"] == 1
    assert summary["config"]["title_match_threshold"] == 90.0


# ---------------------------------------------------------------------------
# EnrichConfig
# ---------------------------------------------------------------------------

def test_enrich_config_user_agent_with_email():
    cfg = EnrichConfig(contact_email="test@example.com")
    assert "mailto:test@example.com" in cfg.user_agent


def test_enrich_config_user_agent_no_email():
    cfg = EnrichConfig()
    assert cfg.user_agent == "citegraph/0.1"


# ---------------------------------------------------------------------------
# works-model behaviors: legacy cache fallback + per-ring summary
# ---------------------------------------------------------------------------
def test_enrich_one_falls_back_to_legacy_cache_filename(tmp_path):
    """A w-<slug> lookup must serve a pre-works-model r-<slug> cache file."""
    from citegraph.enrich import EnrichConfig, _enrich_one

    legacy = {
        "doi": "10.1/x",
        "enrichment_source": "crossref",
        "enrichment_status": "matched",
        "Title": "Cached",
    }
    (tmp_path / "r-ostrom-1990-governing.json").write_text(json.dumps(legacy))
    row = pd.Series({"Title": "Governing", "Authors": "Ostrom", "Year": 1990})

    out = _enrich_one(
        "w-ostrom-1990-governing", row, EnrichConfig(), None, tmp_path
    )
    assert out["doi"] == "10.1/x"  # served from the legacy r- cache, no network


def test_enrichment_summary_breaks_down_by_ring(tmp_path):
    from citegraph.enrich import enrich_works
    from citegraph.io import OutLayout

    df = pd.DataFrame(
        [
            {
                "id": "w-doe-2021-core-paper",
                "ring": 0,
                "Title": "Core Paper",
                "Authors": "Doe, J.",
                "Authors_List": ["Doe, J."],
                "Journal": "J",
                "Year": 2021,
            },
            {
                "id": "w-vaswani-2017-attention",
                "ring": 1,
                "Title": "Attention Is All You Need",
                "Authors": "Ashish Vaswani",
                "Authors_List": ["Ashish Vaswani"],
                "Journal": "",
                "Year": 2017,
            },
        ]
    ).set_index("id")

    layout = OutLayout(tmp_path)
    layout.ensure()

    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = _mock_crossref_response([])

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client))
        enrich_works(df, cfg=_CFG, layout=layout)

    summary = json.loads(layout.enrichment_summary_json.read_text())
    assert summary["by_ring"] == {
        "0": {"n": 1, "n_matched": 0},
        "1": {"n": 1, "n_matched": 0},
    }


def test_normalize_record_unescapes_html_entities():
    """CrossRef returns container titles with raw HTML entities.

    80 works in one corpus carried "Journal of Economic Behavior &amp;
    Organization" as their canonical journal name.
    """
    item = {
        "title": ["Communication, Commitment &amp; Trust"],
        "author": [{"given": "A", "family": "Smith &amp; Co"}],
        "container-title": ["Journal of Economic Behavior &amp; Organization"],
        "issued": {"date-parts": [[2004]]},
        "DOI": "10.0/x",
    }
    rec = _normalize_record(item, "crossref")
    assert rec["Journal"] == "Journal of Economic Behavior & Organization"
    assert rec["Title"] == "Communication, Commitment & Trust"
    assert rec["Authors_List"] == ["A Smith & Co"]


def test_normalize_record_unescapes_openalex_entities():
    item = {
        "title": "Group Processes &amp; Intergroup Relations",
        "authorships": [{"author": {"display_name": "R &amp; B"}}],
        "primary_location": {"source": {"display_name": "Nature &amp; Science"}},
        "publication_year": 2010,
    }
    rec = _normalize_record(item, "openalex")
    assert rec["Title"] == "Group Processes & Intergroup Relations"
    assert rec["Journal"] == "Nature & Science"
    assert rec["Authors_List"] == ["R & B"]
