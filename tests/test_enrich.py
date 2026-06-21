"""Tests for citegraph.enrich — all HTTP calls are mocked."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd

from citegraph.enrich import (
    EnrichConfig,
    _best_match,
    _crossref_lookup,
    _normalize_record,
    _openalex_lookup,
    enrich_references,
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


# ---------------------------------------------------------------------------
# _crossref_lookup
# ---------------------------------------------------------------------------

def test_crossref_happy_path():
    client = MagicMock()
    client.get.return_value = _mock_crossref_response([_CROSSREF_ITEM])
    result = _crossref_lookup("Attention Is All You Need", "Vaswani", 2017, _CFG, client)
    assert result is not None
    assert result["doi"] == "10.48550/arxiv.1706.03762"


def test_crossref_score_below_threshold():
    client = MagicMock()
    client.get.return_value = _mock_crossref_response([_CROSSREF_ITEM])
    result = _crossref_lookup("Totally Unrelated Work on Bananas", "", None, _CFG, client)
    assert result is None


def test_crossref_http_error():
    client = MagicMock()
    client.get.side_effect = Exception("connection refused")
    result = _crossref_lookup("Attention Is All You Need", "", None, _CFG, client)
    assert result is None


def test_crossref_empty_title_returns_none():
    client = MagicMock()
    result = _crossref_lookup("", "Author", 2020, _CFG, client)
    assert result is None
    client.get.assert_not_called()


# ---------------------------------------------------------------------------
# _openalex_lookup
# ---------------------------------------------------------------------------

def test_openalex_happy_path():
    client = MagicMock()
    client.get.return_value = _mock_openalex_response([_OPENALEX_ITEM])
    result = _openalex_lookup("Attention Is All You Need", 2017, _CFG, client)
    assert result is not None
    assert result["doi"] == "10.48550/arxiv.1706.03762"
    assert result["enrichment_source"] == "openalex"


def test_openalex_http_error():
    client = MagicMock()
    client.get.side_effect = Exception("timeout")
    result = _openalex_lookup("Some Title", None, _CFG, client)
    assert result is None


def test_openalex_empty_title_returns_none():
    client = MagicMock()
    result = _openalex_lookup("", None, _CFG, client)
    assert result is None
    client.get.assert_not_called()


# ---------------------------------------------------------------------------
# enrich_references — integration over a DataFrame
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


def test_enrich_references_crossref_match():
    df = _make_df()
    mock_client = MagicMock()
    mock_client.get.return_value = _mock_crossref_response([_CROSSREF_ITEM])
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client))
        result = enrich_references(df, cfg=_CFG)

    assert "doi" in result.columns
    assert result.iloc[0]["doi"] == "10.48550/arxiv.1706.03762"
    assert result.iloc[0]["enrichment_source"] == "crossref"
    assert result.iloc[0]["enrichment_title_score"] == 100.0
    assert result.iloc[0]["enrichment_candidate_title"] == "Attention Is All You Need"
    assert bool(result.iloc[0]["enrichment_year_match"]) is True


def test_enrich_references_openalex_fallback():
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
        result = enrich_references(df, cfg=_CFG)

    assert result.iloc[0]["enrichment_source"] == "openalex"


def test_enrich_references_no_match():
    """Both APIs return nothing; original row preserved with doi=None."""
    df = _make_df()
    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = _mock_crossref_response([])

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client))
        result = enrich_references(df, cfg=_CFG)

    assert result.iloc[0]["doi"] is None
    assert result.iloc[0]["Title"] == "Attention Is All You Need"
    assert result.iloc[0]["enrichment_status"] == "miss"
    assert result.iloc[0]["enrichment_miss_reason"] == "no_candidates"


def test_enrich_references_uses_cache(tmp_path):
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
        result = enrich_references(df, cfg=_CFG, layout=layout)

    mock_client.get.assert_not_called()
    assert result.iloc[0]["doi"] == "cached-doi"


def test_enrich_references_backfills_diagnostics_for_legacy_cache(tmp_path):
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
        result = enrich_references(df, cfg=_CFG, layout=layout)

    assert result.iloc[0]["enrichment_status"] == "matched"
    assert result.iloc[0]["enrichment_miss_reason"] is None
    summary = json.loads(layout.enrichment_summary_json.read_text())
    assert summary["n_matched"] == 1


def test_enrich_references_writes_cache(tmp_path):
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
        enrich_references(df, cfg=_CFG, layout=layout)

    cache_file = layout.enrichment_dir / "r-vaswani-2017-attention.json"
    assert cache_file.exists()
    cached = json.loads(cache_file.read_text())
    assert cached["doi"] == "10.48550/arxiv.1706.03762"
    assert cached["enrichment_status"] == "matched"
    assert cached["enrichment_title_score"] == 100.0


def test_enrich_references_caches_misses(tmp_path):
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
        first = enrich_references(df, cfg=_CFG, layout=layout)

    cache_file = layout.enrichment_dir / "r-vaswani-2017-attention.json"
    assert cache_file.exists()
    cached = json.loads(cache_file.read_text())
    assert cached["enrichment_status"] == "miss"
    assert cached["enrichment_miss_reason"] == "no_candidates"
    assert first.iloc[0]["enrichment_status"] == "miss"

    mock_client_2 = MagicMock()
    mock_client_2.__enter__ = lambda s: mock_client_2
    mock_client_2.__exit__ = MagicMock(return_value=False)

    with patch("citegraph.enrich._try_import_httpx") as mock_httpx:
        mock_httpx.return_value = MagicMock(Client=MagicMock(return_value=mock_client_2))
        second = enrich_references(df, cfg=_CFG, layout=layout)

    mock_client_2.get.assert_not_called()
    assert second.iloc[0]["enrichment_status"] == "miss"


def test_enrich_references_writes_misses_and_summary(tmp_path):
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
        enrich_references(df, cfg=_CFG, layout=layout)

    assert layout.enrichment_misses_csv.exists()
    misses = pd.read_csv(layout.enrichment_misses_csv)
    assert len(misses) == 1
    assert misses.iloc[0]["id"] == "r-vaswani-2017-attention"
    assert misses.iloc[0]["enrichment_miss_reason"] == "no_candidates"

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
