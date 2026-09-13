"""Incremental enrichment cache reuse without provider calls."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

import citegraph.enrich as enrich
from citegraph.enrich import EnrichConfig, enrich_works
from citegraph.io import OutLayout, metadata_fingerprint


def _works(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("id")


def _row(work_id: str, *, title: str = "Original title") -> dict:
    return {
        "id": work_id,
        "ring": 1,
        "source_file": "",
        "Title": title,
        "Authors": "Ada Lovelace",
        "Authors_List": ["Ada Lovelace"],
        "Journal": "Journal",
        "Year": 1843,
    }


def _result(*, doi: str = "10.example/original", author_id: str = "A123") -> dict:
    return {
        "doi": doi,
        "Title": "Original title",
        "Authors": "Ada Lovelace",
        "Authors_List": ["Ada Lovelace"],
        "OpenAlex_Authors": [
            {
                "display_name": "Ada Lovelace",
                "family": None,
                "given": None,
                "openalex_id": author_id,
                "orcid": None,
            }
        ],
        "Journal": "Journal",
        "Year": 1843,
        "enrichment_source": "openalex",
        "enrichment_status": "matched",
        "enrichment_miss_reason": None,
    }


def _write_v2(path: Path, source: dict, result: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "input_fingerprint": metadata_fingerprint(source),
                "result": result,
            }
        ),
        encoding="utf-8",
    )


def _load(works: pd.DataFrame, layout: OutLayout) -> pd.DataFrame:
    loader = getattr(enrich, "load_cached_enrichments", lambda *_a, **_k: pd.DataFrame())
    return loader(works, layout)


def test_new_enrichment_cache_detects_changed_identity_output(tmp_path):
    layout = OutLayout(tmp_path)
    layout.ensure()
    row = _row('w-original')
    path = layout.enrichment_dir / 'w-original.json'
    enrich._write_cache(path, _result(), metadata_fingerprint(row))
    saved = json.loads(path.read_text())
    saved['result']['OpenAlex_Authors'][0]['openalex_id'] = 'WRONG'
    path.write_text(json.dumps(saved))
    assert _load(_works(row), layout).empty


def test_legacy_permanent_miss_does_not_repeat_provider_lookup(tmp_path):
    layout = OutLayout(tmp_path)
    layout.ensure()
    (layout.enrichment_dir / 'w-original.json').write_text(
        json.dumps({'doi': None, 'enrichment_source': None}))
    with patch.object(enrich, '_try_import_httpx', side_effect=AssertionError('cached miss must be reused')):
        result = enrich_works(_works(_row('w-original')), layout=layout)
    assert result.loc['w-original', 'enrichment_status'] == 'miss'


def test_offline_loader_reuses_v2_cache_after_work_id_change(tmp_path: Path) -> None:
    layout = OutLayout(tmp_path)
    layout.ensure()
    old = _row("w-old-id")
    new = _row("w-new-id")
    _write_v2(layout.enrichment_dir / "w-old-id.json", old, _result())

    cached = _load(_works(new), layout)

    assert list(cached.index) == ["w-new-id"]
    assert cached.loc["w-new-id", "doi"] == "10.example/original"
    assert cached.loc["w-new-id", "OpenAlex_Authors"][0]["openalex_id"] == "A123"


def test_offline_loader_skips_changed_corrupt_and_transient_rows_only(
    tmp_path: Path,
) -> None:
    layout = OutLayout(tmp_path)
    layout.ensure()
    valid = _row("w-valid-new")
    changed = _row("w-changed-new", title="Changed title")
    transient = _row("w-transient-new", title="Transient title")
    invalid_output = _row("w-invalid-output-new", title="Invalid output")
    invalid_doi = _row("w-invalid-doi-new", title="Invalid DOI")
    incomplete = _row("w-incomplete-new", title="Incomplete")
    _write_v2(layout.enrichment_dir / "w-valid-old.json", valid, _result())
    _write_v2(layout.enrichment_dir / "w-changed-old.json", _row("w-x"), _result())
    _write_v2(
        layout.enrichment_dir / "w-transient-old.json",
        transient,
        {
            "doi": None,
            "enrichment_source": None,
            "enrichment_status": "miss",
            "enrichment_miss_reason": "http_error",
        },
    )
    _write_v2(
        layout.enrichment_dir / "w-invalid-output-old.json",
        invalid_output,
        {**_result(), "OpenAlex_Authors": "not-a-list"},
    )
    _write_v2(
        layout.enrichment_dir / "w-invalid-doi-old.json",
        invalid_doi,
        {**_result(), "doi": ["not", "a", "string"]},
    )
    _write_v2(
        layout.enrichment_dir / "w-incomplete-old.json",
        incomplete,
        {
            "enrichment_source": "openalex",
            "enrichment_status": "matched",
            "enrichment_miss_reason": None,
        },
    )
    (layout.enrichment_dir / "w-corrupt.json").write_text("not json", encoding="utf-8")

    cached = _load(
        _works(valid, changed, transient, invalid_output, invalid_doi, incomplete),
        layout,
    )

    assert list(cached.index) == ["w-valid-new"]


def test_offline_loader_does_not_choose_between_conflicting_cross_id_caches(
    tmp_path: Path,
) -> None:
    layout = OutLayout(tmp_path)
    layout.ensure()
    current = _row("w-current")
    _write_v2(layout.enrichment_dir / "w-old-a.json", current, _result(doi="10/a"))
    _write_v2(layout.enrichment_dir / "w-old-b.json", current, _result(doi="10/b"))

    cached = _load(_works(current), layout)

    assert cached.empty


def test_offline_loader_prefers_exact_valid_cache_over_conflicting_renamed_caches(
    tmp_path: Path,
) -> None:
    layout = OutLayout(tmp_path)
    layout.ensure()
    current = _row("w-current")
    _write_v2(layout.enrichment_dir / "w-current.json", current, _result(doi="10/current"))
    _write_v2(layout.enrichment_dir / "w-old.json", current, _result(doi="10/old"))

    cached = _load(_works(current), layout)

    assert cached.loc["w-current", "doi"] == "10/current"


def test_offline_loader_indexes_cache_directory_once(tmp_path: Path, monkeypatch) -> None:
    layout = OutLayout(tmp_path)
    layout.ensure()
    source = _row("w-old")
    _write_v2(layout.enrichment_dir / "w-old.json", source, _result())
    works = _works(*(_row(f"w-new-{number}") for number in range(20)))
    cache_path = layout.enrichment_dir / "w-old.json"
    original = Path.read_text
    reads = 0

    def counted_read(path: Path, *args, **kwargs):
        nonlocal reads
        if path == cache_path:
            reads += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted_read)

    cached = _load(works, layout)

    assert len(cached) == 20
    assert reads == 1


def test_enrich_works_reuses_renamed_v2_cache_without_httpx(tmp_path: Path) -> None:
    layout = OutLayout(tmp_path)
    layout.ensure()
    old = _row("w-old-id")
    new = _row("w-new-id")
    _write_v2(layout.enrichment_dir / "w-old-id.json", old, _result())

    with patch("citegraph.enrich._try_import_httpx") as import_httpx:
        import_httpx.return_value = MagicMock()
        enriched = enrich_works(_works(new), cfg=EnrichConfig(), layout=layout)

    import_httpx.assert_not_called()
    assert enriched.loc["w-new-id", "doi"] == "10.example/original"


def test_enrich_works_refreshes_when_metadata_changed(tmp_path: Path) -> None:
    layout = OutLayout(tmp_path)
    layout.ensure()
    original = _row("w-same-id")
    changed = _row("w-same-id", title="Changed title")
    _write_v2(layout.enrichment_dir / "w-same-id.json", original, _result())

    response = MagicMock()
    response.json.return_value = {"message": {"items": []}}
    response.raise_for_status.return_value = None
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    client.get.return_value = response

    with patch("citegraph.enrich._try_import_httpx") as import_httpx:
        import_httpx.return_value = MagicMock(Client=MagicMock(return_value=client))
        enriched = enrich_works(
            _works(changed),
            cfg=EnrichConfig(retry_wait_s=0),
            layout=layout,
        )

    client.get.assert_called()
    assert enriched.loc["w-same-id", "enrichment_status"] == "miss"
