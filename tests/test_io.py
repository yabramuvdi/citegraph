"""Tests for artifact and structured-column I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from citegraph.io import (
    OutLayout,
    parse_authors_list,
    parse_openalex_authors,
    require_columns,
    serialize_structured,
)


def test_parse_authors_list_accepts_lists_repr_and_joined_strings() -> None:
    assert parse_authors_list(["Ada Lovelace", "Grace Hopper"]) == [
        "Ada Lovelace",
        "Grace Hopper",
    ]
    assert parse_authors_list("['Ada Lovelace', 'Grace Hopper']") == [
        "Ada Lovelace",
        "Grace Hopper",
    ]
    assert parse_authors_list("Ada Lovelace; Grace Hopper") == [
        "Ada Lovelace",
        "Grace Hopper",
    ]


def test_openalex_authors_round_trip_as_json() -> None:
    authors = [
        {"display_name": "Ada Lovelace", "openalex_id": "A1", "orcid": None},
        {"display_name": "Grace Hopper", "openalex_id": "A2", "orcid": "0000"},
    ]

    encoded = serialize_structured(authors)

    assert json.loads(encoded)[0]["display_name"] == "Ada Lovelace"
    assert parse_openalex_authors(encoded) == authors


def test_require_columns_reports_missing_columns() -> None:
    df = pd.DataFrame([{"Title": "A"}])

    with pytest.raises(ValueError, match="dedup input.*missing required column.*citing_id"):
        require_columns(df, ["Title", "citing_id"], artifact="dedup input")


def test_out_layout_exposes_artifact_manifest_path(tmp_path: Path) -> None:
    layout = OutLayout(tmp_path / "out")

    assert layout.artifact_manifest_json == tmp_path / "out" / "artifact_manifest.json"


def test_outlayout_works_model_paths(tmp_path: Path) -> None:
    layout = OutLayout(tmp_path)

    assert layout.sources_csv == tmp_path / "sources.csv"
    assert layout.citations_raw_csv == tmp_path / "citations_raw.csv"
    assert layout.works_csv == tmp_path / "works.csv"
    assert layout.enriched_works_csv == tmp_path / "enriched_works.csv"
def test_empty_snapshot_fingerprints_include_schema():
    from citegraph.io import frame_fingerprint
    assert frame_fingerprint(pd.DataFrame(columns=["citing_id", "cited_id"])) != frame_fingerprint(
        pd.DataFrame(columns=["wrong"]))


def test_interrupted_atomic_json_replace_preserves_previous_file(tmp_path, monkeypatch):
    from citegraph.io import write_json
    path = tmp_path / "audit.json"
    write_json(path, {"before": True})
    previous = path.read_bytes()
    def fail_replace(*args):
        raise OSError("interrupted replacement")
    monkeypatch.setattr("citegraph.io.os.replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        write_json(path, {"after": True})
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]
