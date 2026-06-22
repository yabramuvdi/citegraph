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
