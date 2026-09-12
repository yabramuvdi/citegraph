"""Tests for the JSON repair fallback used when Gemini truncates output."""

from __future__ import annotations

import json
from types import SimpleNamespace

from citegraph.llm import fix_incomplete_json_string, parse_structured_response
from citegraph.schemas import PaperMetadata, Reference


def test_repairs_truncated_array():
    truncated = '[{"Title": "A", "Year": 2020}, {"Title": "B", "Yea'
    fixed = fix_incomplete_json_string(truncated)
    assert fixed is not None
    parsed = json.loads(fixed)
    assert parsed == [{"Title": "A", "Year": 2020}]


def test_returns_none_when_no_complete_object():
    assert fix_incomplete_json_string('[{"Title": "A", "Yea') is None
    assert fix_incomplete_json_string("") is None


def test_strips_surrounding_single_quotes():
    s = "'[{\"Title\": \"A\"}]'"
    fixed = fix_incomplete_json_string(s)
    assert fixed is not None
    assert json.loads(fixed) == [{"Title": "A"}]


def test_idempotent_on_already_valid_array():
    valid = '[{"Title": "A"}, {"Title": "B"}]'
    fixed = fix_incomplete_json_string(valid)
    assert fixed is not None
    assert json.loads(fixed) == [{"Title": "A"}, {"Title": "B"}]


def test_parsed_response_is_validated_through_schema():
    metadata = {"Title": "A", "Authors_List": [], "Journal": "", "Year": 2020}
    reference = {"Title": "B", "Authors_List": [], "Journal": "", "Year": 2019}

    parsed_metadata = parse_structured_response(
        SimpleNamespace(parsed=metadata), schema=PaperMetadata
    )
    parsed_references = parse_structured_response(
        SimpleNamespace(parsed=[reference]), schema=Reference
    )

    assert isinstance(parsed_metadata, PaperMetadata)
    assert isinstance(parsed_references[0], Reference)
