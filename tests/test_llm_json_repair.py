"""Tests for the JSON repair fallback used when Gemini truncates output."""

from __future__ import annotations

import json

from citegraph.llm import fix_incomplete_json_string


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
