"""Tests for the references-section slicer."""

from __future__ import annotations

import pytest

from citegraph.extract_references import slice_to_references_section


@pytest.mark.parametrize(
    "header",
    [
        "## References",
        "# References",
        "### Bibliography",
        "## REFERENCES",
        "## References and Notes",
        "## References Cited",
        "## Works Cited",
        "## Literature Cited",
        "## Cited Literature",
        "## 5. References",
        "## V. Bibliography",
    ],
)
def test_slicer_matches_common_header_forms(header: str) -> None:
    body = f"Some body text.\n\n{header}\n\n1. Smith, J. (2020). A paper.\n"
    sliced, found = slice_to_references_section(body)
    assert found is True
    assert sliced.startswith(header)
    assert "1. Smith" in sliced


def test_slicer_uses_last_match_when_multiple_present() -> None:
    body = (
        "## References to Foo\n"
        "Body mentioning prior references.\n"
        "## References\n"
        "1. Real reference.\n"
    )
    sliced, found = slice_to_references_section(body)
    assert found is True
    # The last (real) References header is what we slice from.
    assert sliced.startswith("## References\n")


def test_slicer_rejects_substantive_trailing_text() -> None:
    body = (
        "## References to Prior Work\n"
        "Discussion paragraph.\n"
        "## Methods\n"
        "Methods text.\n"
    )
    sliced, found = slice_to_references_section(body)
    assert found is False
    assert sliced == body  # unchanged fallback


def test_slicer_falls_back_when_no_header() -> None:
    body = "## Introduction\n\nSome content with no bibliography header.\n"
    sliced, found = slice_to_references_section(body)
    assert found is False
    assert sliced == body


def test_slicer_ignores_inline_mentions() -> None:
    body = "We refer the reader to the references [12] for details.\n"
    sliced, found = slice_to_references_section(body)
    assert found is False
    assert sliced == body
