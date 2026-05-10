"""Tests for the inline-citation sanity check."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from citegraph.extract_references import (
    _INLINE_CITATION_MIN_ESTIMATE,
    estimate_inline_citations,
    extract_references_from_markdown,
    split_at_references_header,
)
from citegraph.llm import GeminiClient
from citegraph.schemas import Reference


# ---------------------------------------------------------------------------
# split_at_references_header
# ---------------------------------------------------------------------------
def test_split_returns_body_and_refs() -> None:
    text = "Body before refs.\n\n## References\n1. A paper.\n"
    body, refs = split_at_references_header(text)
    assert body == "Body before refs.\n\n"
    assert refs is not None and refs.startswith("## References")


def test_split_returns_none_refs_when_no_header() -> None:
    text = "Just a body, no bibliography.\n"
    body, refs = split_at_references_header(text)
    assert body == text
    assert refs is None


# ---------------------------------------------------------------------------
# estimate_inline_citations
# ---------------------------------------------------------------------------
def test_estimate_handles_empty_body() -> None:
    assert estimate_inline_citations("") == 0


def test_estimate_counts_unique_numeric_refs() -> None:
    body = "We use [1] and combine it with [2, 3]. Later we revisit [1] and [3]."
    # Unique numbers: {1, 2, 3} -> 3
    assert estimate_inline_citations(body) == 3


def test_estimate_handles_numeric_ranges_as_endpoints() -> None:
    """Ranges like [12-15] are counted by endpoint only — fine for a floor estimate."""
    body = "See [12-15] for details."
    # Captures 12 and 15.
    assert estimate_inline_citations(body) == 2


def test_estimate_counts_parenthetical_author_year() -> None:
    body = (
        "Foundational work was done by (Smith, 2019). Later, (Jones and Lee, 2020) "
        "extended it. Building on (Smith et al., 2019), we ..."
    )
    # Tokens: ("smith", 2019), ("jones and lee", 2020), ("smith et al.", 2019).
    assert estimate_inline_citations(body) == 3


def test_estimate_counts_narrative_author_year() -> None:
    body = "Smith (2019) showed X. Jones and Lee (2020) extended this."
    # Tokens: ("smith", 2019), ("jones and lee", 2020).
    assert estimate_inline_citations(body) == 2


def test_estimate_dedupes_repeated_citations() -> None:
    body = "(Smith, 2019) is cited many times. As (Smith, 2019) explains..."
    assert estimate_inline_citations(body) == 1


def test_estimate_ignores_non_year_parens() -> None:
    """Things like Equation (3.2) or (12) without an author shouldn't count."""
    body = "See Equation (3.2) and Section (4) for details. Also try (12)."
    assert estimate_inline_citations(body) == 0


# ---------------------------------------------------------------------------
# Integration: the warning fires from extract_references_from_markdown
# ---------------------------------------------------------------------------
class _ScriptedClient(GeminiClient):
    """Returns a fixed list of references regardless of input."""

    def __init__(self, refs):
        self._client = object()
        self.model = "fake-model"
        from citegraph.config import Settings

        self._settings = Settings(GOOGLE_API_KEY="fake")  # type: ignore[arg-type]
        self._api_key = "fake"
        self._refs = refs

    def generate_structured(
        self, *, prompt, response_schema, system_instruction=None, max_output_tokens=None
    ):
        return SimpleNamespace(parsed=self._refs, text="", candidates=[])


def _make_paper(tmp_path: Path, body: str, refs_section: str) -> Path:
    p = tmp_path / "paper.md"
    p.write_text(f"{body}\n\n## References\n{refs_section}")
    return p


def _make_reference(idx: int) -> Reference:
    return Reference(
        Title=f"Paper {idx}",
        Authors_List=[f"Author {idx}"],
        Authors=f"Author {idx}",
        Journal="J",
        Year=2020,
    )


def test_undercount_fires_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    body = "We cite many works: " + " ".join(f"[{i}]" for i in range(1, 21))
    md_path = _make_paper(tmp_path, body, "1. dummy.\n")
    client = _ScriptedClient(refs=[_make_reference(1), _make_reference(2)])

    with caplog.at_level(logging.WARNING, logger="citegraph.extract_references"):
        result = extract_references_from_markdown(md_path, client=client)

    assert len(result) == 2
    assert any("undercount" in rec.message for rec in caplog.records)


def test_no_warning_when_count_meets_threshold(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """20 distinct citations, 15 extracted (75%) — above the 50% threshold, no warning."""
    body = "We cite many works: " + " ".join(f"[{i}]" for i in range(1, 21))
    md_path = _make_paper(tmp_path, body, "1. dummy.\n")
    refs = [_make_reference(i) for i in range(15)]
    client = _ScriptedClient(refs=refs)

    with caplog.at_level(logging.WARNING, logger="citegraph.extract_references"):
        extract_references_from_markdown(md_path, client=client)

    assert not any("undercount" in rec.message for rec in caplog.records)


def test_no_warning_for_short_papers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Bodies with very few citations skip the check (estimate is noisy at small N)."""
    body = "A short opinion citing only [1] and [2]."  # estimate = 2, below the floor
    assert _INLINE_CITATION_MIN_ESTIMATE > 2  # guards the assumption
    md_path = _make_paper(tmp_path, body, "1. dummy.\n")
    client = _ScriptedClient(refs=[])

    with caplog.at_level(logging.WARNING, logger="citegraph.extract_references"):
        extract_references_from_markdown(md_path, client=client)

    assert not any("undercount" in rec.message for rec in caplog.records)


def test_no_warning_when_no_refs_header(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """If we couldn't slice (no header), we don't run the check — body would include the bibliography."""
    p = tmp_path / "paper.md"
    body = "We cite many works: " + " ".join(f"[{i}]" for i in range(1, 21))
    p.write_text(body)  # no ## References header
    client = _ScriptedClient(refs=[_make_reference(1)])

    with caplog.at_level(logging.WARNING, logger="citegraph.extract_references"):
        extract_references_from_markdown(p, client=client)

    assert not any("undercount" in rec.message for rec in caplog.records)
