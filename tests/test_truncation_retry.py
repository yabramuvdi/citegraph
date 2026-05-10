"""Tests for truncation detection and the references-stage retry path."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from citegraph.extract_references import extract_references_from_markdown
from citegraph.llm import GeminiClient, response_was_truncated
from citegraph.schemas import Reference


def _fake_response(*, parsed=None, text: str = "", finish_reason: str | None = None):
    """Build a stand-in for a Gemini response.

    ``finish_reason`` may be either a plain string (we wrap it in a namespace
    that exposes a ``.name`` attribute, mirroring the SDK's enum shape) or
    ``None`` to omit the candidates list entirely.
    """
    candidates = []
    if finish_reason is not None:
        candidates = [SimpleNamespace(finish_reason=SimpleNamespace(name=finish_reason))]
    return SimpleNamespace(parsed=parsed, text=text, candidates=candidates)


# ---------------------------------------------------------------------------
# response_was_truncated
# ---------------------------------------------------------------------------
def test_response_was_truncated_true_for_max_tokens() -> None:
    assert response_was_truncated(_fake_response(finish_reason="MAX_TOKENS")) is True


def test_response_was_truncated_false_for_stop() -> None:
    assert response_was_truncated(_fake_response(finish_reason="STOP")) is False


def test_response_was_truncated_false_when_no_candidates() -> None:
    assert response_was_truncated(_fake_response(finish_reason=None)) is False


def test_response_was_truncated_handles_plain_string_finish_reason() -> None:
    """Some SDK versions return finish_reason as a plain string, not an enum."""
    response = SimpleNamespace(
        parsed=None,
        text="",
        candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
    )
    assert response_was_truncated(response) is True


# ---------------------------------------------------------------------------
# Retry path in extract_references_from_markdown
# ---------------------------------------------------------------------------
class _RecordingClient(GeminiClient):
    """Fake client that records the cap used on each call and returns scripted responses."""

    def __init__(self, scripted_responses):
        self._client = object()
        self.model = "fake-model"
        from citegraph.config import Settings

        self._settings = Settings(GOOGLE_API_KEY="fake")  # type: ignore[arg-type]
        self._api_key = "fake"
        self._scripted = list(scripted_responses)
        self.calls: list[int] = []

    def generate_structured(
        self, *, prompt, response_schema, system_instruction=None, max_output_tokens=None
    ):
        self.calls.append(max_output_tokens)
        return self._scripted.pop(0)


@pytest.fixture()
def md_path(tmp_path: Path) -> Path:
    p = tmp_path / "paper.md"
    p.write_text("## References\n1. Smith, J. (2020). A paper. Journal X.\n")
    return p


def test_truncation_triggers_retry_with_doubled_cap(
    md_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    good_refs = [
        Reference(
            Title="A paper",
            Authors_List=["J. Smith"],
            Authors="Smith, J.",
            Journal="Journal X",
            Year=2020,
        )
    ]
    client = _RecordingClient(
        scripted_responses=[
            _fake_response(parsed=None, text="[", finish_reason="MAX_TOKENS"),
            _fake_response(parsed=good_refs, finish_reason="STOP"),
        ]
    )

    with caplog.at_level(logging.WARNING, logger="citegraph.extract_references"):
        result = extract_references_from_markdown(md_path, client=client)

    assert result == good_refs
    # First call uses the client default; second call uses 2x.
    assert client.calls == [
        client.default_max_output_tokens,
        client.default_max_output_tokens * 2,
    ]
    assert any("retrying with cap" in rec.message for rec in caplog.records)


def test_persistent_truncation_logs_error_and_returns_partial(
    md_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """If the retry is also truncated, we log an error and return what we can parse."""
    partial_refs = [
        Reference(
            Title="A paper",
            Authors_List=["J. Smith"],
            Authors="Smith, J.",
            Journal="Journal X",
            Year=2020,
        )
    ]
    client = _RecordingClient(
        scripted_responses=[
            _fake_response(parsed=None, text="[", finish_reason="MAX_TOKENS"),
            _fake_response(parsed=partial_refs, finish_reason="MAX_TOKENS"),
        ]
    )

    with caplog.at_level(logging.ERROR, logger="citegraph.extract_references"):
        result = extract_references_from_markdown(md_path, client=client)

    assert result == partial_refs
    assert len(client.calls) == 2
    assert any("still truncated" in rec.message for rec in caplog.records)


def test_no_retry_when_initial_response_complete(md_path: Path) -> None:
    refs = [
        Reference(
            Title="A paper",
            Authors_List=["J. Smith"],
            Authors="Smith, J.",
            Journal="Journal X",
            Year=2020,
        )
    ]
    client = _RecordingClient(
        scripted_responses=[_fake_response(parsed=refs, finish_reason="STOP")]
    )

    result = extract_references_from_markdown(md_path, client=client)

    assert result == refs
    assert len(client.calls) == 1


def test_no_retry_when_already_at_hard_cap(
    md_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """If the initial cap already meets the hard cap, retry is skipped (and logged)."""
    from citegraph.extract_references import _TRUNCATION_HARD_CAP_TOKENS

    partial = [
        Reference(
            Title="A",
            Authors_List=["X"],
            Authors="X",
            Journal="J",
            Year=2020,
        )
    ]
    client = _RecordingClient(
        scripted_responses=[
            _fake_response(parsed=partial, finish_reason="MAX_TOKENS"),
        ]
    )

    with caplog.at_level(logging.ERROR, logger="citegraph.extract_references"):
        result = extract_references_from_markdown(
            md_path, client=client, max_output_tokens=_TRUNCATION_HARD_CAP_TOKENS
        )

    assert result == partial
    assert client.calls == [_TRUNCATION_HARD_CAP_TOKENS]
    assert any("hard cap" in rec.message for rec in caplog.records)
