"""Tests for the cost estimation module (no network, no Gemini calls)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from citegraph.cost_estimation import (
    _CHARS_PER_TOKEN,
    _METADATA_OUTPUT_TOKENS,
    _METADATA_OVERHEAD_CHARS,
    _REFERENCES_OVERHEAD_CHARS,
    _TOKENS_PER_REFERENCE,
    ExtractionEstimate,
    FileEstimate,
    estimate_extraction_cost,
)
from citegraph.extract_metadata import DEFAULT_METADATA_INPUT_CHARS
from citegraph.io import OutLayout
from citegraph.pipeline import StageNotReadyError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _layout(tmp_path: Path) -> OutLayout:
    layout = OutLayout(tmp_path)
    layout.ensure()
    return layout


def _write_md(layout: OutLayout, name: str, content: str) -> Path:
    path = layout.markdown_dir / name
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# StageNotReady guard
# ---------------------------------------------------------------------------

def test_raises_when_no_markdowns(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    with pytest.raises(StageNotReadyError, match="markdown"):
        estimate_extraction_cost(layout, model="gemini-2.0-flash")


# ---------------------------------------------------------------------------
# Single file, no caches
# ---------------------------------------------------------------------------

def test_single_uncached_file_token_counts(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    # Longer than the metadata cap so we verify truncation.
    content = "A" * 20_000
    _write_md(layout, "paper.md", content)

    est = estimate_extraction_cost(layout, model="gemini-2.0-flash")

    assert est.n_files == 1
    assert est.n_metadata_to_process == 1
    assert est.n_references_to_process == 1

    fe = est.files[0]
    assert not fe.metadata_cached
    assert not fe.references_cached

    expected_meta = math.ceil((DEFAULT_METADATA_INPUT_CHARS + _METADATA_OVERHEAD_CHARS) / _CHARS_PER_TOKEN)
    assert fe.metadata_input_tokens == expected_meta

    # No references header → full content is sent for references.
    expected_refs = math.ceil((20_000 + _REFERENCES_OVERHEAD_CHARS) / _CHARS_PER_TOKEN)
    assert fe.references_input_tokens == expected_refs


def test_short_file_metadata_not_truncated(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    content = "B" * 3_000  # well under the 12 000-char metadata cap
    _write_md(layout, "paper.md", content)

    est = estimate_extraction_cost(layout, model="gemini-2.0-flash")
    fe = est.files[0]

    expected_meta = math.ceil((3_000 + _METADATA_OVERHEAD_CHARS) / _CHARS_PER_TOKEN)
    assert fe.metadata_input_tokens == expected_meta


# ---------------------------------------------------------------------------
# Cache awareness
# ---------------------------------------------------------------------------

def test_fully_cached_file_contributes_zero_tokens(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    _write_md(layout, "paper.md", "C" * 5_000)
    (layout.metadata_dir / "paper.json").write_text("{}", encoding="utf-8")
    (layout.references_dir / "paper.json").write_text("[]", encoding="utf-8")

    est = estimate_extraction_cost(layout, model="gemini-2.0-flash")

    assert est.n_metadata_to_process == 0
    assert est.n_references_to_process == 0
    assert est.total_input_tokens == 0
    assert est.estimated_output_tokens == 0


def test_partial_cache_only_references_cached(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    _write_md(layout, "paper.md", "D" * 3_000)
    (layout.references_dir / "paper.json").write_text("[]", encoding="utf-8")

    est = estimate_extraction_cost(layout, model="gemini-2.0-flash")
    fe = est.files[0]

    assert not fe.metadata_cached
    assert fe.references_cached
    assert fe.metadata_input_tokens > 0
    assert fe.references_input_tokens == 0


def test_partial_cache_only_metadata_cached(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    _write_md(layout, "paper.md", "E" * 3_000)
    (layout.metadata_dir / "paper.json").write_text("{}", encoding="utf-8")

    est = estimate_extraction_cost(layout, model="gemini-2.0-flash")
    fe = est.files[0]

    assert fe.metadata_cached
    assert not fe.references_cached
    assert fe.metadata_input_tokens == 0
    assert fe.references_input_tokens > 0


# ---------------------------------------------------------------------------
# References-section slicing
# ---------------------------------------------------------------------------

def test_references_section_slicing_reduces_tokens(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    body = "X" * 10_000
    refs = "\n## References\n" + "Y" * 2_000
    content = body + refs
    _write_md(layout, "paper.md", content)

    est = estimate_extraction_cost(layout, model="gemini-2.0-flash")
    fe = est.files[0]

    # References section is only ~2 015 chars, so refs tokens < full-doc tokens.
    full_doc_tokens = math.ceil((len(content) + _REFERENCES_OVERHEAD_CHARS) / _CHARS_PER_TOKEN)
    assert fe.references_input_tokens < full_doc_tokens


# ---------------------------------------------------------------------------
# Multi-file aggregation
# ---------------------------------------------------------------------------

def test_multiple_files_aggregate(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    _write_md(layout, "a.md", "A" * 3_000)
    _write_md(layout, "b.md", "B" * 4_000)

    est = estimate_extraction_cost(layout, model="gemini-1.5-flash")

    assert est.n_files == 2
    assert est.n_metadata_to_process == 2
    assert est.n_references_to_process == 2
    assert est.total_input_tokens > 0
    assert est.estimated_output_tokens == 2 * _METADATA_OUTPUT_TOKENS + sum(
        f.estimated_n_references * _TOKENS_PER_REFERENCE for f in est.files
    )


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

def test_cost_usd_known_model() -> None:
    est = ExtractionEstimate(
        model="gemini-2.0-flash",
        files=[
            FileEstimate(
                name="p.md",
                metadata_cached=False,
                references_cached=False,
                metadata_input_tokens=1_000_000,
                references_input_tokens=0,
                estimated_n_references=0,
            )
        ],
    )
    cost = est.cost_usd()
    assert cost is not None
    input_cost, _ = cost
    # 1 M tokens × $0.10/1M = $0.10
    assert abs(input_cost - 0.10) < 1e-9


def test_cost_usd_unknown_model_returns_none() -> None:
    est = ExtractionEstimate(model="unknown-future-model", files=[])
    assert est.cost_usd() is None


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------

def test_format_summary_known_model(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    _write_md(layout, "paper.md", "F" * 5_000)
    est = estimate_extraction_cost(layout, model="gemini-2.0-flash")

    summary = est.format_summary()
    assert "gemini-2.0-flash" in summary
    assert "input tokens" in summary.lower()
    assert "$" in summary
    assert "Estimates are approximate" in summary


def test_format_summary_unknown_model(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    _write_md(layout, "paper.md", "G" * 5_000)
    est = estimate_extraction_cost(layout, model="mystery-model-9")

    summary = est.format_summary()
    assert "unknown" in summary.lower()
    assert "mystery-model-9" in summary


def test_format_summary_shows_cached_count(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    _write_md(layout, "a.md", "H" * 3_000)
    _write_md(layout, "b.md", "I" * 3_000)
    (layout.metadata_dir / "a.json").write_text("{}", encoding="utf-8")

    est = estimate_extraction_cost(layout, model="gemini-2.0-flash")
    summary = est.format_summary()
    assert "1 cached" in summary


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

def test_pipeline_estimate_extraction_cost(tmp_path: Path) -> None:
    from citegraph.pipeline import Pipeline

    md_dir = tmp_path / "out" / "markdown"
    md_dir.mkdir(parents=True)
    (md_dir / "paper.md").write_text("J" * 5_000, encoding="utf-8")

    pipeline = Pipeline(pdf_dir=None, out_dir=tmp_path / "out")
    est = pipeline.estimate_extraction_cost()

    assert est.n_files == 1
    assert est.total_input_tokens > 0


def test_pipeline_estimate_uses_explicit_model(tmp_path: Path) -> None:
    from citegraph.pipeline import Pipeline

    md_dir = tmp_path / "out" / "markdown"
    md_dir.mkdir(parents=True)
    (md_dir / "paper.md").write_text("K" * 3_000, encoding="utf-8")

    pipeline = Pipeline(pdf_dir=None, out_dir=tmp_path / "out", model="gemini-2.5-flash")
    est = pipeline.estimate_extraction_cost()

    assert est.model == "gemini-2.5-flash"
