"""Pre-flight cost estimation for the LLM extraction stages.

Reads the on-disk markdown files (stage-1 output) and estimates Gemini token
usage for stages 2 (metadata) and 3 (references) before making any API calls.
Already-cached files are excluded from the estimate.

Library usage::

    pipeline = Pipeline(pdf_dir=..., out_dir=...)
    estimate = pipeline.estimate_extraction_cost()
    print(estimate.format_summary())

CLI usage::

    citegraph estimate --out ./out
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from citegraph.extract_metadata import DEFAULT_METADATA_INPUT_CHARS
from citegraph.extract_references import slice_to_references_section
from citegraph.io import OutLayout

# Rough heuristic: English/academic prose averages ~4 characters per token.
_CHARS_PER_TOKEN: int = 4

# Static prompt overhead per call (template chars + system instruction chars).
# Measured from the actual prompt strings in extract_metadata.py / extract_references.py.
_METADATA_OVERHEAD_CHARS: int = 560
_REFERENCES_OVERHEAD_CHARS: int = 1_060

# Output-token constants for cost projection.
_METADATA_OUTPUT_TOKENS: int = 300   # PaperMetadata JSON is small and bounded
_TOKENS_PER_REFERENCE: int = 75      # one Reference JSON object
_CHARS_PER_REFERENCE_APPROX: int = 175  # typical reference length in source text

# Pricing table: (input USD / 1M tokens, output USD / 1M tokens).
# Approximate rates as of mid-2025 — check https://ai.google.dev/pricing for
# current numbers before trusting these for budget planning.
_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-pro-preview":   (1.25, 10.00),
    "gemini-2.5-flash-preview": (0.15,  0.60),
    "gemini-2.5-flash":         (0.15,  0.60),
    "gemini-2.0-flash":         (0.10,  0.40),
    "gemini-2.0-flash-lite":    (0.075, 0.30),
    "gemini-1.5-pro":           (1.25,  5.00),
    "gemini-1.5-flash":         (0.075, 0.30),
    "gemini-1.5-flash-lite":    (0.075, 0.30),
    "gemini-3.1-flash-lite":    (0.075, 0.30),
}


def _chars_to_tokens(n: int) -> int:
    return math.ceil(n / _CHARS_PER_TOKEN)


@dataclass
class FileEstimate:
    """Per-file breakdown of estimated token usage."""

    name: str
    metadata_cached: bool
    references_cached: bool
    metadata_input_tokens: int    # 0 when cached
    references_input_tokens: int  # 0 when cached
    estimated_n_references: int   # 0 when cached


@dataclass
class ExtractionEstimate:
    """Aggregate token and cost estimate for LLM extraction stages 2 and 3."""

    model: str
    files: list[FileEstimate] = field(default_factory=list)

    @property
    def n_files(self) -> int:
        return len(self.files)

    @property
    def n_metadata_to_process(self) -> int:
        return sum(1 for f in self.files if not f.metadata_cached)

    @property
    def n_references_to_process(self) -> int:
        return sum(1 for f in self.files if not f.references_cached)

    @property
    def total_input_tokens(self) -> int:
        return sum(f.metadata_input_tokens + f.references_input_tokens for f in self.files)

    @property
    def estimated_output_tokens(self) -> int:
        meta_out = self.n_metadata_to_process * _METADATA_OUTPUT_TOKENS
        refs_out = sum(
            f.estimated_n_references * _TOKENS_PER_REFERENCE
            for f in self.files
            if not f.references_cached
        )
        return meta_out + refs_out

    def cost_usd(self) -> tuple[float, float] | None:
        """Return ``(input_cost_usd, output_cost_usd)``, or ``None`` for unknown models."""
        pricing = _PRICING.get(self.model)
        if pricing is None:
            return None
        in_price, out_price = pricing
        return (
            self.total_input_tokens * in_price / 1_000_000,
            self.estimated_output_tokens * out_price / 1_000_000,
        )

    def format_references_summary(self) -> str:
        """Return a concise cost summary covering only stage 3 (references extraction)."""
        n_cached = self.n_files - self.n_references_to_process
        cached_str = f"  ({n_cached} cached)" if n_cached else ""

        refs_input = sum(f.references_input_tokens for f in self.files)
        refs_output = sum(
            f.estimated_n_references * _TOKENS_PER_REFERENCE
            for f in self.files
            if not f.references_cached
        )

        lines = [
            f"Model:                 {self.model}",
            f"Files to process:      {self.n_references_to_process}{cached_str}",
            f"Est. input tokens:     ~{refs_input:,}",
            f"Est. output tokens:    ~{refs_output:,}",
        ]

        pricing = _PRICING.get(self.model)
        if pricing is not None:
            in_price, out_price = pricing
            in_cost = refs_input * in_price / 1_000_000
            out_cost = refs_output * out_price / 1_000_000
            lines.append(
                f"Est. cost:             ~${in_cost:.4f} input"
                f" + ~${out_cost:.4f} output"
                f" = ~${in_cost + out_cost:.4f} total"
            )
        else:
            lines.append(
                f"Est. cost:             unknown (no pricing data for '{self.model}')"
            )
        lines.append("(Estimates are approximate; actual cost depends on response length.)")
        return "\n".join(lines)

    def format_summary(self) -> str:
        """Return a multi-line human-readable cost summary."""
        n_meta_cached = self.n_files - self.n_metadata_to_process
        n_refs_cached = self.n_files - self.n_references_to_process

        def _cached(n: int) -> str:
            return f"  ({n} cached)" if n else ""

        lines = [
            f"Model:                 {self.model}",
            f"Files found:           {self.n_files}",
            f"Metadata to process:   {self.n_metadata_to_process}{_cached(n_meta_cached)}",
            f"References to process: {self.n_references_to_process}{_cached(n_refs_cached)}",
            f"Est. input tokens:     ~{self.total_input_tokens:,}",
            f"Est. output tokens:    ~{self.estimated_output_tokens:,}",
        ]

        cost = self.cost_usd()
        if cost is not None:
            in_c, out_c = cost
            lines.append(
                f"Est. cost:             ~${in_c:.4f} input"
                f" + ~${out_c:.4f} output"
                f" = ~${in_c + out_c:.4f} total"
            )
        else:
            lines.append(
                f"Est. cost:             unknown (no pricing data for '{self.model}')"
            )
        lines.append("(Estimates are approximate; actual cost depends on response length.)")
        return "\n".join(lines)


def estimate_extraction_cost(
    layout: OutLayout,
    model: str,
    max_input_chars_metadata: int = DEFAULT_METADATA_INPUT_CHARS,
) -> ExtractionEstimate:
    """Estimate LLM token usage for stages 2 and 3 without making any API calls.

    Reads markdown files from ``layout.markdown_dir`` and skips files whose
    metadata / references caches already exist.

    Raises :class:`~citegraph.pipeline.StageNotReadyError` when no markdown
    files are found (stage 1 must run first).
    """
    from citegraph.pipeline import StageNotReadyError

    md_paths = sorted(layout.markdown_dir.glob("*.md")) if layout.markdown_dir.exists() else []
    if not md_paths:
        raise StageNotReadyError(
            f"No markdown files found in {layout.markdown_dir}. "
            "Run `citegraph convert <pdf_dir>` first."
        )

    file_estimates: list[FileEstimate] = []
    for md in md_paths:
        meta_cached = (layout.metadata_dir / f"{md.stem}.json").exists()
        refs_cached = (layout.references_dir / f"{md.stem}.json").exists()

        if meta_cached and refs_cached:
            file_estimates.append(
                FileEstimate(
                    name=md.name,
                    metadata_cached=True,
                    references_cached=True,
                    metadata_input_tokens=0,
                    references_input_tokens=0,
                    estimated_n_references=0,
                )
            )
            continue

        content = md.read_text(encoding="utf-8")

        if meta_cached:
            meta_tokens = 0
        else:
            capped = min(len(content), max_input_chars_metadata) if max_input_chars_metadata else len(content)
            meta_tokens = _chars_to_tokens(capped + _METADATA_OVERHEAD_CHARS)

        if refs_cached:
            refs_tokens = 0
            est_n_refs = 0
        else:
            sliced, _ = slice_to_references_section(content)
            refs_tokens = _chars_to_tokens(len(sliced) + _REFERENCES_OVERHEAD_CHARS)
            est_n_refs = max(1, len(sliced) // _CHARS_PER_REFERENCE_APPROX)

        file_estimates.append(
            FileEstimate(
                name=md.name,
                metadata_cached=meta_cached,
                references_cached=refs_cached,
                metadata_input_tokens=meta_tokens,
                references_input_tokens=refs_tokens,
                estimated_n_references=est_n_refs,
            )
        )

    return ExtractionEstimate(model=model, files=file_estimates)
