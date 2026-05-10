"""citegraph: build a deduplicated citation graph from a folder of PDFs.

See :class:`citegraph.Pipeline` for the high-level entry point::

    from citegraph import Pipeline

    pipe = Pipeline(pdf_dir="./pdfs", out_dir="./out")
    result = pipe.run()
    result.papers, result.references, result.graph
"""

from __future__ import annotations

from citegraph.cost_estimation import ExtractionEstimate
from citegraph.graph import CitationGraph
from citegraph.pipeline import Pipeline
from citegraph.schemas import (
    CitationLink,
    PaperMetadata,
    PipelineResult,
    Reference,
)

__all__ = [
    "Pipeline",
    "CitationGraph",
    "ExtractionEstimate",
    "PipelineResult",
    "PaperMetadata",
    "Reference",
    "CitationLink",
]

__version__ = "0.1.0"
