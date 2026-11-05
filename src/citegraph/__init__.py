"""Build and analyze citation and author networks from academic PDFs.

LLM-assisted extraction supports Gemini, OpenAI, and Anthropic.

See :class:`citegraph.Pipeline` for the high-level entry point::

    from citegraph import Pipeline

    pipe = Pipeline(pdf_dir="./pdfs", out_dir="./out")
    result = pipe.run()
    result.works, result.graph
"""

from __future__ import annotations

from citegraph.authors import (
    AuthorClusterConfig,
    ParsedAuthor,
    normalize_authors,
    parse_author,
)
from citegraph.cost_estimation import ExtractionEstimate
from citegraph.graph import CitationGraph
from citegraph.llm import AnthropicClient, GeminiClient, LLMClient, LLMResponse, OpenAIClient
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
    "AuthorClusterConfig",
    "ParsedAuthor",
    "normalize_authors",
    "parse_author",
    "LLMClient",
    "LLMResponse",
    "GeminiClient",
    "OpenAIClient",
    "AnthropicClient",
]

__version__ = "0.2.1"
