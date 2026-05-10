"""Extract bibliographic metadata of source papers using Gemini."""

from __future__ import annotations

import logging
from pathlib import Path

from citegraph.ids import make_paper_id
from citegraph.llm import GeminiClient, LLMError, parse_structured_response
from citegraph.schemas import PaperMetadata

logger = logging.getLogger(__name__)


SYSTEM_INSTRUCTION = (
    "You are a helpful assistant with great expertise in extracting "
    "bibliographic information from academic papers in markdown format."
)

# Title/authors/journal/year live on the first page of essentially every
# academic paper, so we send only a prefix to keep input tokens small.
DEFAULT_METADATA_INPUT_CHARS = 12_000


def _build_prompt(paper_content: str) -> str:
    return f"""
Extract the basic bibliographic information from the following paper. The information should include:
- Title: The full title of the paper
- Authors_List: A list of all authors
- Journal: The journal or venue where the paper was published
- Year: The publication year as an integer

Make sure to not include any dirty characters that might have come from problems in the PDF.

Here is the paper content:
{paper_content}
"""


def extract_metadata_from_markdown(
    markdown_path: Path | str,
    *,
    client: GeminiClient,
    max_input_chars: int = DEFAULT_METADATA_INPUT_CHARS,
) -> PaperMetadata:
    """Extract :class:`PaperMetadata` from a single markdown-rendered paper.

    Only the first ``max_input_chars`` characters of the markdown are sent to
    the LLM. Pass ``0`` to disable truncation.
    """
    markdown_path = Path(markdown_path)
    content = markdown_path.read_text(encoding="utf-8")
    if max_input_chars and len(content) > max_input_chars:
        logger.debug(
            "Truncating %s from %d to %d chars for metadata extraction",
            markdown_path.name,
            len(content),
            max_input_chars,
        )
        content = content[:max_input_chars]

    response = client.generate_structured(
        prompt=_build_prompt(content),
        response_schema=PaperMetadata,
        system_instruction=SYSTEM_INSTRUCTION,
        max_output_tokens=500,
    )
    parsed = parse_structured_response(response, schema=PaperMetadata)
    if not isinstance(parsed, PaperMetadata):
        raise LLMError(
            f"Expected PaperMetadata for {markdown_path.name}, got {type(parsed)!r}"
        )
    return parsed


def metadata_to_record(meta: PaperMetadata, source_file: str) -> dict:
    """Convert a :class:`PaperMetadata` into a flat dict ready for a DataFrame."""
    record = meta.model_dump()
    record["source_file"] = source_file
    record["id"] = make_paper_id(meta.Authors_List or meta.Authors, meta.Year, meta.Title)
    return record
