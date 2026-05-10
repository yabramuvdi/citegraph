"""Extract the reference list of a paper using Gemini structured output."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from citegraph.llm import GeminiClient, parse_structured_response, response_was_truncated
from citegraph.schemas import Reference

# Long bibliographies sometimes hit ``max_output_tokens``. When that happens we
# retry once with a higher cap before falling back to the JSON-repair path in
# :mod:`citegraph.llm`. The hard cap stops us from exceeding model output limits
# on a runaway retry.
_TRUNCATION_RETRY_MULTIPLIER = 2
_TRUNCATION_HARD_CAP_TOKENS = 80_000

# A sanity-check ratio: warn if the LLM returned fewer than this fraction of the
# refs implied by the body's inline-citation count. Set conservatively so we
# only flag clear shortfalls, not minor undercounts (citation-counting from
# free text is necessarily approximate).
_INLINE_CITATION_RATIO_THRESHOLD = 0.5
# Skip the sanity check on bodies with very few citations — the estimate is
# noisy at small N and the absolute gap isn't meaningful.
_INLINE_CITATION_MIN_ESTIMATE = 10

logger = logging.getLogger(__name__)


# Strict regex for the heading of a paper's bibliography section. Matches
# markdown headers like "## References", "# Bibliography", "### 5. References",
# "## References and Notes", "## Works Cited". Deliberately rejects headers
# with substantive trailing text (e.g. "## References to Prior Work") to avoid
# slicing at the wrong place.
_REF_KEYWORD = (
    r"(?:references?|bibliography|works\s+cited|literature\s+cited|cited\s+literature)"
)
_REF_HEADER_RE = re.compile(
    rf"^\#{{1,6}}\s+(?:[\dIVX]+\.?\s+)?{_REF_KEYWORD}(?:\s+(?:and\s+notes|cited))?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def split_at_references_header(content: str) -> tuple[str, str | None]:
    """Return ``(body, refs)`` split at the bibliography header.

    Locates the *last* :data:`_REF_HEADER_RE` match (bibliographies come at
    the end of a paper). If none is found, returns ``(content, None)`` so
    callers can decide how to react to the missing header.
    """
    matches = list(_REF_HEADER_RE.finditer(content))
    if not matches:
        return content, None
    cut = matches[-1].start()
    return content[:cut], content[cut:]


def slice_to_references_section(content: str) -> tuple[str, bool]:
    """Return ``(text, found)`` — content from the bibliography header onward.

    If no header matches, returns the original ``content`` unchanged with
    ``found=False`` so callers can fall back to sending the full markdown
    rather than dropping the paper.
    """
    body, refs = split_at_references_header(content)
    if refs is None:
        return content, False
    return refs, True


# Numeric in-text citations: \[12\], \[12, 15\], \[12-15\], \[12, 15, 18\], etc.
_INLINE_NUMERIC_RE = re.compile(r"\[([\d,\s\-–]+)\]")
_NUMERIC_SPLIT_RE = re.compile(r"[,\s\-–]+")

# Author-year citations in either parenthetical or narrative form:
#   "(Smith, 2019)", "(Smith and Jones, 2019)", "(Smith et al., 2019)"
#   "Smith (2019)", "Smith and Jones (2019)", "Smith et al. (2019)"
# The author group is captured loosely; we lower-case it for deduplication.
_AUTHOR = r"[A-Z][A-Za-z\-']+(?:\s+(?:and|&)\s+[A-Z][A-Za-z\-']+)?(?:\s+et\s+al\.?)?"
_INLINE_PAREN_RE = re.compile(rf"\(({_AUTHOR}),?\s+(\d{{4}})[a-z]?\)")
_INLINE_NARRATIVE_RE = re.compile(rf"\b({_AUTHOR})\s+\((\d{{4}})[a-z]?\)")


def estimate_inline_citations(body: str) -> int:
    """Approximate the number of distinct references cited inline in ``body``.

    Counts:

    - Unique integers inside ``[...]`` brackets (IEEE/ACM-style numeric).
    - Unique ``(author_lowercased, year)`` tokens from author-year citations,
      both ``(Smith, 2019)`` and narrative ``Smith (2019)`` forms.

    Returns the *sum* of the two — papers typically use one style, so summing
    gives a sensible total without much double-counting. The result is a
    deliberately rough estimate intended for order-of-magnitude sanity checks
    (e.g. "we extracted 12 refs but the body cites 80 — likely a parsing
    failure"), not a precise citation count.
    """
    numeric: set[int] = set()
    for match in _INLINE_NUMERIC_RE.finditer(body):
        for token in _NUMERIC_SPLIT_RE.split(match.group(1)):
            if token.isdigit():
                numeric.add(int(token))

    author_year: set[tuple[str, str]] = set()
    for pattern in (_INLINE_PAREN_RE, _INLINE_NARRATIVE_RE):
        for match in pattern.finditer(body):
            author = re.sub(r"\s+", " ", match.group(1).lower().strip())
            author_year.add((author, match.group(2)))

    return len(numeric) + len(author_year)


SYSTEM_INSTRUCTION = (
    "You are an expert assistant specializing in extracting bibliographic "
    "information from the references or bibliography section of an academic "
    "paper. You can interpret APA, MLA, Chicago and IEEE citation formats, "
    "and you handle complex layouts including double-column pages. Your "
    "goal is to extract every reference comprehensively and return them in "
    "a structured form."
)


def _build_prompt(paper_content: str) -> str:
    return f"""
# Instructions
Please extract the complete bibliographic information for all the papers cited in the references or bibliography section of the document below. For each reference, return:

- Title: The full title of the paper
- Authors_List: A list of all authors
- Journal: The journal or venue where the paper was published
- Year: The publication year as an integer

Take into account the following:

1. Capture every reference. Do not skip any.
2. Focus on the references / bibliography section at the end of the paper.
3. Strip any garbled characters that may have come from PDF extraction artefacts.
4. If a field is genuinely missing in the original, use an empty string for text fields and 0 for Year.

# Paper
{paper_content}
"""


def extract_references_from_markdown(
    markdown_path: Path | str,
    *,
    client: GeminiClient,
    max_output_tokens: int | None = None,
    slice_to_references: bool = True,
) -> list[Reference]:
    """Extract a list of :class:`Reference` from a single markdown-rendered paper.

    By default the markdown is sliced to its bibliography section before being
    sent to the LLM (see :func:`slice_to_references_section`). Pass
    ``slice_to_references=False`` to send the full file.

    If the LLM response is cut off by ``max_output_tokens`` (detected via
    :func:`response_was_truncated`), the call is retried once with the cap
    raised by ``_TRUNCATION_RETRY_MULTIPLIER``, up to
    ``_TRUNCATION_HARD_CAP_TOKENS``. A persistent truncation is logged at
    error level and the partial result is still returned.
    """
    markdown_path = Path(markdown_path)
    content = markdown_path.read_text(encoding="utf-8")

    body_for_sanity_check: str | None = None
    if slice_to_references:
        body, refs = split_at_references_header(content)
        if refs is not None:
            logger.debug(
                "Sliced %s to references section: %d -> %d chars",
                markdown_path.name,
                len(content),
                len(refs),
            )
            content = refs
            body_for_sanity_check = body
        else:
            logger.info(
                "No references-section header found in %s; sending full markdown",
                markdown_path.name,
            )

    prompt = _build_prompt(content)
    initial_cap = max_output_tokens or client.default_max_output_tokens

    def _call(cap: int):
        return client.generate_structured(
            prompt=prompt,
            response_schema=list[Reference],
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=cap,
        )

    response = _call(initial_cap)

    if response_was_truncated(response):
        retry_cap = min(
            initial_cap * _TRUNCATION_RETRY_MULTIPLIER, _TRUNCATION_HARD_CAP_TOKENS
        )
        if retry_cap > initial_cap:
            logger.warning(
                "References response for %s was truncated at %d tokens; "
                "retrying with cap %d",
                markdown_path.name,
                initial_cap,
                retry_cap,
            )
            response = _call(retry_cap)
            if response_was_truncated(response):
                logger.error(
                    "References response for %s still truncated at %d tokens; "
                    "extracted list will be incomplete",
                    markdown_path.name,
                    retry_cap,
                )
        else:
            logger.error(
                "References response for %s truncated at hard cap %d tokens; "
                "cannot retry — extracted list will be incomplete",
                markdown_path.name,
                initial_cap,
            )

    parsed = parse_structured_response(response, schema=Reference)
    if not isinstance(parsed, list):
        logger.warning(
            "Expected a list of references for %s, got %r", markdown_path.name, type(parsed)
        )
        return []

    if body_for_sanity_check is not None:
        _check_extracted_count(markdown_path, body_for_sanity_check, len(parsed))

    return parsed


def _check_extracted_count(
    markdown_path: Path, body: str, extracted_count: int
) -> None:
    """Warn if the extracted count is far below the body's inline-citation estimate."""
    estimate = estimate_inline_citations(body)
    if estimate < _INLINE_CITATION_MIN_ESTIMATE:
        return
    if extracted_count >= estimate * _INLINE_CITATION_RATIO_THRESHOLD:
        return
    logger.warning(
        "Possible references undercount for %s: extracted %d but body cites ~%d "
        "distinct works (threshold %.0f%%). The bibliography may have been "
        "partially parsed or the LLM may have skipped entries.",
        markdown_path.name,
        extracted_count,
        estimate,
        _INLINE_CITATION_RATIO_THRESHOLD * 100,
    )
