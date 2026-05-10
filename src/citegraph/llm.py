"""Thin Gemini wrapper used by the extraction modules.

We keep all of the provider-specific code in this single file so that the
rest of the package only deals with Pydantic models and plain Python
objects. The wrapper handles:

- Lazy client construction with a configurable API key/model.
- Retry/backoff via :mod:`tenacity` for transient errors.
- A :func:`fix_incomplete_json_string` JSON repair fallback for the rare
  cases where Gemini truncates a response.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from citegraph.config import Settings, get_settings

logger = logging.getLogger(__name__)


def fix_incomplete_json_string(input_string: str) -> str | None:
    """Attempt to recover a truncated JSON-array string.

    The current Gemini structured-output path almost always returns valid
    JSON, but on long bibliographies the response can hit the output-token
    limit and arrive truncated. This function:

    1. Strips surrounding single quotes (a quirk we sometimes see).
    2. Cuts the string at the last complete ``}``.
    3. Closes the trailing ``]`` so the result parses as a JSON array.

    Returns ``None`` if no complete object boundary is found.
    """
    cleaned = input_string
    if cleaned.startswith("'"):
        cleaned = cleaned[1:]
    if cleaned.endswith("'"):
        cleaned = cleaned[:-1]

    last_obj_end = cleaned.rfind("}")
    if last_obj_end == -1:
        return None
    return cleaned[: last_obj_end + 1] + "\n]"


class LLMError(RuntimeError):
    """Raised when the LLM call fails after all retries."""


def response_was_truncated(response: Any) -> bool:
    """Return True if the Gemini response was cut off by ``max_output_tokens``.

    Inspects ``response.candidates[*].finish_reason`` and matches the value
    "MAX_TOKENS" (the SDK may expose it as an enum or a plain string). Returns
    ``False`` defensively when the response shape is unexpected — callers can
    fall back to other signals (e.g. JSON parse failure) in that case.
    """
    try:
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            reason = getattr(cand, "finish_reason", None)
            name = getattr(reason, "name", None) or str(reason or "")
            if "MAX_TOKENS" in name:
                return True
    except Exception:  # noqa: BLE001
        return False
    return False


class GeminiClient:
    """Minimal Gemini client used by the extraction modules."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._api_key = api_key or self._settings.require_api_key()
        self.model = model or self._settings.citegraph_model
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    @property
    def default_max_output_tokens(self) -> int:
        """Output-token cap used when ``generate_structured`` gets ``None``."""
        return self._settings.citegraph_max_output_tokens

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _generate(self, *, contents: str, config: dict[str, Any]) -> Any:
        return self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

    def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: Any,
        system_instruction: str | None = None,
        max_output_tokens: int | None = None,
    ) -> Any:
        """Run a structured-output Gemini call and return the response object.

        ``response_schema`` is passed through to Gemini unchanged. Use a
        Pydantic ``BaseModel`` subclass for a single-object response or a
        list type (e.g. ``list[Reference]``) for an array response.
        """
        config: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": response_schema,
            "max_output_tokens": (
                max_output_tokens or self._settings.citegraph_max_output_tokens
            ),
        }
        if system_instruction:
            config["system_instruction"] = system_instruction
        try:
            return self._generate(contents=prompt, config=config)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Gemini call failed: {exc}") from exc


def parse_structured_response(
    response: Any,
    schema: type[BaseModel] | None = None,
) -> Any:
    """Parse a Gemini response, falling back to JSON repair if needed.

    - If ``response.parsed`` is populated, return it.
    - Otherwise try ``json.loads(response.text)``.
    - On failure, attempt :func:`fix_incomplete_json_string` and retry.

    When ``schema`` is provided, list elements / single objects are
    validated through that Pydantic model before being returned.
    """
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return parsed

    raw = getattr(response, "text", None)
    if not raw:
        raise LLMError("Empty Gemini response (no .parsed and no .text).")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Gemini returned invalid JSON, attempting repair: %s", exc)
        repaired = fix_incomplete_json_string(raw)
        if repaired is None:
            raise LLMError("Could not repair truncated JSON response.") from exc
        data = json.loads(repaired)

    if schema is None:
        return data

    if isinstance(data, list):
        return [schema.model_validate(item) for item in data]
    return schema.model_validate(data)
