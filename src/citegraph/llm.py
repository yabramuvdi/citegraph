"""Small provider adapters for schema-constrained bibliographic extraction."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, TypeAdapter
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from citegraph.config import LLMProvider, Settings, get_settings, output_budgets, resolve_model

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


@dataclass
class LLMResponse:
    parsed: Any = None
    text: str | None = None
    truncated: bool = False


class LLMClient(Protocol):
    provider: str
    model: str

    @property
    def default_max_output_tokens(self) -> int: ...

    def generate_structured(
        self, *, prompt: str, response_schema: Any,
        system_instruction: str | None = None, max_output_tokens: int | None = None,
    ) -> LLMResponse: ...


def response_was_truncated(response: Any) -> bool:
    """Return True if the Gemini response was cut off by ``max_output_tokens``.

    Inspects ``response.candidates[*].finish_reason`` and matches the value
    "MAX_TOKENS" (the SDK may expose it as an enum or a plain string). Returns
    ``False`` defensively when the response shape is unexpected — callers can
    fall back to other signals (e.g. JSON parse failure) in that case.
    """
    if isinstance(response, LLMResponse):
        return response.truncated
    # Compatibility for callers supplying older Gemini-shaped response objects.
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


def _retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int):
        return status in (408, 409, 429) or status >= 500
    return isinstance(exc, (TimeoutError, ConnectionError)) or type(exc).__name__ in {
        "APIConnectionError", "APITimeoutError", "ConnectError", "ReadError",
        "ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout",
        "RemoteProtocolError",
    }


class _ProviderClient:
    provider: str

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.model = resolve_model(self.provider, model, self._settings)
        self._api_key = api_key or self._settings.require_api_key(self.provider)
        self._client: Any | None = None
        if self.max_retry_output_tokens < self.default_max_output_tokens:
            raise ValueError("CITEGRAPH_MAX_RETRY_OUTPUT_TOKENS must be at least the initial output budget.")

    @property
    def default_max_output_tokens(self) -> int:
        return output_budgets(self.provider, self._settings)["references"]

    @property
    def max_retry_output_tokens(self) -> int:
        return output_budgets(self.provider, self._settings)["retry"]

    @property
    def metadata_max_output_tokens(self) -> int:
        return output_budgets(self.provider, self._settings)["metadata"]

    @retry(
        retry=retry_if_exception(_retryable), stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30), reraise=True,
    )
    def _request(self, **kwargs: Any) -> Any:
        return self._generate(**kwargs)


class GeminiClient(_ProviderClient):
    """Gemini adapter; existing constructor and injected-client usage remain valid."""

    provider = "gemini"

    @property
    def client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key, http_options={
                "timeout": int(self._settings.citegraph_request_timeout_s * 1000),
                "retry_options": {"attempts": 1},
            })
        return self._client

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
                max_output_tokens or self.default_max_output_tokens
            ),
        }
        if system_instruction:
            config["system_instruction"] = system_instruction
        try:
            response = self._request(contents=prompt, config=config)
            return LLMResponse(
                parsed=getattr(response, "parsed", None),
                text=getattr(response, "text", None),
                truncated=response_was_truncated(response),
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Gemini call failed: {exc}") from exc


def _output_schema(response_schema: Any) -> dict:
    """Use an object root for providers that cannot constrain a top-level array."""
    schema = TypeAdapter(response_schema).json_schema()

    def close_objects(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object":
                value["additionalProperties"] = False
                value["required"] = list(value.get("properties", {}))
            for nested in value.values():
                close_objects(nested)
        elif isinstance(value, list):
            for nested in value:
                close_objects(nested)

    # References in the nested result still resolve from the document root.
    definitions = schema.pop("$defs", {})
    wrapped = {"type": "object", "properties": {"result": schema},
               "required": ["result"], "additionalProperties": False}
    if definitions:
        wrapped["$defs"] = definitions
    close_objects(wrapped)
    return wrapped


def _unwrap(text: str, *, truncated: bool, response_schema: Any) -> LLMResponse:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        if not truncated:
            raise LLMError("Provider returned invalid structured JSON.") from exc
        # Keep the existing last-complete-reference repair for a cut-off envelope.
        return LLMResponse(
            text=re.sub(r'^\s*\{\s*"result"\s*:\s*', "", text), truncated=True,
        )
    if not isinstance(data, dict) or "result" not in data:
        raise LLMError("Provider response is missing its structured result.")
    value = data["result"] if truncated else TypeAdapter(response_schema).validate_python(data["result"])
    return LLMResponse(parsed=value, truncated=truncated)


class OpenAIClient(_ProviderClient):
    """OpenAI Responses API adapter using strict JSON-schema outputs."""

    provider = "openai"

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError('Install OpenAI support with: pip install "citegraph[openai]"') from exc
            self._client = OpenAI(
                api_key=self._api_key, timeout=self._settings.citegraph_request_timeout_s,
                max_retries=0,
            )
        return self._client

    def _generate(self, **kwargs: Any) -> Any:
        return self.client.responses.create(**kwargs)

    def generate_structured(
        self, *, prompt: str, response_schema: Any,
        system_instruction: str | None = None, max_output_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs = {
            "model": self.model, "input": prompt, "store": False,
            "max_output_tokens": max_output_tokens or self.default_max_output_tokens,
            "text": {"format": {"type": "json_schema", "name": "bibliography",
                                "strict": True, "schema": _output_schema(response_schema)}},
        }
        if system_instruction:
            kwargs["instructions"] = system_instruction
        try:
            result = self._request(**kwargs)
        except Exception as exc:
            raise LLMError(f"OpenAI call failed: {exc}") from exc
        for item in result.output:
            for block in getattr(item, "content", []):
                if block.type == "refusal":
                    raise LLMError("OpenAI refused the extraction request.")
        reason = getattr(getattr(result, "incomplete_details", None), "reason", None)
        truncated = result.status == "incomplete" and reason == "max_output_tokens"
        if result.status != "completed" and not truncated:
            raise LLMError(f"OpenAI extraction did not complete ({result.status}, {reason}).")
        return _unwrap(result.output_text, truncated=truncated, response_schema=response_schema)


class AnthropicClient(_ProviderClient):
    """Anthropic Messages API adapter using JSON-schema outputs."""

    provider = "anthropic"

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise ImportError('Install Anthropic support with: pip install "citegraph[anthropic]"') from exc
            self._client = Anthropic(
                api_key=self._api_key, timeout=self._settings.citegraph_request_timeout_s,
                max_retries=0,
            )
        return self._client

    def _generate(self, **kwargs: Any) -> Any:
        return self.client.messages.create(**kwargs)

    def generate_structured(
        self, *, prompt: str, response_schema: Any,
        system_instruction: str | None = None, max_output_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs = {
            "model": self.model, "max_tokens": max_output_tokens or self.default_max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"format": {"type": "json_schema", "schema": _output_schema(response_schema)}},
        }
        if system_instruction:
            kwargs["system"] = system_instruction
        try:
            result = self._request(**kwargs)
        except Exception as exc:
            raise LLMError(f"Anthropic call failed: {exc}") from exc
        if result.stop_reason == "refusal":
            raise LLMError("Anthropic refused the extraction request.")
        if result.stop_reason not in ("end_turn", "max_tokens"):
            raise LLMError(f"Anthropic extraction did not complete ({result.stop_reason}).")
        text = "".join(block.text for block in result.content if block.type == "text")
        return _unwrap(text, truncated=result.stop_reason == "max_tokens", response_schema=response_schema)


def create_llm_client(
    *, provider: str | None = None, model: str | None = None, settings: Settings | None = None,
) -> LLMClient:
    settings = settings or get_settings()
    selected = LLMProvider(provider or settings.citegraph_provider).value
    cls = {"gemini": GeminiClient, "openai": OpenAIClient, "anthropic": AnthropicClient}[selected]
    return cls(model=model, settings=settings)


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
        data = parsed
    else:
        raw = getattr(response, "text", None)
        if not raw:
            raise LLMError("Empty LLM response (no .parsed and no .text).")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("LLM returned invalid JSON, attempting repair: %s", exc)
            repaired = fix_incomplete_json_string(raw)
            if repaired is None:
                raise LLMError("Could not repair truncated JSON response.") from exc
            data = json.loads(repaired)

    if schema is None:
        return data

    if isinstance(data, list):
        return [schema.model_validate(item) for item in data]
    return schema.model_validate(data)
