"""Runtime configuration for citegraph.

Reads from environment variables and an optional ``.env`` file via
``pydantic-settings``. No file paths or secrets should be hardcoded
elsewhere in the package.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    """Global, environment-driven settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    google_api_key: str | None = Field(
        default=None,
        validation_alias="GOOGLE_API_KEY",
        description="API key for the Google Gemini API.",
    )
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    citegraph_provider: LLMProvider = Field(
        default=LLMProvider.GEMINI, validation_alias="CITEGRAPH_PROVIDER",
    )
    openalex_api_key: str | None = Field(
        default=None,
        validation_alias="OPENALEX_API_KEY",
        description="API key for the OpenAlex API (raises the daily credit budget).",
    )
    citegraph_model: str | None = Field(
        default=None,
        validation_alias="CITEGRAPH_MODEL",
        description="Model identifier for the selected LLM provider.",
    )
    citegraph_max_output_tokens: int | None = Field(
        default=None, gt=0,
        validation_alias="CITEGRAPH_MAX_OUTPUT_TOKENS",
    )
    citegraph_max_retry_output_tokens: int | None = Field(
        default=None, gt=0, validation_alias="CITEGRAPH_MAX_RETRY_OUTPUT_TOKENS",
    )
    citegraph_metadata_max_output_tokens: int | None = Field(
        default=None, gt=0, validation_alias="CITEGRAPH_METADATA_MAX_OUTPUT_TOKENS",
    )
    citegraph_request_timeout_s: float = Field(
        default=120.0, gt=0,
        validation_alias="CITEGRAPH_REQUEST_TIMEOUT_S",
    )
    citegraph_llm_concurrency: int = Field(
        default=4,
        ge=1,
        validation_alias="CITEGRAPH_LLM_CONCURRENCY",
        description="Maximum concurrent LLM extraction calls.",
    )

    def require_api_key(self, provider: str = "gemini") -> str:
        """Return only the selected provider's key or raise a helpful error."""
        field, env_var = {
            "gemini": ("google_api_key", "GOOGLE_API_KEY"),
            "openai": ("openai_api_key", "OPENAI_API_KEY"),
            "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
        }[LLMProvider(provider).value]
        key = getattr(self, field)
        if not key:
            raise RuntimeError(
                f"{env_var} is not set. Export it in your environment or "
                "place it in a .env file in your working directory."
            )
        return key


def resolve_model(provider: str, model: str | None, settings: Settings) -> str:
    provider = LLMProvider(provider).value
    selected = model or settings.citegraph_model
    if selected:
        return selected
    if provider == "gemini":
        return "gemini-3.1-flash-lite"
    raise ValueError(
        f"Choose a model for {provider} with --model, Pipeline(model=...), or CITEGRAPH_MODEL."
    )


def output_budgets(provider: str, settings: Settings) -> dict[str, int]:
    references = settings.citegraph_max_output_tokens or (40_000 if provider == "gemini" else 8_000)
    return {
        "references": references,
        "retry": settings.citegraph_max_retry_output_tokens or max(
            references, 80_000 if provider == "gemini" else 16_000,
        ),
        "metadata": settings.citegraph_metadata_max_output_tokens or min(
            references, 500 if provider == "gemini" else 4096,
        ),
    }


def get_settings() -> Settings:
    """Load settings (cheap; safe to call repeatedly)."""
    return Settings()


def default_out_dir() -> Path:
    """Default output directory used when the caller doesn't specify one."""
    return Path.cwd() / "out"
