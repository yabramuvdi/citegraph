"""Runtime configuration for citegraph.

Reads from environment variables and an optional ``.env`` file via
``pydantic-settings``. No file paths or secrets should be hardcoded
elsewhere in the package.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    citegraph_model: str = Field(
        default="gemini-3.1-flash-lite",
        validation_alias="CITEGRAPH_MODEL",
        description="Default Gemini model identifier.",
    )
    citegraph_max_output_tokens: int = Field(
        default=40_000,
        validation_alias="CITEGRAPH_MAX_OUTPUT_TOKENS",
    )
    citegraph_request_timeout_s: float = Field(
        default=120.0,
        validation_alias="CITEGRAPH_REQUEST_TIMEOUT_S",
    )
    citegraph_llm_concurrency: int = Field(
        default=4,
        ge=1,
        validation_alias="CITEGRAPH_LLM_CONCURRENCY",
        description="Maximum concurrent LLM extraction calls.",
    )

    def require_api_key(self) -> str:
        """Return the Gemini API key or raise a helpful error."""
        if not self.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Export it in your environment or "
                "place it in a .env file in your working directory."
            )
        return self.google_api_key


def get_settings() -> Settings:
    """Load settings (cheap; safe to call repeatedly)."""
    return Settings()


def default_out_dir() -> Path:
    """Default output directory used when the caller doesn't specify one."""
    return Path.cwd() / "out"
