"""Configuration for dstools.

All settings are environment variables (a ``.env`` file is loaded automatically).
Sensible defaults mean the keyless parts — web search (DuckDuckGo) and page fetching —
work out of the box. The DeepSeek API key is required only for ``deep_research``; a
vision provider is required only for ``analyze_image`` (DeepSeek-V4 is text-only).

Settings are cached via :func:`get_settings` so they are parsed once per process.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Runtime settings, populated from the environment / a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- DeepSeek (the LLM brain; OpenAI-compatible API) -------------------
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    # DeepSeek-V4 models. (deepseek-chat / deepseek-reasoner are deprecated
    # 2026-07-24 and map to v4-flash non-thinking / thinking respectively.)
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_fast_model: str = "deepseek-v4-flash"
    # V4 thinking mode for *hard* steps (planning, synthesis): auto | on | off.
    deepseek_thinking: Literal["auto", "on", "off"] = "auto"
    deepseek_reasoning_effort: Literal["low", "medium", "high", "max", "xhigh"] = "high"
    deepseek_timeout: float = 120.0

    # --- Vision provider (any OpenAI-compatible multimodal endpoint) --------
    # Required for analyze_image. DeepSeek-V4 cannot see images natively.
    vision_base_url: str = ""
    vision_api_key: str = ""
    vision_model: str = ""
    vision_timeout: float = 60.0
    vision_max_tokens: int = 1024

    # --- Web search --------------------------------------------------------
    search_provider: Literal["duckduckgo", "tavily", "brave"] = "duckduckgo"
    tavily_api_key: str = ""
    brave_api_key: str = ""
    search_timeout: float = 20.0
    search_max_results: int = 10
    # Seconds to back off before retrying a rate-limited keyless search.
    search_retry_attempts: int = 3

    # --- Deep research tuning ----------------------------------------------
    research_breadth: int = Field(default=3, ge=1, le=10)
    research_depth: int = Field(default=2, ge=1, le=5)
    research_max_sources: int = Field(default=8, ge=1, le=30)
    research_fetch_timeout: float = 20.0
    # Per-page character budget fed to the reranker (keeps cost down; V4 has
    # 1M context so this is conservative).
    research_per_page_chars: int = 6000
    research_total_chars: int = 120_000
    # Per-step model overrides (empty = sensible default: flash for plan/refine/
    # rerank, pro for synthesis). Lets cost-sensitive users run all-flash.
    research_plan_model: str = ""
    research_refine_model: str = ""
    research_rerank_model: str = ""
    research_synth_model: str = ""

    # --- Server / misc -----------------------------------------------------
    log_level: LogLevel = "INFO"
    mcp_transport: Literal["stdio", "sse", "streamable-http"] = "stdio"
    http_host: str = "127.0.0.1"
    http_port: int = 8000
    # User-Agent used for outbound web fetches.
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )

    # --- Capability flags (derived) ---------------------------------------
    @property
    def has_deepseek(self) -> bool:
        return bool(self.deepseek_api_key)

    @property
    def has_vision(self) -> bool:
        return bool(self.vision_base_url and self.vision_model)

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def has_brave(self) -> bool:
        return bool(self.brave_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached, process-wide :class:`Settings` instance."""
    return Settings()


def reload_settings() -> Settings:
    """Re-read settings (mainly for tests). Clears the cache first."""
    get_settings.cache_clear()
    return get_settings()
