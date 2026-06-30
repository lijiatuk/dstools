"""Pluggable web-search providers."""

from __future__ import annotations

from ..config import Settings
from .base import SearchProvider, SearchResult
from .duckduckgo import DuckDuckGoSearchProvider
from .tavily import TavilySearchProvider

__all__ = ["SearchProvider", "SearchResult", "get_search_provider"]


def get_search_provider(settings: Settings) -> SearchProvider:
    """Build the configured search provider from *settings*."""
    if settings.search_provider == "tavily":
        return TavilySearchProvider(settings)
    return DuckDuckGoSearchProvider(settings)
