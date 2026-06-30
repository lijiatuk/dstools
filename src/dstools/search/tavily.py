"""Tavily search provider (optional, requires an API key).

Install with ``pip install dstools[tavily]`` (pulls in ``tavily-python``) and set
``SEARCH_PROVIDER=tavily`` + ``TAVILY_API_KEY``. Tavily generally returns
higher-quality, cleaner results than the keyless DuckDuckGo scraper.
"""

from __future__ import annotations

from ..config import Settings
from ..exceptions import ConfigError, SearchError
from ..logging_setup import get_logger
from .base import SearchResult

_logger = get_logger("search.tavily")


class TavilySearchProvider:
    """Search via the Tavily API."""

    name = "tavily"

    def __init__(self, settings: Settings) -> None:
        if not settings.has_tavily:
            raise ConfigError(
                "Tavily is selected (SEARCH_PROVIDER=tavily) but TAVILY_API_KEY is "
                "not set. Set it or switch back to SEARCH_PROVIDER=duckduckgo."
            )
        self._settings = settings

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        try:
            from tavily import AsyncTavilyClient
        except ImportError as exc:  # pragma: no cover - exercised only when extra missing
            raise ConfigError(
                "The 'tavily-python' package is not installed. Install it with "
                "`pip install dstools[tavily]`."
            ) from exc

        client = AsyncTavilyClient(api_key=self._settings.tavily_api_key)
        try:
            resp = await client.search(
                query=query, max_results=max_results, search_depth="basic"
            )
        except Exception as exc:
            raise SearchError(f"Tavily search failed: {exc}") from exc

        results: list[SearchResult] = []
        for item in resp.get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                )
            )
        _logger.debug("Tavily search %r -> %d results", query, len(results))
        return results[:max_results]
