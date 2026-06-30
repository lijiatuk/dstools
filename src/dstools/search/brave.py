"""Brave Search provider (optional, requires an API key).

Set ``SEARCH_PROVIDER=brave`` and ``BRAVE_API_KEY`` (free tier: 2000 queries/month).
Brave returns clean, ad-free results and is more reliable than the keyless
DuckDuckGo scraper — recommended for heavy/reliable use.
"""

from __future__ import annotations

import httpx

from ..config import Settings
from ..exceptions import ConfigError, SearchError
from ..logging_setup import get_logger
from .base import SearchResult

_logger = get_logger("search.brave")

_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


class BraveSearchProvider:
    """Search via the Brave Search API."""

    name = "brave"

    def __init__(self, settings: Settings) -> None:
        if not settings.has_brave:
            raise ConfigError(
                "Brave is selected (SEARCH_PROVIDER=brave) but BRAVE_API_KEY is "
                "not set. Set it or switch to SEARCH_PROVIDER=duckduckgo."
            )
        self._settings = settings

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        headers = {
            "X-Subscription-Token": self._settings.brave_api_key,
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        }
        # Brave's count is capped at 20 per request.
        params: dict[str, str | int] = {"q": query, "count": min(max(max_results, 1), 20)}
        try:
            async with httpx.AsyncClient(
                timeout=self._settings.search_timeout, headers=headers
            ) as client:
                resp = await client.get(_ENDPOINT, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise SearchError(f"Brave search failed: {exc}") from exc

        results: list[SearchResult] = []
        web = data.get("web") or {}
        for item in (web.get("results") or [])[:max_results]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                )
            )
        _logger.debug("Brave search %r -> %d results", query, len(results))
        return results
