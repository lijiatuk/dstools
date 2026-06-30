"""DuckDuckGo search provider (keyless, HTML endpoint).

Scrapes DuckDuckGo's lightweight HTML endpoint — no API key, works out of the
box. Robustness measures:

* Tries the HTML endpoint, then the Lite endpoint (layout fallback).
* Filters sponsored/ad results (DuckDuckGo interleaves ``y.js`` ad redirects).
* Retries with exponential backoff when rate-limited / empty (DuckDuckGo 429s
  under load) up to ``search_retry_attempts``.

This is the default :data:`SEARCH_PROVIDER`; for heavy/reliable use switch to
``brave`` or ``tavily``.
"""

from __future__ import annotations

import asyncio
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from ..config import Settings
from ..exceptions import SearchError
from ..logging_setup import get_logger
from .base import SearchResult

_logger = get_logger("search.duckduckgo")

_HTML_URL = "https://html.duckduckgo.com/html/"
_LITE_URL = "https://lite.duckduckgo.com/lite/"


def _decode_result_url(href: object) -> str:
    """Decode a DuckDuckGo redirect URL to the underlying target URL."""
    if not isinstance(href, str) or not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in (parsed.netloc or ""):
        # Sponsored results use a /y.js ad redirect — not a real result.
        if parsed.path.startswith("/y.js"):
            return ""
        qs = parse_qs(parsed.query)
        uddg = qs.get("uddg", [""])[0]
        if uddg:
            return unquote(uddg)
    return href


def _is_ad_anchor(anchor) -> bool:  # pragma: no cover - defensive secondary filter
    """True if *anchor* is inside a DuckDuckGo sponsored/ad result block.

    Primary ad filtering is done in :func:`_decode_result_url` (drops ``/y.js``
    ad redirects); this is a secondary check for ads that reuse the normal
    redirect.
    """
    parent = anchor.find_parent("div", class_="result--ad")
    return parent is not None


def _parse_html(html: str) -> list[SearchResult]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    seen: set[str] = set()

    for anchor in soup.select("a.result__a"):
        if _is_ad_anchor(anchor):
            continue
        href = anchor.get("href", "")
        url = _decode_result_url(href)
        if not url or url in seen:
            continue
        title = anchor.get_text(" ", strip=True)
        snippet = ""
        snip_node = anchor.find_parent("div", class_="result")
        if snip_node:
            snip = snip_node.select_one("a.result__snippet")
            if snip:
                snippet = snip.get_text(" ", strip=True)
        seen.add(url)
        results.append(SearchResult(title=title, url=url, snippet=snippet))

    return results


def _parse_lite(html: str) -> list[SearchResult]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    seen: set[str] = set()
    for anchor in soup.select("a.result-link"):
        href = anchor.get("href", "")
        url = _decode_result_url(href)
        if not url or url in seen:
            continue
        title = anchor.get_text(" ", strip=True)
        seen.add(url)
        results.append(SearchResult(title=title, url=url, snippet=""))
    return results


class DuckDuckGoSearchProvider:
    """Keyless DuckDuckGo search via the HTML/Lite endpoints, with retries."""

    name = "duckduckgo"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        headers = {
            "User-Agent": self._settings.user_agent,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        timeout = self._settings.search_timeout
        attempts = max(1, self._settings.search_retry_attempts)

        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, headers=headers
        ) as client:
            for attempt in range(attempts):
                results = await self._try_once(client, query)
                if results:
                    _logger.debug(
                        "DDG search %r -> %d results (attempt %d)",
                        query,
                        len(results),
                        attempt + 1,
                    )
                    return results[:max_results]
                if attempt < attempts - 1:
                    backoff = 1.5 ** (attempt + 1)
                    _logger.debug("DDG empty/rate-limited; retrying in %.1fs", backoff)
                    await asyncio.sleep(backoff)

        raise SearchError(
            f"DuckDuckGo returned no results for: {query!r} after {attempts} attempts "
            "(likely rate-limited). Set SEARCH_PROVIDER=brave or tavily for reliability."
        )

    async def _try_once(
        self, client: httpx.AsyncClient, query: str
    ) -> list[SearchResult]:
        # 1) HTML endpoint (POST is more reliable than GET here).
        try:
            resp = await client.post(_HTML_URL, data={"q": query, "b": ""})
            if resp.status_code == 200:
                results = _parse_html(resp.text)
                if results:
                    return results
        except httpx.HTTPError as exc:
            _logger.debug("DDG HTML endpoint failed: %s", exc)

        # 2) Lite fallback.
        try:
            resp = await client.post(_LITE_URL, data={"q": query})
            if resp.status_code == 200:
                return _parse_lite(resp.text)
        except httpx.HTTPError as exc:
            _logger.debug("DDG Lite endpoint failed: %s", exc)

        return []
