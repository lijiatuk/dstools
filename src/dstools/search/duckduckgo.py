"""DuckDuckGo search provider (keyless, HTML endpoint).

Scrapes DuckDuckGo's lightweight HTML endpoint — no API key, works out of the
box. Two endpoints are tried (HTML then Lite) for robustness against layout
changes. This is the default :data:`SEARCH_PROVIDER`.
"""

from __future__ import annotations

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
        qs = parse_qs(parsed.query)
        uddg = qs.get("uddg", [""])[0]
        if uddg:
            return unquote(uddg)
    return href


def _parse_html(html: str) -> list[SearchResult]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    seen: set[str] = set()

    for anchor in soup.select("a.result__a"):
        href = anchor.get("href", "")
        url = _decode_result_url(href)
        if not url or url in seen:
            continue
        title = anchor.get_text(" ", strip=True)
        # Snippet: sibling result__snippet (DuckDuckGo nests it in a.result__snippet).
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
    # The Lite endpoint lays results out in a table; links land in .result-link.
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
    """Keyless DuckDuckGo search via the HTML/Lite endpoints."""

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
        results: list[SearchResult] = []

        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, headers=headers
        ) as client:
            # 1) Try the HTML endpoint (POST is more reliable than GET here).
            try:
                resp = await client.post(
                    _HTML_URL, data={"q": query, "b": ""}
                )
                if resp.status_code == 200:
                    results = _parse_html(resp.text)
            except httpx.HTTPError as exc:
                _logger.debug("DDG HTML endpoint failed: %s", exc)

            # 2) Fall back to the Lite endpoint if HTML returned nothing.
            if not results:
                try:
                    resp = await client.post(_LITE_URL, data={"q": query})
                    if resp.status_code == 200:
                        results = _parse_lite(resp.text)
                except httpx.HTTPError as exc:
                    _logger.debug("DDG Lite endpoint failed: %s", exc)

        if not results:
            raise SearchError(
                f"DuckDuckGo returned no results for: {query!r} "
                "(the endpoint may be rate-limiting; try again or set SEARCH_PROVIDER=tavily)."
            )

        _logger.debug("DDG search %r -> %d results", query, len(results))
        return results[:max_results]
