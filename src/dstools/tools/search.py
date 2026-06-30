"""Web-search tool (keyless DuckDuckGo by default)."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from ..config import Settings, get_settings
from ..exceptions import DSToolsError
from ..logging_setup import get_logger
from ..runtime import get_search_provider
from ..search import SearchProvider, SearchResult
from ._ctx import ctx_info

_logger = get_logger("tools.search")


async def web_search_logic(
    query: str,
    *,
    max_results: int | None = None,
    settings: Settings | None = None,
    search: SearchProvider | None = None,
) -> str:
    """Run a web search and return ranked results as markdown."""
    settings = settings or get_settings()
    search = search or get_search_provider()
    max_results = max_results or settings.search_max_results

    results = await search.search(query, max_results=max_results)
    return _format_results(query, results)


def _format_results(query: str, results: list[SearchResult]) -> str:
    if not results:
        return f"No results found for: {query!r}"
    lines = [f"**{len(results)} web results for:** `{query}`", ""]
    for i, r in enumerate(results, 1):
        lines.append(r.to_markdown(i))
    return "\n".join(lines)


def register(mcp: FastMCP) -> None:
    """Register the web_search tool on *mcp*."""

    @mcp.tool(name="web_search")
    async def web_search(
        query: str,
        ctx: Context,
        max_results: int = 10,
    ) -> str:
        """Search the web and return ranked results (title, URL, snippet).

        Use this for live, up-to-date information. Returns up to `max_results`
        hits as a numbered markdown list. Follow up with `fetch_page` to read any
        result in full. Works without any API key (DuckDuckGo).
        """
        await ctx_info(ctx, f"web_search: {query}")
        try:
            return await web_search_logic(query, max_results=max_results)
        except DSToolsError as exc:
            return f"web_search failed: {exc}"


__all__: list[Any] = ["register", "web_search_logic"]
