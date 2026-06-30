"""Page-fetch tool: URL → clean readable Markdown."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from ..config import Settings, get_settings
from ..exceptions import DSToolsError
from ..logging_setup import get_logger
from ..runtime import get_fetcher
from ..web.fetcher import Fetcher
from ._ctx import ctx_info

_logger = get_logger("tools.fetch")


async def fetch_page_logic(
    url: str,
    *,
    max_chars: int = 20_000,
    settings: Settings | None = None,
    fetcher: Fetcher | None = None,
) -> str:
    """Fetch *url* and return its main content as Markdown."""
    settings = settings or get_settings()
    fetcher = fetcher or get_fetcher()
    page = await fetcher.fetch(url, max_chars=max_chars)
    return page.to_markdown(max_chars)


def register(mcp: FastMCP) -> None:
    """Register the fetch_page tool on *mcp*."""

    @mcp.tool(name="fetch_page")
    async def fetch_page(
        url: str,
        ctx: Context,
        max_chars: int = 20000,
    ) -> str:
        """Fetch a web page and return clean, readable Markdown.

        Use this to read the full content of a URL (e.g. a search result). Strips
        navigation/ads and extracts the main article text. Output is capped to
        `max_chars` characters. Works without any API key.
        """
        await ctx_info(ctx, f"fetch_page: {url}")
        try:
            return await fetch_page_logic(url, max_chars=max_chars)
        except DSToolsError as exc:
            return f"fetch_page failed: {exc}"


__all__: list[Any] = ["fetch_page_logic", "register"]
