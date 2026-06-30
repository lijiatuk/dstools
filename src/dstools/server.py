"""The FastMCP server: builds the server instance and registers all tools."""

from __future__ import annotations

from typing import cast

from mcp.server.fastmcp import FastMCP

from .config import LogLevel, get_settings
from .logging_setup import configure_logging, get_logger
from .tools import fetch, image, research, search

_logger = get_logger("server")

INSTRUCTIONS = (
    "dstools augments DeepSeek-V4 (a text-only model) with two capabilities it "
    "lacks natively: image understanding and deep web research.\n\n"
    "Tools:\n"
    "- analyze_image / ocr_image: turn an image into text (vision provider required;\n"
    "  falls back to metadata + OCR).\n"
    "- web_search / fetch_page: live, keyless web search and page reading.\n"
    "- deep_research: a multi-step, citation-backed report (needs DEEPSEEK_API_KEY).\n\n"
    "Search and fetch work without any API key. deep_research uses DeepSeek-V4 for\n"
    "planning and synthesis."
)


def create_server(
    *, host: str | None = None, port: int | None = None, log_level: str | None = None
) -> FastMCP:
    """Build a fresh FastMCP server with all dstools tools registered."""
    settings = get_settings()
    server = FastMCP(
        name="dstools",
        instructions=INSTRUCTIONS,
        host=host or settings.http_host,
        port=port or settings.http_port,
        log_level=cast(LogLevel, log_level or settings.log_level),
    )
    image.register(server)
    search.register(server)
    fetch.register(server)
    research.register(server)
    _logger.debug("registered tools on FastMCP server")
    return server


# A module-level instance for convenience (e.g. `from dstools.server import mcp`).
mcp = create_server()


def run(transport: str = "stdio") -> None:
    """Configure logging and run the module-level server over *transport*."""
    configure_logging()
    _logger.info("starting dstools MCP server (transport=%s)", transport)
    mcp.run(transport=transport)  # type: ignore[arg-type]


__all__ = ["INSTRUCTIONS", "create_server", "mcp", "run"]
