"""Command-line interface for dstools.

Usage::

    dstools serve                       # run the MCP server (stdio)
    dstools serve --transport http      # Streamable HTTP on 127.0.0.1:8000
    dstools inspect                     # list registered tools + schemas
    dstools doctor                      # check what's configured
    dstools version
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from rich.console import Console
from rich.table import Table

from . import __version__
from .config import get_settings
from .logging_setup import configure_logging
from .server import create_server

_console = Console()
# Status banners for `serve` go to stderr so they never corrupt the MCP stdio
# protocol stream (stdout is reserved for MCP messages in stdio transport).
_err = Console(stderr=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dstools",
        description="DeepSeek-V4 MCP toolkit (image understanding + deep research).",
    )
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Run the MCP server.")
    serve.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http", "http"],
        default=None,
        help="MCP transport (default: from config / stdio). 'http' = streamable-http.",
    )
    serve.add_argument("--host", default=None, help="HTTP host (default: 127.0.0.1).")
    serve.add_argument("--port", type=int, default=None, help="HTTP port (default: 8000).")
    serve.add_argument("--log-level", default=None, help="Log level (e.g. DEBUG, INFO).")

    sub.add_parser("inspect", help="List registered tools and their input schemas.")
    sub.add_parser("doctor", help="Check configuration and which capabilities are ready.")
    sub.add_parser("version", help="Print the version and exit.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = args.command or "serve"

    if command == "version":
        _console.print(f"dstools {__version__}")
        return 0
    if command == "serve":
        return _serve(args)
    if command == "inspect":
        return _inspect()
    if command == "doctor":
        return _doctor()
    parser.print_help()
    return 1


def _serve(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    settings = get_settings()
    transport = args.transport or settings.mcp_transport
    if transport == "http":  # friendly alias
        transport = "streamable-http"
    server = create_server(host=args.host, port=args.port, log_level=args.log_level)
    _err.print(
        f"[bold]dstools[/bold] serving over [cyan]{transport}[/cyan]", style="dim"
    )
    if transport != "stdio":
        host = args.host or settings.http_host
        port = args.port or settings.http_port
        _err.print(f"  → http://{host}:{port}/mcp", style="dim")
    server.run(transport=transport)
    return 0


def _inspect() -> int:
    server = create_server()
    tools = asyncio.run(server.list_tools())
    table = Table(title=f"dstools tools ({len(tools)})", show_lines=False)
    table.add_column("Tool", style="bold cyan", no_wrap=True)
    table.add_column("Description", overflow="fold")
    table.add_column("Params", style="green")
    for tool in tools:
        schema = tool.inputSchema or {}
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        params = ", ".join(
            f"{k}{'*' if k in required else ''}" for k in props if k != "ctx"
        )
        desc = (tool.description or "").strip().split("\n", 1)[0]
        table.add_row(tool.name, desc, params or "—")
    _console.print(table)
    return 0


def _doctor() -> int:
    settings = get_settings()
    table = Table(title="dstools configuration")
    table.add_column("Capability", style="bold")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")

    rows = [
        (
            "deep_research (LLM brain)",
            "ready" if settings.has_deepseek else "MISSING",
            f"model={settings.deepseek_model} / fast={settings.deepseek_fast_model} "
            f"@ {settings.deepseek_base_url}",
        ),
        (
            "image understanding (vision)",
            "ready" if settings.has_vision else "missing",
            f"{settings.vision_model or '—'} @ {settings.vision_base_url or '—'}"
            + ("" if settings.has_vision else "  (set VISION_* to enable)"),
        ),
        (
            "web search",
            "ready",
            f"provider={settings.search_provider}"
            + (" (key set)" if settings.search_provider == "tavily" and settings.has_tavily else ""),
        ),
        (
            "page fetch",
            "ready",
            "keyless (direct HTTP + readability)",
        ),
    ]
    for cap, status, detail in rows:
        style = "green" if status == "ready" else ("red" if "MISSING" in status else "yellow")
        table.add_row(cap, f"[{style}]{status}[/{style}]", detail)
    _console.print(table)

    _console.print(
        "\n[dim]Tip: deep_research needs DEEPSEEK_API_KEY; analyze_image needs VISION_*; "
        "web_search/fetch_page need nothing.[/dim]"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
