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
    cache = sub.add_parser("cache", help="Show stats for (or clear) the retrieval cache.")
    cache.add_argument("--clear", action="store_true", help="Delete all cached entries.")
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
    if command == "cache":
        return _cache(args)
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

    search_key_ok = {
        "tavily": settings.has_tavily,
        "brave": settings.has_brave,
        "duckduckgo": True,
    }.get(settings.search_provider, False)

    rows = [
        (
            "deep_research (LLM brain)",
            "ready" if settings.has_deepseek else "MISSING",
            f"synth={settings.research_synth_model or settings.deepseek_model} / "
            f"fast={settings.deepseek_fast_model} @ {settings.deepseek_base_url}",
        ),
        (
            "image understanding (vision)",
            "ready" if settings.has_vision else "missing",
            f"{settings.vision_model or '—'} @ {settings.vision_base_url or '—'}"
            + ("" if settings.has_vision else "  (set VISION_* to enable)"),
        ),
        (
            "web search",
            "ready" if search_key_ok else "no key",
            f"provider={settings.search_provider}"
            + ("" if search_key_ok else "  (set the provider's _API_KEY)"),
        ),
        (
            "page fetch",
            "ready",
            "keyless (direct HTTP + readability)",
        ),
        (
            "retrieval cache",
            "on" if settings.research_cache_enabled else "off",
            f"dir={settings.research_cache_dir} ttl={settings.research_cache_ttl}s",
        ),
    ]
    for cap, status, detail in rows:
        style = "green" if status == "ready" else ("red" if "MISSING" in status else "yellow")
        table.add_row(cap, f"[{style}]{status}[/{style}]", detail)
    _console.print(table)

    if settings.has_deepseek:
        _console.print(f"\n[dim]deep_research cost (est): {_estimate_cost(settings)}[/dim]")
    _console.print(
        "[dim]Tip: deep_research needs DEEPSEEK_API_KEY; analyze_image needs VISION_*; "
        "web_search/fetch_page need nothing. For reliable search use brave/tavily.[/dim]"
    )
    return 0


def _estimate_cost(settings) -> str:
    """Rough per-research USD estimate from default breadth/depth/max_sources."""
    breadth = settings.research_breadth
    depth = settings.research_depth
    sources = settings.research_max_sources
    # Light-step calls: plan + refine*(depth-1) + rerank*sources.
    light_calls = 1 + max(0, depth - 1) + sources
    light_out_tokens = light_calls * 400
    # Rerank reads each page (~per_page_chars/3 tokens); plan/refine add ~800.
    light_in_tokens = sources * (settings.research_per_page_chars // 3) + 800
    # Synthesis reads the reranked excerpts (~sources*400 tokens) + ~1500 out.
    synth_in = sources * 400 + 200
    synth_out = 1500
    # Pricing per 1M tokens (cache miss).
    f_in, f_out = 0.14, 0.28  # flash
    p_in, p_out = 0.435, 0.87  # pro
    light_cost = (light_in_tokens / 1_000_000) * f_in + (light_out_tokens / 1_000_000) * f_out
    synth_cost = (synth_in / 1_000_000) * p_in + (synth_out / 1_000_000) * p_out
    total = light_cost + synth_cost
    synth_model = settings.research_synth_model or settings.deepseek_model
    return (
        f"~${total:.3f} (light={light_calls}x flash, synth=1x {synth_model}; "
        f"breadth={breadth} depth={depth} max_sources={sources}). Varies with content."
    )


def _cache(args: argparse.Namespace) -> int:
    settings = get_settings()
    if not settings.research_cache_enabled:
        _console.print("[yellow]Cache is disabled[/yellow] (RESEARCH_CACHE_ENABLED=false).")
        _console.print("[dim]Enable to cache search results + page markdown (LLM never cached).[/dim]")
        return 0
    from .cache import build_cache

    cache = build_cache(settings)
    if cache is None:  # pragma: no cover - guarded above
        return 1
    if args.clear:
        removed = cache.clear()
        _console.print(f"Cleared {removed} cached entries from {cache.stats()['dir']}.")
        return 0
    stats = cache.stats()
    _console.print(f"Cache dir:  {stats['dir']}")
    _console.print(f"Entries:    {stats['entries']}")
    _console.print(f"Size:       {stats['size_bytes']} bytes")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
