"""Minimal MCP client demo for dstools.

Launches the dstools MCP server as a stdio subprocess, lists its tools, and
calls the two keyless tools (``web_search`` + ``fetch_page``) end-to-end over the
real MCP protocol — no API key required.

Run from the project root:

    uv run python examples/mcp_client_demo.py

It doubles as a smoke test that the server speaks MCP correctly over stdio.
"""

from __future__ import annotations

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "dstools", "serve"],
        env=None,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected to dstools MCP server.\n")

            tools = await session.list_tools()
            print(f"Tools ({len(tools.tools)}):")
            for t in tools.tools:
                print(f"  - {t.name}")
            print()

            print("Calling web_search('DeepSeek V4 release')...")
            res = await session.call_tool(
                "web_search", {"query": "DeepSeek V4 release", "max_results": 3}
            )
            print(_text(res), end="\n\n")

            print("Calling fetch_page('https://example.com')...")
            res = await session.call_tool(
                "fetch_page", {"url": "https://example.com", "max_chars": 400}
            )
            print(_text(res)[:400])


def _text(result) -> str:
    parts = []
    for block in getattr(result, "content", []) or []:
        txt = getattr(block, "text", None)
        if txt:
            parts.append(txt)
    return "\n".join(parts)


if __name__ == "__main__":
    asyncio.run(main())
