"""Defensive MCP Context helpers.

``Context.info`` / ``Context.report_progress`` require an active request session,
which is absent when a tool is invoked directly (tests, ad-hoc calls) or via a
host that doesn't wire up progress. These helpers swallow those failures so a
logging/progress hiccup can never crash a tool's real work.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context


async def ctx_info(ctx: Context, message: str) -> None:
    """Best-effort ``ctx.info``; never raises."""
    with contextlib.suppress(Exception):
        await ctx.info(message)


async def ctx_progress(
    ctx: Context, *, progress: float, total: float | None, message: str
) -> None:
    """Best-effort ``ctx.report_progress``; never raises."""
    with contextlib.suppress(Exception):
        await ctx.report_progress(progress=progress, total=total, message=message)
