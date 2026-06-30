"""Deep-research tool: a multi-step, citation-backed research pipeline.

Pipeline (uses DeepSeek-V4 as the planning + synthesis brain):

1. **Plan**  — V4-flash decomposes the question into `breadth` search queries.
2. **Search** — run each query (concurrent) via the configured search provider.
3. **Fetch**  — fetch & extract the top `max_sources` pages (concurrent).
4. **Synthesize** — V4-pro writes a structured, citation-backed report from the
   gathered context (thinking mode on for hard synthesis; stable system prompt
   for DeepSeek's automatic prefix caching).

The granular tools (``web_search``, ``fetch_page``, ``analyze_image``) let a host
agent run its own agentic loop; ``deep_research`` is the one-shot orchestrator.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from ..config import Settings, get_settings
from ..exceptions import ConfigError, DSToolsError
from ..llm import DeepSeekClient
from ..logging_setup import get_logger
from ..runtime import get_deepseek_client, get_fetcher, get_search_provider
from ..search import SearchProvider, SearchResult
from ..utils.text import extract_json_list, truncate
from ..web.fetcher import PageFetcher
from ._ctx import ctx_info, ctx_progress

_logger = get_logger("tools.research")

ProgressFn = Callable[[str, str], Awaitable[None] | None]

# Stable system prompts (kept constant across calls so DeepSeek's automatic
# context caching can reuse the prefix -> cheaper repeated research).
_PLAN_SYSTEM = (
    "You are an expert research planner. You output ONLY a JSON array of "
    "search-engine query strings — no prose, no code fences, no explanation."
)
_SYNTH_SYSTEM = (
    "You are dstools' deep-research analyst, powered by DeepSeek-V4. You write "
    "thorough, accurate, well-structured markdown reports from gathered web sources.\n\n"
    "Rules:\n"
    "- Address the user's research question directly and comprehensively.\n"
    "- Use markdown headings and bullet points for readability.\n"
    "- Cite every factual claim with an inline [n] marker matching the numbered "
    "source list provided.\n"
    "- Where sources disagree, surface the disagreement and give your best judgment.\n"
    "- Use ONLY information present in the provided sources. If something is missing, "
    "say so explicitly rather than guessing.\n"
    "- Write in the same language as the user's question.\n"
    "- End with a '## Key findings' section of 3-6 concise bullet points."
)


async def deep_research_logic(
    query: str,
    *,
    breadth: int | None = None,
    depth: int | None = None,
    max_sources: int | None = None,
    settings: Settings | None = None,
    llm: DeepSeekClient | None = None,
    search: SearchProvider | None = None,
    fetcher: PageFetcher | None = None,
    on_progress: ProgressFn | None = None,
) -> str:
    """Run the full research pipeline and return a cited markdown report."""
    settings = settings or get_settings()
    llm = llm or get_deepseek_client()
    search = search or get_search_provider()
    fetcher = fetcher or get_fetcher()

    if not llm.configured:
        raise ConfigError(
            "deep_research requires a DeepSeek API key (DEEPSEEK_API_KEY). The "
            "keyless search/fetch steps work without it, but planning & synthesis use V4."
        )

    breadth = breadth or settings.research_breadth
    depth = depth or settings.research_depth
    max_sources = max_sources or settings.research_max_sources
    per_page = settings.research_per_page_chars
    total_budget = settings.research_total_chars

    await _maybe_await(on_progress, "plan", f"Decomposing question into {breadth} queries")
    sub_queries = await _plan_queries(llm, query, breadth, settings)

    # --- Search -----------------------------------------------------------
    await _maybe_await(on_progress, "search", f"Searching {len(sub_queries)} queries")
    all_results: list[SearchResult] = []
    for _round in range(max(1, depth)):
        round_results = await _search_concurrent(search, sub_queries, settings.search_max_results)
        all_results.extend(round_results)

    sources = _dedupe_and_cap(all_results, max_sources)
    if not sources:
        return (
            f"deep_research could not find any web results for: {query!r}. "
            "Try rephrasing, or check that the search provider is reachable."
        )
    await _maybe_await(on_progress, "search", f"{len(sources)} sources selected")

    # --- Fetch ------------------------------------------------------------
    await _maybe_await(on_progress, "fetch", f"Fetching {len(sources)} pages")
    texts = await _fetch_concurrent(fetcher, sources, per_page)

    context = _assemble_context(sources, texts, total_budget)

    # --- Synthesize -------------------------------------------------------
    await _maybe_await(on_progress, "synthesize", "Writing report with DeepSeek-V4")
    report = await _synthesize(llm, query, context, settings)

    sources_md = "\n".join(f"{i}. [{s.title or s.url}]({s.url})" for i, s in enumerate(sources, 1))
    await _maybe_await(on_progress, "done", "Report complete")

    return f"{report.strip()}\n\n---\n## Sources\n\n{sources_md}\n"


# --- pipeline steps --------------------------------------------------------


async def _plan_queries(
    llm: DeepSeekClient, query: str, breadth: int, settings: Settings
) -> list[str]:
    """Use V4-flash (non-thinking, JSON mode) to generate search sub-queries."""
    user = (
        f"Research question: {query}\n\n"
        f"Generate {breadth} diverse, specific web-search queries that together "
        f"thoroughly investigate this question. Each should target a different facet "
        f"(definition, recent developments, comparisons, evidence, primary sources). "
        f"Queries must be in the same language as the question. Output a JSON array "
        f"of {breadth} strings, nothing else."
    )
    try:
        resp = await llm.complete(
            messages=[
                {"role": "system", "content": _PLAN_SYSTEM},
                {"role": "user", "content": user},
            ],
            model=settings.deepseek_fast_model,
            thinking="off",
            json_mode=True,
            max_tokens=512,
        )
        queries = extract_json_list(resp.content) or []
        queries = [str(q).strip() for q in queries if str(q).strip()]
    except DSToolsError as exc:
        _logger.warning("planning failed (%s); falling back to simple queries", exc)
        queries = []

    if not queries:
        queries = [query, f"{query} overview", f"{query} latest"]

    # Always include the original question, dedup, cap to breadth.
    queries = [query, *[q for q in queries if q != query]]
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique[:breadth] if len(unique) >= breadth else unique + [query] * (breadth - len(unique))


async def _search_concurrent(
    search: SearchProvider, queries: list[str], max_per_query: int
) -> list[SearchResult]:
    sem = asyncio.Semaphore(4)

    async def one(q: str) -> list[SearchResult]:
        async with sem:
            try:
                return await search.search(q, max_results=max_per_query)
            except Exception as exc:  # one query failing is fine
                _logger.debug("search failed for %r: %s", q, exc)
                return []

    gathered = await asyncio.gather(*(one(q) for q in queries))
    return [r for sub in gathered for r in sub]


def _dedupe_and_cap(results: list[SearchResult], max_sources: int) -> list[SearchResult]:
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in results:
        if not r.url or r.url in seen:
            continue
        seen.add(r.url)
        out.append(r)
        if len(out) >= max_sources:
            break
    return out


async def _fetch_concurrent(
    fetcher: PageFetcher, sources: list[SearchResult], per_page_chars: int
) -> list[str]:
    sem = asyncio.Semaphore(5)
    results: list[str] = [""] * len(sources)

    async def one(idx: int, src: SearchResult) -> None:
        async with sem:
            try:
                page = await fetcher.fetch(src.url, max_chars=per_page_chars)
                results[idx] = truncate(page.text, per_page_chars)
            except Exception as exc:
                _logger.debug("fetch failed for %s: %s", src.url, exc)
                results[idx] = src.snippet  # best-effort fallback

    await asyncio.gather(*(one(i, s) for i, s in enumerate(sources)))
    return results


def _assemble_context(
    sources: list[SearchResult], texts: list[str], total_budget: int
) -> str:
    blocks: list[str] = []
    used = 0
    for i, (src, text) in enumerate(zip(sources, texts, strict=True), 1):
        body = (text or src.snippet or "(no content retrieved)").strip()
        block = f"[{i}] {src.title or src.url}\nURL: {src.url}\n{body}"
        if used + len(block) > total_budget:
            remaining = total_budget - used
            if remaining > 200:
                blocks.append(truncate(block, remaining))
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


async def _synthesize(
    llm: DeepSeekClient, query: str, context: str, settings: Settings
) -> str:
    # Use the heavy model; thinking defaults to "auto" which the client resolves
    # from the global setting (enabled unless the user explicitly disabled it).
    user = (
        f"# Research question\n{query}\n\n"
        f"# Sources (gathered from the web)\n{context}\n\n"
        f"# Task\nWrite the report now, citing every claim as [n]."
    )
    resp = await llm.complete(
        messages=[
            {"role": "system", "content": _SYNTH_SYSTEM},
            {"role": "user", "content": user},
        ],
        model=settings.deepseek_model,
        reasoning_effort=settings.deepseek_reasoning_effort,
    )
    if not resp.content.strip():
        raise DSToolsError("Synthesis produced an empty report.")
    return resp.content


# --- helpers ---------------------------------------------------------------


async def _maybe_await(fn: ProgressFn | None, step: str, message: str) -> None:
    if fn is None:
        return
    result = fn(step, message)
    if asyncio.iscoroutine(result):
        await result


# --- MCP registration ------------------------------------------------------


def register(mcp: FastMCP) -> None:
    """Register the deep_research tool on *mcp*."""

    @mcp.tool(name="deep_research")
    async def deep_research(
        query: str,
        ctx: Context,
        breadth: int = 0,
        depth: int = 0,
        max_sources: int = 0,
    ) -> str:
        """Run a multi-step deep-research investigation and return a cited report.

        Decomposes `query` into search queries, searches the web, fetches and reads
        the top pages, then synthesises a structured markdown report with inline
        [n] citations and a numbered source list. Ideal for questions needing
        up-to-date, multi-source synthesis (uses DeepSeek-V4 for planning/synthesis).

        Tunables (0 = use defaults from config): `breadth` sub-queries per round
        (default 3), `depth` rounds (default 2), `max_sources` pages read
        (default 8). Higher values are more thorough but slower/costlier.
        Requires DEEPSEEK_API_KEY; search/fetch are keyless.
        """
        progress_total = 5

        async def on_progress(step: str, message: str) -> None:
            step_index = {"plan": 1, "search": 2, "fetch": 3, "synthesize": 4, "done": 5}.get(
                step, 0
            )
            await ctx_progress(
                ctx, progress=step_index, total=progress_total, message=message
            )
            await ctx_info(ctx, f"deep_research [{step}]: {message}")

        await ctx_info(ctx, f"deep_research: {query}")
        try:
            return await deep_research_logic(
                query,
                breadth=breadth or None,
                depth=depth or None,
                max_sources=max_sources or None,
                on_progress=on_progress,
            )
        except DSToolsError as exc:
            return f"deep_research failed: {exc}"


__all__: list[Any] = ["deep_research_logic", "register"]
