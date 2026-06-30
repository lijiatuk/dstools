"""Deep-research tool: a multi-step, citation-backed research pipeline.

Pipeline (DeepSeek-V4 is the planning + extraction + synthesis brain):

1. **Plan**    — V4-flash decomposes the question into `breadth` search queries.
2. **Round loop** (`depth` rounds):
   a. **Search** each query (concurrent) via the configured search provider.
   b. **Fetch**  new pages (concurrent) and extract text.
   c. **Refine** — between rounds, V4-flash reads findings-so-far and generates
      next-round queries targeting uncovered facets (STORM-style).
3. **Rerank**  — V4-flash extracts the passages most relevant to the question
   from each fetched page (always-on; quality over the raw "stuff everything"
   approach).
4. **Synthesize** — V4-pro writes a structured, citation-backed report from the
   reranked context (thinking on; stable system prompts for DeepSeek's
   automatic prefix caching).

Per-step models are configurable (``RESEARCH_{PLAN,REFINE,RERANK,SYNTH}_MODEL``);
defaults are flash for the light steps and pro for synthesis.

The granular tools (``web_search``, ``fetch_page``, ``analyze_image``) let a host
agent run its own agentic loop; ``deep_research`` is the one-shot orchestrator.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from ..config import Settings, get_settings
from ..exceptions import ConfigError, DSToolsError
from ..llm import DeepSeekClient
from ..logging_setup import get_logger
from ..runtime import get_deepseek_client, get_fetcher, get_search_provider
from ..search import SearchProvider, SearchResult
from ..utils.text import extract_json_list, truncate
from ..web.fetcher import Fetcher
from ._ctx import ctx_info, ctx_progress

_logger = get_logger("tools.research")

ProgressFn = Callable[[str, str], Awaitable[None] | None]

# Stable system prompts (constant across calls -> DeepSeek prefix caching).
_PLAN_SYSTEM = (
    "You are an expert research planner. You output ONLY a JSON array of "
    "search-engine query strings — no prose, no code fences, no explanation."
)
_REFINE_SYSTEM = (
    "You are a research query refiner. Given the question, the queries already "
    "tried, and a brief summary of findings so far, output ONLY a JSON array of "
    "NEW search-engine query strings that explore UNCOVERED facets — no prose, "
    "no code fences."
)
_RERANK_SYSTEM = (
    "You are a relevance extractor. From the provided page content, extract the "
    "passages most relevant to answering the research question. Return concise, "
    "faithful excerpts preserving key facts, numbers, and quotes. Do NOT add "
    "information. If nothing is relevant, return exactly: NO_RELEVANT_CONTENT."
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

_NO_RELEVANT = "NO_RELEVANT_CONTENT"


@dataclass
class _Fetched:
    """A search result plus its extracted page text."""

    result: SearchResult
    text: str  # truncated page text, or snippet fallback


async def deep_research_logic(
    query: str,
    *,
    breadth: int | None = None,
    depth: int | None = None,
    max_sources: int | None = None,
    settings: Settings | None = None,
    llm: DeepSeekClient | None = None,
    search: SearchProvider | None = None,
    fetcher: Fetcher | None = None,
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
    queries = await _plan_queries(llm, query, breadth, settings)

    fetched: list[_Fetched] = []
    seen_urls: set[str] = set()

    for rnd in range(1, depth + 1):
        await _maybe_await(
            on_progress, "search", f"Round {rnd}/{depth}: searching {len(queries)} queries"
        )
        results = await _search_concurrent(search, queries, settings.search_max_results)
        new_results = [r for r in _dedupe(results) if r.url not in seen_urls]
        remaining = max(0, max_sources - len(fetched))
        new_results = new_results[:remaining]
        for r in new_results:
            seen_urls.add(r.url)

        if new_results:
            await _maybe_await(
                on_progress, "fetch", f"Round {rnd}: fetching {len(new_results)} pages"
            )
            texts = await _fetch_concurrent(fetcher, new_results, per_page)
            fetched.extend(_Fetched(r, t) for r, t in zip(new_results, texts, strict=True))

        if rnd < depth and fetched:
            await _maybe_await(
                on_progress, "refine", f"Refining queries from round {rnd} findings"
            )
            queries = await _refine_queries(llm, query, breadth, queries, fetched, settings)

    if not fetched:
        return (
            f"deep_research could not find any web results for: {query!r}. "
            "Try rephrasing, or check that the search provider is reachable."
        )

    fetched = fetched[:max_sources]

    await _maybe_await(
        on_progress, "rerank", f"Extracting relevant passages from {len(fetched)} sources"
    )
    excerpts = await _rerank_concurrent(llm, query, fetched, settings)

    context = _assemble_context(fetched, excerpts, total_budget)

    await _maybe_await(on_progress, "synthesize", "Writing report with DeepSeek-V4")
    report = await _synthesize(llm, query, context, settings)

    sources_md = "\n".join(
        f"{i}. [{f.result.title or f.result.url}]({f.result.url})"
        for i, f in enumerate(fetched, 1)
    )
    await _maybe_await(on_progress, "done", "Report complete")

    return f"{report.strip()}\n\n---\n## Sources\n\n{sources_md}\n"


# --- pipeline steps --------------------------------------------------------


def _fast_model(settings: Settings, override: str) -> str:
    return override or settings.deepseek_fast_model


async def _plan_queries(
    llm: DeepSeekClient, query: str, breadth: int, settings: Settings
) -> list[str]:
    """V4-flash (non-thinking, JSON mode) -> initial search sub-queries."""
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
            model=_fast_model(settings, settings.research_plan_model),
            thinking="off",
            json_mode=True,
            max_tokens=512,
        )
        queries = [str(q).strip() for q in (extract_json_list(resp.content) or []) if str(q).strip()]
    except DSToolsError as exc:
        _logger.warning("planning failed (%s); falling back to simple queries", exc)
        queries = []

    return _stabilise_queries(query, queries, breadth)


async def _refine_queries(
    llm: DeepSeekClient,
    query: str,
    breadth: int,
    prev_queries: list[str],
    fetched: list[_Fetched],
    settings: Settings,
) -> list[str]:
    """V4-flash (non-thinking, JSON) -> next-round queries from findings so far."""
    summary = "\n".join(
        f"- {f.result.title or f.result.url}: {f.result.snippet}".strip()
        for f in fetched[:12]
    )
    tried = "; ".join(prev_queries)
    user = (
        f"Research question: {query}\n"
        f"Queries already tried: {tried}\n"
        f"Findings so far:\n{summary}\n\n"
        f"Generate {breadth} NEW search queries targeting facets NOT yet covered. "
        f"Same language as the question. Output a JSON array of {breadth} strings, "
        f"nothing else."
    )
    try:
        resp = await llm.complete(
            messages=[
                {"role": "system", "content": _REFINE_SYSTEM},
                {"role": "user", "content": user},
            ],
            model=_fast_model(settings, settings.research_refine_model),
            thinking="off",
            json_mode=True,
            max_tokens=512,
        )
        queries = [str(q).strip() for q in (extract_json_list(resp.content) or []) if str(q).strip()]
    except DSToolsError as exc:
        _logger.warning("refine failed (%s); reusing previous queries", exc)
        queries = list(prev_queries)

    return _stabilise_queries(query, queries, breadth, allow_original=False)


def _stabilise_queries(
    query: str, queries: list[str], breadth: int, *, allow_original: bool = True
) -> list[str]:
    """Dedup, optionally prepend the original question, pad/truncate to breadth."""
    seed = [query] if allow_original else []
    ordered = seed + [q for q in queries if q]
    seen: set[str] = set()
    unique: list[str] = []
    for q in ordered:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    if len(unique) >= breadth:
        return unique[:breadth]
    padder = query if allow_original else (unique[-1] if unique else query)
    return unique + [padder] * (breadth - len(unique))


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


def _dedupe(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in results:
        if not r.url or r.url in seen:
            continue
        seen.add(r.url)
        out.append(r)
    return out


async def _fetch_concurrent(
    fetcher: Fetcher, sources: list[SearchResult], per_page_chars: int
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


async def _rerank_concurrent(
    llm: DeepSeekClient, query: str, fetched: list[_Fetched], settings: Settings
) -> list[str]:
    """For each source, V4-flash extracts the passages relevant to *query*."""
    sem = asyncio.Semaphore(4)
    results: list[str] = [""] * len(fetched)
    model = _fast_model(settings, settings.research_rerank_model)

    async def one(idx: int, src: _Fetched) -> None:
        async with sem:
            text = src.text or src.result.snippet or "(no content retrieved)"
            user = (
                f"Research question: {query}\n\n"
                f"Page content:\n{truncate(text, settings.research_per_page_chars)}\n\n"
                f"Extract the most relevant passages."
            )
            try:
                resp = await llm.complete(
                    messages=[
                        {"role": "system", "content": _RERANK_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    model=model,
                    thinking="off",
                    max_tokens=1024,
                )
                results[idx] = resp.content
            except DSToolsError as exc:
                _logger.debug("rerank failed for %s: %s", src.result.url, exc)
                results[idx] = ""  # assemble will fall back to the raw text

    await asyncio.gather(*(one(i, f) for i, f in enumerate(fetched)))
    return results


def _assemble_context(
    fetched: list[_Fetched], excerpts: list[str], total_budget: int
) -> str:
    blocks: list[str] = []
    used = 0
    for i, (src, excerpt) in enumerate(zip(fetched, excerpts, strict=True), 1):
        exc = excerpt.strip()
        body = exc if exc and _NO_RELEVANT not in exc else ""
        if not body:
            body = (src.text or src.result.snippet or "(no content retrieved)").strip()
        block = f"[{i}] {src.result.title or src.result.url}\nURL: {src.result.url}\n{body}"
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
    # Heavy model; thinking defaults to "auto" (client resolves from global setting).
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
        model=settings.research_synth_model or settings.deepseek_model,
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

        Pipeline: decompose `query` into search queries -> search the web -> fetch &
        read pages -> (between rounds) refine queries from findings -> extract the
        most relevant passages -> synthesise a structured markdown report with inline
        [n] citations and a numbered source list. Uses DeepSeek-V4 (flash for
        plan/refine/rerank, pro for synthesis).

        Tunables (0 = use defaults from config): `breadth` sub-queries per round
        (default 3), `depth` rounds (default 2), `max_sources` pages read
        (default 8). Higher = more thorough but slower/costlier.
        Requires DEEPSEEK_API_KEY; search/fetch are keyless (DuckDuckGo).
        """
        settings = get_settings()
        eff_depth = depth or settings.research_depth
        # Rough total for progress: plan + (search+fetch)*depth + refine*(depth-1)
        # + rerank + synth + done.
        total = 1 + eff_depth * 2 + max(0, eff_depth - 1) + 3
        state = {"n": 0}

        async def on_progress(step: str, message: str) -> None:
            state["n"] += 1
            await ctx_progress(
                ctx, progress=state["n"], total=total, message=message
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
