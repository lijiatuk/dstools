"""Tests for the deep_research pipeline (logic-level, with fakes)."""

from __future__ import annotations

from urllib.parse import quote

import pytest

from dstools.exceptions import ConfigError
from dstools.llm import DeepSeekClient, LLMResponse
from dstools.search import SearchResult
from dstools.tools.research import (
    _assemble_context,
    _dedupe,
    _Fetched,
    _stabilise_queries,
    deep_research_logic,
)
from dstools.web.fetcher import PageData

from .conftest import make_settings


class FakeLLM:
    """Routes responses by the system-prompt marker of each call."""

    def __init__(
        self, *, plan: str, refine: str, rerank: str, synth: str, configured: bool = True
    ) -> None:
        self._plan, self._refine, self._rerank, self._synth = plan, refine, rerank, synth
        self._configured = configured
        self.calls: list[dict] = []

    @property
    def configured(self) -> bool:
        return self._configured

    async def complete(self, *, messages, **kwargs):
        sys_prompt = messages[0]["content"] if messages else ""
        self.calls.append({**kwargs, "_sys": sys_prompt, "_user": messages[-1]["content"] if messages else ""})
        if "research planner" in sys_prompt:
            content = self._plan
        elif "research query refiner" in sys_prompt:
            content = self._refine
        elif "relevance extractor" in sys_prompt:
            content = self._rerank
        elif "deep-research analyst" in sys_prompt:
            content = self._synth
        else:
            content = "?"
        return LLMResponse(content=content, model="fake")


class PlanFailingLLM:
    """Planning/refine (json) fail; rerank/synth succeed."""

    configured = True

    def __init__(self, *, rerank: str, synth: str) -> None:
        self._rerank, self._synth = rerank, synth
        self.calls: list[dict] = []

    async def complete(self, *, messages, **kwargs):
        sys_prompt = messages[0]["content"] if messages else ""
        self.calls.append(kwargs)
        from dstools.exceptions import LLMError

        if "research planner" in sys_prompt or "research query refiner" in sys_prompt:
            raise LLMError("plan/refine boom")
        if "relevance extractor" in sys_prompt:
            return LLMResponse(content=self._rerank, model="fake")
        return LLMResponse(content=self._synth, model="fake")


class FakeSearch:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        self.calls.append(query)
        return [
            SearchResult(
                title=f"Title: {query}",
                url=f"https://example.com/{quote(query)}",
                snippet=f"snippet for {query}",
            )
        ]


class FakeFetcher:
    async def fetch(self, url: str, *, max_chars: int = 20000) -> PageData:
        return PageData(
            url=url,
            final_url=url,
            status=200,
            content_type="text/html",
            title=f"Page {url}",
            text=f"Body content for {url} mentioning key facts.",
            truncated=False,
        )


def _llm() -> FakeLLM:
    return FakeLLM(
        plan='["sub q1","sub q2"]',
        refine='["refined q1","refined q2"]',
        rerank="EXCERPT: DeepSeek-V4 has 1M context and thinking mode.",
        synth="# Report\nDeepSeek-V4 is a model. [1]\n\n## Key findings\n- a",
    )


async def test_pipeline_produces_cited_report_with_sources():
    settings = make_settings(research_breadth=2, research_depth=1, research_max_sources=3)
    llm = _llm()
    report = await deep_research_logic(
        "What is DeepSeek V4?", settings=settings, llm=llm,
        search=FakeSearch(), fetcher=FakeFetcher(),
    )
    assert "# Report" in report
    assert "## Sources" in report
    assert "https://example.com/" in report
    # rerank ran for each fetched source (2 queries -> 2 sources).
    rerank_calls = [c for c in llm.calls if "relevance extractor" in c["_sys"]]
    assert len(rerank_calls) >= 2


async def test_pipeline_planning_uses_flash_and_json():
    settings = make_settings(research_breadth=2, research_depth=1, research_max_sources=3)
    llm = _llm()
    await deep_research_logic(
        "q", settings=settings, llm=llm, search=FakeSearch(), fetcher=FakeFetcher()
    )
    plan = next(c for c in llm.calls if "research planner" in c["_sys"])
    assert plan["json_mode"] is True
    assert plan["thinking"] == "off"
    assert plan["model"] == "deepseek-v4-flash"  # fast model
    synth = next(c for c in llm.calls if "deep-research analyst" in c["_sys"])
    assert synth["model"] == "deepseek-v4-pro"  # heavy model


async def test_pipeline_rerank_excerpt_reaches_synthesis():
    settings = make_settings(research_breadth=2, research_depth=1, research_max_sources=3)
    llm = _llm()
    await deep_research_logic(
        "q", settings=settings, llm=llm, search=FakeSearch(), fetcher=FakeFetcher()
    )
    synth = next(c for c in llm.calls if "deep-research analyst" in c["_sys"])
    # The rerank excerpt is part of the context handed to synthesis.
    assert "EXCERPT" in synth["_user"]


async def test_pipeline_refines_between_rounds_at_depth_2():
    settings = make_settings(research_breadth=2, research_depth=2, research_max_sources=4)
    llm = _llm()
    search = FakeSearch()
    await deep_research_logic(
        "q", settings=settings, llm=llm, search=search, fetcher=FakeFetcher()
    )
    refine_calls = [c for c in llm.calls if "research query refiner" in c["_sys"]]
    assert len(refine_calls) == 1  # one refine between round 1 and round 2
    # Round 2 used the refined queries (they were searched).
    assert "refined q1" in search.calls


async def test_pipeline_planning_fallback_still_reports():
    settings = make_settings(research_breadth=2, research_depth=1, research_max_sources=3)
    llm = PlanFailingLLM(rerank="EXCERPT: fallback.", synth="# Report\nbody [1]")
    report = await deep_research_logic(
        "q", settings=settings, llm=llm, search=FakeSearch(), fetcher=FakeFetcher()
    )
    assert "# Report" in report


async def test_pipeline_no_sources_returns_message():
    settings = make_settings(research_breadth=2, research_depth=1, research_max_sources=2)

    class EmptySearch:
        name = "empty"

        async def search(self, query, max_results=10):
            return []

    llm = _llm()
    report = await deep_research_logic(
        "q", settings=settings, llm=llm, search=EmptySearch(), fetcher=FakeFetcher()
    )
    assert "could not find any web results" in report


async def test_pipeline_raises_config_error_without_llm():
    settings = make_settings()  # no key
    llm = DeepSeekClient(settings)  # not configured
    with pytest.raises(ConfigError):
        await deep_research_logic(
            "q", settings=settings, llm=llm, search=FakeSearch(), fetcher=FakeFetcher()
        )


def test_dedupe_dedupes_and_skips_empty_urls():
    r = [
        SearchResult("a", "u1", "s"),
        SearchResult("b", "u1", "s2"),
        SearchResult("c", "u2", ""),
        SearchResult("d", "", ""),
    ]
    assert [o.url for o in _dedupe(r)] == ["u1", "u2"]


def test_stabilise_queries_prepends_original_and_pads():
    out = _stabilise_queries("Q", ["a", "a", "b"], 3)
    assert out == ["Q", "a", "b"]


def test_stabilise_queries_pads_when_short():
    out = _stabilise_queries("Q", ["a"], 3)
    assert out == ["Q", "a", "Q"]


def test_assemble_context_uses_excerpt_over_raw_text():
    fetched = [_Fetched(SearchResult("a", "u1", "snip"), "RAW TEXT BODY")]
    excerpts = ["RELEVANT EXCERPT"]
    ctx = _assemble_context(fetched, excerpts, 5000)
    assert "RELEVANT EXCERPT" in ctx
    assert "RAW TEXT BODY" not in ctx


def test_assemble_context_falls_back_to_text_when_no_excerpt():
    fetched = [_Fetched(SearchResult("a", "u1", "snip"), "RAW TEXT BODY")]
    ctx = _assemble_context(fetched, [""], 5000)
    assert "RAW TEXT BODY" in ctx


def test_assemble_context_respects_budget():
    fetched = [_Fetched(SearchResult("a", "u1", ""), "BODY" * 5000)]
    ctx = _assemble_context(fetched, [""], 500)
    assert len(ctx) <= 510
    assert "u1" in ctx
