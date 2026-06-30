"""Tests for the deep_research pipeline (logic-level, with fakes)."""

from __future__ import annotations

import pytest

from dstools.exceptions import ConfigError
from dstools.llm import DeepSeekClient, LLMResponse
from dstools.search import SearchResult
from dstools.tools.research import (
    _assemble_context,
    _dedupe_and_cap,
    deep_research_logic,
)
from dstools.web.fetcher import PageData

from .conftest import make_settings


class FakeLLM:
    def __init__(self, plan: str, synth: str, *, configured: bool = True) -> None:
        self._plan = plan
        self._synth = synth
        self._configured = configured
        self.calls: list[dict] = []

    @property
    def configured(self) -> bool:
        return self._configured

    async def complete(self, *, messages, **kwargs):
        self.calls.append({**kwargs, "_n_messages": len(messages)})
        if kwargs.get("json_mode"):
            return LLMResponse(content=self._plan, model="fake")
        return LLMResponse(content=self._synth, model="fake")


class PlanFailingLLM:
    """Planning (json) fails, synthesis succeeds — exercises the fallback path."""

    configured = True

    def __init__(self, synth: str) -> None:
        self._synth = synth
        self.calls: list[dict] = []

    async def complete(self, *, messages, **kwargs):
        self.calls.append(kwargs)
        from dstools.exceptions import LLMError

        if kwargs.get("json_mode"):
            raise LLMError("plan boom")
        return LLMResponse(content=self._synth, model="fake")


class FakeSearch:
    name = "fake"

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        return list(self._results)


class FakeFetcher:
    async def fetch(self, url: str, *, max_chars: int = 20000) -> PageData:
        return PageData(
            url=url,
            final_url=url,
            status=200,
            content_type="text/html",
            title=f"Title {url}",
            text=f"Body content for {url}",
            truncated=False,
        )


def _results() -> list[SearchResult]:
    return [
        SearchResult("A", "https://a.com", "snippet a"),
        SearchResult("B", "https://b.com", "snippet b"),
        SearchResult("A2", "https://a.com", "dup"),  # duplicate URL
    ]


async def test_pipeline_produces_cited_report_with_sources():
    settings = make_settings(research_breadth=2, research_depth=1, research_max_sources=3)
    llm = FakeLLM('["sub q1","sub q2"]', "# Report\nDeepSeek V4 is a model. [1]\n\n## Key findings\n- a")
    report = await deep_research_logic(
        "What is DeepSeek V4?",
        settings=settings,
        llm=llm,
        search=FakeSearch(_results()),
        fetcher=FakeFetcher(),
    )
    assert "# Report" in report
    assert "## Sources" in report
    assert "https://a.com" in report
    assert "https://b.com" in report
    # Dedup: a.com appears once in sources list (one numbered entry per unique url).
    assert report.count("https://a.com") == 1


async def test_pipeline_planning_uses_json_mode_and_fast_model():
    settings = make_settings(research_breadth=2, research_depth=1, research_max_sources=2)
    llm = FakeLLM('["q1","q2"]', "# Report\nbody [1]")
    await deep_research_logic(
        "q", settings=settings, llm=llm, search=FakeSearch(_results()), fetcher=FakeFetcher()
    )
    # First call = planning: json mode on, thinking off, fast model.
    planning = llm.calls[0]
    assert planning["json_mode"] is True
    assert planning["thinking"] == "off"
    assert planning["model"] == "deepseek-v4-flash"
    # Synthesis call: not json, uses the heavy (pro) model.
    synth = next(c for c in llm.calls if not c.get("json_mode"))
    assert synth["model"] == "deepseek-v4-pro"


async def test_pipeline_planning_fallback_on_llm_error():
    settings = make_settings(research_breadth=2, research_depth=1, research_max_sources=2)
    llm = PlanFailingLLM("# Report\nsynthesised from fallback queries [1]")
    report = await deep_research_logic(
        "q", settings=settings, llm=llm, search=FakeSearch(_results()), fetcher=FakeFetcher()
    )
    assert "# Report" in report
    assert "## Sources" in report


async def test_pipeline_no_sources_returns_message():
    settings = make_settings(research_breadth=2, research_depth=1, research_max_sources=2)
    llm = FakeLLM('["q1"]', "should not be used")
    report = await deep_research_logic(
        "q", settings=settings, llm=llm, search=FakeSearch([]), fetcher=FakeFetcher()
    )
    assert "could not find any web results" in report


async def test_pipeline_raises_config_error_without_llm():
    settings = make_settings()  # no key
    llm = DeepSeekClient(settings)  # not configured
    with pytest.raises(ConfigError):
        await deep_research_logic(
            "q", settings=settings, llm=llm, search=FakeSearch([]), fetcher=FakeFetcher()
        )


def test_dedupe_and_cap_dedupes_and_skips_empty_urls():
    r = [
        SearchResult("a", "u1", "s"),
        SearchResult("b", "u1", "s2"),  # dup url
        SearchResult("c", "u2", ""),
        SearchResult("d", "", ""),  # empty url skipped
    ]
    out = _dedupe_and_cap(r, 5)
    assert [o.url for o in out] == ["u1", "u2"]


def test_dedupe_and_cap_respects_limit():
    r = [SearchResult(str(i), f"u{i}", "") for i in range(10)]
    assert len(_dedupe_and_cap(r, 3)) == 3


def test_assemble_context_respects_budget():
    sources = [SearchResult("a", "u1", "snip")]
    texts = ["BODY" * 5000]
    ctx = _assemble_context(sources, texts, 500)
    assert len(ctx) <= 510
    assert "u1" in ctx


def test_assemble_context_uses_snippet_when_text_empty():
    sources = [SearchResult("a", "u1", "fallback snippet text")]
    ctx = _assemble_context(sources, [""], 5000)
    assert "fallback snippet text" in ctx
