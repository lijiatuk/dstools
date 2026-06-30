"""Real DeepSeek-V4 integration test — skipped unless DEEPSEEK_API_KEY is set.

Run locally with a key to validate the full pipeline against the live API:

    DEEPSEEK_API_KEY=sk-... uv run pytest tests/test_integration_live.py -q -s

It is skipped in CI (no key) to avoid cost / network flakiness.
"""

from __future__ import annotations

import os

import pytest

from dstools.tools.research import deep_research_logic

pytestmark = pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not set — live integration test is opt-in",
)


async def test_deep_research_real_produces_report() -> None:
    report = await deep_research_logic(
        "What is DeepSeek-V4? Summarize its key models and context length.",
        breadth=2,
        depth=1,
        max_sources=3,
    )
    assert "## Sources" in report
    assert len(report) > 300
    # The report must cite at least one source.
    assert "[1]" in report or "https://" in report
