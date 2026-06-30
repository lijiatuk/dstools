"""Tests for dstools.config."""

from __future__ import annotations

from dstools.config import Settings


def test_defaults():
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.deepseek_base_url == "https://api.deepseek.com"
    assert s.deepseek_model == "deepseek-v4-pro"
    assert s.deepseek_fast_model == "deepseek-v4-flash"
    assert s.deepseek_thinking == "auto"
    assert s.search_provider == "duckduckgo"
    assert s.research_breadth == 3
    assert not s.has_deepseek
    assert not s.has_vision


def test_capability_flags():
    s = Settings(
        _env_file=None,
        deepseek_api_key="k",
        vision_base_url="http://x",
        vision_model="m",
        tavily_api_key="t",
    )  # type: ignore[call-arg]
    assert s.has_deepseek
    assert s.has_vision
    assert s.has_tavily


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "envkey")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("RESEARCH_BREADTH", "7")
    from dstools.config import reload_settings

    s = reload_settings()
    assert s.deepseek_api_key == "envkey"
    assert s.deepseek_model == "deepseek-v4-flash"
    assert s.research_breadth == 7
    assert s.has_deepseek
