"""Tests for the retrieval cache (workstream D)."""

from __future__ import annotations

import os
import time
from pathlib import Path

from dstools.cache import CachingPageFetcher, CachingSearchProvider, RetrievalCache, build_cache
from dstools.search import SearchResult
from dstools.web.fetcher import PageData

from .conftest import make_settings


def _cache(tmp_path: Path, ttl: float = 3600) -> RetrievalCache:
    return RetrievalCache(str(tmp_path / "cache"), ttl=ttl)


def test_cache_set_get_roundtrip(tmp_path):
    c = _cache(tmp_path)
    assert c.get("k") is None
    c.set("k", {"a": 1, "b": [1, 2]})
    assert c.get("k") == {"a": 1, "b": [1, 2]}


def test_cache_ttl_expires(tmp_path):
    c = _cache(tmp_path, ttl=1)
    c.set("k", "v")
    # Backdate the entry so it is older than the TTL.
    path = c._path("k")  # type: ignore[attr-defined]
    past = time.time() - 10
    os.utime(path, (past, past))
    assert c.get("k") is None  # expired -> miss (and file removed)


def test_cache_ttl_zero_means_no_expiry(tmp_path):
    c = _cache(tmp_path, ttl=0)
    c.set("k", "v")
    path = c._path("k")  # type: ignore[attr-defined]
    past = time.time() - 9999
    os.utime(path, (past, past))
    assert c.get("k") == "v"  # ttl=0 => never expires


def test_cache_clear(tmp_path):
    c = _cache(tmp_path)
    c.set("a", 1)
    c.set("b", 2)
    assert c.stats()["entries"] == 2
    removed = c.clear()
    assert removed == 2
    assert c.get("a") is None
    assert c.stats()["entries"] == 0


def test_build_cache_disabled_by_default():
    s = make_settings()
    assert build_cache(s) is None


def test_build_cache_enabled(tmp_path):
    s = make_settings(research_cache_enabled=True, research_cache_dir=str(tmp_path / "c"))
    cache = build_cache(s)
    assert cache is not None
    assert cache.stats()["dir"] == str(tmp_path / "c")


class _CountingSearch:
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query, max_results=10):
        self.calls += 1
        return [SearchResult("t", "https://x.com", "s")]


async def test_caching_search_provider_hits_on_second_call(tmp_path):
    cache = _cache(tmp_path)
    inner = _CountingSearch()
    wrapper = CachingSearchProvider(inner, cache)
    r1 = await wrapper.search("q")
    r2 = await wrapper.search("q")
    assert r1 == r2
    assert inner.calls == 1  # second call served from cache


class _CountingFetcher:
    def __init__(self) -> None:
        self.calls = 0

    async def fetch(self, url, *, max_chars=20000):
        self.calls += 1
        return PageData(
            url=url, final_url=url, status=200, content_type="text/html",
            title="T", text="body", truncated=False,
        )

    async def fetch_markdown(self, url, *, max_chars=20000):
        page = await self.fetch(url, max_chars=max_chars)
        return page.to_markdown(max_chars)


async def test_caching_page_fetcher_hits_on_second_call(tmp_path):
    cache = _cache(tmp_path)
    inner = _CountingFetcher()
    wrapper = CachingPageFetcher(inner, cache)
    p1 = await wrapper.fetch("https://x.com")
    p2 = await wrapper.fetch("https://x.com")
    assert p1.url == p2.url
    assert inner.calls == 1  # second call served from cache
