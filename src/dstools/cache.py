"""Opt-in on-disk retrieval cache for deep_research.

When ``RESEARCH_CACHE_ENABLED=true``, search results and fetched page markdown are
cached to disk (default TTL 24h) so repeated / iterative research doesn't re-pay
for the same searches and fetches. LLM responses are **never** cached (they are
non-deterministic and freshness matters).

The cache is a simple key→JSON-file store keyed by a SHA-256 of the lookup key;
TTL is enforced via file mtime. Disable any time with ``RESEARCH_CACHE_ENABLED=false``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .config import Settings
from .logging_setup import get_logger
from .search import SearchResult
from .search.base import SearchProvider
from .web.fetcher import PageData, PageFetcher

_logger = get_logger("cache")


class RetrievalCache:
    """Key→JSON-file disk cache with TTL."""

    def __init__(self, cache_dir: str, ttl: float) -> None:
        self._dir = Path(cache_dir).expanduser()
        self._ttl = ttl
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._dir / f"{digest}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        if self._ttl > 0 and (time.time() - path.stat().st_mtime) > self._ttl:
            with contextlib.suppress(OSError):
                path.unlink()
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(value, fh, ensure_ascii=False)
        except OSError as exc:  # pragma: no cover - disk full / permissions
            _logger.debug("cache write failed: %s", exc)

    def clear(self) -> int:
        """Remove all cached entries; return the count removed."""
        removed = 0
        for path in self._dir.glob("*.json"):
            with contextlib.suppress(OSError):
                path.unlink()
                removed += 1
        return removed

    def stats(self) -> dict[str, Any]:
        files = list(self._dir.glob("*.json"))
        size = sum(p.stat().st_size for p in files if p.exists())
        return {"entries": len(files), "size_bytes": size, "dir": str(self._dir)}


class CachingSearchProvider:
    """Wraps a :class:`SearchProvider` with an on-disk result cache."""

    name = "cached"

    def __init__(self, inner: SearchProvider, cache: RetrievalCache) -> None:
        self._inner = inner
        self._cache = cache

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        key = f"search|{query}|{max_results}"
        cached = self._cache.get(key)
        if cached is not None:
            _logger.debug("cache hit (search): %s", query)
            return [SearchResult(**r) for r in cached]
        results = await self._inner.search(query, max_results=max_results)
        self._cache.set(key, [dataclasses.asdict(r) for r in results])
        return results


class CachingPageFetcher:
    """Wraps a :class:`PageFetcher` with an on-disk page-text cache."""

    def __init__(self, inner: PageFetcher, cache: RetrievalCache) -> None:
        self._inner = inner
        self._cache = cache

    async def fetch(self, url: str, *, max_chars: int = 20_000) -> PageData:
        key = f"page|{url}|{max_chars}"
        cached = self._cache.get(key)
        if cached is not None:
            _logger.debug("cache hit (page): %s", url)
            return PageData(**cached)
        page = await self._inner.fetch(url, max_chars=max_chars)
        self._cache.set(key, dataclasses.asdict(page))
        return page

    async def fetch_markdown(self, url: str, *, max_chars: int = 20_000) -> str:
        page = await self.fetch(url, max_chars=max_chars)
        return page.to_markdown(max_chars)


def build_cache(settings: Settings) -> RetrievalCache | None:
    """Return a configured cache, or None if caching is disabled."""
    if not settings.research_cache_enabled:
        return None
    return RetrievalCache(settings.research_cache_dir, settings.research_cache_ttl)
