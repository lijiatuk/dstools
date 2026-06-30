"""Process-wide lazy singletons (clients & providers).

Avoids MCP lifespan complexity: clients are constructed on first use from the
cached :class:`Settings` and reused. :func:`reset_runtime` clears the cache
(mainly for tests). When retrieval caching is enabled, the search provider and
page fetcher are wrapped transparently.
"""

from __future__ import annotations

from functools import lru_cache

from .cache import CachingPageFetcher, CachingSearchProvider, RetrievalCache, build_cache
from .config import get_settings
from .llm import DeepSeekClient, VisionClient
from .search import get_search_provider as _build_search_provider
from .web.fetcher import Fetcher, PageFetcher


@lru_cache(maxsize=1)
def get_deepseek_client() -> DeepSeekClient:
    return DeepSeekClient(get_settings())


@lru_cache(maxsize=1)
def get_vision_client() -> VisionClient:
    return VisionClient(get_settings())


@lru_cache(maxsize=1)
def get_cache() -> RetrievalCache | None:
    return build_cache(get_settings())


@lru_cache(maxsize=1)
def get_search_provider():
    """Return the configured SearchProvider (keyless DuckDuckGo by default),
    wrapped in the retrieval cache when ``RESEARCH_CACHE_ENABLED=true``."""
    provider = _build_search_provider(get_settings())
    cache = get_cache()
    if cache is not None:
        return CachingSearchProvider(provider, cache)
    return provider


@lru_cache(maxsize=1)
def get_fetcher() -> Fetcher:
    fetcher = PageFetcher(get_settings())
    cache = get_cache()
    if cache is not None:
        return CachingPageFetcher(fetcher, cache)
    return fetcher


def reset_runtime() -> None:
    """Clear all cached clients/providers (for tests after changing settings)."""
    for fn in (
        get_deepseek_client,
        get_vision_client,
        get_cache,
        get_search_provider,
        get_fetcher,
    ):
        fn.cache_clear()
