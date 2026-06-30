"""Process-wide lazy singletons (clients & providers).

Avoids MCP lifespan complexity: clients are constructed on first use from the
cached :class:`Settings` and reused. :func:`reset_runtime` clears the cache
(mainly for tests).
"""

from __future__ import annotations

from functools import lru_cache

from .config import get_settings
from .llm import DeepSeekClient, VisionClient
from .search import get_search_provider as _build_search_provider
from .web.fetcher import PageFetcher


@lru_cache(maxsize=1)
def get_deepseek_client() -> DeepSeekClient:
    return DeepSeekClient(get_settings())


@lru_cache(maxsize=1)
def get_vision_client() -> VisionClient:
    return VisionClient(get_settings())


@lru_cache(maxsize=1)
def get_search_provider():
    """Return the configured :class:`SearchProvider` (keyless DuckDuckGo by default)."""
    return _build_search_provider(get_settings())


@lru_cache(maxsize=1)
def get_fetcher() -> PageFetcher:
    return PageFetcher(get_settings())


def reset_runtime() -> None:
    """Clear all cached clients/providers (for tests after changing settings)."""
    for fn in (
        get_deepseek_client,
        get_vision_client,
        get_search_provider,
        get_fetcher,
    ):
        fn.cache_clear()
