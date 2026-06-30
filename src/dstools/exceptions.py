"""Custom exception hierarchy for dstools.

All errors raised by the toolkit derive from :class:`DSToolsError` so that the MCP
tool layer can catch them uniformly and return a clean, LLM-friendly message instead
of a stack trace.
"""

from __future__ import annotations


class DSToolsError(Exception):
    """Base class for all dstools errors."""


class ConfigError(DSToolsError):
    """A required setting is missing or invalid (e.g. no DeepSeek API key)."""


class LLMError(DSToolsError):
    """A call to the DeepSeek (or vision) LLM failed."""


class VisionNotConfiguredError(ConfigError):
    """No vision provider is configured, so image understanding is unavailable."""


class SearchError(DSToolsError):
    """A web-search backend failed."""


class FetchError(DSToolsError):
    """Fetching or extracting a web page failed."""


class ImageLoadError(DSToolsError):
    """An image source could not be loaded, decoded, or validated."""
