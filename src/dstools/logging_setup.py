"""Logging setup.

MCP servers communicate over **stdout**, so all logs are written to **stderr** to
avoid corrupting the protocol stream. A ``rich`` handler gives readable, colored
output; the level is taken from :class:`dstools.config.Settings`.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from rich.console import Console
from rich.logging import RichHandler

from .config import get_settings

_CONFIGURED = False
# stderr (never stdout): stdout is the MCP stdio protocol channel.
_STDERR = Console(stderr=True)


@lru_cache(maxsize=1)
def _logger() -> logging.Logger:
    return logging.getLogger("dstools")


def configure_logging(level: str | None = None) -> None:
    """Configure the ``dstools`` logger to write colored logs to stderr."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    lvl = (level or settings.log_level).upper()
    logger = _logger()
    logger.setLevel(lvl)
    logger.propagate = False
    if not logger.handlers:
        handler = RichHandler(
            console=_STDERR,
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            markup=False,
        )
        handler.setLevel(lvl)
        logger.addHandler(handler)
    # Silence noisy HTTP-client logs (httpx prints every request at INFO by default).
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the ``dstools`` namespace."""
    configure_logging()
    base = _logger()
    return base.getChild(name) if name else base
