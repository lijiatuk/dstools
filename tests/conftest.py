"""Shared test fixtures and helpers."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from dstools.config import Settings, get_settings
from dstools.runtime import reset_runtime


def make_settings(**overrides: object) -> Settings:
    """Build a Settings instance with no .env loading and explicit overrides."""
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _isolate_caches() -> None:
    """Clear cached settings/runtime before each test to avoid cross-test bleed."""
    get_settings.cache_clear()
    reset_runtime()
    yield
    get_settings.cache_clear()
    reset_runtime()


@pytest.fixture
def png_bytes() -> bytes:
    """A small valid PNG (60x40, green)."""
    buf = io.BytesIO()
    Image.new("RGB", (60, 40), (10, 200, 10)).save(buf, "PNG")
    return buf.getvalue()
