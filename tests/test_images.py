"""Tests for dstools.utils.images."""

from __future__ import annotations

import base64

import respx

from dstools.exceptions import ImageLoadError
from dstools.utils.images import load_image


async def test_load_image_base64(png_bytes):
    b64 = base64.b64encode(png_bytes).decode()
    img = await load_image(b64, user_agent="test")
    assert img.width == 60 and img.height == 40
    assert img.mime == "image/png"
    assert img.source_kind == "base64"
    assert img.data == png_bytes
    assert img.data_url.startswith("data:image/png;base64,")


async def test_load_image_data_uri(png_bytes):
    b64 = base64.b64encode(png_bytes).decode()
    img = await load_image(f"data:image/png;base64,{b64}")
    assert img.source_kind == "data-uri"
    assert img.width == 60


async def test_load_image_file_path(png_bytes, tmp_path):
    path = tmp_path / "img.png"
    path.write_bytes(png_bytes)
    img = await load_image(str(path))
    assert img.source_kind == "path"
    assert img.mime == "image/png"


async def test_load_image_url(png_bytes):
    url = "https://example.com/img.png"
    with respx.mock:
        respx.get(url).respond(200, content=png_bytes)
        img = await load_image(url, user_agent="test")
    assert img.source_kind == "url"
    assert img.data == png_bytes


async def test_load_image_empty_raises():
    try:
        await load_image("   ")
    except ImageLoadError:
        return
    raise AssertionError("expected ImageLoadError")


async def test_load_image_invalid_raises():
    try:
        await load_image("not-a-path-or-valid-base64@@@")
    except ImageLoadError:
        return
    raise AssertionError("expected ImageLoadError")


async def test_load_image_url_failure():
    with respx.mock:
        respx.get("https://example.com/missing.png").respond(404)
        try:
            await load_image("https://example.com/missing.png", user_agent="t")
        except ImageLoadError:
            return
    raise AssertionError("expected ImageLoadError")
