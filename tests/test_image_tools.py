"""Tests for the image tools' logic (vision-configured + fallback paths)."""

from __future__ import annotations

from dstools.llm import VisionClient
from dstools.tools.image import analyze_image_logic, ocr_image_logic

from .conftest import make_settings


class FakeVision:
    """A configured vision client that returns a canned description."""

    configured = True

    def __init__(self, description: str) -> None:
        self._description = description
        self.calls: list[dict] = []

    async def describe(self, *, data_url: str, prompt: str, **kwargs) -> str:
        self.calls.append({"data_url": data_url, "prompt": prompt})
        return self._description


async def test_analyze_image_with_vision(png_bytes, tmp_path):
    path = tmp_path / "i.png"
    path.write_bytes(png_bytes)
    vision = FakeVision("A solid green rectangle with no text.")
    out = await analyze_image_logic(
        str(path), question="What is in this image?", vision=vision
    )
    assert "Image analysis" in out
    assert "green rectangle" in out
    assert vision.calls  # the vision model was actually called
    assert vision.calls[0]["data_url"].startswith("data:image/png;base64,")


async def test_analyze_image_fallback_without_vision(png_bytes, tmp_path):
    path = tmp_path / "i.png"
    path.write_bytes(png_bytes)
    settings = make_settings()
    out = await analyze_image_logic(str(path), vision=VisionClient(settings))
    assert "metadata" in out
    assert "vision" in out.lower()


async def test_ocr_image_with_vision_fallback(png_bytes, tmp_path):
    path = tmp_path / "i.png"
    path.write_bytes(png_bytes)
    vision = FakeVision("NO_TEXT_FOUND")
    out = await ocr_image_logic(str(path), vision=vision)
    # No local OCR (pytesseract not installed in tests) -> vision path used.
    assert "OCR (via vision model)" in out
    assert vision.calls


async def test_ocr_image_no_backend(png_bytes, tmp_path):
    path = tmp_path / "i.png"
    path.write_bytes(png_bytes)
    settings = make_settings()
    out = await ocr_image_logic(str(path), settings=settings, vision=VisionClient(settings))
    assert "No OCR" in out or "vision provider" in out
