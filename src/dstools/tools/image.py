"""Image-understanding tools.

Because DeepSeek-V4 is text-only, these tools turn images into text the V4 model
can reason over:

* ``analyze_image`` — a vision model describes/answers questions about an image.
* ``ocr_image``     — extract text from an image (local Tesseract, or the vision
  model as a fallback).

Both accept an image as a **file path**, **HTTP(S) URL**, **data URI**, or raw
**base64** string.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from ..config import Settings, get_settings
from ..exceptions import DSToolsError, ImageLoadError
from ..llm import VisionClient
from ..logging_setup import get_logger
from ..runtime import get_vision_client
from ..utils.images import ImageData, load_image
from ._ctx import ctx_info

_logger = get_logger("tools.image")

_DEFAULT_DESC_QUESTION = (
    "Describe this image in thorough detail: the main subjects, the scene/context, "
    "any visible text (transcribe it), colors, composition/layout, and notable details "
    "or anomalies. Be precise and factual."
)


async def analyze_image_logic(
    image: str,
    *,
    question: str | None = None,
    settings: Settings | None = None,
    vision: VisionClient | None = None,
) -> str:
    """Load *image* and produce a text description/answer via the vision provider."""
    settings = settings or get_settings()
    vision = vision or get_vision_client()
    question = question or _DEFAULT_DESC_QUESTION

    img = await load_image(
        image, timeout=settings.search_timeout, user_agent=settings.user_agent
    )

    if vision.configured:
        desc = await vision.describe(data_url=img.data_url, prompt=question)
        return _format_vision_result(img, question, desc)

    # No vision provider: degrade gracefully (metadata + OCR if available).
    return _format_fallback(img, question)


async def ocr_image_logic(
    image: str,
    *,
    lang: str = "eng",
    settings: Settings | None = None,
    vision: VisionClient | None = None,
) -> str:
    """Extract text from *image* using local OCR, falling back to the vision model."""
    settings = settings or get_settings()
    img = await load_image(
        image, timeout=settings.search_timeout, user_agent=settings.user_agent
    )

    ocr_text = _try_local_ocr(img, lang=lang)
    if ocr_text.strip():
        return f"OCR result ({lang}):\n{ocr_text.strip()}"

    vision = vision or get_vision_client()
    if vision.configured:
        prompt = (
            "Extract ALL text visible in this image verbatim, preserving original "
            "reading order and line breaks. Output only the extracted text; if there "
            "is no text, output exactly: NO_TEXT_FOUND."
        )
        result = await vision.describe(data_url=img.data_url, prompt=prompt)
        return f"OCR (via vision model):\n{result.strip()}"

    return (
        "No OCR text could be extracted and no vision provider is configured. "
        "Install `pytesseract` (and the Tesseract binary) or set VISION_* to enable "
        "text extraction from images."
    )


# --- helpers ---------------------------------------------------------------


def _format_vision_result(img: ImageData, question: str, desc: str) -> str:
    return (
        f"**Image analysis** — {_image_facts(img)}\n\n"
        f"Question: {question}\n\n{desc.strip()}"
    )


def _format_fallback(img: ImageData, question: str) -> str:
    ocr_text = _try_local_ocr(img)
    parts = [f"**Image metadata** — {_image_facts(img)}"]
    if ocr_text.strip():
        parts.append(f"\n**Extracted text (OCR):**\n{ocr_text.strip()}")
    parts.append(
        "\n**Note:** No vision provider (VISION_*) is configured, so a full visual "
        "description is unavailable. Configure one, or `pip install dstools[ocr]` for "
        "text extraction, to enable richer image understanding. "
        "Requested question was: " + question
    )
    return "\n".join(parts)


def _image_facts(img: ImageData) -> str:
    """One-line factual metadata: dimensions, format, size, aspect ratio, avg color."""
    ar = round(img.width / img.height, 2) if img.height else 0
    facts = (
        f"{img.width}x{img.height} {img.mime} ({img.source_kind}); "
        f"{len(img.data)} bytes; aspect={ar}"
    )
    color = _avg_color(img)
    if color:
        facts += f"; avg_color={color}"
    return facts


def _avg_color(img: ImageData) -> str:
    """Hex of the image's average colour (cheap, via 1x1 downscale)."""
    try:
        from PIL import Image as _PILImage

        pil = _PILImage.open(BytesIO(img.data)).convert("RGB").resize((1, 1))
        pixel = pil.getpixel((0, 0))
        if not isinstance(pixel, (tuple, list)) or len(pixel) < 3:
            return ""
        r, g, b = (int(c) for c in pixel[:3])
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception as exc:  # pragma: no cover - environment dependent
        _logger.debug("avg_color failed: %s", exc)
        return ""


def _try_local_ocr(img: ImageData, *, lang: str = "eng") -> str:
    """Local OCR without a vision API. Prefers rapidocr (pip-installable, no
    system binary); falls back to pytesseract (needs the Tesseract binary).

    Returns the recognised text, or "" if no backend is available / no text found.
    """
    # 1. rapidocr-onnxruntime — recommended, `pip install dstools[ocr]`.
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        pass
    else:
        try:
            engine = RapidOCR()
            result, _elapse = engine(img.data)
            if isinstance(result, list):
                lines = [
                    item[1]
                    for item in result
                    if isinstance(item, (list, tuple)) and len(item) > 1 and item[1]
                ]
                return "\n".join(lines)
            return ""
        except Exception as exc:  # pragma: no cover - environment dependent
            _logger.debug("rapidocr failed: %s", exc)

    # 2. pytesseract — legacy, needs the system Tesseract binary.
    try:
        import pytesseract
        from PIL import Image as _PILImage
    except ImportError:
        return ""
    try:
        pil = _PILImage.open(BytesIO(img.data))
        return pytesseract.image_to_string(pil, lang=lang)
    except Exception as exc:  # pragma: no cover - environment dependent
        _logger.debug("tesseract OCR failed: %s", exc)
        return ""


# --- MCP registration ------------------------------------------------------


def register(mcp: FastMCP) -> None:
    """Register the image tools on *mcp*."""

    @mcp.tool(name="analyze_image")
    async def analyze_image(
        image: str,
        ctx: Context,
        question: str = "",
    ) -> str:
        """Understand an image and return a detailed text description.

        Use this when you need to "see" an image — DeepSeek-V4 cannot read images
        directly. Accepts an image as a local file path, an HTTP(S) URL, a data URI,
        or a base64 string. Optionally pass `question` to focus the analysis
        (e.g. "What error is shown in this screenshot?", "Read the chart values.").
        Returns structured text you can reason over. Requires a configured vision
        provider (VISION_*); degrades to metadata + OCR without one.
        """
        await ctx_info(ctx, f"analyze_image: {image[:80]}")
        try:
            return await analyze_image_logic(image, question=question or None)
        except (ImageLoadError, DSToolsError) as exc:
            return f"analyze_image failed: {exc}"

    @mcp.tool(name="ocr_image")
    async def ocr_image(
        image: str,
        ctx: Context,
        lang: str = "eng",
    ) -> str:
        """Extract readable text from an image (OCR).

        Accepts a file path, URL, data URI, or base64 string. Uses local Tesseract
        when installed; otherwise falls back to the configured vision provider.
        `lang` is a Tesseract language code (e.g. 'eng', 'chi_sim', 'eng+chi_sim').
        """
        await ctx_info(ctx, f"ocr_image: {image[:80]}")
        try:
            return await ocr_image_logic(image, lang=lang)
        except (ImageLoadError, DSToolsError) as exc:
            return f"ocr_image failed: {exc}"


__all__: list[Any] = ["analyze_image_logic", "ocr_image_logic", "register"]
