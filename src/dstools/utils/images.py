"""Image loading, validation, and encoding.

Accepts an image from a **file path**, **HTTP(S) URL**, **data URI**, or a raw
**base64** string, validates it with Pillow, and returns normalised bytes + MIME
type + dimensions. Everything downstream (the vision client, OCR) works on this
normalised :class:`ImageData`.
"""

from __future__ import annotations

import base64
import binascii
import os
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Literal

import httpx
from PIL import Image, UnidentifiedImageError

from ..exceptions import ImageLoadError

# Pillow format -> MIME type.
_FORMAT_TO_MIME: dict[str, str] = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "JPG": "image/jpeg",
    "GIF": "image/gif",
    "WEBP": "image/webp",
    "BMP": "image/bmp",
}

_DATA_URI_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$", re.DOTALL)
_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB safety cap.

SourceKind = Literal["path", "url", "data-uri", "base64"]


@dataclass(frozen=True)
class ImageData:
    """A normalised, validated image ready for vision/OCR processing."""

    data: bytes
    mime: str
    b64: str
    width: int
    height: int
    source_kind: SourceKind
    source_desc: str

    @property
    def data_url(self) -> str:
        """The image as a ``data:`` URL suitable for OpenAI-compatible vision APIs."""
        return f"data:{self.mime};base64,{self.b64}"


def _detect_kind(source: str) -> SourceKind:
    if source.startswith("data:"):
        return "data-uri"
    if source.startswith(("http://", "https://")):
        return "url"
    if os.path.exists(source):
        return "path"
    return "base64"


def _decode_pil(data: bytes) -> tuple[Image.Image, str]:
    """Open bytes with Pillow and return (image, mime)."""
    try:
        img = Image.open(BytesIO(data))
        img.load()
    except UnidentifiedImageError as exc:  # not an image
        raise ImageLoadError("Bytes are not a recognisable image") from exc
    except Exception as exc:  # truncated / corrupt
        raise ImageLoadError(f"Could not decode image: {exc}") from exc
    fmt = (img.format or "").upper()
    mime = _FORMAT_TO_MIME.get(fmt, "image/png")
    return img, mime


async def _fetch_url(url: str, *, timeout: float, user_agent: str) -> bytes:
    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, headers={"User-Agent": user_agent}
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content
    except httpx.HTTPError as exc:
        raise ImageLoadError(f"Failed to download image from {url}: {exc}") from exc


def load_image_sync(source: str, *, timeout: float = 20.0, user_agent: str = "") -> ImageData:
    """Synchronous image loader (used when a caller is not in an event loop)."""
    import asyncio

    return asyncio.run(_load_image(source, timeout=timeout, user_agent=user_agent))


async def load_image(
    source: str,
    *,
    timeout: float = 20.0,
    user_agent: str = "",
) -> ImageData:
    """Load and validate an image from a path, URL, data URI, or base64 string."""
    return await _load_image(source, timeout=timeout, user_agent=user_agent)


async def _load_image(source: str, *, timeout: float, user_agent: str) -> ImageData:
    source = source.strip()
    if not source:
        raise ImageLoadError("Empty image source")

    kind = _detect_kind(source)

    if kind == "data-uri":
        m = _DATA_URI_RE.match(source)
        if not m:
            raise ImageLoadError("Malformed data URI")
        mime = m.group(1)
        try:
            data = base64.b64decode(m.group(2))
        except binascii.Error as exc:
            raise ImageLoadError(f"Invalid base64 in data URI: {exc}") from exc
    elif kind == "url":
        data = await _fetch_url(source, timeout=timeout, user_agent=user_agent)
        mime = ""  # determined below from bytes
    elif kind == "path":
        try:
            with open(source, "rb") as fh:
                data = fh.read()
        except OSError as exc:
            raise ImageLoadError(f"Could not read image file {source}: {exc}") from exc
        mime = ""
    else:  # base64
        try:
            data = base64.b64decode(source)
        except (binascii.Error, ValueError) as exc:
            raise ImageLoadError(
                "Source is not a path, URL, or data URI and could not be "
                f"decoded as base64: {exc}"
            ) from exc
        mime = ""

    if len(data) > _MAX_IMAGE_BYTES:
        raise ImageLoadError(
            f"Image is too large ({len(data)} bytes); limit is {_MAX_IMAGE_BYTES} bytes"
        )
    if not data:
        raise ImageLoadError("Image source resolved to empty data")

    img, detected_mime = _decode_pil(data)
    if not mime:
        mime = detected_mime

    return ImageData(
        data=data,
        mime=mime,
        b64=base64.b64encode(data).decode("ascii"),
        width=img.width,
        height=img.height,
        source_kind=kind,
        source_desc=source if kind in {"url", "path"} else f"<{kind}>",
    )
