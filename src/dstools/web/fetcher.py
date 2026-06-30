"""Async web fetcher + HTML→Markdown extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx
from bs4 import BeautifulSoup

from ..config import Settings
from ..exceptions import FetchError
from ..logging_setup import get_logger
from ..utils.text import clean_whitespace, truncate

_logger = get_logger("web.fetcher")

# Hard cap on downloaded bytes (prevents memory bombs on huge pages).
_MAX_DOWNLOAD_BYTES = 3 * 1024 * 1024  # 3 MB

# Tags that carry navigation/boilerplate rather than content.
_NOISE_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "button",
    "svg",
    "iframe",
)


@dataclass
class PageData:
    """The result of fetching and extracting a single page."""

    url: str
    final_url: str
    status: int
    content_type: str
    title: str
    text: str
    truncated: bool = False

    def to_markdown(self, max_chars: int) -> str:
        """Render the page as markdown, capped to *max_chars*."""
        header = f"# {self.title}\nSource: {self.final_url}\n"
        body = truncate(self.text, max_chars)
        return header + body


class Fetcher(Protocol):
    """Async page-fetcher interface (``PageFetcher`` and the caching wrapper both satisfy).

    Only :meth:`fetch` is required; ``fetch_markdown`` is a convenience on the
    concrete classes.
    """

    async def fetch(self, url: str, *, max_chars: int = 20_000) -> PageData: ...


class PageFetcher:
    """Fetches URLs and extracts clean, readable Markdown."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def fetch(self, url: str, *, max_chars: int = 20_000) -> PageData:
        headers = {
            "User-Agent": self._settings.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        timeout = self._settings.research_fetch_timeout

        raw = b""
        status = 0
        content_type = ""
        final_url = url

        try:
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=True, headers=headers
            ) as client, client.stream("GET", url) as resp:
                status = resp.status_code
                final_url = str(resp.url)
                content_type = resp.headers.get("content-type", "")
                if status >= 400:
                    raise FetchError(
                        f"HTTP {status} fetching {url}"
                        + (f" ({resp.reason_phrase})" if resp.reason_phrase else "")
                    )
                async for chunk in resp.aiter_bytes():
                    if len(raw) + len(chunk) > _MAX_DOWNLOAD_BYTES:
                        raw += chunk[: _MAX_DOWNLOAD_BYTES - len(raw)]
                        break
                    raw += chunk
        except httpx.HTTPError as exc:
            raise FetchError(f"Failed to fetch {url}: {exc}") from exc

        is_html = "text/html" in content_type or "xhtml" in content_type
        if is_html:
            title, text = self._extract_html(raw)
        elif "text/" in content_type or "json" in content_type or "xml" in content_type:
            title, text = "", raw.decode("utf-8", errors="replace")
        else:
            title, text = "", (
                f"[Non-HTML content ({content_type or 'unknown type'}), "
                f"{len(raw)} bytes — not extracted]"
            )

        truncated = len(text) > max_chars
        return PageData(
            url=url,
            final_url=final_url,
            status=status,
            content_type=content_type,
            title=title,
            text=clean_whitespace(text),
            truncated=truncated,
        )

    @staticmethod
    def _extract_html(raw: bytes) -> tuple[str, str]:
        try:
            html = raw.decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - decode rarely fails with replace
            html = raw.decode("latin-1", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(list(_NOISE_TAGS)):
            tag.decompose()

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Prefer semantic main-content containers; fall back to the whole body.
        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find(id="content")
            or soup.find(class_="content")
            or soup.body
            or soup
        )

        import html2text  # local import: only needed when extracting HTML

        converter = html2text.HTML2Text()
        converter.body_width = 0  # do not hard-wrap lines
        converter.ignore_images = True
        converter.ignore_emphasis = False
        converter.protect_links = True
        text = converter.handle(str(main))
        return title, text

    async def fetch_markdown(self, url: str, *, max_chars: int = 20_000) -> str:
        """Convenience: fetch and return only the markdown text."""
        page = await self.fetch(url, max_chars=max_chars)
        return page.to_markdown(max_chars)
