"""Tests for dstools.web.fetcher."""

from __future__ import annotations

import respx

from dstools.exceptions import FetchError
from dstools.web.fetcher import PageFetcher

from .conftest import make_settings

_HTML = """<html><head><title>My Page</title></head><body>
<nav>nav links</nav>
<article>
  <h1>Hello</h1>
  <p>This is the main content about DeepSeek V4.</p>
  <script>SECRETJS=1</script>
</article>
<footer>footer stuff</footer>
</body></html>"""


async def test_fetch_html_extracts_main_content():
    settings = make_settings()
    url = "https://example.com/article"
    with respx.mock:
        respx.get(url).respond(200, text=_HTML, headers={"content-type": "text/html"})
        page = await PageFetcher(settings).fetch(url, max_chars=5000)
    assert page.status == 200
    assert page.title == "My Page"
    assert "main content about DeepSeek V4" in page.text
    assert "SECRETJS" not in page.text  # script stripped


async def test_fetch_non_html_passthrough():
    settings = make_settings()
    url = "https://example.com/data.txt"
    with respx.mock:
        respx.get(url).respond(
            200, content=b"plain text body", headers={"content-type": "text/plain"}
        )
        page = await PageFetcher(settings).fetch(url)
    assert page.title == ""
    assert "plain text body" in page.text


async def test_fetch_404_raises():
    settings = make_settings()
    url = "https://example.com/missing"
    with respx.mock:
        respx.get(url).respond(404)
        try:
            await PageFetcher(settings).fetch(url)
        except FetchError:
            return
    raise AssertionError("expected FetchError")


async def test_to_markdown_caps_size():
    settings = make_settings()
    url = "https://example.com/big"
    big = "<html><body><article>" + ("word " * 5000) + "</article></body></html>"
    with respx.mock:
        respx.get(url).respond(200, text=big, headers={"content-type": "text/html"})
        page = await PageFetcher(settings).fetch(url, max_chars=1000)
    md = page.to_markdown(1000)
    assert len(md) <= 1100
