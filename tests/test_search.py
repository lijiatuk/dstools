"""Tests for dstools.search (DuckDuckGo HTML parsing + Tavily guard)."""

from __future__ import annotations

import respx

from dstools.exceptions import ConfigError, SearchError
from dstools.search.duckduckgo import DuckDuckGoSearchProvider, _decode_result_url
from dstools.search.tavily import TavilySearchProvider

from .conftest import make_settings

_DD_HTML = """<html><body>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F1&rut=x">Result One</a>
  <a class="result__snippet" href="#">Snippet one</a>
</div>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F2">Result Two</a>
</div>
</body></html>"""

_LITE_HTML = """<html><body><table>
<tr><td><a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Flite.example%2Fx">Lite One</a></td></tr>
</table></body></html>"""


def test_decode_result_url():
    assert _decode_result_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Ffoo.com%2Fp") == (
        "https://foo.com/p"
    )
    assert _decode_result_url("https://plain.example.com/a") == "https://plain.example.com/a"
    assert _decode_result_url("") == ""


async def test_duckduckgo_html_parse():
    settings = make_settings()
    with respx.mock:
        respx.post("https://html.duckduckgo.com/html/").respond(200, text=_DD_HTML)
        provider = DuckDuckGoSearchProvider(settings)
        results = await provider.search("deepseek v4", max_results=10)
    assert len(results) == 2
    assert results[0].url == "https://example.com/1"
    assert results[0].title == "Result One"
    assert results[0].snippet == "Snippet one"
    assert results[1].url == "https://example.com/2"


async def test_duckduckgo_lite_fallback():
    settings = make_settings()
    with respx.mock:
        # HTML endpoint returns an empty body -> fall back to Lite.
        respx.post("https://html.duckduckgo.com/html/").respond(200, text="<html></html>")
        respx.post("https://lite.duckduckgo.com/lite/").respond(200, text=_LITE_HTML)
        provider = DuckDuckGoSearchProvider(settings)
        results = await provider.search("anything")
    assert len(results) == 1
    assert results[0].url == "https://lite.example/x"


async def test_duckduckgo_no_results_raises():
    settings = make_settings()
    with respx.mock:
        respx.post("https://html.duckduckgo.com/html/").respond(200, text="<html></html>")
        respx.post("https://lite.duckduckgo.com/lite/").respond(200, text="<html></html>")
        provider = DuckDuckGoSearchProvider(settings)
        try:
            await provider.search("nothing")
        except SearchError:
            return
    raise AssertionError("expected SearchError")


def test_tavily_without_key_raises():
    settings = make_settings(search_provider="tavily")  # no tavily_api_key
    try:
        TavilySearchProvider(settings)
    except ConfigError:
        return
    raise AssertionError("expected ConfigError")
