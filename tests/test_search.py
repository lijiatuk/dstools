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


def test_decode_result_url_filters_ad_redirects():
    # Sponsored results use a /y.js ad redirect and must be dropped.
    assert (
        _decode_result_url(
            "//duckduckgo.com/y.js?ad_domain=qubrid.com&ad_provider=bingv7aa"
        )
        == ""
    )


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


_DD_HTML_WITH_AD = """<html><body>
<div class="result results_links result--ad">
  <a class="result__a" href="//duckduckgo.com/y.js?ad_domain=adsite.com">Sponsored Ad</a>
  <a class="result__snippet" href="#">buy now</a>
</div>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Freal.example%2F1">Real Result</a>
</div>
</body></html>"""


async def test_duckduckgo_filters_ads():
    settings = make_settings()
    with respx.mock:
        respx.post("https://html.duckduckgo.com/html/").respond(200, text=_DD_HTML_WITH_AD)
        provider = DuckDuckGoSearchProvider(settings)
        results = await provider.search("anything")
    assert len(results) == 1
    assert results[0].url == "https://real.example/1"
    assert "Sponsored" not in results[0].title


async def test_duckduckgo_no_results_raises():
    settings = make_settings(search_retry_attempts=1)  # avoid slow retry backoff
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


def test_brave_without_key_raises():
    settings = make_settings(search_provider="brave")  # no brave_api_key
    try:
        from dstools.search.brave import BraveSearchProvider

        BraveSearchProvider(settings)
    except ConfigError:
        return
    raise AssertionError("expected ConfigError")


async def test_brave_search_parses_results():
    from dstools.search.brave import BraveSearchProvider

    settings = make_settings(search_provider="brave", brave_api_key="test-key")
    payload = {
        "web": {
            "results": [
                {"title": "DeepSeek V4", "url": "https://a.com/1", "description": "snip a"},
                {"title": "V4 Pro", "url": "https://a.com/2", "description": "snip b"},
            ]
        }
    }
    with respx.mock:
        respx.get("https://api.search.brave.com/res/v1/web/search").respond(
            200, json=payload
        )
        provider = BraveSearchProvider(settings)
        results = await provider.search("deepseek v4", max_results=5)
    assert len(results) == 2
    assert results[0].url == "https://a.com/1"
    assert results[0].snippet == "snip a"
