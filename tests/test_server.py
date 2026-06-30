"""Integration tests for the FastMCP server (tool registration + call_tool)."""

from __future__ import annotations

from dstools.llm import DeepSeekClient, VisionClient
from dstools.search import SearchResult
from dstools.server import create_server

from .conftest import make_settings


def _extract_text(result) -> str:
    """Extract a text blob from a FastMCP call_tool result."""
    content = result[0] if isinstance(result, tuple) else getattr(result, "content", result)
    if isinstance(content, list):
        parts = []
        for block in content:
            txt = getattr(block, "text", None)
            if txt is None and isinstance(block, dict):
                txt = block.get("text")
            if txt:
                parts.append(txt)
        return "\n".join(parts)
    return str(result)[:500]


def test_server_registers_five_tools():
    server = create_server()
    import asyncio

    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {"analyze_image", "ocr_image", "web_search", "fetch_page", "deep_research"}


def test_tool_schemas_exclude_ctx_and_mark_required():
    server = create_server()
    import asyncio

    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    ws = tools["web_search"].inputSchema
    props = set(ws.get("properties", {}))
    assert "ctx" not in props
    assert "query" in set(ws.get("required", []))


async def test_call_analyze_image_falls_back_without_vision(monkeypatch, png_bytes, tmp_path):
    from dstools.tools import image as imod

    s = make_settings()
    monkeypatch.setattr(imod, "get_vision_client", lambda: VisionClient(s))
    path = tmp_path / "i.png"
    path.write_bytes(png_bytes)

    server = create_server()
    result = await server.call_tool("analyze_image", {"image": str(path)})
    txt = _extract_text(result)
    assert "metadata" in txt
    assert "vision" in txt.lower() or "VISION" in txt


async def test_call_deep_research_reports_missing_key(monkeypatch):
    from dstools.tools import research as rmod

    s = make_settings()
    monkeypatch.setattr(rmod, "get_settings", lambda: s)
    monkeypatch.setattr(rmod, "get_deepseek_client", lambda: DeepSeekClient(s))

    server = create_server()
    result = await server.call_tool("deep_research", {"query": "anything"})
    txt = _extract_text(result)
    assert "deep_research failed" in txt
    assert "DEEPSEEK_API_KEY" in txt


async def test_call_web_search_with_mocked_provider(monkeypatch):
    from dstools.tools import search as smod

    class _StaticSearch:
        name = "static"

        def __init__(self, results):
            self._results = results

        async def search(self, query, max_results=10):
            return list(self._results)

    monkeypatch.setattr(
        smod,
        "get_search_provider",
        lambda: _StaticSearch([SearchResult("DeepSeek V4", "https://a.com", "snippet a")]),
    )
    server = create_server()
    result = await server.call_tool("web_search", {"query": "x"})
    txt = _extract_text(result)
    assert "https://a.com" in txt
    assert "DeepSeek V4" in txt
