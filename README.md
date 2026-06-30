# dstools — DeepSeek-V4 MCP Toolkit

> Give **DeepSeek-V4** models eyes and a research desk.
>
> An [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server that augments
> DeepSeek's text models with two capabilities they don't have natively:
>
> 1. **Image content understanding** — DeepSeek-V4 is a text-only model. `dstools` adds a
>    vision tool that turns any image into rich, structured text the V4 model can reason over
>    (leveraging its 1M-token context and world-class reasoning).
> 2. **Deep Research** — a multi-step, citation-backed research pipeline that uses V4 as the
>    planning + synthesis brain over live web search and page extraction.

`dstools` is a productizable, installable Python package. It speaks MCP over **stdio** and
**Streamable HTTP**, so any MCP-capable host (Claude Code, Claude Desktop, Cherry Studio, a
custom agent, …) can connect a DeepSeek-V4 backend to it and immediately call these tools.

---

## Why this exists

DeepSeek-V4 (`deepseek-v4-flash` / `deepseek-v4-pro`, released 2026-04-24) is an outstanding
text model with 1M context, strong agentic/tool-calling ability, and an automatic context
cache — but the official chat API is **text-only** (no multimodal vision). `dstools` closes
exactly that gap:

| DeepSeek-V4 strength | What's missing | What `dstools` adds |
| --- | --- | --- |
| 1M context, top reasoning | Can't see images | `analyze_image` → vision-to-text |
| Agentic, tool-calling | No live web access | `web_search`, `fetch_page`, `deep_research` |
| Automatic prompt caching | — | Stable-prefix prompts to maximise cache hits |
| Thinking mode (`thinking={"type":"enabled"}`) | — | Used selectively for hard synthesis steps |

The toolkit is **deeply adapted to V4**: it defaults to `deepseek-v4-pro` for synthesis and
`deepseek-v4-flash` for cheap sub-steps, toggles V4's native thinking mode per call, structures
prompts for cache hits, and uses V4's JSON-output mode for structured extraction.

## Tools exposed

| Tool | Description | Needs a key? |
| --- | --- | --- |
| `analyze_image` | Describe/understand an image (path, URL, or base64). Returns structured text. | Vision provider key (or local model) |
| `ocr_image` | Extract text from an image (OCR). | Optional `pytesseract` |
| `web_search` | Run a web search, return ranked results (title, url, snippet). | **No** (DuckDuckGo, keyless) |
| `fetch_page` | Fetch a URL and return clean, readable Markdown. | **No** |
| `deep_research` | Full pipeline: plan → search → fetch → select → synthesize, with citations. | DeepSeek API key |

Granular tools (`web_search`, `fetch_page`, `analyze_image`) let the host agent run its own
agentic loop; `deep_research` is a one-shot orchestrator for when you just want a cited report.

## Quick start

```bash
# 1. Install (Python 3.10+)
uv sync                # or: pip install -e .

# 2. Configure
cp .env.example .env   # then edit: set DEEPSEEK_API_KEY and a vision provider

# 3. Run the MCP server (stdio — for local hosts like Claude Code/Desktop)
uv run dstools serve

# …or over Streamable HTTP (for remote hosts)
uv run dstools serve --transport http --port 8000
```

Connect from Claude Code:

```bash
claude mcp add --transport stdio dstools -- uv run --directory /path/to/dstools dstools serve
```

A ready-made `examples/claude_desktop_config.json` is included for Claude Desktop.

## Configuration

All settings are environment variables (`.env` supported). Sensible defaults mean the
keyless parts (search + fetch) work out of the box.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key (required for `deep_research`) |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI-compatible endpoint |
| `DEEPSEEK_MODEL` | `deepseek-v4-pro` | Synthesis / heavy model |
| `DEEPSEEK_FAST_MODEL` | `deepseek-v4-flash` | Cheap sub-step model |
| `DEEPSEEK_THINKING` | `auto` | `auto`/`on`/`off` — V4 thinking mode for hard steps |
| `DEEPSEEK_REASONING_EFFORT` | `high` | `low`/`medium`/`high` |
| `VISION_BASE_URL` | — | OpenAI-compatible **vision** endpoint (any multimodal model) |
| `VISION_API_KEY` | — | Key for the vision endpoint |
| `VISION_MODEL` | — | e.g. `gpt-4o`, `qwen-vl-max`, `glm-4v`, a local `qwen2.5-vl` via Ollama |
| `SEARCH_PROVIDER` | `duckduckgo` | `duckduckgo` (keyless) / `tavily` |
| `TAVILY_API_KEY` | — | Required if `SEARCH_PROVIDER=tavily` |
| `RESEARCH_BREADTH` | `3` | Sub-queries generated per round |
| `RESEARCH_DEPTH` | `2` | Research rounds |
| `RESEARCH_MAX_SOURCES` | `8` | Pages fetched & synthesised |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### Vision providers (for `analyze_image`)

Since DeepSeek-V4 can't see images, point `VISION_*` at any OpenAI-compatible multimodal model:

- **OpenAI**: `VISION_BASE_URL=https://api.openai.com/v1`, `VISION_MODEL=gpt-4o` / `gpt-4o-mini`
- **Alibaba Qwen-VL** (DashScope, OpenAI-compat): `VISION_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`, `VISION_MODEL=qwen-vl-max`
- **Zhipu GLM-4V**: `VISION_BASE_URL=https://open.bigmodel.cn/api/paas/v4`, `VISION_MODEL=glm-4v`
- **Local (Ollama)**: `VISION_BASE_URL=http://localhost:11434/v1`, `VISION_MODEL=qwen2.5-vl` (no key needed)

Without a vision provider, `analyze_image` degrades to image metadata + OCR (if `pytesseract`
is installed) and returns a clear note — it never crashes.

## Development

```bash
uv sync --extra dev
make lint        # ruff
make typecheck   # mypy
make test        # pytest
make serve       # run the server (stdio)
```

## Project layout

```
src/dstools/
  server.py          # FastMCP server + tool registration
  cli.py             # `dstools` CLI (serve / inspect / doctor)
  config.py          # pydantic-settings config
  llm/               # DeepSeek (OpenAI-compat) + vision clients, V4 thinking-aware
  search/            # pluggable search providers (DuckDuckGo default, Tavily optional)
  web/               # async page fetcher + HTML→Markdown extraction
  tools/             # image / search / fetch / research tools
  utils/             # image I/O & encoding, text chunking
tests/               # pytest suite (network & LLM mocked)
examples/            # claude_desktop_config.json, mcp client demo
```

## License

MIT.
