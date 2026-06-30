# CLAUDE.md — dstools

A reference for anyone (human or AI) working in this repo.

## What this is

`dstools` is an **MCP server** that gives **DeepSeek-V4** (a text-only model)
two capabilities it lacks natively: **image understanding** and **deep
research**. It exposes 5 tools over stdio / Streamable HTTP / SSE.

## DeepSeek-V4 specifics (verified 2026-06-30 via the official API docs)

- API is **OpenAI-compatible**; `base_url=https://api.deepseek.com` (Anthropic
  format at `/anthropic`). Use the `openai` SDK.
- Models: **`deepseek-v4-flash`** (cheap/fast) and **`deepseek-v4-pro`** (heavy).
  The legacy `deepseek-chat` / `deepseek-reasoner` are **deprecated 2026-07-24**.
- **Thinking mode**: toggle via `extra_body={"thinking": {"type": "enabled"|"disabled"}}`
  (default enabled); effort via `reasoning_effort` (`high`/`max`). In thinking
  mode `temperature`/`top_p` are silently ignored; CoT is returned as
  `message.reasoning_content`.
- **JSON output**: `response_format={"type": "json_object"}`. **Tool calls**:
  standard OpenAI function format. **Context caching**: automatic (keep system
  prompts constant to benefit).

Adaptations live in `src/dstools/llm/deepseek.py`: per-call thinking toggle,
`reasoning_content` capture, JSON mode, retry with backoff, pro/flash model
split (pro for synthesis, flash for cheap planning).

## Commands

```bash
uv sync --extra dev          # install
uv run dstools serve         # run MCP server (stdio)
uv run dstools serve --transport http --port 8000
uv run dstools inspect       # list tools + param schemas
uv run dstools doctor        # what's configured (keys/providers)
make lint && make typecheck && make test
uv run python examples/mcp_client_demo.py   # end-to-end stdio smoke test
```

## Architecture

- `server.py` — `FastMCP` instance + tool registration (`create_server()`).
- `cli.py` — `dstools` CLI (serve / inspect / doctor / version).
- `config.py` — `Settings` (pydantic-settings, env + `.env`).
- `runtime.py` — lazy singletons for clients/providers (no MCP lifespan).
- `llm/` — `DeepSeekClient` (V4-aware) + `VisionClient` (pluggable multimodal).
- `search/` — providers: `DuckDuckGoSearchProvider` (keyless, default) +
  `TavilySearchProvider` (optional).
- `web/fetcher.py` — async page fetch (streamed, size-capped) + HTML→Markdown.
- `tools/` — each module has **logic functions** (testable, dependency-injected)
  + a `register(mcp)` thin wrapper that adds `ctx` logging/progress and error
  handling. `_ctx.py` makes ctx calls defensive (never crash on missing session).
- `utils/` — image I/O (path/URL/data-uri/base64 → validated `ImageData`) and
  text helpers (chunking, truncation, JSON extraction).

## Adding a tool

1. Write `async def your_tool_logic(...) -> str` in `tools/your_tool.py`,
   taking injected `settings`/clients (defaults from `runtime`).
2. Add `register(mcp)` with `@mcp.tool(name=...)`, a clear docstring (it becomes
   the LLM-visible description), and `ctx: Context` for logging (use `ctx_info`).
3. Call `your_tool.register(server)` in `server.py` and add the module to
   `tools/__init__.py`.
4. Test the logic function with fakes (see `tests/test_research.py`).

## Conventions

- Python ≥3.10, src layout, type hints everywhere, ruff + mypy clean.
- All logs/banners go to **stderr** (stdout is the MCP stdio channel).
- Tools return **markdown strings** (broad host compatibility) and degrade
  gracefully (clear messages, never stack traces) when a backend is missing.
- Tests are hermetic: network mocked with `respx`, LLM/search/fetch via fakes.
