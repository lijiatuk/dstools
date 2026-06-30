# Self-host dstools MCP server over Streamable HTTP.
#   docker build -t dstools .
#   docker run --rm -p 8000:8000 -e DEEPSEEK_API_KEY=... -e VISION_*... dstools
# Configure via env vars (see .env.example). Logs go to stderr; stdout is MCP.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install the package (core deps only; add --extra ocr or --extra tavily if needed).
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY README.md LICENSE ./
RUN uv sync --no-dev

EXPOSE 8000

ENTRYPOINT ["uv", "run", "dstools", "serve", \
            "--transport", "streamable-http", \
            "--host", "0.0.0.0", "--port", "8000"]
