"""DeepSeek-V4 LLM client (OpenAI-compatible).

Deeply adapted to DeepSeek-V4:

* Models default to ``deepseek-v4-pro`` / ``deepseek-v4-flash`` (the legacy
  ``deepseek-chat`` / ``deepseek-reasoner`` names are deprecated 2026-07-24).
* V4 **thinking mode** is toggled via ``extra_body={"thinking": {"type": ...}}``
  and effort via the ``reasoning_effort`` parameter. In thinking mode the model
  returns ``reasoning_content`` alongside ``content`` — captured here for
  debugging but kept out of tool output by default.
* Sampling params (``temperature`` etc.) are omitted in thinking mode (they are
  silently ignored by V4 there).
* V4's **JSON output** mode (``response_format={"type": "json_object"}``) is used
  for structured extraction.
* Transient errors are retried with exponential backoff.
"""

from __future__ import annotations

from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import Settings
from ..exceptions import ConfigError, LLMError
from ..logging_setup import get_logger
from .base import LLMResponse, Message, ThinkingMode

_logger = get_logger("llm.deepseek")

# Errors that are safe to retry (transient).
_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


def _get_reasoning(message: Any) -> str:
    """Extract V4 ``reasoning_content`` from an OpenAI SDK chat message."""
    rc = getattr(message, "reasoning_content", None)
    if rc:
        return rc
    try:
        data = message.model_dump()
    except Exception:
        return ""
    return data.get("reasoning_content") or ""


class DeepSeekClient:
    """Async OpenAI-compatible client targeting DeepSeek-V4."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncOpenAI | None = None
        if settings.has_deepseek:
            self._client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                timeout=settings.deepseek_timeout,
            )

    @property
    def configured(self) -> bool:
        return self._client is not None

    def _require(self) -> AsyncOpenAI:
        if self._client is None:
            raise ConfigError(
                "DeepSeek API key is not configured. Set DEEPSEEK_API_KEY (see "
                ".env.example) — it is required for deep_research."
            )
        return self._client

    def _resolve_thinking(self, thinking: ThinkingMode) -> bool:
        """Resolve an 'auto' thinking request against the global setting."""
        if thinking == "auto":
            # 'auto'/'on' in settings => enabled (V4 default); 'off' => disabled.
            return self._settings.deepseek_thinking != "off"
        return thinking == "on"

    async def complete(
        self,
        *,
        messages: list[Message],
        model: str | None = None,
        thinking: ThinkingMode = "auto",
        reasoning_effort: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        client = self._require()
        use_thinking = self._resolve_thinking(thinking)
        effort = reasoning_effort or self._settings.deepseek_reasoning_effort
        model = model or self._settings.deepseek_model

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        extra_body: dict[str, Any] = {"thinking": {"type": "enabled" if use_thinking else "disabled"}}
        if use_thinking:
            # reasoning_effort is a recognised OpenAI param; V4 accepts it directly.
            kwargs["reasoning_effort"] = effort
        else:
            # Sampling params are only meaningful (and not silently ignored) in
            # non-thinking mode.
            if temperature is not None:
                kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        kwargs["extra_body"] = extra_body

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(4),
                wait=wait_exponential(multiplier=1.5, max=20),
                retry=retry_if_exception_type(_RETRYABLE),
                reraise=True,
            ):
                with attempt:
                    resp = await client.chat.completions.create(**kwargs)
        except _RETRYABLE as exc:
            raise LLMError(f"DeepSeek API transient failure after retries: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"DeepSeek API call failed: {exc}") from exc

        choice = resp.choices[0]
        message = choice.message
        content = message.content or ""
        reasoning = _get_reasoning(message)
        usage = resp.usage
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0

        _logger.debug(
            "deepseek complete model=%s thinking=%s effort=%s in=%d out=%d reasoning=%d chars",
            model,
            use_thinking,
            effort if use_thinking else "-",
            in_tok,
            out_tok,
            len(reasoning),
        )
        return LLMResponse(
            content=content,
            reasoning_content=reasoning,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )
