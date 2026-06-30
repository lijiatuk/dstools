"""Pluggable vision client for image understanding.

DeepSeek-V4 is text-only, so image understanding is provided by **any**
OpenAI-compatible multimodal model pointed at via ``VISION_BASE_URL`` /
``VISION_API_KEY`` / ``VISION_MODEL`` (OpenAI gpt-4o, Qwen-VL, GLM-4V, a local
Ollama vision model, …). The client sends a standard
``{type: image_url, image_url: {url: data:...}}`` multimodal message.
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
from ..exceptions import LLMError, VisionNotConfiguredError
from ..logging_setup import get_logger

_logger = get_logger("llm.vision")

_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


class VisionClient:
    """Async OpenAI-compatible multimodal (vision) client."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncOpenAI | None = None
        if settings.has_vision:
            self._client = AsyncOpenAI(
                api_key=settings.vision_api_key or "not-set",
                base_url=settings.vision_base_url,
                timeout=settings.vision_timeout,
            )

    @property
    def configured(self) -> bool:
        return self._client is not None

    def _require(self) -> AsyncOpenAI:
        if self._client is None:
            raise VisionNotConfiguredError(
                "No vision provider is configured. DeepSeek-V4 cannot see images "
                "natively, so set VISION_BASE_URL / VISION_API_KEY / VISION_MODEL "
                "to any OpenAI-compatible multimodal model (see .env.example)."
            )
        return self._client

    async def describe(
        self,
        *,
        data_url: str,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Ask the vision model *prompt* about the image at *data_url*."""
        client = self._require()
        model = self._settings.vision_model
        max_tokens = max_tokens or self._settings.vision_max_tokens

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1.5, max=15),
                retry=retry_if_exception_type(_RETRYABLE),
                reraise=True,
            ):
                with attempt:
                    resp = await client.chat.completions.create(**kwargs)
        except _RETRYABLE as exc:
            raise LLMError(f"Vision API transient failure after retries: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"Vision API call failed: {exc}") from exc

        message = resp.choices[0].message
        return message.content or ""
