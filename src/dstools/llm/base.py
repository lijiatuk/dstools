"""LLM client abstractions.

A thin, provider-agnostic interface so the tools can talk to DeepSeek-V4 (text)
and to any OpenAI-compatible vision model through the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict


class SystemMessage(TypedDict):
    role: str
    content: str


class UserMessage(TypedDict):
    role: str
    content: str


ThinkingMode = Literal["auto", "on", "off"]

# OpenAI-style message dict (role + content, or multimodal content lists).
Message = dict[str, Any]


@dataclass
class LLMResponse:
    """A normalised completion result."""

    content: str
    reasoning_content: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    """Minimal async chat-completion interface used by the tools."""

    @property
    def configured(self) -> bool: ...

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
    ) -> LLMResponse: ...
