"""Text utilities: chunking, truncation, and rough token estimation.

Token estimation is approximate (no tokenizer dependency) and deliberately
*over*-estimates so context budgets are respected conservatively.
"""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"[ \t]+")
_NEWLINE_RE = re.compile(r"\n{3,}")


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token for latin text; over-estimates CJK)."""
    return max(1, len(text) // 3)


def truncate(text: str, max_chars: int, *, suffix: str = "\n…[truncated]") -> str:
    """Truncate *text* to *max_chars*, appending *suffix* if cut."""
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - len(suffix))
    return text[:keep].rstrip() + suffix


def chunk_text(text: str, max_chars: int, *, overlap: int = 0) -> list[str]:
    """Split *text* into chunks of at most *max_chars*.

    Splits on paragraph boundaries where possible, falling back to hard cuts.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    paragraphs = text.split("\n\n")
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".lstrip("\n") if current else para
            continue

        # Flush current chunk.
        if current:
            chunks.append(current)
            current = ""

        # Paragraph itself too long: hard-split.
        while len(para) > max_chars:
            chunks.append(para[:max_chars])
            para = para[max_chars - overlap :] if overlap else para[max_chars:]
        if para:
            current = para

    if current:
        chunks.append(current)
    return chunks


def clean_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and excessive blank lines."""
    text = _WHITESPACE_RE.sub(" ", text)
    text = _NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# LLM-output JSON extraction
# ---------------------------------------------------------------------------

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_LIST_RE = re.compile(r"\[.*\]", re.DOTALL)


def _strip_fence(text: str) -> str:
    """Strip ```json / ``` code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from LLM output.

    Returns ``None`` if no valid object can be found.
    """
    import json

    candidate = _strip_fence(text)
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        pass
    match = _JSON_OBJ_RE.search(candidate)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def extract_json_list(text: str) -> list | None:
    """Best-effort extraction of a JSON list from LLM output."""
    import json

    candidate = _strip_fence(text)
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, list) else None
    except (json.JSONDecodeError, ValueError):
        pass
    match = _JSON_LIST_RE.search(candidate)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, list) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None
