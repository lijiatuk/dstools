"""Tests for dstools.utils.text."""

from __future__ import annotations

from dstools.utils.text import (
    chunk_text,
    clean_whitespace,
    estimate_tokens,
    extract_json_list,
    extract_json_object,
    truncate,
)


def test_estimate_tokens_positive():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1  # 4 chars -> ~1 token
    assert estimate_tokens("a" * 300) == 100


def test_truncate_noop_when_short():
    assert truncate("short", 100) == "short"


def test_truncate_cuts_and_marks():
    out = truncate("x" * 100, 20)
    assert len(out) <= 20
    assert out.endswith("…[truncated]") or "…" in out


def test_chunk_text_small():
    assert chunk_text("hello", 100) == ["hello"]


def test_chunk_text_splits_long_paragraph():
    para = "w" * 250
    chunks = chunk_text(para, 100)
    assert len(chunks) >= 3
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(c.replace("\n", "") for c in chunks)  # non-empty


def test_chunk_text_paragraph_boundaries():
    text = "para one\n\npara two\n\npara three"
    chunks = chunk_text(text, 15)
    assert len(chunks) >= 2


def test_clean_whitespace_collapses():
    assert clean_whitespace("a    b\n\n\n\nc") == "a b\n\nc"


def test_extract_json_object_plain():
    obj = extract_json_object('{"a": 1, "b": "x"}')
    assert obj == {"a": 1, "b": "x"}


def test_extract_json_object_fenced():
    obj = extract_json_object("```json\n{\"k\": [1,2,3]}\n```")
    assert obj == {"k": [1, 2, 3]}


def test_extract_json_object_with_prose():
    obj = extract_json_object('Here you go: {"q": "deepseek v4"} thanks')
    assert obj == {"q": "deepseek v4"}


def test_extract_json_object_invalid():
    assert extract_json_object("not json at all") is None


def test_extract_json_list_plain():
    assert extract_json_list('["a", "b", "c"]') == ["a", "b", "c"]


def test_extract_json_list_fenced_with_prose():
    assert extract_json_list('Queries:\n```json\n["q1", "q2"]\n```') == ["q1", "q2"]


def test_extract_json_list_invalid():
    assert extract_json_list("no list here") is None
