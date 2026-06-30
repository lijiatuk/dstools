"""Search-provider abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
    """A single search hit."""

    title: str
    url: str
    snippet: str = ""

    def to_markdown(self, index: int) -> str:
        """Render as a markdown list item with an index for citation."""
        title = self.title or "(no title)"
        line = f"{index}. **{title}**"
        if self.url:
            line += f" — {self.url}"
        if self.snippet:
            line += f"\n   {self.snippet}"
        return line


class SearchProvider(Protocol):
    """Async web-search backend."""

    name: str

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]: ...
