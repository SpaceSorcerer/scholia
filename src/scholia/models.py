"""Shared record types for Scholia."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Paper:
    """One library item parsed from a Zotero literature-mirror note."""

    id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: str = ""
    doi: str = ""
    zotero_key: str = ""
    zotero_link: str = ""
    abstract: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def embedding_text(self) -> str:
        """The text fed to the embedder: title and abstract together."""
        return f"{self.title}\n\n{self.abstract}"
