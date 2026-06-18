"""Read-only loader for the Zotero literature-mirror markdown corpus."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from scholia.models import Paper

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _extract_abstract(body: str) -> str:
    """Return the text under '## Abstract' up to the next '##' heading."""
    m = re.search(r"##\s+Abstract\s*\n(.*?)(?:\n##\s+|\Z)", body, re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip()


def parse_mirror_note(path: Path) -> Paper:
    """Parse one mirror markdown note into a Paper. Read-only."""
    text = Path(path).read_text(encoding="utf-8")
    fm_match = _FRONTMATTER_RE.match(text)
    if not fm_match:
        raise ValueError(f"No YAML frontmatter found in {path}")

    meta = yaml.safe_load(fm_match.group(1)) or {}
    body = text[fm_match.end():]

    def _str(key: str) -> str:
        val = meta.get(key, "")
        return "" if val is None else str(val).strip()

    def _list(key: str) -> list[str]:
        val = meta.get(key) or []
        if not isinstance(val, list):
            return []
        return [str(v).strip() for v in val if v is not None and str(v).strip()]

    zotero_key = _str("zotero_key") or Path(path).stem

    return Paper(
        id=zotero_key,
        title=_str("title"),
        authors=_list("authors"),
        year=_str("year"),
        doi=_str("doi"),
        zotero_key=zotero_key,
        zotero_link=_str("zotero_link"),
        abstract=_extract_abstract(body),
        tags=_list("tags"),
    )


def load_corpus(corpus_dir: Path) -> list[Paper]:
    """Parse every *.md note in corpus_dir (sorted) into Papers. Read-only.

    Files without valid frontmatter are skipped. Papers with neither title
    nor abstract are skipped (nothing to embed).
    """
    corpus_dir = Path(corpus_dir)
    papers: list[Paper] = []
    for md_path in sorted(corpus_dir.glob("*.md")):
        try:
            paper = parse_mirror_note(md_path)
        except ValueError:
            continue
        if not paper.title and not paper.abstract:
            continue
        papers.append(paper)
    return papers
