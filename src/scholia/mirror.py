"""Build the Scholia corpus from ANY user's Zotero library (Web API v3).

Scholia indexes a directory of markdown "mirror notes" — one note per library
item, in the exact format ``scholia.corpus.parse_mirror_note`` reads. Until now
only the author's private exporter could produce that corpus. ``scholia mirror``
closes the loop: any user can turn THEIR Zotero library into the corpus with
nothing but their numeric user (or group) id and a read-only API key.

INTEGRITY / SCOPE
-----------------
This is an IMPORTER, not a writer of scholarship. It only ever GETs from the
Zotero Web API (read-only) and writes plain markdown notes locally. It never
generates prose, never invents metadata, and never mutates the Zotero library.
Fields are copied straight from the API response; missing fields become empty
(never fabricated).

PRIVACY
-------
The API key is supplied by the caller (CLI flag or the ``ZOTERO_API_KEY``
environment variable) — never hardcoded, never written into a note, never
logged. Only the key, the user/group id, and the standard request headers leave
the machine, all to ``api.zotero.org`` over HTTPS.

NETWORK
-------
Standard library only (``urllib``) — no new dependency, the same pattern as
``discovery.py``. Pagination is followed to completion via the ``Link``
``rel="next"`` header (falling back to ``Total-Results`` + ``start``). A
``FakeZoteroFetcher`` makes the whole module deterministic and offline for unit
tests; the real fetcher is only constructed when an actual fetch is requested.

YAML SAFETY
-----------
Frontmatter is emitted with ``yaml.safe_dump(allow_unicode=True,
sort_keys=False)`` so it is ALWAYS valid YAML. Hand-built frontmatter (the bug
that silently broke the author's mirror — e.g. an unescaped colon or quote in a
title producing ``tags:[]``-class invalid YAML) is structurally impossible here:
every note round-trips back through ``parse_mirror_note``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml

# --- Configuration ----------------------------------------------------------

_API_ROOT = "https://api.zotero.org"
_API_VERSION = "3"
_PAGE_LIMIT = 100  # Zotero's maximum page size for /items.
_USER_AGENT = "scholia-mirror/0.1 (local; mailto:gsdewson@utmb.edu)"

# Item types that carry citeable scholarship. Everything else returned by the
# /items endpoint (attachments, standalone notes, annotations) is skipped — it
# has no abstract/title to ground against. This is an allow-list on purpose:
# unknown future scholarly types are skipped rather than silently mis-imported,
# and the list is easy to extend.
CITEABLE_ITEM_TYPES = frozenset(
    {
        "journalArticle",
        "preprint",
        "conferencePaper",
        "book",
        "bookSection",
        "thesis",
        "report",
        "manuscript",
        "document",
        "magazineArticle",
        "newspaperArticle",
        "dataset",
    }
)


class ZoteroUnavailable(RuntimeError):
    """Raised when the Zotero Web API cannot be reached (offline / auth / rate)."""


# --- Fetcher protocol (mockable) --------------------------------------------


@runtime_checkable
class ZoteroFetcher(Protocol):
    """Fetches raw item pages from a Zotero library.

    ``fetch_page(start, limit)`` returns ``(items, has_next)`` where ``items`` is
    the parsed JSON list for that page and ``has_next`` says whether another page
    follows. Implementations handle their own transport; tests inject a fake.
    """

    def fetch_page(self, start: int, limit: int) -> tuple[list[dict], bool]:
        ...


class HttpZoteroFetcher:
    """Real fetcher: read-only GETs against the Zotero Web API v3 (stdlib only).

    Builds the user- or group-scoped ``/items`` URL, sends the API key in the
    ``Zotero-API-Key`` header (never in the URL, never logged), and parses one
    page. Pagination is decided from the ``Link: rel="next"`` header when present
    (authoritative), falling back to ``Total-Results`` vs. ``start + len``.
    """

    def __init__(
        self,
        api_key: str,
        user_id: str | None = None,
        group_id: str | None = None,
        timeout: int = 30,
    ) -> None:
        if bool(user_id) == bool(group_id):
            raise ValueError(
                "Provide exactly one of user_id or group_id (not both/neither)."
            )
        if not api_key:
            raise ValueError("A Zotero API key is required.")
        self._api_key = api_key
        self._timeout = timeout
        scope = f"users/{user_id}" if user_id else f"groups/{group_id}"
        self._items_url = f"{_API_ROOT}/{scope}/items"

    def fetch_page(self, start: int, limit: int) -> tuple[list[dict], bool]:
        params = urllib.parse.urlencode(
            {"format": "json", "limit": int(limit), "start": int(start)}
        )
        url = f"{self._items_url}?{params}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Zotero-API-Version": _API_VERSION,
                "Zotero-API-Key": self._api_key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
                link_header = resp.headers.get("Link", "") or ""
                total_header = resp.headers.get("Total-Results")
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            # Never echo the key; the URL we report is key-free by construction.
            raise ZoteroUnavailable(f"GET {url!r} failed: {exc}") from exc
        try:
            items = json.loads(body)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ZoteroUnavailable(f"Bad JSON from {url!r}: {exc}") from exc
        if not isinstance(items, list):
            raise ZoteroUnavailable(f"Unexpected response shape from {url!r}.")

        has_next = 'rel="next"' in link_header
        if not link_header and total_header is not None:
            try:
                total = int(total_header)
                has_next = (start + len(items)) < total
            except (TypeError, ValueError):
                has_next = False
        return items, has_next


# --- Field mapping (Zotero item -> mirror-note frontmatter) ------------------


def _creator_name(creator: dict) -> str:
    """Render one Zotero creator as ``"Last, First"`` (the mirror convention).

    Zotero creators are either two-field (``lastName``/``firstName``) or
    single-field (``name``, used for institutions). Editors are kept alongside
    authors so editor-only items (e.g. edited books) still get a credit; the
    raw author/editor order from Zotero is preserved by the caller.
    """
    last = str(creator.get("lastName", "") or "").strip()
    first = str(creator.get("firstName", "") or "").strip()
    if last and first:
        return f"{last}, {first}"
    if last:
        return last
    # Single-field creator (institution / "name" only).
    return str(creator.get("name", "") or "").strip()


def _authors(data: dict) -> list[str]:
    """Authors (then editors) as ``"Last, First"`` strings, order preserved."""
    out: list[str] = []
    for ct in ("author", "editor"):
        for c in data.get("creators") or []:
            if str(c.get("creatorType", "")) != ct:
                continue
            name = _creator_name(c)
            if name:
                out.append(name)
    return out


def _year(date: str) -> str:
    """Extract a 4-digit year from a free-form Zotero date string.

    Zotero dates are messy ("2021", "2021-05", "May 2021", "2021/06/01"). Scan
    for the first 4-digit run; return "" when none is present (never guess).
    """
    digits = ""
    for ch in str(date or ""):
        if ch.isdigit():
            digits += ch
            if len(digits) == 4:
                return digits
        else:
            digits = ""
    return ""


def _tags(data: dict) -> list[str]:
    """Tag strings from Zotero's ``tags: [{"tag": ...}, ...]`` structure."""
    out: list[str] = []
    for t in data.get("tags") or []:
        tag = str((t or {}).get("tag", "") or "").strip()
        if tag:
            out.append(tag)
    return out


def map_item_to_note(item: dict) -> str:
    """Render one Zotero API item as a mirror-note markdown string.

    The output is byte-for-byte in the format ``parse_mirror_note`` reads:
    ``safe_dump``-ed YAML frontmatter (always valid), then ``# <title>``, a
    ``## Abstract`` section, and a ``## Links`` section. Every field is copied
    straight from the item; absent fields become empty strings/lists (never
    invented).
    """
    data = item.get("data") or {}
    key = str(item.get("key") or data.get("key") or "").strip()
    title = str(data.get("title", "") or "").strip()
    doi = str(data.get("DOI", "") or "").strip()
    item_type = str(data.get("itemType", "") or "").strip()
    publication = str(data.get("publicationTitle", "") or "").strip()
    abstract = str(data.get("abstractNote", "") or "").strip()
    date_added = str(data.get("dateAdded", "") or "").strip()
    zotero_link = (
        f"zotero://select/library/items/{key}" if key else ""
    )

    # Build the frontmatter as a dict and let safe_dump produce VALID YAML.
    # sort_keys=False keeps the human-friendly order documented in the README.
    frontmatter = {
        "title": title,
        "authors": _authors(data),
        "year": _year(data.get("date", "")),
        "publication": publication,
        "doi": doi,
        "item_type": item_type,
        "zotero_key": key,
        "zotero_link": zotero_link,
        "tags": _tags(data),
        "date_added": date_added,
    }
    fm_yaml = yaml.safe_dump(
        frontmatter, allow_unicode=True, sort_keys=False, default_flow_style=False
    )

    parts = [
        "---",
        fm_yaml.rstrip("\n"),
        "---",
        "",
        f"# {title}".rstrip(),
        "",
        "## Abstract",
        "",
        abstract,
        "",
        "## Links",
        "",
    ]
    links: list[str] = []
    if zotero_link:
        links.append(f"- [Open in Zotero]({zotero_link})")
    if doi:
        links.append(f"- [DOI](https://doi.org/{doi})")
    parts.extend(links)
    return "\n".join(parts).rstrip("\n") + "\n"


def is_citeable(item: dict) -> bool:
    """True when the item is a citeable scholarly type with a usable key.

    Attachments, standalone notes, and annotations are skipped (no
    title/abstract to ground). A keyless item is also skipped — the key is the
    note's filename and identity.
    """
    data = item.get("data") or {}
    key = str(item.get("key") or data.get("key") or "").strip()
    if not key:
        return False
    return str(data.get("itemType", "")) in CITEABLE_ITEM_TYPES


# --- Orchestration ----------------------------------------------------------


def fetch_all_items(fetcher: ZoteroFetcher, page_limit: int = _PAGE_LIMIT) -> list[dict]:
    """Page through the whole library via ``fetcher``, returning every raw item.

    Follows pagination to completion (``has_next`` from the fetcher). A hard cap
    on the number of pages guards against a misbehaving server looping forever.
    """
    items: list[dict] = []
    start = 0
    # Generous safety cap: 10,000 pages * 100 = 1,000,000 items.
    for _ in range(10_000):
        page, has_next = fetcher.fetch_page(start=start, limit=page_limit)
        items.extend(page)
        if not has_next or not page:
            break
        start += len(page)
    return items


def write_corpus(
    fetcher: ZoteroFetcher,
    out_dir: Path,
    page_limit: int = _PAGE_LIMIT,
) -> tuple[int, int]:
    """Fetch the whole library and write one mirror note per citeable item.

    Idempotent: each note is named ``<zotero_key>.md`` and OVERWRITTEN on every
    run (the key is stable), so re-running refreshes in place without
    duplicating. Non-citeable items (attachments/notes/annotations) and keyless
    items are skipped.

    Returns ``(written, skipped)``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = fetch_all_items(fetcher, page_limit=page_limit)
    written = 0
    skipped = 0
    for item in items:
        if not is_citeable(item):
            skipped += 1
            continue
        data = item.get("data") or {}
        key = str(item.get("key") or data.get("key") or "").strip()
        note = map_item_to_note(item)
        (out_dir / f"{key}.md").write_text(note, encoding="utf-8")
        written += 1
    return written, skipped


def build_corpus(
    out_dir: Path,
    api_key: str,
    user_id: str | None = None,
    group_id: str | None = None,
    page_limit: int = _PAGE_LIMIT,
) -> tuple[int, int]:
    """Convenience wrapper: build the REAL HTTP fetcher and write the corpus.

    Validates that exactly one of ``user_id``/``group_id`` and an ``api_key`` are
    present (clear errors otherwise — the CLI surfaces these). Returns
    ``(written, skipped)``. The key is used only to construct the fetcher's
    request header; it is never returned, stored, or logged here.
    """
    fetcher = HttpZoteroFetcher(
        api_key=api_key, user_id=user_id, group_id=group_id
    )
    return write_corpus(fetcher, out_dir, page_limit=page_limit)


# --- FakeZoteroFetcher (deterministic, offline; test-only) ------------------


class FakeZoteroFetcher:
    """Deterministic, offline fetcher for unit tests (no network).

    Serves a fixed list of raw Zotero items, paginated by ``page_size`` so tests
    can prove that pagination is followed to completion. ``calls`` records the
    ``(start, limit)`` of every page request for assertions.
    """

    def __init__(self, items: list[dict], page_size: int = 2) -> None:
        self._items = list(items)
        self._page_size = max(int(page_size), 1)
        self.calls: list[tuple[int, int]] = []

    def fetch_page(self, start: int, limit: int) -> tuple[list[dict], bool]:
        self.calls.append((start, limit))
        page = self._items[start:start + self._page_size]
        has_next = (start + self._page_size) < len(self._items)
        return page, has_next
