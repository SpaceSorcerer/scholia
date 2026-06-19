"""Unit tests for `scholia mirror` — build the corpus from a Zotero library.

All tests here are deterministic and offline: they use ``FakeZoteroFetcher``
(no network) and tie the produced notes back to the REAL corpus parser
(``parse_mirror_note`` / ``load_corpus``). The real Zotero Web API call lives
behind ``@pytest.mark.integration`` in test_integration_mirror.py (deselected by
default) and performs a READ-ONLY GET only — it never mutates the library.

The contract under test: a Zotero API item, mapped to a note, must round-trip
cleanly back through the parser into the expected ``Paper`` — the same format
``scholia index`` consumes.
"""

from __future__ import annotations

import pytest
import yaml

from scholia.corpus import load_corpus, parse_mirror_note
from scholia.mirror import (
    FakeZoteroFetcher,
    HttpZoteroFetcher,
    ZoteroFetcher,
    build_corpus,
    fetch_all_items,
    is_citeable,
    map_item_to_note,
    write_corpus,
)


# --- Fixtures: representative raw Zotero API items ---------------------------


def _journal_item(
    key="ABCD1234",
    title="QKI regulates alternative splicing in cardiomyocytes",
    doi="10.1038/s41467-020-20327-5",
):
    return {
        "key": key,
        "data": {
            "key": key,
            "itemType": "journalArticle",
            "title": title,
            "creators": [
                {"creatorType": "author", "firstName": "Xinyun",
                 "lastName": "Chen"},
                {"creatorType": "author", "firstName": "Ying",
                 "lastName": "Liu"},
            ],
            "date": "2021-05-04",
            "publicationTitle": "Nature Communications",
            "DOI": doi,
            "abstractNote": (
                "The RNA-binding protein QKI controls pre-mRNA alternative "
                "splicing during cardiomyocyte differentiation."
            ),
            "tags": [{"tag": "Cardiology"}, {"tag": "RNA"}],
            "dateAdded": "2023-05-04T19:54:06Z",
        },
    }


def _attachment_item(key="ATT00001"):
    return {
        "key": key,
        "data": {
            "key": key,
            "itemType": "attachment",
            "title": "Full Text PDF",
            "parentItem": "ABCD1234",
        },
    }


def _note_item(key="NOTE0001"):
    return {
        "key": key,
        "data": {
            "key": key,
            "itemType": "note",
            "note": "<p>a standalone note, not citeable</p>",
        },
    }


# --- Field mapping ----------------------------------------------------------


def test_map_item_produces_valid_yaml_frontmatter():
    """The frontmatter must parse as YAML (this is the bug class that broke the
    author's hand-built mirror — safe_dump prevents it)."""
    note = map_item_to_note(_journal_item())
    assert note.startswith("---\n")
    fm = note.split("---", 2)[1]
    meta = yaml.safe_load(fm)  # must not raise
    assert isinstance(meta, dict)
    assert meta["zotero_key"] == "ABCD1234"
    assert meta["item_type"] == "journalArticle"


def test_map_item_round_trips_through_parser(tmp_path):
    """A mapped note, parsed by the REAL corpus parser, yields the right Paper.

    This ties the exporter to the canonical reader — if the format ever drifts,
    this test breaks.
    """
    note = map_item_to_note(_journal_item())
    path = tmp_path / "ABCD1234.md"
    path.write_text(note, encoding="utf-8")

    paper = parse_mirror_note(path)
    assert paper.zotero_key == "ABCD1234"
    assert paper.id == "ABCD1234"
    assert paper.title == "QKI regulates alternative splicing in cardiomyocytes"
    assert paper.year == "2021"
    assert paper.doi == "10.1038/s41467-020-20327-5"
    assert paper.zotero_link == "zotero://select/library/items/ABCD1234"
    assert paper.authors == ["Chen, Xinyun", "Liu, Ying"]
    assert paper.tags == ["Cardiology", "RNA"]
    assert paper.abstract.startswith("The RNA-binding protein QKI controls")
    # The Links section must not bleed into the abstract.
    assert "doi.org" not in paper.abstract
    assert "Open in Zotero" not in paper.abstract


def test_map_item_handles_tricky_title_chars(tmp_path):
    """A title with a colon, quotes, and unicode must still produce VALID YAML
    and round-trip — exactly the case hand-built frontmatter mangles."""
    item = _journal_item(
        title='QKI: a "critical" regulator of β-cell splicing — review',
    )
    note = map_item_to_note(item)
    path = tmp_path / "ABCD1234.md"
    path.write_text(note, encoding="utf-8")
    paper = parse_mirror_note(path)  # must not raise
    assert paper.title == 'QKI: a "critical" regulator of β-cell splicing — review'


def test_map_item_year_extracted_from_messy_date():
    for date, expect in [
        ("2021-05-04", "2021"),
        ("May 2021", "2021"),
        ("2019", "2019"),
        ("", ""),
        ("n.d.", ""),
    ]:
        item = _journal_item()
        item["data"]["date"] = date
        meta = yaml.safe_load(map_item_to_note(item).split("---", 2)[1])
        assert meta["year"] == expect, f"date={date!r}"


def test_map_item_missing_fields_become_empty_never_invented(tmp_path):
    """Sparse item: only key + itemType + title. Nothing is fabricated."""
    item = {
        "key": "SPARSE01",
        "data": {"key": "SPARSE01", "itemType": "journalArticle",
                 "title": "A minimal record"},
    }
    note = map_item_to_note(item)
    path = tmp_path / "SPARSE01.md"
    path.write_text(note, encoding="utf-8")
    paper = parse_mirror_note(path)
    assert paper.title == "A minimal record"
    assert paper.authors == []
    assert paper.tags == []
    assert paper.doi == ""
    assert paper.year == ""
    assert paper.abstract == ""


def test_map_item_single_field_creator_institution():
    """A 'name'-only creator (institution) is rendered as its name."""
    item = _journal_item()
    item["data"]["creators"] = [
        {"creatorType": "author", "name": "The QKI Consortium"}
    ]
    meta = yaml.safe_load(map_item_to_note(item).split("---", 2)[1])
    assert meta["authors"] == ["The QKI Consortium"]


def test_map_item_includes_editors_after_authors():
    item = _journal_item()
    item["data"]["creators"] = [
        {"creatorType": "author", "firstName": "A", "lastName": "Author"},
        {"creatorType": "editor", "firstName": "E", "lastName": "Editor"},
    ]
    meta = yaml.safe_load(map_item_to_note(item).split("---", 2)[1])
    assert meta["authors"] == ["Author, A", "Editor, E"]


# --- Citeable filtering -----------------------------------------------------


def test_is_citeable_accepts_journal_article():
    assert is_citeable(_journal_item()) is True


def test_is_citeable_rejects_attachment_and_note():
    assert is_citeable(_attachment_item()) is False
    assert is_citeable(_note_item()) is False


def test_is_citeable_rejects_keyless_item():
    assert is_citeable({"data": {"itemType": "journalArticle"}}) is False


# --- Pagination -------------------------------------------------------------


def test_fetcher_satisfies_protocol():
    assert isinstance(FakeZoteroFetcher([]), ZoteroFetcher)


def test_fetch_all_items_follows_pagination():
    """A multi-page library returns ALL items, not just the first page."""
    items = [_journal_item(key=f"K{i:04d}") for i in range(5)]
    fetcher = FakeZoteroFetcher(items, page_size=2)
    out = fetch_all_items(fetcher, page_limit=2)
    assert len(out) == 5
    assert [it["key"] for it in out] == [f"K{i:04d}" for i in range(5)]
    # Paged: 5 items / page_size 2 -> 3 page requests (starts 0, 2, 4).
    assert [c[0] for c in fetcher.calls] == [0, 2, 4]


def test_fetch_all_items_single_page():
    fetcher = FakeZoteroFetcher([_journal_item()], page_size=100)
    out = fetch_all_items(fetcher, page_limit=100)
    assert len(out) == 1
    assert fetcher.calls == [(0, 100)]


def test_fetch_all_items_empty_library():
    fetcher = FakeZoteroFetcher([], page_size=2)
    assert fetch_all_items(fetcher) == []


# --- write_corpus: end-to-end through the real loader -----------------------


def test_write_corpus_writes_one_note_per_citeable_item(tmp_path):
    items = [
        _journal_item(key="AAAA0001", doi="10.1/a"),
        _journal_item(key="BBBB0002", doi="10.1/b"),
        _attachment_item(),  # skipped
        _note_item(),        # skipped
    ]
    fetcher = FakeZoteroFetcher(items, page_size=2)
    written, skipped = write_corpus(fetcher, tmp_path)
    assert written == 2
    assert skipped == 2
    md = sorted(p.name for p in tmp_path.glob("*.md"))
    assert md == ["AAAA0001.md", "BBBB0002.md"]


def test_write_corpus_output_loads_via_load_corpus(tmp_path):
    """The produced directory IS a valid Scholia corpus — load_corpus reads it
    cleanly into the expected Papers. This is the full importer contract."""
    items = [
        _journal_item(key="AAAA0001", title="Paper A", doi="10.1/a"),
        _journal_item(key="BBBB0002", title="Paper B", doi="10.1/b"),
        _attachment_item(),
    ]
    fetcher = FakeZoteroFetcher(items, page_size=10)
    write_corpus(fetcher, tmp_path)

    papers = load_corpus(tmp_path)
    assert len(papers) == 2
    ids = sorted(p.id for p in papers)
    assert ids == ["AAAA0001", "BBBB0002"]
    titles = {p.title for p in papers}
    assert titles == {"Paper A", "Paper B"}


def test_write_corpus_is_idempotent(tmp_path):
    """Re-running overwrites by key — no duplicate notes accumulate."""
    items = [_journal_item(key="AAAA0001")]
    fetcher = FakeZoteroFetcher(items, page_size=10)
    write_corpus(fetcher, tmp_path)
    # Mutate the title, re-run; the same file is overwritten in place.
    items2 = [_journal_item(key="AAAA0001", title="Updated title")]
    write_corpus(FakeZoteroFetcher(items2, page_size=10), tmp_path)
    md = list(tmp_path.glob("*.md"))
    assert len(md) == 1
    assert parse_mirror_note(md[0]).title == "Updated title"


def test_write_corpus_creates_missing_output_dir(tmp_path):
    out = tmp_path / "new" / "corpus"
    fetcher = FakeZoteroFetcher([_journal_item()], page_size=10)
    written, _ = write_corpus(fetcher, out)
    assert written == 1
    assert out.is_dir()


def test_write_corpus_does_not_leak_api_key_into_notes(tmp_path):
    """Sanity: notes never contain anything key-shaped (the key is request-only)."""
    fetcher = FakeZoteroFetcher([_journal_item()], page_size=10)
    write_corpus(fetcher, tmp_path)
    for md in tmp_path.glob("*.md"):
        text = md.read_text(encoding="utf-8")
        assert "api_key" not in text.lower()
        assert "Zotero-API-Key" not in text


# --- build_corpus / HttpZoteroFetcher input validation (no network) ---------


def test_http_fetcher_requires_exactly_one_of_user_or_group():
    with pytest.raises(ValueError):
        HttpZoteroFetcher(api_key="k")  # neither
    with pytest.raises(ValueError):
        HttpZoteroFetcher(api_key="k", user_id="1", group_id="2")  # both


def test_http_fetcher_requires_api_key():
    with pytest.raises(ValueError):
        HttpZoteroFetcher(api_key="", user_id="123")


def test_build_corpus_validates_missing_key(tmp_path):
    with pytest.raises(ValueError):
        build_corpus(out_dir=tmp_path, api_key="", user_id="123")


def test_build_corpus_validates_missing_id(tmp_path):
    with pytest.raises(ValueError):
        build_corpus(out_dir=tmp_path, api_key="k")
