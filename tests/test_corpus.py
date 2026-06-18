from pathlib import Path

import pytest

from scholia.corpus import parse_mirror_note

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_mirror_note_reads_frontmatter():
    p = parse_mirror_note(FIXTURES / "sample_note.md")
    assert p.zotero_key == "BP3TYXHJ"
    assert p.id == "BP3TYXHJ"
    assert p.title.startswith("QKI is a critical")
    assert p.year == "2021"
    assert p.doi == "10.1038/s41467-020-20327-5"
    assert p.zotero_link == "zotero://select/library/items/BP3TYXHJ"
    assert p.authors == ["Chen, Xinyun", "Liu, Ying"]
    assert p.tags == ["Cardiology", "RNA"]


def test_parse_mirror_note_reads_abstract_only():
    p = parse_mirror_note(FIXTURES / "sample_note.md")
    assert p.abstract.startswith("The RNA-binding protein QKI")
    # The Links section must NOT bleed into the abstract.
    assert "Open in Zotero" not in p.abstract
    assert "doi.org" not in p.abstract


def test_parse_mirror_note_without_frontmatter_raises(tmp_path):
    bad = tmp_path / "bad.md"
    bad.write_text("# No frontmatter here\n\njust text", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_mirror_note(bad)


from scholia.corpus import load_corpus


def test_load_corpus_reads_all_notes():
    papers = load_corpus(FIXTURES / "corpus")
    assert len(papers) == 3
    ids = [p.id for p in papers]
    assert ids == ["AAAAAAAA", "BBBBBBBB", "CCCCCCCC"]  # sorted by filename


def test_load_corpus_skips_malformed_md(tmp_path):
    (tmp_path / "good.md").write_text(
        (FIXTURES / "corpus" / "paperA.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "junk.md").write_text("no frontmatter", encoding="utf-8")
    papers = load_corpus(tmp_path)
    assert len(papers) == 1
    assert papers[0].id == "AAAAAAAA"
