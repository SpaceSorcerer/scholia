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


from scholia.corpus import load_corpus, load_corpus_reporting


def test_load_corpus_reads_all_notes():
    # The fixture corpus also contains paperD_malformed.md (invalid YAML); it
    # must be skipped, leaving the three well-formed papers.
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


def test_parse_mirror_note_malformed_yaml_raises_value_error(tmp_path):
    """A present-but-invalid YAML frontmatter (`tags:[]`) must raise ValueError,
    not a bare yaml.YAMLError, so load_corpus's skip path catches it."""
    bad = tmp_path / "malformed.md"
    bad.write_text(
        '---\ntitle: "x"\nzotero_key: "ZZ"\ntags:[]\n---\n\n## Abstract\n\nbody\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        parse_mirror_note(bad)


def test_load_corpus_skips_malformed_yaml_and_keeps_good(tmp_path):
    """A note with malformed YAML frontmatter is skipped; good papers remain."""
    (tmp_path / "good.md").write_text(
        (FIXTURES / "corpus" / "paperA.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "malformed.md").write_text(
        (FIXTURES / "corpus" / "paperD_malformed.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    papers = load_corpus(tmp_path)
    assert len(papers) == 1
    assert papers[0].id == "AAAAAAAA"


def test_load_corpus_reporting_surfaces_skip_count(tmp_path):
    """load_corpus_reporting returns (papers, skipped_count); the malformed-YAML
    note and the no-frontmatter note both count as skips."""
    (tmp_path / "good.md").write_text(
        (FIXTURES / "corpus" / "paperA.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "malformed.md").write_text(
        (FIXTURES / "corpus" / "paperD_malformed.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "junk.md").write_text("no frontmatter", encoding="utf-8")
    papers, skipped = load_corpus_reporting(tmp_path)
    assert len(papers) == 1
    assert papers[0].id == "AAAAAAAA"
    assert skipped == 2
