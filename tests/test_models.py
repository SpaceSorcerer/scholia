from scholia.models import Paper


def _paper(**over):
    base = dict(
        id="BP3TYXHJ",
        title="QKI is a splicing regulator",
        authors=["Chen, Xinyun", "Liu, Ying"],
        year="2021",
        doi="10.1038/s41467-020-20327-5",
        zotero_key="BP3TYXHJ",
        zotero_link="zotero://select/library/items/BP3TYXHJ",
        abstract="The RNA-binding protein QKI regulates alternative splicing.",
        tags=["RNA", "Cardiology"],
    )
    base.update(over)
    return Paper(**base)


def test_paper_holds_fields():
    p = _paper()
    assert p.id == "BP3TYXHJ"
    assert p.zotero_key == "BP3TYXHJ"
    assert p.authors == ["Chen, Xinyun", "Liu, Ying"]


def test_embedding_text_combines_title_and_abstract():
    p = _paper(title="T", abstract="A")
    assert p.embedding_text == "T\n\nA"


def test_paper_is_frozen():
    p = _paper()
    try:
        p.title = "mutated"
    except Exception as e:
        assert "frozen" in str(type(e)).lower() or "FrozenInstanceError" in str(type(e))
    else:
        raise AssertionError("Paper should be immutable")
