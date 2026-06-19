"""READ-ONLY integration test for the real Zotero Web API mirror fetch.

Deselected by default (``-m 'not integration'`` in pyproject). Run with
``pytest -m integration`` AND a real key:

    ZOTERO_API_KEY=<key> ZOTERO_USER_ID=<id> pytest -m integration -k mirror

This performs a live, READ-ONLY GET against ``api.zotero.org`` — it NEVER adds,
edits, or deletes anything in the library. It skips (rather than fails) when no
credentials are configured or the API is unreachable, so the suite stays green
without a key or connectivity.
"""

from __future__ import annotations

import os

import pytest

from scholia.corpus import load_corpus
from scholia.mirror import ZoteroUnavailable, build_corpus


@pytest.mark.integration
def test_real_zotero_mirror_readonly(tmp_path):
    api_key = os.environ.get("ZOTERO_API_KEY")
    user_id = os.environ.get("ZOTERO_USER_ID")
    group_id = os.environ.get("ZOTERO_GROUP_ID")
    if not api_key or not (user_id or group_id):
        pytest.skip(
            "Set ZOTERO_API_KEY and ZOTERO_USER_ID (or ZOTERO_GROUP_ID) to run "
            "the real-API mirror test (read-only)."
        )
    try:
        written, skipped = build_corpus(
            out_dir=tmp_path,
            api_key=api_key,
            user_id=user_id,
            group_id=group_id,
        )
    except ZoteroUnavailable as exc:
        pytest.skip(f"Zotero Web API unavailable: {exc}")

    # The produced directory must be a valid Scholia corpus.
    papers = load_corpus(tmp_path)
    assert written >= 0
    assert len(papers) <= written  # parser may skip title/abstract-less notes
    for p in papers:
        assert p.zotero_key
        assert p.zotero_link.startswith("zotero://select/library/items/")
