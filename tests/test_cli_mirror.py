"""CLI tests for `scholia mirror` (offline; real fetcher MONKEYPATCHED).

CRITICAL: no test here ever performs a real Zotero Web API call. The network
fetcher (``HttpZoteroFetcher``) is replaced with a deterministic
``FakeZoteroFetcher``, or the built-in ``--fake-source`` path is used. We assert
the field mapping, the "next step" hint, and that missing key/id error cleanly.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

import scholia.cli as cli_mod
from scholia.cli import cli
from scholia.corpus import load_corpus
from scholia.mirror import FakeZoteroFetcher


def _journal_item(key="ABCD1234", title="Mirrored paper", doi="10.1/a"):
    return {
        "key": key,
        "data": {
            "key": key, "itemType": "journalArticle", "title": title,
            "creators": [{"creatorType": "author", "firstName": "Jane",
                          "lastName": "Doe"}],
            "date": "2021", "DOI": doi,
            "abstractNote": "An abstract.", "tags": [{"tag": "RNA"}],
            "dateAdded": "2021-01-01T00:00:00Z",
        },
    }


def test_mirror_fake_source_writes_a_valid_corpus(tmp_path):
    """`mirror --fake-source` writes notes and prints the next-step hint; the
    output directory loads cleanly through the real corpus loader."""
    runner = CliRunner()
    out = tmp_path / "corpus"
    res = runner.invoke(cli, ["mirror", "--fake-source", "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert "Wrote 1 notes" in res.output
    assert "scholia index" in res.output
    papers = load_corpus(out)
    assert len(papers) == 1
    assert papers[0].zotero_key == "FAKE0001"


def test_mirror_maps_fields_and_round_trips(tmp_path, monkeypatch):
    """A real-shaped Zotero item flows through the CLI into a Paper with the
    right fields (the network fetcher is swapped for a FakeZoteroFetcher)."""
    runner = CliRunner()
    out = tmp_path / "corpus"

    items = [_journal_item(key="K0001", title="Paper One", doi="10.1/one"),
             _journal_item(key="K0002", title="Paper Two", doi="10.1/two")]

    def _fake_http(*args, **kwargs):
        return FakeZoteroFetcher(items, page_size=1)  # force pagination

    monkeypatch.setattr(cli_mod, "HttpZoteroFetcher", _fake_http)

    res = runner.invoke(
        cli,
        ["mirror", "--user-id", "12345", "--api-key", "secret-key",
         "--out", str(out)],
    )
    assert res.exit_code == 0, res.output
    assert "Wrote 2 notes" in res.output

    papers = sorted(load_corpus(out), key=lambda p: p.id)
    assert [p.id for p in papers] == ["K0001", "K0002"]
    assert papers[0].title == "Paper One"
    assert papers[0].authors == ["Doe, Jane"]
    assert papers[0].year == "2021"
    assert papers[0].doi == "10.1/one"
    assert papers[0].zotero_link == "zotero://select/library/items/K0001"

    # The secret key must never appear in any produced note.
    for md in out.glob("*.md"):
        assert "secret-key" not in md.read_text(encoding="utf-8")


def test_mirror_missing_key_errors_cleanly(tmp_path):
    """No --api-key and no env var -> clean error pointing to the keys page."""
    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["mirror", "--user-id", "12345", "--out", str(tmp_path / "c")],
        env={"ZOTERO_API_KEY": ""},
    )
    assert res.exit_code != 0
    assert "Traceback" not in res.output
    assert "api key" in res.output.lower()
    assert "zotero.org/settings/keys" in res.output


def test_mirror_missing_id_errors_cleanly(tmp_path):
    """A key but neither --user-id nor --group-id -> clean error."""
    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["mirror", "--api-key", "k", "--out", str(tmp_path / "c")],
    )
    assert res.exit_code != 0
    assert "Traceback" not in res.output
    assert "user-id" in res.output.lower() or "group-id" in res.output.lower()


def test_mirror_both_ids_errors_cleanly(tmp_path):
    """Both --user-id and --group-id -> clean error (exactly one)."""
    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["mirror", "--api-key", "k", "--user-id", "1", "--group-id", "2",
         "--out", str(tmp_path / "c")],
    )
    assert res.exit_code != 0
    assert "exactly one" in res.output.lower()


def test_mirror_reads_key_from_env(tmp_path, monkeypatch):
    """The API key is picked up from ZOTERO_API_KEY when --api-key is absent."""
    runner = CliRunner()
    out = tmp_path / "corpus"
    captured = {}

    def _fake_http(api_key, user_id=None, group_id=None, **kwargs):
        captured["api_key"] = api_key
        captured["user_id"] = user_id
        return FakeZoteroFetcher([_journal_item()], page_size=10)

    monkeypatch.setattr(cli_mod, "HttpZoteroFetcher", _fake_http)

    res = runner.invoke(
        cli,
        ["mirror", "--user-id", "999", "--out", str(out)],
        env={"ZOTERO_API_KEY": "from-env-key"},
    )
    assert res.exit_code == 0, res.output
    assert captured["api_key"] == "from-env-key"
    assert captured["user_id"] == "999"


def test_mirror_reports_unavailable_api_cleanly(tmp_path, monkeypatch):
    """A ZoteroUnavailable from the fetcher is reported, not crashed."""
    runner = CliRunner()

    class _BoomFetcher:
        def fetch_page(self, start, limit):
            from scholia.mirror import ZoteroUnavailable
            raise ZoteroUnavailable("offline")

    monkeypatch.setattr(cli_mod, "HttpZoteroFetcher",
                        lambda *a, **k: _BoomFetcher())

    res = runner.invoke(
        cli,
        ["mirror", "--user-id", "1", "--api-key", "k",
         "--out", str(tmp_path / "c")],
    )
    assert res.exit_code != 0
    assert "Traceback" not in res.output
    assert "unavailable" in res.output.lower()
