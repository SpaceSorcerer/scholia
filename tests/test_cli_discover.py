"""CLI tests for `scholia discover` (offline, --fake-source; --add mocked).

CRITICAL: no test here ever performs a real add-to-Zotero. The --add path shells
out to zotero_ingest.py, and that subprocess call is MOCKED — we assert the exact
command (the DOI is passed) and never run a real ingest.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

import scholia.cli as cli_mod
from scholia.cli import cli

FIXTURES = Path(__file__).parent / "fixtures"


def _build_index(runner, idx_dir):
    res = runner.invoke(
        cli,
        ["index", "--corpus", str(FIXTURES / "corpus"),
         "--index-dir", str(idx_dir), "--fake-embedder"],
    )
    assert res.exit_code == 0, res.output


def test_discover_fake_source_prints_new_candidates(tmp_path):
    """`discover --fake-source` lists ranked NEW candidate papers, clearly marked
    as not in the library."""
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _build_index(runner, idx_dir)

    res = runner.invoke(
        cli,
        ["discover", "QKI regulates alternative splicing",
         "--index-dir", str(idx_dir), "--fake-source", "--limit", "5"],
    )
    assert res.exit_code == 0, res.output
    # Clearly framed as NOT in the library.
    assert "not in your library" in res.output.lower()
    # Shows candidate fields (a DOI and a source label).
    assert "doi" in res.output.lower()
    assert "source" in res.output.lower() or "[" in res.output


def test_discover_dedupes_against_library(tmp_path):
    """A candidate whose DOI/title is already in the indexed library is not shown.

    We index the fixture corpus, then craft the fake source to emit one of the
    corpus papers; the CLI must filter it out.
    """
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _build_index(runner, idx_dir)

    # paperA.md has doi 10.1038/aaa — discover must never surface it as "new".
    res = runner.invoke(
        cli,
        ["discover", "QKI splicing cardiomyocytes",
         "--index-dir", str(idx_dir), "--fake-source", "--limit", "8"],
    )
    assert res.exit_code == 0, res.output
    assert "10.1038/aaa" not in res.output


def test_discover_can_dedupe_against_corpus_dir(tmp_path):
    """--corpus dedupes against the markdown mirror directly (no prebuilt index)."""
    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["discover", "QKI splicing",
         "--corpus", str(FIXTURES / "corpus"), "--fake-source", "--limit", "8"],
    )
    assert res.exit_code == 0, res.output
    assert "10.1038/aaa" not in res.output


def test_discover_no_library_source_still_works(tmp_path):
    """With neither --index-dir nor --corpus, discover still runs (empty library)."""
    runner = CliRunner()
    monkeypatch_env = {}  # ensure no env-driven index is picked up
    res = runner.invoke(
        cli,
        ["discover", "QKI splicing", "--fake-source", "--limit", "3"],
        env={"SCHOLIA_INDEX_DIR": str(tmp_path / "nope"),
             "SCHOLIA_CORPUS": ""},
    )
    assert res.exit_code == 0, res.output
    assert "not in your library" in res.output.lower()


def test_discover_add_invokes_ingest_subprocess_with_doi(tmp_path, monkeypatch):
    """`discover ... --add <DOI>` shells out to zotero_ingest.py with that DOI.

    The subprocess is MOCKED — we assert the exact command and DO NOT run a real
    add. This is the integrity-safe wiring test.
    """
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _build_index(runner, idx_dir)

    captured = {}

    class _FakeCompleted:
        returncode = 0
        stdout = "DRY-RUN-OK"
        stderr = ""

    def _fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FakeCompleted()

    monkeypatch.setattr(cli_mod.subprocess, "run", _fake_run)

    res = runner.invoke(
        cli,
        ["discover", "QKI splicing", "--index-dir", str(idx_dir),
         "--fake-source", "--add", "10.1234/new-paper"],
    )
    assert res.exit_code == 0, res.output

    cmd = captured["cmd"]
    # The ingest script and the DOI must both appear in the command.
    joined = " ".join(str(x) for x in cmd)
    assert "zotero_ingest.py" in joined
    assert "--doi" in cmd
    assert "10.1234/new-paper" in cmd
    # And it reminds the user to re-index after adding.
    assert "re-index" in res.output.lower() or "scholia index" in res.output


def test_discover_add_reports_ingest_failure_without_crashing(tmp_path, monkeypatch):
    """If the ingest subprocess exits non-zero, discover reports it cleanly."""
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _build_index(runner, idx_dir)

    class _FailCompleted:
        returncode = 1
        stdout = ""
        stderr = "validation failed: DOI not found"

    monkeypatch.setattr(
        cli_mod.subprocess, "run", lambda cmd, *a, **k: _FailCompleted()
    )

    res = runner.invoke(
        cli,
        ["discover", "QKI", "--index-dir", str(idx_dir),
         "--fake-source", "--add", "10.9999/bad"],
    )
    # Non-zero exit, friendly message, no traceback.
    assert res.exit_code != 0
    assert "Traceback" not in res.output
    assert "failed" in res.output.lower() or "could not" in res.output.lower()
