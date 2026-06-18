from pathlib import Path

from click.testing import CliRunner

from scholia.cli import cli

FIXTURES = Path(__file__).parent / "fixtures"


def test_index_then_cite_supported(tmp_path):
    runner = CliRunner()
    idx_dir = tmp_path / "idx"

    res_index = runner.invoke(
        cli,
        ["index", "--corpus", str(FIXTURES / "corpus"),
         "--index-dir", str(idx_dir), "--fake-embedder"],
    )
    assert res_index.exit_code == 0, res_index.output
    assert "Indexed 3 papers" in res_index.output

    # Query with a paper's own text -> high self-similarity -> SUPPORTED.
    passage = "QKI regulates alternative splicing in cardiomyocytes\n\nQKI is an RNA-binding protein that controls pre-mRNA alternative splicing during cardiac differentiation."
    res_cite = runner.invoke(
        cli,
        ["cite", passage, "--index-dir", str(idx_dir),
         "--threshold", "0.5", "--fake-embedder"],
    )
    assert res_cite.exit_code == 0, res_cite.output
    assert "AAAAAAAA" in res_cite.output
    assert "CLAIM-CHECK: SUPPORTED" in res_cite.output


def test_cite_unsupported_with_high_threshold(tmp_path):
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    runner.invoke(
        cli,
        ["index", "--corpus", str(FIXTURES / "corpus"),
         "--index-dir", str(idx_dir), "--fake-embedder"],
    )
    res = runner.invoke(
        cli,
        ["cite", "completely unrelated topic", "--index-dir", str(idx_dir),
         "--threshold", "0.99", "--fake-embedder"],
    )
    assert res.exit_code == 0, res.output
    assert "CLAIM-CHECK: UNSUPPORTED" in res.output


# --- Finding B: friendly error paths ---

def test_cite_missing_index_dir_friendly_message(tmp_path):
    """cite against a nonexistent index dir must print a friendly message, no Traceback, non-zero exit."""
    runner = CliRunner()
    missing = tmp_path / "no_index_here"
    res = runner.invoke(
        cli,
        ["cite", "some passage", "--index-dir", str(missing), "--fake-embedder"],
    )
    assert res.exit_code != 0
    assert "Traceback" not in (res.output or "")
    # The message must mention how to fix it
    combined = (res.output or "") + str(res.exception or "")
    assert "scholia index" in combined or "No index at" in combined


def test_cite_dim_mismatch_friendly_message(tmp_path):
    """cite with wrong embedder dim must print a friendly mismatch message, non-zero exit."""
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    # Build index with FakeEmbedder(dim=16)
    runner.invoke(
        cli,
        ["index", "--corpus", str(FIXTURES / "corpus"),
         "--index-dir", str(idx_dir), "--fake-embedder"],
    )
    # Query with a different dim by patching FakeEmbedder — we simulate mismatch
    # by directly testing the error path: build index, then call cite with a
    # FakeEmbedder of different dim via monkeypatching
    import scholia.cli as cli_mod
    from scholia.embedders import FakeEmbedder as FE
    orig_make = cli_mod._make_embedder
    cli_mod._make_embedder = lambda fake, model: FE(dim=32)  # wrong dim
    try:
        res = runner.invoke(
            cli,
            ["cite", "some passage", "--index-dir", str(idx_dir), "--fake-embedder"],
        )
    finally:
        cli_mod._make_embedder = orig_make
    assert res.exit_code != 0
    assert "Traceback" not in (res.output or "")
    combined = (res.output or "") + str(res.exception or "")
    assert "dimension mismatch" in combined.lower() or "mismatch" in combined.lower()


# --- Finding D: env-var defaults ---

def test_index_respects_scholia_corpus_env(tmp_path, monkeypatch):
    """SCHOLIA_CORPUS env var is used as corpus when --corpus not passed."""
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    monkeypatch.setenv("SCHOLIA_CORPUS", str(FIXTURES / "corpus"))
    monkeypatch.setenv("SCHOLIA_INDEX_DIR", str(idx_dir))
    res = runner.invoke(cli, ["index", "--fake-embedder"])
    assert res.exit_code == 0, res.output
    assert "Indexed 3 papers" in res.output


def test_index_no_corpus_no_env_exits_cleanly(tmp_path, monkeypatch):
    """Without --corpus and no SCHOLIA_CORPUS env var, index exits non-zero with a helpful message."""
    runner = CliRunner()
    monkeypatch.delenv("SCHOLIA_CORPUS", raising=False)
    monkeypatch.setenv("SCHOLIA_INDEX_DIR", str(tmp_path / "idx"))
    res = runner.invoke(cli, ["index", "--fake-embedder"])
    assert res.exit_code != 0
    assert "Traceback" not in (res.output or "")
    combined = (res.output or "") + str(res.exception or "")
    assert "SCHOLIA_CORPUS" in combined or "--corpus" in combined


def test_index_respects_scholia_index_dir_env(tmp_path, monkeypatch):
    """SCHOLIA_INDEX_DIR env var is used as index dir when --index-dir not passed."""
    runner = CliRunner()
    idx_dir = tmp_path / "env_idx"
    monkeypatch.setenv("SCHOLIA_CORPUS", str(FIXTURES / "corpus"))
    monkeypatch.setenv("SCHOLIA_INDEX_DIR", str(idx_dir))
    res = runner.invoke(cli, ["index", "--fake-embedder"])
    assert res.exit_code == 0, res.output
    assert idx_dir.exists()
