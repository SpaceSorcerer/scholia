import subprocess
import sys
from pathlib import Path

from click.testing import CliRunner

from scholia.cli import cli

FIXTURES = Path(__file__).parent / "fixtures"


def test_python_m_scholia_is_not_a_silent_noop():
    """`python -m scholia` must invoke the CLI (prints help), not silently exit."""
    res = subprocess.run(
        [sys.executable, "-m", "scholia", "--help"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    assert "Scholia" in (res.stdout + res.stderr)


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


# --- Item 1: malformed-note skip warning ---

def test_index_warns_on_skipped_malformed_notes(tmp_path):
    """index over a corpus with a malformed-YAML note still builds and warns."""
    runner = CliRunner()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "good.md").write_text(
        (FIXTURES / "corpus" / "paperA.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (corpus / "malformed.md").write_text(
        (FIXTURES / "corpus" / "paperD_malformed.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    res = runner.invoke(
        cli,
        ["index", "--corpus", str(corpus),
         "--index-dir", str(tmp_path / "idx"), "--fake-embedder"],
    )
    assert res.exit_code == 0, res.output
    assert "Indexed 1 papers" in res.output
    assert "skipped 1 malformed notes" in res.output


# --- Item 4: embedder-aware default threshold + adopt stored embedder ---

def test_default_threshold_for_picks_per_embedder():
    from scholia.cli import default_threshold_for
    assert default_threshold_for("nomic-ai/nomic-embed-text-v1.5") == 0.73
    assert default_threshold_for("sentence-transformers/all-MiniLM-L6-v2") == 0.45
    assert default_threshold_for("FakeEmbedder") == 0.45
    assert default_threshold_for("some-unknown-model") == 0.45
    assert default_threshold_for("") == 0.45


def test_cite_uses_minilm_default_threshold_in_output(tmp_path):
    """With a Fake/MiniLM index and no --threshold, the claim-check line shows 0.45."""
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
         "--fake-embedder"],
    )
    assert res.exit_code == 0, res.output
    assert "0.45" in res.output


def test_cite_adopts_stored_embedder_model(tmp_path, monkeypatch):
    """cite without --model adopts the index's stored embedder_model."""
    import scholia.cli as cli_mod
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    runner.invoke(
        cli,
        ["index", "--corpus", str(FIXTURES / "corpus"),
         "--index-dir", str(idx_dir), "--fake-embedder"],
    )

    captured = {}
    orig_make = cli_mod._make_embedder

    def _spy(fake, model):
        captured["model"] = model
        return orig_make(fake, model)

    monkeypatch.setattr(cli_mod, "_make_embedder", _spy)
    res = runner.invoke(
        cli,
        ["cite", "some passage", "--index-dir", str(idx_dir), "--fake-embedder"],
    )
    assert res.exit_code == 0, res.output
    # The stored embedder_model ("FakeEmbedder") must have been adopted.
    assert captured["model"] == "FakeEmbedder"


def test_cite_explicit_threshold_overrides_default(tmp_path):
    """An explicit --threshold is honored over the embedder-aware default."""
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    runner.invoke(
        cli,
        ["index", "--corpus", str(FIXTURES / "corpus"),
         "--index-dir", str(idx_dir), "--fake-embedder"],
    )
    res = runner.invoke(
        cli,
        ["cite", "unrelated", "--index-dir", str(idx_dir),
         "--threshold", "0.99", "--fake-embedder"],
    )
    assert res.exit_code == 0, res.output
    assert "0.99" in res.output
    assert "UNSUPPORTED" in res.output


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
