"""CLI tests for the cross-encoder re-rank stage (FakeReranker, no download)."""

from pathlib import Path

from click.testing import CliRunner

from scholia.cli import cli, default_reranker_threshold_for

FIXTURES = Path(__file__).parent / "fixtures"


def _index(runner, idx_dir):
    return runner.invoke(
        cli,
        ["index", "--corpus", str(FIXTURES / "corpus"),
         "--index-dir", str(idx_dir), "--fake-embedder"],
    )


def test_cite_rerank_with_fake_reranker_supported(tmp_path):
    """cite --rerank --fake-reranker re-scores and still returns the QKI paper SUPPORTED."""
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _index(runner, idx_dir)
    passage = ("QKI regulates alternative splicing in cardiomyocytes\n\n"
               "QKI is an RNA-binding protein that controls pre-mRNA "
               "alternative splicing during cardiac differentiation.")
    res = runner.invoke(
        cli,
        ["cite", passage, "--index-dir", str(idx_dir),
         "--rerank", "--fake-reranker", "--fake-embedder",
         "--threshold", "0.0001"],
    )
    assert res.exit_code == 0, res.output
    assert "AAAAAAAA" in res.output
    assert "CLAIM-CHECK: SUPPORTED" in res.output
    # The output must indicate reranking is on.
    assert "rerank" in res.output.lower()


def test_cite_rerank_unsupported_with_high_threshold(tmp_path):
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _index(runner, idx_dir)
    res = runner.invoke(
        cli,
        ["cite", "completely unrelated nonsense topic", "--index-dir", str(idx_dir),
         "--rerank", "--fake-reranker", "--fake-embedder",
         "--threshold", "999999"],
    )
    assert res.exit_code == 0, res.output
    assert "CLAIM-CHECK: UNSUPPORTED" in res.output


def test_cite_no_rerank_matches_bi_encoder_path(tmp_path):
    """--no-rerank must use the bi-encoder path (cosine threshold 0.45 for Fake)."""
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _index(runner, idx_dir)
    res = runner.invoke(
        cli,
        ["cite", "completely unrelated topic", "--index-dir", str(idx_dir),
         "--no-rerank", "--fake-embedder"],
    )
    assert res.exit_code == 0, res.output
    # Bi-encoder default threshold shows 0.45 in the claim-check line.
    assert "0.45" in res.output


def test_cite_candidate_k_option_accepted(tmp_path):
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _index(runner, idx_dir)
    res = runner.invoke(
        cli,
        ["cite", "QKI splicing", "--index-dir", str(idx_dir),
         "--rerank", "--fake-reranker", "--fake-embedder",
         "--candidate-k", "10", "--threshold", "0.0001"],
    )
    assert res.exit_code == 0, res.output


def test_cite_rerank_graceful_fallback_when_model_unloadable(tmp_path, monkeypatch):
    """If the real reranker can't load, cite must fall back to the bi-encoder,
    print a one-line notice, and NOT crash."""
    import scholia.cli as cli_mod

    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _index(runner, idx_dir)

    # Force the real reranker constructor to be used (not fake), and make its
    # load fail as if the model could not be downloaded/loaded.
    from scholia.rerank import CrossEncoderReranker

    def _boom_load(self):
        raise RuntimeError("simulated: model not available offline")

    monkeypatch.setattr(CrossEncoderReranker, "_load_backend", _boom_load)

    res = runner.invoke(
        cli,
        ["cite", "QKI splicing", "--index-dir", str(idx_dir),
         "--rerank", "--fake-embedder"],  # real reranker, fake embedder
    )
    assert res.exit_code == 0, res.output
    assert "Traceback" not in res.output
    # A one-line fallback notice must be printed.
    assert "fall" in res.output.lower() or "bi-encoder" in res.output.lower()


def test_default_reranker_threshold_for_known_models():
    """Per-reranker-model threshold map (analogous to the embedder-aware map).

    Values derived empirically on the real 361-paper library (see reranker
    report): MiniLM logit margin 6.98 straddling 0.0; bge prob margin 0.40
    straddling 0.20."""
    assert default_reranker_threshold_for("cross-encoder/ms-marco-MiniLM-L-6-v2") == 0.0
    assert default_reranker_threshold_for("BAAI/bge-reranker-v2-m3") == 0.20
    # Unknown reranker falls back to a finite default.
    t_unknown = default_reranker_threshold_for("some-unknown-reranker")
    assert isinstance(t_unknown, float)
