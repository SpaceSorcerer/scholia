"""CLI tests for the entailment support-verification stage (FakeEntailmentChecker,
no download). Verification is ON by default; --fake-embedder keeps it model-free."""

from pathlib import Path

from click.testing import CliRunner

from scholia.cli import cli, default_entailment_threshold_for

FIXTURES = Path(__file__).parent / "fixtures"


def _index(runner, idx_dir):
    return runner.invoke(
        cli,
        ["index", "--corpus", str(FIXTURES / "corpus"),
         "--index-dir", str(idx_dir), "--fake-embedder"],
    )


def test_cite_verify_supported_when_text_supports(tmp_path):
    """A passage that is the paper's own text -> similarity SUPPORTED AND the
    fake entailment checker (claim-token recall) confirms support."""
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _index(runner, idx_dir)
    passage = ("QKI regulates alternative splicing in cardiomyocytes\n\n"
               "QKI is an RNA-binding protein that controls pre-mRNA "
               "alternative splicing during cardiac differentiation.")
    res = runner.invoke(
        cli,
        ["cite", passage, "--index-dir", str(idx_dir),
         "--verify", "--fake-entailment", "--fake-embedder",
         "--rerank", "--fake-reranker", "--threshold", "0.0001"],
    )
    assert res.exit_code == 0, res.output
    assert "CLAIM-CHECK: SUPPORTED" in res.output
    assert "VERIFIED" in res.output


def test_cite_verify_flags_retrieved_but_not_supported(tmp_path):
    """Similarity ranks a paper highly (low threshold) but the claim's content
    words are absent from the matched paper -> the honest non-support flag.

    The passage's distinctive tokens (zebrafish, photoreceptor) do not appear in
    any fixture abstract, so the fake checker's claim-token recall is ~0, while a
    near-zero similarity threshold still calls it SUPPORTED-by-similarity."""
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _index(runner, idx_dir)
    res = runner.invoke(
        cli,
        ["cite", "zebrafish photoreceptor regeneration commitment",
         "--index-dir", str(idx_dir),
         "--verify", "--fake-entailment", "--fake-embedder",
         "--rerank", "--fake-reranker", "--threshold", "-1"],
    )
    assert res.exit_code == 0, res.output
    assert "by similarity" in res.output
    assert "clearly support" in res.output
    assert "verify the source" in res.output
    # Honest framing: never claims contradiction.
    assert "contradict" not in res.output.lower()


def test_cite_no_verify_omits_entailment(tmp_path):
    """--no-verify reverts to the similarity-only claim-check line."""
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _index(runner, idx_dir)
    passage = ("QKI regulates alternative splicing in cardiomyocytes\n\n"
               "QKI is an RNA-binding protein that controls pre-mRNA "
               "alternative splicing during cardiac differentiation.")
    res = runner.invoke(
        cli,
        ["cite", passage, "--index-dir", str(idx_dir),
         "--no-verify", "--fake-embedder", "--no-rerank", "--threshold", "0.5"],
    )
    assert res.exit_code == 0, res.output
    assert "CLAIM-CHECK: SUPPORTED" in res.output
    assert "VERIFIED" not in res.output
    assert "verify the source" not in res.output


def test_cite_verify_unsupported_stays_unsupported(tmp_path):
    """Similarity UNSUPPORTED -> UNSUPPORTED; no spurious entailment flag."""
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _index(runner, idx_dir)
    res = runner.invoke(
        cli,
        ["cite", "completely unrelated nonsense topic", "--index-dir", str(idx_dir),
         "--verify", "--fake-entailment", "--fake-embedder",
         "--rerank", "--fake-reranker", "--threshold", "999999"],
    )
    assert res.exit_code == 0, res.output
    assert "CLAIM-CHECK: UNSUPPORTED" in res.output
    assert "verify the source" not in res.output


def test_cite_verify_graceful_fallback_when_model_unloadable(tmp_path, monkeypatch):
    """If the real entailment model can't load, cite reports the similarity
    verdict with a one-line notice and does NOT crash."""
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _index(runner, idx_dir)

    from scholia.entailment import MiniCheckEntailmentChecker

    def _boom_load(self):
        raise RuntimeError("simulated: MiniCheck model not available offline")

    monkeypatch.setattr(MiniCheckEntailmentChecker, "_load_backend", _boom_load)

    passage = ("QKI regulates alternative splicing in cardiomyocytes\n\n"
               "QKI is an RNA-binding protein that controls pre-mRNA "
               "alternative splicing during cardiac differentiation.")
    # Real entailment checker (no --fake-entailment), fake embedder + reranker.
    res = runner.invoke(
        cli,
        ["cite", passage, "--index-dir", str(idx_dir),
         "--verify", "--verify-model", "lytang/MiniCheck-Flan-T5-Large",
         "--fake-embedder", "--rerank", "--fake-reranker", "--threshold", "0.0001"],
    )
    assert res.exit_code == 0, res.output
    assert "Traceback" not in res.output
    assert "support-verification unavailable" in res.output
    # Falls back to the plain similarity verdict.
    assert "CLAIM-CHECK: SUPPORTED" in res.output


def test_cite_verify_threshold_option_accepted(tmp_path):
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _index(runner, idx_dir)
    res = runner.invoke(
        cli,
        ["cite", "QKI splicing", "--index-dir", str(idx_dir),
         "--verify", "--fake-entailment", "--fake-embedder",
         "--rerank", "--fake-reranker",
         "--verify-threshold", "0.9", "--threshold", "0.0001"],
    )
    assert res.exit_code == 0, res.output


def test_default_entailment_threshold_for_known_models():
    """Per-model entailment threshold map (analogous to the reranker map)."""
    assert default_entailment_threshold_for("lytang/MiniCheck-Flan-T5-Large") == 0.50
    t_unknown = default_entailment_threshold_for("some-unknown-checker")
    assert isinstance(t_unknown, float)


def test_cite_verify_supported_names_supporting_paper(tmp_path):
    """When at least one paper supports, the VERIFIED line names it (author/year/title)."""
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _index(runner, idx_dir)
    passage = ("QKI regulates alternative splicing in cardiomyocytes\n\n"
               "QKI is an RNA-binding protein that controls pre-mRNA "
               "alternative splicing during cardiac differentiation.")
    res = runner.invoke(
        cli,
        ["cite", passage, "--index-dir", str(idx_dir),
         "--verify", "--fake-entailment", "--fake-embedder",
         "--rerank", "--fake-reranker", "--threshold", "0.0001"],
    )
    assert res.exit_code == 0, res.output
    assert "CLAIM-CHECK: SUPPORTED" in res.output
    assert "VERIFIED" in res.output
    # The VERIFIED line should name a paper (author name + year or title fragment).
    assert "supports the claim" in res.output
    assert "best support=" in res.output


def test_cite_verify_not_supported_shows_count(tmp_path):
    """The non-support flag shows how many papers were retrieved but none supported."""
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _index(runner, idx_dir)
    res = runner.invoke(
        cli,
        ["cite", "zebrafish photoreceptor regeneration commitment",
         "--index-dir", str(idx_dir),
         "--verify", "--fake-entailment", "--fake-embedder",
         "--rerank", "--fake-reranker", "--threshold", "-1"],
    )
    assert res.exit_code == 0, res.output
    assert "retrieved" in res.output
    assert "paper(s)" in res.output
    assert "clearly support" in res.output
