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
