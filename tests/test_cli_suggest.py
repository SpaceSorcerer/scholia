"""CLI tests for `scholia suggest` (writing-partner gap suggestions).

Offline + model-free: --backend fake + --fake-embedder. The cloud gate is tested
by asserting that --backend cloud WITHOUT --allow-cloud refuses (non-zero exit,
clear institutional-sign-off message) and never reaches a network call.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from scholia.cli import cli

FIXTURES = Path(__file__).parent / "fixtures"


def _build_index(runner, idx_dir):
    res = runner.invoke(
        cli,
        ["index", "--corpus", str(FIXTURES / "corpus"),
         "--index-dir", str(idx_dir), "--fake-embedder"],
    )
    assert res.exit_code == 0, res.output


def test_suggest_fake_backend_prints_pointer_suggestions(tmp_path):
    """`suggest --backend fake` prints gap pointers + supporting library papers,
    and NEVER prints rewritten prose."""
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _build_index(runner, idx_dir)

    res = runner.invoke(
        cli,
        ["suggest", "QKI regulates alternative splicing in cardiomyocytes",
         "--index-dir", str(idx_dir), "--backend", "fake", "--fake-embedder"],
    )
    assert res.exit_code == 0, res.output
    out = res.output
    # Pointer sections + grounding papers are shown.
    assert "Missing topics" in out
    assert "Supporting papers from your library:" in out
    # The assist-not-ghostwrite contract is stated to the user.
    assert "never writes manuscript prose" in out.lower()


def test_suggest_default_backend_is_local(tmp_path):
    """The default backend is local (on-device). Help text documents local as the
    default."""
    runner = CliRunner()
    res = runner.invoke(cli, ["suggest", "--help"])
    assert res.exit_code == 0, res.output
    assert "local" in res.output
    # Default shown for --backend is 'local'.
    assert "default: local" in res.output.lower() or "[default: local]" in res.output


def test_suggest_cloud_without_allow_cloud_refuses(tmp_path):
    """--backend cloud WITHOUT --allow-cloud refuses: non-zero exit, clear message
    about institutional sign-off, and no traceback. Must not call the network."""
    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _build_index(runner, idx_dir)

    res = runner.invoke(
        cli,
        ["suggest", "QKI splicing", "--index-dir", str(idx_dir),
         "--backend", "cloud", "--fake-embedder"],
    )
    assert res.exit_code != 0
    assert "Traceback" not in (res.output or "")
    low = (res.output or "").lower()
    assert "allow-cloud" in low
    assert "sign-off" in low or "institution" in low


def test_suggest_cloud_with_allow_cloud_warns_about_sending_to_anthropic(
    tmp_path, monkeypatch
):
    """--backend cloud --allow-cloud prints a clear one-line warning that the
    passage text leaves the machine (we stub the cloud model so no real call is
    made)."""
    import scholia.cli as cli_mod

    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _build_index(runner, idx_dir)

    class _StubCloud:
        def __init__(self, *a, **k):
            pass

        def complete(self, system, user):
            return (
                "MISSING TOPICS:\n- stubbed pointer\n"
                "NEEDS CITATION:\n- stubbed citation pointer\n"
            )

    monkeypatch.setattr(cli_mod, "CloudClaudeLLM", _StubCloud)

    res = runner.invoke(
        cli,
        ["suggest", "QKI splicing", "--index-dir", str(idx_dir),
         "--backend", "cloud", "--allow-cloud", "--fake-embedder"],
    )
    assert res.exit_code == 0, res.output
    low = res.output.lower()
    assert "anthropic" in low
    assert "sent to" in low or "leave" in low or "off your machine" in low


def test_suggest_missing_index_friendly_message(tmp_path):
    """suggest against a nonexistent index prints a friendly message, non-zero exit."""
    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["suggest", "some passage", "--index-dir", str(tmp_path / "nope"),
         "--backend", "fake", "--fake-embedder"],
    )
    assert res.exit_code != 0
    assert "Traceback" not in (res.output or "")
    combined = (res.output or "") + str(res.exception or "")
    assert "scholia index" in combined or "No index at" in combined


def test_suggest_never_prints_rewritten_prose(tmp_path):
    """Even when the model returns a long drafted paragraph, the CLI never echoes
    it (the parser drops prose lines)."""
    import scholia.cli as cli_mod

    runner = CliRunner()
    idx_dir = tmp_path / "idx"
    _build_index(runner, idx_dir)

    drafted = (
        "This is a fully drafted manuscript paragraph the model should never have "
        "produced. It runs across several sentences of polished prose. It even "
        "ends with a confident concluding flourish."
    )

    class _ProseLLM:
        def complete(self, system, user):
            return f"MISSING TOPICS:\n- a short pointer\n- {drafted}\n"

    monkeypatch_attr = cli_mod.FakeLLM
    cli_mod.FakeLLM = lambda *a, **k: _ProseLLM()
    try:
        res = runner.invoke(
            cli,
            ["suggest", "QKI splicing", "--index-dir", str(idx_dir),
             "--backend", "fake", "--fake-embedder"],
        )
    finally:
        cli_mod.FakeLLM = monkeypatch_attr

    assert res.exit_code == 0, res.output
    assert "a short pointer" in res.output
    assert drafted not in res.output
