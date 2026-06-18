"""Scholia command-line interface: `scholia index` and `scholia cite`."""

from __future__ import annotations

import os
from pathlib import Path

import click

from scholia.corpus import load_corpus_reporting
from scholia.embedders import FakeEmbedder, NomicEmbedder
from scholia.grounding import claim_check, format_citation_suggestions
from scholia.index import ScholiaIndex, build_index
from scholia.retrieval import retrieve

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_THRESHOLD = 0.45

# Embedder-aware default claim-check thresholds (used when the user does not
# pass --threshold). MiniLM / FakeEmbedder show textbook separation around 0.45
# (off-domain <=0.38, on-domain >=0.54).
#
# nomic-embed-v1.5 has an inflated similarity floor even WITH the
# search_document:/search_query: prefixes (Item 3). Calibrated against the REAL
# 210-doc library (.scholia_recalibrate.py): genuine off-domain queries top out
# at 0.71 (Mediterranean-diet hooking CVD papers) and gibberish at ~0.61, while
# on-domain hits sit >=0.74 (rMATS) up to 0.92. Empty/whitespace queries score
# ~0.74 because a blank vector lands near the corpus centroid -- those are
# short-circuited to UNSUPPORTED in retrieve(), so they are out of the
# threshold's job. 0.73 then sits cleanly above all real negatives (<=0.71) and
# below all positives (>=0.74). See README "Models".
_MINILM_DEFAULT_THRESHOLD = 0.45
_NOMIC_DEFAULT_THRESHOLD = 0.73


def default_threshold_for(model_name: str) -> float:
    """Pick an embedder-appropriate default claim-check threshold by model name.

    nomic models -> 0.73 (inflated floor). MiniLM / FakeEmbedder / anything
    unknown -> 0.45.
    """
    name = (model_name or "").lower()
    if "nomic" in name:
        return _NOMIC_DEFAULT_THRESHOLD
    return _MINILM_DEFAULT_THRESHOLD


# Portable default index directory (~/.scholia/index).
_DEFAULT_INDEX_DIR = Path.home() / ".scholia" / "index"


def _default_corpus() -> Path | None:
    env = os.environ.get("SCHOLIA_CORPUS")
    return Path(env) if env else None


def _default_index_dir() -> Path:
    env = os.environ.get("SCHOLIA_INDEX_DIR")
    return Path(env) if env else _DEFAULT_INDEX_DIR


def _make_embedder(fake: bool, model_name: str):
    return FakeEmbedder() if fake else NomicEmbedder(model_name=model_name)


@click.group()
def cli() -> None:
    """Scholia Brain — local citation grounding over your Zotero library."""


@cli.command()
@click.option("--corpus", "corpus_dir", type=click.Path(path_type=Path),
              default=None,
              help="Directory of Zotero mirror markdown notes (read-only). "
                   "Overrides SCHOLIA_CORPUS env var.")
@click.option("--index-dir", type=click.Path(path_type=Path),
              default=None,
              help="Where to write the FAISS index + metadata. "
                   "Overrides SCHOLIA_INDEX_DIR env var.")
@click.option("--model", "model_name", default=DEFAULT_MODEL, show_default=True)
@click.option("--fake-embedder", is_flag=True,
              help="Use the deterministic test embedder (no model download).")
@click.pass_context
def index(ctx: click.Context, corpus_dir: Path | None, index_dir: Path | None,
          model_name: str, fake_embedder: bool) -> None:
    """Build/refresh the FAISS index from the corpus."""
    resolved_corpus = corpus_dir or _default_corpus()
    if resolved_corpus is None:
        click.echo(
            "Error: no corpus specified. Pass --corpus <path> or set the "
            "SCHOLIA_CORPUS environment variable.",
            err=True,
        )
        ctx.exit(1)
        return

    resolved_index_dir = index_dir or _default_index_dir()

    papers, skipped = load_corpus_reporting(resolved_corpus)
    if skipped:
        click.echo(
            f"Warning: skipped {skipped} malformed notes "
            f"(missing/invalid YAML frontmatter or no title/abstract).",
            err=True,
        )
    embedder = _make_embedder(fake_embedder, model_name)
    build_index(papers, embedder, resolved_index_dir)
    click.echo(f"Indexed {len(papers)} papers -> {resolved_index_dir}")


@cli.command()
@click.argument("passage")
@click.option("--index-dir", type=click.Path(path_type=Path),
              default=None,
              help="FAISS index directory. Overrides SCHOLIA_INDEX_DIR env var.")
@click.option("--k", default=5, show_default=True, help="Number of papers to return.")
@click.option("--threshold", default=None, type=float,
              help="Claim-check cosine threshold. Default is embedder-aware "
                   "(0.45 for MiniLM/Fake, 0.73 for nomic).")
@click.option("--model", "model_name", default=None,
              help="Embedder model. Default: adopt the index's stored embedder.")
@click.option("--fake-embedder", is_flag=True)
@click.pass_context
def cite(ctx: click.Context, passage: str, index_dir: Path | None, k: int,
         threshold: float | None, model_name: str | None,
         fake_embedder: bool) -> None:
    """Print ranked supporting papers for PASSAGE, plus a claim-check line."""
    resolved_index_dir = index_dir or _default_index_dir()

    try:
        scholia_index = ScholiaIndex.load(resolved_index_dir)
    except FileNotFoundError as exc:
        click.echo(str(exc), err=True)
        ctx.exit(1)
        return

    # Adopt the index's stored embedder unless the user overrode --model. This
    # spares the user from repeating --model and prevents dim mismatch by
    # construction (the load-time catch below remains as a backstop).
    resolved_model = model_name or scholia_index.embedder_model or DEFAULT_MODEL

    # Embedder-aware default threshold when --threshold was not passed.
    resolved_threshold = (
        threshold if threshold is not None
        else default_threshold_for(resolved_model)
    )

    embedder = _make_embedder(fake_embedder, resolved_model)

    try:
        hits = retrieve(passage, embedder, scholia_index, k=k)
    except (AssertionError, ValueError) as exc:
        # faiss raises AssertionError on dimension mismatch (query dim != index dim).
        click.echo(
            "Embedder/index dimension mismatch — rebuild the index with the "
            "same embedder (`scholia index`).",
            err=True,
        )
        ctx.exit(1)
        return

    click.echo(format_citation_suggestions(passage, hits))
    verdict = claim_check(hits, threshold=resolved_threshold)
    if verdict.supported:
        click.echo(
            f"\nCLAIM-CHECK: SUPPORTED "
            f"(top={verdict.top_score:.3f} >= {resolved_threshold})"
        )
    else:
        click.echo(
            f"\nCLAIM-CHECK: UNSUPPORTED by your library "
            f"(top={verdict.top_score:.3f} < {resolved_threshold})"
        )


if __name__ == "__main__":  # pragma: no cover - allows `python -m scholia.cli`
    cli()
