"""Scholia command-line interface: `scholia index` and `scholia cite`."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from scholia.corpus import load_corpus
from scholia.embedders import FakeEmbedder, NomicEmbedder
from scholia.grounding import claim_check, format_citation_suggestions
from scholia.index import ScholiaIndex, build_index
from scholia.retrieval import retrieve

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_THRESHOLD = 0.45

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

    papers = load_corpus(resolved_corpus)
    embedder = _make_embedder(fake_embedder, model_name)
    build_index(papers, embedder, resolved_index_dir)
    click.echo(f"Indexed {len(papers)} papers -> {resolved_index_dir}")


@cli.command()
@click.argument("passage")
@click.option("--index-dir", type=click.Path(path_type=Path),
              default=None,
              help="FAISS index directory. Overrides SCHOLIA_INDEX_DIR env var.")
@click.option("--k", default=5, show_default=True, help="Number of papers to return.")
@click.option("--threshold", default=DEFAULT_THRESHOLD, show_default=True,
              help="Claim-check cosine threshold.")
@click.option("--model", "model_name", default=DEFAULT_MODEL, show_default=True)
@click.option("--fake-embedder", is_flag=True)
@click.pass_context
def cite(ctx: click.Context, passage: str, index_dir: Path | None, k: int,
         threshold: float, model_name: str, fake_embedder: bool) -> None:
    """Print ranked supporting papers for PASSAGE, plus a claim-check line."""
    resolved_index_dir = index_dir or _default_index_dir()
    embedder = _make_embedder(fake_embedder, model_name)

    try:
        scholia_index = ScholiaIndex.load(resolved_index_dir)
    except FileNotFoundError as exc:
        click.echo(str(exc), err=True)
        ctx.exit(1)
        return

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
    verdict = claim_check(hits, threshold=threshold)
    if verdict.supported:
        click.echo(
            f"\nCLAIM-CHECK: SUPPORTED (top={verdict.top_score:.3f} >= {threshold})"
        )
    else:
        click.echo(
            f"\nCLAIM-CHECK: UNSUPPORTED by your library "
            f"(top={verdict.top_score:.3f} < {threshold})"
        )
