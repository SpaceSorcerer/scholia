"""Scholia command-line interface: `scholia index` and `scholia cite`."""

from __future__ import annotations

from pathlib import Path

import click

from scholia.corpus import load_corpus
from scholia.embedders import FakeEmbedder, NomicEmbedder
from scholia.grounding import claim_check, format_citation_suggestions
from scholia.index import ScholiaIndex, build_index
from scholia.retrieval import retrieve

DEFAULT_CORPUS = Path(
    r"C:\Users\ambur\.claude\projects\E--\memory\literature"
)
DEFAULT_INDEX_DIR = Path(r"E:\Claude\scholia\.scholia_index")
DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_THRESHOLD = 0.45


def _make_embedder(fake: bool, model_name: str):
    return FakeEmbedder() if fake else NomicEmbedder(model_name=model_name)


@click.group()
def cli() -> None:
    """Scholia Brain — local citation grounding over your Zotero library."""


@cli.command()
@click.option("--corpus", "corpus_dir", type=click.Path(path_type=Path),
              default=DEFAULT_CORPUS, show_default=True,
              help="Directory of Zotero mirror markdown notes (read-only).")
@click.option("--index-dir", type=click.Path(path_type=Path),
              default=DEFAULT_INDEX_DIR, show_default=True,
              help="Where to write the FAISS index + metadata.")
@click.option("--model", "model_name", default=DEFAULT_MODEL, show_default=True)
@click.option("--fake-embedder", is_flag=True,
              help="Use the deterministic test embedder (no model download).")
def index(corpus_dir: Path, index_dir: Path, model_name: str,
          fake_embedder: bool) -> None:
    """Build/refresh the FAISS index from the corpus."""
    papers = load_corpus(corpus_dir)
    embedder = _make_embedder(fake_embedder, model_name)
    build_index(papers, embedder, index_dir)
    click.echo(f"Indexed {len(papers)} papers -> {index_dir}")


@cli.command()
@click.argument("passage")
@click.option("--index-dir", type=click.Path(path_type=Path),
              default=DEFAULT_INDEX_DIR, show_default=True)
@click.option("--k", default=5, show_default=True, help="Number of papers to return.")
@click.option("--threshold", default=DEFAULT_THRESHOLD, show_default=True,
              help="Claim-check cosine threshold.")
@click.option("--model", "model_name", default=DEFAULT_MODEL, show_default=True)
@click.option("--fake-embedder", is_flag=True)
def cite(passage: str, index_dir: Path, k: int, threshold: float,
         model_name: str, fake_embedder: bool) -> None:
    """Print ranked supporting papers for PASSAGE, plus a claim-check line."""
    embedder = _make_embedder(fake_embedder, model_name)
    scholia_index = ScholiaIndex.load(index_dir)
    hits = retrieve(passage, embedder, scholia_index, k=k)
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
