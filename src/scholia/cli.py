"""Scholia command-line interface: `scholia index` and `scholia cite`."""

from __future__ import annotations

import os
from pathlib import Path

import click

from scholia.corpus import load_corpus_reporting
from scholia.embedders import FakeEmbedder, NomicEmbedder
from scholia.grounding import claim_check, format_citation_suggestions
from scholia.index import ScholiaIndex, build_index
from scholia.rerank import CrossEncoderReranker, FakeReranker
from scholia.retrieval import retrieve, retrieve_reranked

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_THRESHOLD = 0.45

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CANDIDATE_K = 30

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


# Cross-encoder re-rank thresholds. Cross-encoder relevance is a DIFFERENT scale
# than cosine, so the SUPPORTED/UNSUPPORTED cutoff is re-derived per reranker.
#
# Derived empirically against the REAL 361-paper nomic index
# (.scholia_reranker_calib.py, 2026-06-18; candidate_k=30):
#
#   ms-marco-MiniLM-L-6-v2 (relevance LOGIT, centred at 0):
#     on-domain top-1   +2.11 .. +8.47   (min +2.11)
#     off-domain/gibber  -11.04 .. -4.87  (max -4.87)
#     -> margin 6.98 logits; 0.0 sits cleanly in the gap. THRESHOLD = 0.0.
#     (Compare the bi-encoder nomic-cosine margin on the same library: only
#      0.777 - 0.689 = 0.089. The cross-encoder widens it ~78x.)
#
#   bge-reranker-v2-m3 (relevance prob in [0,1], sigmoid output):
#     on-domain top-1   0.439 .. 0.999   (min 0.439)
#     off-domain/gibber  0.001 .. 0.038  (max 0.038)
#     -> margin 0.40; 0.20 sits cleanly in the gap. THRESHOLD = 0.20.
#     (bge is far more accurate but ~42 s/query on CPU vs ~2.2 s for MiniLM, so
#      MiniLM is the default; bge is offered via --rerank-model.)
#
# See the reranker report (.git/sdd/reranker-report.md) + README "Re-ranking".
_MINILM_RERANKER_THRESHOLD = 0.0
_BGE_RERANKER_THRESHOLD = 0.20
_DEFAULT_RERANKER_THRESHOLD = 0.0


def default_reranker_threshold_for(reranker_model: str) -> float:
    """Pick a reranker-appropriate default claim-check threshold by model name.

    ms-marco MiniLM cross-encoders emit a relevance LOGIT centred at 0 (on-domain
    positive, off-domain negative) -> 0.0 cutoff. bge-reranker-v2-m3 emits a
    relevance PROBABILITY in [0,1] (on-domain >=0.44, off-domain <=0.04) -> 0.20
    cutoff. Unknown rerankers fall back to 0.0. The FakeReranker (token-overlap,
    scores in [0,1]) is test-only and not in this map; the CLI uses a small
    positive cutoff for it inline.
    """
    name = (reranker_model or "").lower()
    if "bge-reranker" in name:
        return _BGE_RERANKER_THRESHOLD
    if "ms-marco" in name or "minilm" in name:
        return _MINILM_RERANKER_THRESHOLD
    return _DEFAULT_RERANKER_THRESHOLD


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


def _make_reranker(fake: bool, model_name: str):
    return FakeReranker() if fake else CrossEncoderReranker(model_name=model_name)


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
              help="Claim-check threshold (overrides the default). SCALE DEPENDS "
                   "ON RANKING: with --rerank (default) it is the cross-encoder "
                   "relevance score (default ~0.0 for ms-marco-MiniLM); with "
                   "--no-rerank it is cosine (0.45 MiniLM/Fake, 0.73 nomic). The "
                   "'Ranking signal' line shows which scale is live.")
@click.option("--model", "model_name", default=None,
              help="Embedder model. Default: adopt the index's stored embedder.")
@click.option("--fake-embedder", is_flag=True)
@click.option("--rerank/--no-rerank", "rerank_flag", default=True,
              help="Cross-encoder re-rank the FAISS candidates for a cleaner "
                   "relevance signal + wider claim-check margin. ON by default; "
                   "falls back to the bi-encoder if the reranker can't load.")
@click.option("--rerank-model", "rerank_model", default=DEFAULT_RERANKER_MODEL,
              show_default=True, help="Cross-encoder reranker model.")
@click.option("--candidate-k", default=DEFAULT_CANDIDATE_K, show_default=True,
              help="FAISS candidate pool size fed to the reranker.")
@click.option("--fake-reranker", is_flag=True,
              help="Use the deterministic test reranker (no model download).")
@click.pass_context
def cite(ctx: click.Context, passage: str, index_dir: Path | None, k: int,
         threshold: float | None, model_name: str | None,
         fake_embedder: bool, rerank_flag: bool, rerank_model: str,
         candidate_k: int, fake_reranker: bool) -> None:
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

    embedder = _make_embedder(fake_embedder, resolved_model)

    # Test-mode coherence (mirrors the Fake/real embedder split): when the user
    # asks for the fake embedder but does NOT explicitly select a reranker, use
    # the deterministic FakeReranker so the run stays fully model-free and
    # offline. An explicit --fake-reranker / --rerank-model / --no-rerank always
    # wins. This keeps unit tests download-free without changing production
    # behaviour (default = real cross-encoder).
    src = ctx.get_parameter_source
    if (fake_embedder and not fake_reranker
            and src("rerank_model").name == "DEFAULT"
            and src("rerank_flag").name == "DEFAULT"):
        fake_reranker = True

    # --rerank requested: try the cross-encoder path. If the reranker model
    # can't load (offline, missing weights), fall back to the bi-encoder with a
    # one-line notice rather than crashing. --fake-reranker forces the test path.
    use_rerank = rerank_flag
    reranked = False
    hits: list = []
    if use_rerank:
        reranker = _make_reranker(fake_reranker, rerank_model)
        try:
            hits = retrieve_reranked(
                passage, embedder, scholia_index, reranker,
                candidate_k=candidate_k, top_k=k,
            )
            reranked = True
        except (AssertionError, ValueError):
            # Dimension mismatch is an embedder/index problem, not a reranker
            # one — surface the same friendly message as the bi-encoder path.
            click.echo(
                "Embedder/index dimension mismatch — rebuild the index with the "
                "same embedder (`scholia index`).",
                err=True,
            )
            ctx.exit(1)
            return
        except Exception as exc:  # noqa: BLE001 - reranker load/scoring failure
            # Reranker model could not load (e.g. offline). Degrade gracefully.
            click.echo(
                f"Notice: reranker unavailable ({type(exc).__name__}); "
                f"falling back to the bi-encoder.",
                err=True,
            )
            use_rerank = False

    if not reranked:
        try:
            hits = retrieve(passage, embedder, scholia_index, k=k)
        except (AssertionError, ValueError):
            click.echo(
                "Embedder/index dimension mismatch — rebuild the index with the "
                "same embedder (`scholia index`).",
                err=True,
            )
            ctx.exit(1)
            return

    # Threshold scale depends on which signal produced the scores. Reranked
    # scores are cross-encoder relevance (reranker-aware threshold); bi-encoder
    # scores are cosine (embedder-aware threshold). --threshold overrides either.
    if reranked:
        default_thr = (
            0.001 if fake_reranker
            else default_reranker_threshold_for(rerank_model)
        )
    else:
        default_thr = default_threshold_for(resolved_model)
    resolved_threshold = threshold if threshold is not None else default_thr

    signal = "reranked (cross-encoder)" if reranked else "bi-encoder (cosine)"
    click.echo(format_citation_suggestions(passage, hits))
    click.echo(f"\nRanking signal: {signal}")
    verdict = claim_check(hits, threshold=resolved_threshold)
    if verdict.supported:
        click.echo(
            f"CLAIM-CHECK: SUPPORTED "
            f"(top={verdict.top_score:.3f} >= {resolved_threshold})"
        )
    else:
        click.echo(
            f"CLAIM-CHECK: UNSUPPORTED by your library "
            f"(top={verdict.top_score:.3f} < {resolved_threshold})"
        )


if __name__ == "__main__":  # pragma: no cover - allows `python -m scholia.cli`
    cli()
