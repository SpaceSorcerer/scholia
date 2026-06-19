"""Scholia command-line interface: `scholia index` and `scholia cite`."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click

from scholia.corpus import load_corpus, load_corpus_reporting
from scholia.discovery import (
    DiscoveryUnavailable,
    FakeDiscoverySource,
    PubMedSource,
    SemanticScholarSource,
    build_query,
    discover,
)
from scholia.embedders import FakeEmbedder, NomicEmbedder
from scholia.grounding import claim_check, format_citation_suggestions
from scholia.index import ScholiaIndex, build_index
from scholia.llm import CloudClaudeLLM, FakeLLM, LLMUnavailable, LocalLLM
from scholia.models import Paper
from scholia.rerank import CrossEncoderReranker, FakeReranker
from scholia.retrieval import retrieve, retrieve_reranked
from scholia.writing_partner import format_gap_report, suggest_gaps

# Path to the existing triple-validating ingest tool (do not modify it). The
# --add path shells out to this so the user's real library is only ever mutated
# by the vetted ingester, never by Scholia directly.
_ZOTERO_INGEST = Path(r"E:\Claude\zotero-tools\zotero_ingest.py")

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_THRESHOLD = 0.45

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CANDIDATE_K = 30

# Writing-partner (LLM) defaults. PRIVACY: local is the default backend; the
# cloud path (sending the user's prose to Anthropic) is opt-in/off and gated
# behind --allow-cloud + a printed warning (institutional sign-off required).
DEFAULT_LOCAL_LLM_URL = "http://localhost:1234/v1"  # LM Studio default
DEFAULT_CLOUD_MODEL = "claude-opus-4-8"
DEFAULT_SUGGEST_K = 8

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


def _load_library(
    corpus_dir: Path | None, index_dir: Path | None
) -> list[Paper]:
    """Load the user's library for dedup: prefer --corpus (mirror notes), else
    the prebuilt index's metadata. Returns ``[]`` when neither is available
    (discover then surfaces everything as 'new')."""
    if corpus_dir is not None:
        try:
            return load_corpus(corpus_dir)
        except (OSError, ValueError):
            return []
    if index_dir is not None:
        try:
            return list(ScholiaIndex.load(index_dir)._papers)
        except (FileNotFoundError, OSError, ValueError):
            return []
    return []


def _format_candidates(query: str, candidates: list) -> str:
    """Render discovered NEW candidates as a readable, clearly-framed block."""
    lines = [f"Search query (only this left your machine): {query}", ""]
    if not candidates:
        lines.append("No NEW candidate papers found (everything relevant is "
                     "already in your library, or the search returned nothing).")
        return "\n".join(lines)
    lines.append(
        f"{len(candidates)} candidate paper(s) NOT in your library "
        f"(suggestions only — validate before adding):"
    )
    for rank, c in enumerate(candidates, 1):
        first_author = (c.authors[0].split(",")[0].strip()
                        if c.authors else "Unknown")
        lines.append(
            f"  {rank}. [{c.source} {c.score:.3f}] {first_author} "
            f"({c.year or 'n.d.'}) — {c.title}"
        )
        if c.doi:
            lines.append(f"     doi: https://doi.org/{c.doi}")
        else:
            lines.append("     doi: (none reported)")
        if c.abstract_snippet:
            lines.append(f"     {c.abstract_snippet}")
    return "\n".join(lines)


@cli.command()
@click.argument("passage")
@click.option("--limit", default=8, show_default=True,
              help="Max NEW candidate papers to return.")
@click.option("--corpus", "corpus_dir", type=click.Path(path_type=Path),
              default=None,
              help="Zotero mirror dir to dedup against (read-only). Overrides "
                   "SCHOLIA_CORPUS. Preferred over --index-dir for dedup.")
@click.option("--index-dir", type=click.Path(path_type=Path), default=None,
              help="Prebuilt FAISS index dir to dedup against. Overrides "
                   "SCHOLIA_INDEX_DIR.")
@click.option("--fake-source", is_flag=True,
              help="Use the deterministic offline source (tests/offline). No "
                   "network.")
@click.option("--add", "add_doi", default=None,
              help="Validate + add this DOI to Zotero via the existing "
                   "zotero_ingest.py (triple-validates, then adds). Re-index "
                   "afterwards.")
@click.pass_context
def discover_cmd(ctx: click.Context, passage: str, limit: int,
                 corpus_dir: Path | None, index_dir: Path | None,
                 fake_source: bool, add_doi: str | None) -> None:
    """Find relevant papers NOT yet in your library for PASSAGE.

    Suggestions only — Scholia never writes prose or auto-adds. Only a short
    keyword query (not your draft) is sent to the search APIs; no cloud LLM is
    used. Use --add <DOI> to validate + add a pick via zotero_ingest.py.
    """
    resolved_corpus = corpus_dir or _default_corpus()
    resolved_index_dir = index_dir or _default_index_dir()

    # If --add is given, skip the search entirely and route to the ingester.
    if add_doi:
        _run_add(ctx, add_doi)
        return

    library = _load_library(resolved_corpus, resolved_index_dir)

    if fake_source:
        sources = [FakeDiscoverySource(source_name="semanticscholar"),
                   FakeDiscoverySource(source_name="pubmed")]
    else:
        sources = [SemanticScholarSource(), PubMedSource()]

    try:
        candidates = discover(passage, sources=sources, library=library,
                              limit=limit)
    except DiscoveryUnavailable as exc:
        click.echo(
            f"Discovery sources unavailable (offline or rate-limited): {exc}",
            err=True,
        )
        ctx.exit(1)
        return

    query = build_query(passage)
    click.echo(_format_candidates(query, candidates))
    click.echo(
        "\nTip: `scholia discover \"<passage>\" --add <DOI>` validates + adds a "
        "pick to Zotero, then run `scholia index` to re-index."
    )


def _run_add(ctx: click.Context, doi: str) -> None:
    """Shell out to the existing zotero_ingest.py to validate + add a DOI.

    Scholia never mutates Zotero itself: the vetted, triple-validating ingester
    is the only writer. On success we remind the user to re-index; on failure we
    surface a clean message (no traceback) and exit non-zero.
    """
    cmd = [sys.executable, str(_ZOTERO_INGEST), "--doi", doi]
    click.echo(f"Validating + adding {doi} via zotero_ingest.py …")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        click.echo(f"Could not launch zotero_ingest.py: {exc}", err=True)
        ctx.exit(1)
        return
    if getattr(proc, "stdout", ""):
        click.echo(proc.stdout)
    if proc.returncode != 0:
        if getattr(proc, "stderr", ""):
            click.echo(proc.stderr, err=True)
        click.echo(
            f"Add failed (zotero_ingest.py exited {proc.returncode}). Nothing "
            f"was added.",
            err=True,
        )
        ctx.exit(1)
        return
    click.echo(
        f"Added {doi}. Re-index to pick it up: `scholia index`."
    )


# Register under the user-facing name `discover` (the function is named
# discover_cmd to avoid shadowing the imported discover() orchestrator).
cli.add_command(discover_cmd, name="discover")


@cli.command()
@click.option("--index-dir", type=click.Path(path_type=Path),
              default=None,
              help="FAISS index directory. Overrides SCHOLIA_INDEX_DIR env var.")
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Interface to bind. Localhost only — never expose to a network.")
@click.option("--port", default=8765, show_default=True, type=int,
              help="TCP port to listen on.")
@click.option("--no-rerank", "no_rerank", is_flag=True,
              help="Disable cross-encoder re-ranking (bi-encoder cosine only).")
@click.option("--fake-embedder", is_flag=True,
              help="Use the deterministic test embedder (no model download). "
                   "Implies --fake-reranker.")
@click.option("--fake-source", is_flag=True,
              help="Use the deterministic offline discovery source (no network).")
@click.pass_context
def serve(ctx: click.Context, index_dir: Path | None, host: str, port: int,
          no_rerank: bool, fake_embedder: bool, fake_source: bool) -> None:
    """Start a localhost JSON API bridge (cite/discover) for UI clients.

    Loads the index + models once at startup so every request is fast. Binds
    127.0.0.1 only — nothing leaves the machine except discovery's keyword
    queries to scholarly APIs (unchanged from the CLI). No prose is generated.

    Endpoints:
      GET  /health   → {"status":"ok","papers":N,"embedder":...}
      POST /cite     → {"passage":str,"k"?:int,"threshold"?:float,"rerank"?:bool}
      POST /discover → {"passage":str,"limit"?:int}
    """
    from scholia.server import load_state, serve as _serve

    if host != "127.0.0.1":
        click.echo(
            "Warning: binding to a non-localhost address exposes your library "
            "to the local network.",
            err=True,
        )

    resolved_index_dir = index_dir or _default_index_dir()
    state = load_state(
        resolved_index_dir,
        no_rerank=no_rerank,
        fake_embedder=fake_embedder,
        fake_source=fake_source,
    )

    click.echo(f"Scholia serving on http://{host}:{port}")
    click.echo(
        f"  {len(state.index._papers)} papers | embedder: "
        f"{state.index.embedder_model or 'unknown'}"
    )
    httpd = _serve(host, port, state)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nShutting down.")
        httpd.shutdown()


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Bridge host. Must be 127.0.0.1 (localhost only).")
@click.option("--port", default=8765, show_default=True, type=int,
              help="Bridge port.")
@click.option("--start-server", "start_server", is_flag=True,
              help="Auto-launch `scholia serve` if the bridge is not running.")
def overlay(host: str, port: int, start_server: bool) -> None:
    """Launch the always-on-top desktop overlay (requires PySide6).

    Install the optional extra first:

        pip install "scholia[overlay]"

    Then start the bridge in one terminal and the overlay in another:

        scholia serve
        scholia overlay

    Or start both at once:

        scholia overlay --start-server

    Workflow: type or paste a passage, then click Ground or Discover.
    The "Ground clipboard" button grabs whatever you last copied (e.g. from
    Word Online) and grounds it immediately.  Ctrl+Enter triggers Ground.
    """
    try:
        from scholia.overlay import run_overlay
    except ImportError:
        click.echo(
            "PySide6 is not installed.\n"
            "Run:  pip install \"scholia[overlay]\"\n"
            "Then: scholia overlay",
            err=True,
        )
        raise SystemExit(1)
    run_overlay(host=host, port=port, start_server=start_server)


@cli.command()
@click.argument("passage")
@click.option("--index-dir", type=click.Path(path_type=Path), default=None,
              help="FAISS index directory. Overrides SCHOLIA_INDEX_DIR env var.")
@click.option("--backend", type=click.Choice(["local", "cloud", "fake"]),
              default="local", show_default=True,
              help="Which language model runs the gap analysis. 'local' (DEFAULT) "
                   "and 'fake' stay fully on-device; 'cloud' sends your prose to "
                   "Anthropic and REQUIRES --allow-cloud.")
@click.option("--local-url", default=DEFAULT_LOCAL_LLM_URL, show_default=True,
              help="Base URL of a local OpenAI-compatible server (LM Studio / "
                   "Ollama).")
@click.option("--model", "model_name", default=None,
              help="LLM name. local -> the served model id; cloud -> a Claude "
                   "model (default claude-opus-4-8).")
@click.option("--allow-cloud", is_flag=True,
              help="REQUIRED to use --backend cloud. Acknowledges that your "
                   "passage text will be sent to Anthropic (institutional "
                   "sign-off required).")
@click.option("--k", default=DEFAULT_SUGGEST_K, show_default=True,
              help="Number of library papers to retrieve as grounding context.")
@click.option("--embedder-model", "embedder_model", default=None,
              help="Embedder model. Default: adopt the index's stored embedder.")
@click.option("--fake-embedder", is_flag=True,
              help="Use the deterministic test embedder (no model download).")
@click.pass_context
def suggest(ctx: click.Context, passage: str, index_dir: Path | None,
            backend: str, local_url: str, model_name: str | None,
            allow_cloud: bool, k: int, embedder_model: str | None,
            fake_embedder: bool) -> None:
    """Suggest GAPS in PASSAGE — missing topics, where citations are needed, and
    next angles — grounded in your own library.

    Scholia SUGGESTS; it never writes manuscript prose or rewrites your
    sentences. The local + fake backends are fully on-device; the cloud backend
    sends your prose to Anthropic and is opt-in/off (requires --allow-cloud).
    """
    # PRIVACY GATE: cloud is opt-in/off. Refuse --backend cloud without the
    # explicit --allow-cloud acknowledgement (institutional sign-off required).
    if backend == "cloud" and not allow_cloud:
        click.echo(
            "Refusing --backend cloud without --allow-cloud.\n"
            "The cloud path sends your unpublished passage text to Anthropic, "
            "which requires your institution's sign-off. Re-run with "
            "--allow-cloud to acknowledge, or use the on-device default "
            "(--backend local).",
            err=True,
        )
        ctx.exit(2)
        return

    resolved_index_dir = index_dir or _default_index_dir()
    try:
        scholia_index = ScholiaIndex.load(resolved_index_dir)
    except FileNotFoundError as exc:
        click.echo(str(exc), err=True)
        ctx.exit(1)
        return

    # Adopt the index's stored embedder unless overridden (prevents dim mismatch).
    resolved_embedder_model = (
        embedder_model or scholia_index.embedder_model or DEFAULT_MODEL
    )
    embedder = _make_embedder(fake_embedder, resolved_embedder_model)

    # Build the chosen backend. Cloud prints a clear one-line privacy warning.
    if backend == "fake":
        model = FakeLLM()
    elif backend == "cloud":
        click.echo(
            "WARNING: --backend cloud — your passage text will be SENT TO "
            "ANTHROPIC (off your machine) for this analysis.",
            err=True,
        )
        model = CloudClaudeLLM(model=model_name or DEFAULT_CLOUD_MODEL)
    else:  # local (default) — fully on-device
        model = LocalLLM(base_url=local_url, model=model_name or "local-model")

    try:
        report = suggest_gaps(passage, scholia_index, embedder, model, k=k)
    except (AssertionError, ValueError):
        click.echo(
            "Embedder/index dimension mismatch — rebuild the index with the "
            "same embedder (`scholia index`).",
            err=True,
        )
        ctx.exit(1)
        return
    except LLMUnavailable as exc:
        click.echo(f"Language model unavailable: {exc}", err=True)
        ctx.exit(1)
        return

    click.echo(format_gap_report(report))


if __name__ == "__main__":  # pragma: no cover - allows `python -m scholia.cli`
    cli()
