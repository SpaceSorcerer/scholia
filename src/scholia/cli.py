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
from scholia.entailment import FakeEntailmentChecker, MiniCheckEntailmentChecker
from scholia.grounding import (
    claim_check,
    format_citation_suggestions,
    verified_claim_check,
)
from scholia.index import ScholiaIndex, build_index
from scholia.llm import CloudClaudeLLM, FakeLLM, LLMUnavailable, LocalLLM
from scholia.mirror import (
    FakeZoteroFetcher,
    HttpZoteroFetcher,
    ZoteroUnavailable,
    write_corpus,
)
from scholia.models import Paper
from scholia.rerank import CrossEncoderReranker, FakeReranker
from scholia.retrieval import retrieve, retrieve_reranked
from scholia.writing_partner import format_gap_report, suggest_gaps

# Env var pointing at the user's external triple-validating ingest tool. The
# --add path shells out to THIS so the user's real library is only ever mutated
# by their own vetted ingester, never by Scholia directly. There is intentionally
# NO baked-in default path — Scholia ships no ingester. The command is resolved
# at --add time from --ingest-cmd or SCHOLIA_INGEST_CMD (see _resolve_ingest_cmd).
_INGEST_CMD_ENV = "SCHOLIA_INGEST_CMD"

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


# Entailment / support-verification defaults. This is an INDEPENDENT check on top
# of similarity/rerank: it asks whether the top paper's text actually *supports*
# the claim, not just scores similar. ON by default — warm cost is ~0.26 s/query
# on CPU (one short seq2seq forward), below the rerank step it follows; the model
# downloads once (MIT-licensed MiniCheck-Flan-T5-Large), same posture as the
# embedder.
#
# The support score is MiniCheck's supported-probability in [0,1] (grounding, not
# strict NLI — see entailment.py for why generic NLI was rejected). Calibrated on
# the REAL 361-paper library (.scholia_entailment_calib.py): genuine support lands
# ~0.96-0.98, off-topic / wrong-paper ~0.01-0.04. The 0.50 cutoff sits in that
# ~0.9-wide gap. See the entailment report + README "Verified grounding".
DEFAULT_ENTAILMENT_MODEL = "lytang/MiniCheck-Flan-T5-Large"
_MINICHECK_ENTAILMENT_THRESHOLD = 0.50
_DEFAULT_ENTAILMENT_THRESHOLD = 0.50


def default_entailment_threshold_for(entail_model: str) -> float:
    """Pick an entailment-appropriate default support threshold by model name.

    MiniCheck models emit a supported PROBABILITY in [0,1] with a wide on/off
    gap -> 0.50 cutoff. Unknown checkers fall back to 0.50. The
    FakeEntailmentChecker (claim-token recall, scores in [0,1]) is test-only and
    not in this map; the CLI uses its own default inline.
    """
    name = (entail_model or "").lower()
    if "minicheck" in name:
        return _MINICHECK_ENTAILMENT_THRESHOLD
    return _DEFAULT_ENTAILMENT_THRESHOLD


def _first_author_cli(paper) -> str:
    """Extract the first author's last name from a Paper for one-line summaries."""
    if not paper.authors:
        return "Unknown"
    return paper.authors[0].split(",")[0].strip()


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


def _make_entailment_checker(fake: bool, model_name: str, threshold: float):
    return (
        FakeEntailmentChecker(threshold=threshold)
        if fake
        else MiniCheckEntailmentChecker(model_name=model_name, threshold=threshold)
    )


@click.group()
def cli() -> None:
    """Scholia Brain — local citation grounding over your Zotero library."""


@cli.command()
@click.option("--user-id", "user_id", default=None,
              help="Your numeric Zotero user ID (from zotero.org/settings/keys, "
                   "shown as 'Your userID for use in API calls'). Use this OR "
                   "--group-id, not both.")
@click.option("--group-id", "group_id", default=None,
              help="A Zotero group library's numeric ID (mirror a group instead "
                   "of your personal library). Use this OR --user-id.")
@click.option("--api-key", "api_key", default=None,
              help="A Zotero API key with read access. Defaults to the "
                   "ZOTERO_API_KEY environment variable. Create one at "
                   "https://www.zotero.org/settings/keys (read-only is enough).")
@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=None,
              help="Corpus output directory (markdown notes). Overrides "
                   "SCHOLIA_CORPUS; defaults to ~/.scholia/corpus.")
@click.option("--fake-source", is_flag=True,
              help="Use the deterministic offline fetcher (tests/offline). No "
                   "network, no key required.")
@click.pass_context
def mirror(ctx: click.Context, user_id: str | None, group_id: str | None,
           api_key: str | None, out_dir: Path | None,
           fake_source: bool) -> None:
    """Build the corpus from YOUR Zotero library via the Zotero Web API.

    Fetches every citeable item in your personal (--user-id) or a group
    (--group-id) library — READ-ONLY — and writes one markdown note per item, in
    the exact format `scholia index` consumes. Idempotent (notes are named by
    Zotero key and overwritten). Your draft never leaves the machine; only the
    API key + library id reach api.zotero.org. The key is read from --api-key or
    the ZOTERO_API_KEY env var and is never stored or logged.
    """
    resolved_out = out_dir or _default_corpus() or (Path.home() / ".scholia" / "corpus")

    if fake_source:
        # Offline/deterministic: a tiny fixed library, no network, no key.
        fetcher = FakeZoteroFetcher(
            [
                {"key": "FAKE0001",
                 "data": {"key": "FAKE0001", "itemType": "journalArticle",
                          "title": "A fake mirrored paper",
                          "creators": [{"creatorType": "author",
                                        "firstName": "Jane", "lastName": "Doe"}],
                          "date": "2021", "DOI": "10.9999/fake",
                          "abstractNote": "A deterministic offline abstract.",
                          "tags": [{"tag": "Fake"}],
                          "dateAdded": "2021-01-01T00:00:00Z"}},
            ]
        )
    else:
        resolved_key = api_key or os.environ.get("ZOTERO_API_KEY")
        if not resolved_key:
            click.echo(
                "Error: no Zotero API key. Pass --api-key or set the "
                "ZOTERO_API_KEY environment variable.\n"
                "Create a read-only key at https://www.zotero.org/settings/keys",
                err=True,
            )
            ctx.exit(1)
            return
        if bool(user_id) == bool(group_id):
            click.echo(
                "Error: provide exactly one of --user-id or --group-id.\n"
                "Your numeric userID is shown at "
                "https://www.zotero.org/settings/keys",
                err=True,
            )
            ctx.exit(1)
            return
        fetcher = HttpZoteroFetcher(
            api_key=resolved_key, user_id=user_id, group_id=group_id
        )

    try:
        written, skipped = write_corpus(fetcher, resolved_out)
    except ZoteroUnavailable as exc:
        click.echo(
            f"Zotero Web API unavailable (offline, bad key, or rate-limited): "
            f"{exc}",
            err=True,
        )
        ctx.exit(1)
        return

    if skipped:
        click.echo(
            f"(skipped {skipped} non-citeable items: attachments/notes/etc.)"
        )
    click.echo(
        f"Wrote {written} notes -> {resolved_out}; next: "
        f"scholia index --corpus {resolved_out}"
    )


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
@click.option("--verify/--no-verify", "verify_flag", default=True,
              help="Verify that at least one of the retrieved papers actually "
                   "SUPPORTS the claim (textual entailment, aggregated over "
                   "all top-k hits). Reports SUPPORTED if ANY paper entails; "
                   "flags 'verify the source' only when NONE do. ON by default "
                   "(~0.26s/paper CPU); falls back silently if the model can't "
                   "load. NEVER claims a paper 'contradicts' — only flags "
                   "non-support.")
@click.option("--verify-model", "verify_model", default=DEFAULT_ENTAILMENT_MODEL,
              show_default=True,
              help="Grounding/entailment verifier model for support-verification.")
@click.option("--verify-threshold", "verify_threshold", default=None, type=float,
              help="Support-verification cutoff (support probability, 0-1). "
                   "Default 0.50 for the grounding/entailment verifier.")
@click.option("--fake-entailment", is_flag=True,
              help="Use the deterministic test entailment checker (no download).")
@click.pass_context
def cite(ctx: click.Context, passage: str, index_dir: Path | None, k: int,
         threshold: float | None, model_name: str | None,
         fake_embedder: bool, rerank_flag: bool, rerank_model: str,
         candidate_k: int, fake_reranker: bool, verify_flag: bool,
         verify_model: str, verify_threshold: float | None,
         fake_entailment: bool) -> None:
    """Print ranked supporting papers for PASSAGE, plus a claim-check line.

    With --verify (default ON) ALL retrieved papers are checked for textual
    SUPPORT of the claim (top-k aggregation). SUPPORTED is reported if ANY of
    the top-k papers' abstracts support the claim; the honest "retrieved but not
    clearly supported" flag fires only when NONE do. The output names which
    paper(s) provided the entailment support. Never emits "contradicts."
    """
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

    # Same coherence for the entailment checker: --fake-embedder without an
    # explicit --verify-model keeps verification fully model-free and offline.
    if (fake_embedder and not fake_entailment
            and src("verify_model").name == "DEFAULT"):
        fake_entailment = True

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

    # --verify (default): textual support-verification on the TOP hit. The
    # entailment cutoff: 0.001 for the Fake checker (token-recall, test/offline),
    # else the model-aware default unless the user overrode --verify-threshold.
    # If the real model can't load, degrade to the similarity-only verdict with a
    # one-line notice (mirrors the reranker fallback) — never crash.
    entail_threshold = (
        0.001 if fake_entailment
        else default_entailment_threshold_for(verify_model)
    )
    if verify_threshold is not None:
        entail_threshold = verify_threshold

    verified = None
    if verify_flag:
        checker = _make_entailment_checker(
            fake_entailment, verify_model, entail_threshold
        )
        try:
            verified = verified_claim_check(
                hits, checker, claim=passage,
                threshold=resolved_threshold, entail_threshold=entail_threshold,
            )
        except Exception as exc:  # noqa: BLE001 - entailment load/scoring failure
            click.echo(
                f"Notice: support-verification unavailable "
                f"({type(exc).__name__}); reporting the similarity verdict only.",
                err=True,
            )
            verified = None

    verdict = claim_check(hits, threshold=resolved_threshold)
    base_line = (
        f"(top={verdict.top_score:.3f} >= {resolved_threshold})"
        if verdict.supported
        else f"(top={verdict.top_score:.3f} < {resolved_threshold})"
    )

    if verified is not None and verified.checked:
        es = verified.entail_score
        if verified.status == "SUPPORTED":
            # Report which paper(s) provided the entailment support.
            sup_papers = verified.supporting_papers
            if sup_papers:
                first = sup_papers[0]
                fa = _first_author_cli(first)
                sup_line = (
                    f"{fa} ({first.year}) — {first.title[:60]}"
                    f"{'…' if len(first.title) > 60 else ''}"
                )
                if len(sup_papers) > 1:
                    sup_line += f" (+{len(sup_papers) - 1} more)"
            else:
                sup_line = "top paper"
            click.echo(
                f"CLAIM-CHECK: SUPPORTED {base_line} | "
                f"VERIFIED: {sup_line} supports the claim "
                f"(best support={es:.3f} >= {entail_threshold})"
            )
        elif verified.status == "RETRIEVED_NOT_SUPPORTED":
            click.echo(
                f"CLAIM-CHECK: SUPPORTED by similarity {base_line}"
            )
            click.echo(
                f"⚠ retrieved {len(hits)} paper(s) but none clearly support "
                f"this claim — verify the source "
                f"(best support={es:.3f} < {entail_threshold})."
            )
        else:  # UNSUPPORTED
            click.echo(
                f"CLAIM-CHECK: UNSUPPORTED by your library {base_line}"
            )
    else:
        # Verification off or unavailable: original similarity-only behaviour.
        if verdict.supported:
            click.echo(f"CLAIM-CHECK: SUPPORTED {base_line}")
        else:
            click.echo(f"CLAIM-CHECK: UNSUPPORTED by your library {base_line}")


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
              help="Validate + add this DOI to your library via YOUR external "
                   "triple-validating ingester (set --ingest-cmd or the "
                   "SCHOLIA_INGEST_CMD env var). Re-index afterwards.")
@click.option("--ingest-cmd", "ingest_cmd", default=None,
              help="Path to your external triple-validating ingest script/command "
                   "used by --add (invoked as `<cmd> --doi <DOI>`). Overrides the "
                   "SCHOLIA_INGEST_CMD env var. Scholia ships no ingester.")
@click.pass_context
def discover_cmd(ctx: click.Context, passage: str, limit: int,
                 corpus_dir: Path | None, index_dir: Path | None,
                 fake_source: bool, add_doi: str | None,
                 ingest_cmd: str | None) -> None:
    """Find relevant papers NOT yet in your library for PASSAGE.

    Suggestions only — Scholia never writes prose or auto-adds. Only a short
    keyword query (not your draft) is sent to the search APIs; no cloud LLM is
    used. Use --add <DOI> to validate + add a pick via your own external
    triple-validating ingester (set --ingest-cmd or SCHOLIA_INGEST_CMD).
    """
    resolved_corpus = corpus_dir or _default_corpus()
    resolved_index_dir = index_dir or _default_index_dir()

    # If --add is given, skip the search entirely and route to the ingester.
    if add_doi:
        _run_add(ctx, add_doi, ingest_cmd)
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


def _resolve_ingest_cmd(ingest_cmd: str | None) -> str | None:
    """Resolve the external ingester command: --ingest-cmd > SCHOLIA_INGEST_CMD.

    Returns the resolved command string (a path to an ingest script/executable),
    or ``None`` when neither source is set. Scholia ships no ingester and bakes
    in no default path; --add is unavailable until the user points at their own.
    """
    if ingest_cmd:
        return ingest_cmd
    env = os.environ.get(_INGEST_CMD_ENV)
    return env if env else None


def _run_add(ctx: click.Context, doi: str, ingest_cmd: str | None) -> None:
    """Shell out to the user's external ingester to validate + add a DOI.

    Scholia never mutates Zotero itself: the user's own vetted, triple-validating
    ingester is the only writer. The command is taken from --ingest-cmd or the
    SCHOLIA_INGEST_CMD env var; if neither is set we fail with a clear message
    (Scholia ships no ingester). On success we remind the user to re-index; on
    failure we surface a clean message (no traceback) and exit non-zero.

    The ingester is invoked as ``<resolved-cmd> --doi <doi>``. A ``.py`` script is
    run with the current interpreter; anything else is executed directly.
    """
    resolved = _resolve_ingest_cmd(ingest_cmd)
    if not resolved:
        click.echo(
            "`--add` needs an external triple-validating ingester; set "
            f"{_INGEST_CMD_ENV} or pass --ingest-cmd <path>. See README.",
            err=True,
        )
        ctx.exit(1)
        return

    if resolved.lower().endswith(".py"):
        cmd = [sys.executable, resolved, "--doi", doi]
    else:
        cmd = [resolved, "--doi", doi]
    click.echo(f"Validating + adding {doi} via the external ingester …")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        click.echo(f"Could not launch the external ingester: {exc}", err=True)
        ctx.exit(1)
        return
    if getattr(proc, "stdout", ""):
        click.echo(proc.stdout)
    if proc.returncode != 0:
        if getattr(proc, "stderr", ""):
            click.echo(proc.stderr, err=True)
        click.echo(
            f"Add failed (ingester exited {proc.returncode}). Nothing "
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


def _trust_cert_windows(cert_path: Path) -> tuple[bool, str]:
    """Install cert_path into the CurrentUser\\Root store silently via certutil.

    Thin shim that delegates to ``scholia.server._install_cert_trust``, which
    holds the implementation.  Kept here for backward-compatibility with tests
    that mock ``scholia.cli._trust_cert_windows``.

    Returns (success, message).
    """
    from scholia.server import _install_cert_trust

    return _install_cert_trust(cert_path)


@cli.command(name="trust-cert")
@click.option("--cert", "cert_path", type=click.Path(path_type=Path), default=None,
              help="Path to the PEM certificate to trust. Defaults to "
                   "~/.scholia/localhost.crt (auto-generated if missing).")
@click.option("--key", "key_path", type=click.Path(path_type=Path), default=None,
              help="Path to the PEM private-key file paired with --cert. "
                   "Defaults to a .key file next to the cert.")
@click.pass_context
def trust_cert(ctx: click.Context, cert_path: Path | None,
               key_path: Path | None) -> None:
    """Trust the Scholia localhost certificate in the OS store (one-time setup).

    Ensures the self-signed localhost cert exists (generates it if not), then
    installs it into the CurrentUser Trusted Root store — no admin elevation
    required.  Idempotent: safe to re-run if the cert is already trusted.

    Required once before the Word add-in task pane will load without a security
    warning.  After this, just run `scholia serve --serve-addin` normally.
    """
    from scholia.server import generate_localhost_cert

    default_cert_dir = Path.home() / ".scholia"
    resolved_cert = cert_path or (default_cert_dir / "localhost.crt")

    # Derive the key path: if --key is given use it; if --cert was given derive
    # from its sibling (same dir, .key extension); else fall back to the default
    # dir.  This ensures cert and key always live together — no cross-dir pairing.
    if key_path is not None:
        resolved_key = key_path
    elif cert_path is not None:
        resolved_key = cert_path.with_suffix(".key")
    else:
        resolved_key = default_cert_dir / "localhost.key"

    # (a) Ensure the cert exists — generate if missing.
    if not resolved_cert.exists() or not resolved_key.exists():
        click.echo(
            f"Certificate not found at {resolved_cert}. Generating …"
        )
        try:
            generate_localhost_cert(resolved_cert, resolved_key)
        except ImportError as exc:
            click.echo(str(exc), err=True)
            ctx.exit(1)
            return
        click.echo(f"  Generated: {resolved_cert}")
    else:
        click.echo(f"Certificate found: {resolved_cert}")

    # (b) Install into CurrentUser\Root (Windows only; no admin).
    if sys.platform != "win32":
        click.echo(
            "Note: automatic cert trust is Windows-only.  On macOS/Linux, add "
            f"{resolved_cert} to your system trust store manually."
        )
        return

    click.echo("Installing into CurrentUser\\Root store …")
    ok, msg = _trust_cert_windows(resolved_cert)
    if ok:
        click.echo(f"  {msg}")
        click.echo(
            f"\nDone.  Certificate trusted at:\n  {resolved_cert}\n"
            "You can now run `scholia serve --serve-addin` and the Word add-in\n"
            "task pane will load without a security warning."
        )
    else:
        click.echo(f"  ERROR: {msg}", err=True)
        ctx.exit(1)


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
@click.option("--serve-addin", "serve_addin", is_flag=True,
              help="Serve the Word add-in task-pane files over HTTPS. "
                   "Enables --https automatically. Generates a self-signed "
                   "localhost cert on first run (requires the 'cryptography' "
                   "package). The cert must be trusted in Windows before the "
                   "task pane will load — see word-addin/SIDELOAD_WORD.md.")
@click.option("--https", "use_https", is_flag=True,
              help="Wrap the server in TLS (HTTPS). "
                   "Provide --cert and --key, or let --serve-addin auto-generate them.")
@click.option("--cert", "cert_path", type=click.Path(path_type=Path), default=None,
              help="Path to a PEM certificate file. Used with --https / --serve-addin. "
                   "Auto-generated when absent (stored in ~/.scholia/localhost.crt).")
@click.option("--key", "key_path", type=click.Path(path_type=Path), default=None,
              help="Path to a PEM private-key file. Auto-generated with the cert.")
@click.pass_context
def serve(ctx: click.Context, index_dir: Path | None, host: str, port: int,
          no_rerank: bool, fake_embedder: bool, fake_source: bool,
          serve_addin: bool, use_https: bool,
          cert_path: Path | None, key_path: Path | None) -> None:
    """Start a localhost JSON API bridge (cite/discover) for UI clients.

    Loads the index + models once at startup so every request is fast. Binds
    127.0.0.1 only — nothing leaves the machine except discovery's keyword
    queries to scholarly APIs (unchanged from the CLI). No prose is generated.

    Endpoints:
      GET  /health   → {"status":"ok","papers":N,"embedder":...}
      POST /cite     → {"passage":str,"k"?:int,"threshold"?:float,"rerank"?:bool}
      POST /discover → {"passage":str,"limit"?:int}

    Add --serve-addin to also serve the Word task-pane static files over HTTPS
    (required for the Office.js add-in; see word-addin/SIDELOAD_WORD.md).
    """
    from scholia.server import (
        _ADDIN_DIR,
        generate_localhost_cert,
        load_state,
        serve as _serve,
    )

    if host != "127.0.0.1":
        click.echo(
            "Warning: binding to a non-localhost address exposes your library "
            "to the local network.",
            err=True,
        )

    # --serve-addin implies HTTPS.
    if serve_addin:
        use_https = True

    # Resolve cert/key paths; auto-generate when running in add-in / HTTPS mode.
    resolved_cert: Path | None = None
    resolved_key: Path | None = None
    if use_https:
        default_cert_dir = Path.home() / ".scholia"
        resolved_cert = cert_path or (default_cert_dir / "localhost.crt")
        resolved_key  = key_path  or (default_cert_dir / "localhost.key")

        if not resolved_cert.exists() or not resolved_key.exists():
            click.echo(
                "Generating self-signed localhost certificate "
                f"(first-run, stored in {default_cert_dir}) …"
            )
            try:
                generate_localhost_cert(resolved_cert, resolved_key)
            except ImportError as exc:
                click.echo(str(exc), err=True)
                ctx.exit(1)
                return
            click.echo(
                f"  Certificate: {resolved_cert}\n"
                f"  Private key: {resolved_key}\n"
                "\nIMPORTANT: Trust the certificate before sideloading the add-in.\n"
                "  See word-addin/SIDELOAD_WORD.md for the trust step."
            )
        else:
            click.echo(f"Using existing certificate: {resolved_cert}")

    resolved_index_dir = index_dir or _default_index_dir()
    state = load_state(
        resolved_index_dir,
        no_rerank=no_rerank,
        fake_embedder=fake_embedder,
        fake_source=fake_source,
    )

    # Set the addin_dir on state so the handler can serve static files.
    if serve_addin:
        addin_dir = _ADDIN_DIR
        if not addin_dir.is_dir():
            click.echo(
                f"Warning: word-addin directory not found at {addin_dir}. "
                "Static file serving disabled.",
                err=True,
            )
        else:
            state.addin_dir = addin_dir

    scheme = "https" if use_https else "http"
    click.echo(f"Scholia serving on {scheme}://{host}:{port}")
    click.echo(
        f"  {len(state.index._papers)} papers | embedder: "
        f"{state.index.embedder_model or 'unknown'}"
    )
    if serve_addin and state.addin_dir:
        click.echo(
            f"  Task pane: {scheme}://{host}:{port}/taskpane.html"
        )

    httpd = _serve(host, port, state,
                   ssl_certfile=resolved_cert, ssl_keyfile=resolved_key)

    # Warm models in the background so /health answers immediately and the
    # first /cite is fast once warming completes.  Fake embedder (test mode)
    # skips warming — FakeEmbedder has no weights to load.
    if not fake_embedder:
        from scholia.server import warm_models_async

        def _on_warm_done() -> None:
            click.echo("  Models warm — bridge fully ready.")

        click.echo(
            "  Warming models in background (first use ~15 s warm, "
            "~50 s on first-ever download)…"
        )
        warm_models_async(state, on_done=_on_warm_done)
    else:
        # Fake embedder: models are trivially ready immediately.
        state.models_ready.set()

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


@cli.command(name="app")
@click.option("--index-dir", type=click.Path(path_type=Path),
              default=None,
              help="FAISS index directory. Overrides SCHOLIA_INDEX_DIR env var.")
def app_cmd(index_dir: Path | None) -> None:
    """Launch the Scholia desktop app (system tray + results panel + Ctrl+Alt+G hotkey).

    Loads the index in-process — no separate ``scholia serve`` window needed.
    Requires PySide6 and pynput:

        pip install "scholia[overlay]"
        pip install pynput

    The app runs in the system tray.  Press Ctrl+Alt+G from any app to ground
    whatever text is selected (or copied) and pop the results panel.  Double-click
    the tray icon to re-open the panel.
    """
    try:
        from scholia.app import run_app
    except ImportError as exc:
        click.echo(
            f"Could not import scholia.app: {exc}\n"
            "Run: pip install \"scholia[overlay]\" && pip install pynput",
            err=True,
        )
        raise SystemExit(1)

    resolved_index_dir = index_dir or _default_index_dir()
    run_app(index_dir=resolved_index_dir)


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
