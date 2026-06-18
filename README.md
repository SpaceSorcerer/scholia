# Scholia Brain — local citation/grounding engine

Given a passage of draft text, Scholia returns the best-matching papers from
**your own Zotero library** (via the local literature mirror) with the claim
each supports, and flags passages your library does not support. 100% local —
no paper content leaves the machine.

## Install

```bash
pip install -e .
```

## Build the index

```bash
# Option A — pass paths explicitly:
scholia index --corpus "/path/to/zotero-mirror" --index-dir "/path/to/index"

# Option B — set env vars once (e.g. in your shell profile):
export SCHOLIA_CORPUS="/path/to/zotero-mirror"
export SCHOLIA_INDEX_DIR="/path/to/index"
scholia index
```

If neither `--corpus` nor `SCHOLIA_CORPUS` is provided, `scholia index` exits with a helpful error. The default index directory (when `SCHOLIA_INDEX_DIR` is unset) is `~/.scholia/index`.

## Cite-ground a passage

```bash
scholia cite "QKI controls alternative splicing during cardiomyocyte maturation."
scholia cite "<passage>" --k 5 --threshold 0.45
```

Output: ranked supporting papers (first author, year, title, Zotero key,
`zotero://` link, DOI) followed by a `CLAIM-CHECK` line. Below the threshold,
the passage is flagged `UNSUPPORTED by your library`.

## Discovery — find relevant papers *not* in your library

While `cite` grounds a passage in papers you **already have**, `discover` does
the opposite: it searches public scholarly APIs for relevant papers that are
**not yet in your library**, so you can validate and add them.

```bash
scholia discover "QKI controls alternative splicing during cardiomyocyte differentiation."
scholia discover "<passage>" --limit 8 --corpus "/path/to/zotero-mirror"
scholia discover "<passage>" --index-dir "/path/to/index"   # dedup against the index
scholia discover "<passage>" --fake-source                  # offline/deterministic
```

It queries **Semantic Scholar** (Academic Graph) and **PubMed** (E-utilities),
merges and de-dupes the results, drops anything already in your library (matched
by DOI, or by title when no DOI), and prints the ranked **new** candidates — each
clearly framed as *not in your library, suggestions only*. Scholia never writes
prose and never auto-adds; discovery only finds and suggests papers.

Add a pick (validate first, then add) via the existing triple-validating
ingester:

```bash
scholia discover "<passage>" --add 10.1242/jcs.230276
```

`--add` shells out to `zotero_ingest.py`, which triple-validates the DOI
(CrossRef + PubMed), de-dupes against Zotero, and writes the Obsidian mirror
note. Re-index afterwards (`scholia index`) so the new paper is searchable.

### Privacy

Only a **short keyword query** ever leaves your machine — never your draft.
`discover` extracts a focused key-term string locally (stopword-filtered, capped
at a handful of content words) and sends *only that string* to the search APIs.
The draft passage itself is never transmitted, and **no cloud LLM is involved**
at any step. The search backends use the Python standard library (`urllib`)
only — no extra dependency.

## Models

Default embedder: `nomic-ai/nomic-embed-text-v1.5` (CPU). Faster fallback:
`scholia index --model sentence-transformers/all-MiniLM-L6-v2`.

The embedder identity (model name + dimension) is recorded in the index's
`metadata.json` at build time, so `scholia cite` adopts the same embedder by
default — you do not need to repeat `--model` (and dimension mismatch is
prevented by construction).

`nomic-embed-text-v1.5` requires task prefixes: corpus text is embedded as
`search_document: <text>` and queries as `search_query: <text>`. Scholia applies
these automatically for nomic models (no-op for MiniLM/other models).

### Claim-check threshold (embedder-aware)

When you do not pass `--threshold`, Scholia picks a default suited to the
embedder:

| Embedder | Default threshold | Rationale |
|---|---|---|
| `all-MiniLM-L6-v2` / FakeEmbedder / unknown | **0.45** | textbook separation (off-domain ≤0.38, on-domain ≥0.54) |
| `nomic-embed-text-v1.5` | **0.73** | even *with* task prefixes the nomic floor is high; calibrated on the real library, genuine off-domain queries top out ~0.71 and gibberish ~0.61, while on-domain hits sit ≥0.74 |

The nomic default was derived empirically (after the prefix fix) against the
real 210-document library: off-domain and gibberish queries top out at ~0.71,
on-domain hits start at ~0.74, so **0.73** cleanly separates them. Empty or
whitespace-only passages are short-circuited to `UNSUPPORTED` before embedding
(a blank vector otherwise floats near the corpus centroid and scores ~0.74).
Override anytime with `--threshold`.

## Re-ranking (cross-encoder)

The bi-encoder embeds the query and each document independently, so its cosine
margins are tight: on the real 361-paper library the *weakest* genuine
on-domain hit (0.777) sits only ~0.09 above the *strongest* off-domain hit
(0.689). A cross-encoder scores each `(query, paper)` pair **jointly**, which is
far more discriminative. `scholia cite` therefore re-ranks the FAISS top
`--candidate-k` (default 30) candidates with a cross-encoder and reports the
re-ranked scores. Re-ranking is **on by default**; pass `--no-rerank` for the
plain bi-encoder path. If the reranker model cannot load (offline / missing
weights), `cite` prints a one-line notice and falls back to the bi-encoder.

```bash
scholia cite "<passage>"                  # rerank on by default (MiniLM)
scholia cite "<passage>" --no-rerank       # bi-encoder cosine only
scholia cite "<passage>" --candidate-k 50  # widen the rerank pool
scholia cite "<passage>" --rerank-model BAAI/bge-reranker-v2-m3
```

Default reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache-2.0, ~22 M
params, ~2.2 s/query on CPU). Higher-accuracy option:
`BAAI/bge-reranker-v2-m3` (Apache-2.0, but ~42 s/query on CPU). Both download
once on first use (local CPU only, no cloud — same posture as the embedder).

### Re-rank claim-check threshold (reranker-aware)

Cross-encoder scores are a **different scale** than cosine, so the
SUPPORTED/UNSUPPORTED cutoff is re-derived per reranker (empirically, on the
real 361-paper library):

| Reranker | Score type | On-domain top-1 | Off-domain/gibberish top-1 | Default threshold |
|---|---|---|---|---|
| `ms-marco-MiniLM-L-6-v2` | relevance logit | +2.11 … +8.47 | −11.04 … −4.87 | **0.0** |
| `bge-reranker-v2-m3` | relevance prob (0–1) | 0.439 … 0.999 | 0.001 … 0.038 | **0.20** |

The MiniLM cross-encoder widens the genuine-vs-off-domain margin from the
bi-encoder's **0.089** to **6.98 logits** (~78× wider), with `0.0` sitting
cleanly in the gap. `--threshold` overrides the reranker-aware default.

## Serve / API

`scholia serve` loads the index + models **once** at startup and exposes a small
JSON API over localhost for UI clients (overlay apps, browser extensions, etc.)
so they get fast, stateful responses without reloading models per query.

```bash
scholia serve --index-dir ./index          # default port 8765
scholia serve --index-dir ./index --port 9000
scholia serve --index-dir ./index --no-rerank         # bi-encoder only
scholia serve --index-dir ./index --fake-embedder --fake-source  # offline/test
```

### Endpoints (127.0.0.1 only)

**`GET /health`**
```json
{"status": "ok", "papers": 210, "embedder": "nomic-ai/nomic-embed-text-v1.5"}
```

**`POST /cite`** — body: `{"passage": str, "k"?: int, "threshold"?: float, "rerank"?: bool}`
```json
{
  "suggestions": [
    {"rank": 1, "score": 3.21, "first_author": "Chen", "year": "2021",
     "title": "QKI regulates...", "zotero_key": "ABCD1234",
     "zotero_link": "zotero://select/library/items/ABCD1234", "doi": "10.1038/..."}
  ],
  "claim_check": {"supported": true, "top_score": 3.21, "threshold": 0.0},
  "ranking_signal": "reranked (cross-encoder)"
}
```

**`POST /discover`** — body: `{"passage": str, "limit"?: int}`
```json
{
  "candidates": [
    {"title": "...", "authors": ["Smith, J."], "year": "2022",
     "doi": "10.1234/...", "snippet": "...", "source": "semanticscholar"}
  ],
  "query": "QKI RNA binding splicing"
}
```

### Privacy

The server **binds 127.0.0.1 only** — never 0.0.0.0. Nothing in your draft
leaves the machine. The only outbound traffic is (a) discovery's short keyword
query to scholarly APIs (identical to `scholia discover`), and (b) nothing else.
No cloud LLM is involved. No authentication is required for a localhost-only
binding.

## Tests

```bash
pytest                 # unit tests only (deterministic FakeEmbedder; no download)
pytest -m integration  # add the real-model end-to-end test (downloads weights)
```

## Attribution

Scholia depends on [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
and [transformers](https://github.com/huggingface/transformers), both licensed under the
Apache License 2.0. Their respective NOTICE files and license terms are included in their
distribution packages.
