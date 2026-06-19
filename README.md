# Scholia

![CI](https://github.com/SpaceSorcerer/scholia/actions/workflows/ci.yml/badge.svg)

**A privacy-first, local citation/grounding engine for scientific writing — grounded in *your own* validated library.**

Scholia is a local-first assistant that checks whether the claims in your scientific writing are actually supported by papers you already trust. Point it at your Zotero literature mirror; it builds an on-device semantic index and, given any passage, returns the best-matching papers from **your library** — each with its Zotero link and DOI — plus a clear SUPPORTED / UNSUPPORTED verdict. It can also *discover* relevant papers you don't yet have, so you can validate and add them. Grounding, indexing, and ranking run 100% on your machine. Scholia never writes your prose, and it cannot invent a citation.

---

## Why Scholia

- **Your library is the engine, not a bolt-on.** Retrieval runs over *your* validated Zotero mirror as the primary corpus — not a generic web index your papers were sprinkled into. What it surfaces is what you've already vetted.
- **It structurally cannot fabricate a citation.** There is no generative citation path. Scholia only *resolves* a passage to real papers already in your library (or, for discovery, to real records returned by Semantic Scholar / PubMed). It never produces a plausible-looking reference out of thin air.
- **Grounding is 100% on-device.** Indexing, embedding, re-ranking, and the SUPPORTED/UNSUPPORTED check all run locally. No cloud LLM is ever contacted.
- **Assist, don't ghostwrite.** Scholia suggests and checks; it never writes sentences for you. You stay the author.
- **A discovery loop that grows your library.** When a passage isn't supported, discovery finds candidate papers *not yet* in your library and routes adds through your existing triple-validating ingester — so only vetted papers enter.
- **MIT licensed.**

---

## Quickstart

```bash
# 1. Install (Python 3.11+)
pip install -e .

# 2. Build the index from your Zotero literature mirror
scholia index --corpus "/path/to/zotero-mirror"
#   (or set it once: export SCHOLIA_CORPUS="/path/to/zotero-mirror"  →  scholia index)

# 3. Ground a passage against your library
scholia cite "QKI controls alternative splicing during cardiomyocyte maturation."

# 4. Discover relevant papers you DON'T yet have
scholia discover "QKI controls alternative splicing during cardiomyocyte maturation."

# 5. (optional) Run the local bridge for fast, stateful UI clients
scholia serve

# 6. (optional) Always-on-top desktop overlay
pip install "scholia[overlay]"
scholia overlay --start-server
```

`scholia index` reads `--corpus` (or the `SCHOLIA_CORPUS` env var) and writes a FAISS index +
metadata sidecar to `--index-dir` (or `SCHOLIA_INDEX_DIR`, defaulting to `~/.scholia/index`).
The embedder identity is stored in the index, so later commands adopt the same embedder
automatically — you don't need to repeat `--model`.

---

## Commands

| Command | What it does |
|---|---|
| `scholia index --corpus <path>` | Embed your Zotero mirror notes and build the local FAISS index. |
| `scholia cite "<passage>"` | Return ranked supporting papers from your library + a claim-check verdict. |
| `scholia discover "<passage>"` | Find relevant papers **not** in your library (Semantic Scholar + PubMed). |
| `scholia discover "<passage>" --add <DOI>` | Validate + add a pick via the triple-validating `zotero_ingest.py`. |
| `scholia serve` | Start the localhost JSON bridge (loads index + models once). |
| `scholia overlay [--start-server]` | Launch the always-on-top desktop window (requires the `overlay` extra). |

### Grounding a passage

```bash
scholia cite "<passage>"                  # rerank ON by default
scholia cite "<passage>" --k 5            # number of papers to return
scholia cite "<passage>" --no-rerank      # plain bi-encoder cosine
scholia cite "<passage>" --candidate-k 50 # widen the rerank pool
```

Output is a ranked list (first author, year, title, Zotero key, `zotero://` link, DOI),
a `Ranking signal` line (which scoring scale is live), and a final `CLAIM-CHECK`
line. Below the active threshold the passage is flagged `UNSUPPORTED by your library`.

### Discovery

`discover` queries **Semantic Scholar** (Academic Graph) and **PubMed** (E-utilities) via the
Python standard library only, merges and de-dupes the results, drops anything already in your
library (matched by DOI, or by title when no DOI), and prints the ranked **new** candidates —
each clearly framed as *not in your library, suggestions only*. Scholia never auto-adds and
never writes prose. Use `--fake-source` for an offline/deterministic run.

To add a pick:

```bash
scholia discover "<passage>" --add 10.1242/jcs.230276
```

`--add` shells out to the existing `zotero_ingest.py`, which triple-validates the DOI
(CrossRef + PubMed), de-dupes against Zotero, and writes the Obsidian mirror note. Re-run
`scholia index` afterwards so the new paper becomes searchable.

---

## How it works

```
Zotero mirror  ──▶  embed  ──▶  FAISS (cosine)  ──▶  cross-encoder re-rank  ──▶  claim-check
   (.md notes)      (bi-encoder)   top-candidate_k        top-k, joint scoring     SUPPORTED?
```

1. **Embed.** Each mirror note (title + abstract) is embedded into a normalized vector. The
   nomic model applies `search_document:` / `search_query:` task prefixes automatically.
2. **Retrieve.** A FAISS inner-product index fetches the top-`candidate_k` candidates by cosine
   similarity (the fast but coarse bi-encoder stage).
3. **Re-rank.** A cross-encoder re-scores each `(query, paper)` pair *jointly* — far more
   discriminative — and returns the top-`k`. On by default; falls back to the bi-encoder if the
   model can't load.
4. **Claim-check.** The top score is compared against a scale-appropriate threshold to produce
   the SUPPORTED / UNSUPPORTED verdict.

**Local bridge & overlay.** `scholia serve` loads the index and models once and exposes a small
localhost JSON API (`/health`, `/cite`, `/discover`) so UI clients respond fast without reloading
models per query. The `scholia overlay` desktop window is a thin client of that bridge: type or
paste a passage (or click **Ground clipboard** to grab whatever you last copied from any editor —
Word Online, VS Code, Obsidian), then **Ground** or **Discover**.

**Pluggable by design.** Embedder, Reranker, and DiscoverySource are simple `Protocol`s. A
third-party embedder needs only `dim` and `embed(texts)`; a reranker needs only
`rerank(query, papers, top_k)`; a discovery source needs only `search(query, limit)`. Swap in
your own without touching the pipeline.

### Bridge API (127.0.0.1 only)

```
GET  /health   → {"status":"ok","papers":N,"embedder":"..."}
POST /cite     → body {"passage":str,"k"?:int,"threshold"?:float,"rerank"?:bool}
POST /discover → body {"passage":str,"limit"?:int}
```

---

## Privacy

Scholia is built so your draft stays on your machine:

- **Indexing, retrieval, re-ranking, and grounding are 100% local.** Nothing about a passage you
  ground is transmitted anywhere.
- **Models download once, then run offline.** The embedder and cross-encoder weights are fetched
  from HuggingFace on first use, then run locally on CPU with no further network calls.
- **Discovery sends only a short keyword query — never your draft.** `discover` extracts a focused,
  stopword-filtered key-term string locally (capped at a handful of content words) and sends *only
  that string* to PubMed / Semantic Scholar. The passage itself never leaves the machine.
- **No cloud LLM is ever contacted**, at any step.
- **The bridge binds `127.0.0.1` only** — never `0.0.0.0`. Binding elsewhere prints a warning.

---

## Models & thresholds

Because a passage is "supported" when its top match clears a threshold, and different scoring
stages live on different scales, Scholia picks a scale-appropriate default for you. The active
scale is always printed on the `Ranking signal` line; `--threshold` overrides it.

**Embedder (bi-encoder, cosine scale — used with `--no-rerank`):**

| Embedder | Default | Notes |
|---|---|---|
| `nomic-ai/nomic-embed-text-v1.5` *(default)* | **0.73** | High similarity floor even with task prefixes; calibrated so genuine off-domain/gibberish queries stay below and on-domain hits clear it. |
| `all-MiniLM-L6-v2` (`--model …all-MiniLM-L6-v2`) / Fake / unknown | **0.45** | Textbook separation (off-domain low, on-domain high). |

**Re-ranker (cross-encoder, relevance scale — ON by default):**

| Re-ranker | Score type | Default | Notes |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` *(default)* | relevance logit | **0.0** | ~22M params, ~2.2 s/query on CPU. Logit centred at 0: on-domain positive, off-domain negative. |
| `BAAI/bge-reranker-v2-m3` (`--rerank-model …`) | relevance prob (0–1) | **0.20** | Higher accuracy but ~42 s/query on CPU. |

> **Note:** the meaning of `--threshold` depends on whether re-ranking is on. With re-rank (default)
> it's a cross-encoder relevance score; with `--no-rerank` it's a cosine similarity. The threshold
> calibrations above were derived empirically against the real local library.

All weights are Apache-2.0 and download once on first use (local CPU, no cloud).

---

## Status & roadmap

**v0.2.x** — the core is shipped and tested:

- ✅ Local citation/grounding engine (embed → FAISS → cross-encoder re-rank → claim-check)
- ✅ Discovery (Semantic Scholar + PubMed) with library de-dup and validated `--add`
- ✅ Localhost JSON bridge (`scholia serve`)
- ✅ Desktop overlay v0 (`scholia overlay`)

**Coming:**

- Live Word-Online capture (ground as you write, no copy/paste round-trip)
- A gap / structure writing-partner (flag under-supported sections, surface structural gaps)

> The overlay GUI is **early (v0)**: a thin, functional always-on-top window driven by the local
> bridge. It works, but expect rough edges and limited polish.

---

## Tests

```bash
pytest                 # unit tests only (deterministic FakeEmbedder; no model download)
pytest -m integration  # real-model end-to-end + GUI smoke tests (downloads weights)
```

The default `pytest` invocation deselects the `integration` marker (configured in
`pyproject.toml`), so unit tests stay fast and download-free — that's also what CI runs.

---

## License & attribution

Scholia is released under the **MIT License** (see [`LICENSE`](LICENSE)).

It depends on [sentence-transformers](https://github.com/UKPLab/sentence-transformers) and
[transformers](https://github.com/huggingface/transformers), both licensed under the
**Apache License 2.0**; their NOTICE files and license terms ship with their distribution
packages. The default and optional model weights (`nomic-embed-text-v1.5`,
`ms-marco-MiniLM-L-6-v2`, `bge-reranker-v2-m3`) are likewise Apache-2.0.
