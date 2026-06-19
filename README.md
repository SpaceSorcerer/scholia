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
| `scholia suggest "<passage>"` | **Writing partner:** flag gaps — missing topics, where a citation is needed, next angles — grounded in your library. Suggests, never drafts. |
| `scholia serve` | Start the localhost JSON bridge (loads index + models once). |
| `scholia overlay [--start-server]` | Launch the always-on-top desktop window (requires the `overlay` extra). |

### Grounding a passage

```bash
scholia cite "<passage>"                  # rerank + verify ON by default
scholia cite "<passage>" --k 5            # number of papers to return
scholia cite "<passage>" --no-rerank      # plain bi-encoder cosine
scholia cite "<passage>" --candidate-k 50 # widen the rerank pool
scholia cite "<passage>" --no-verify      # skip the support-verification pass
```

Output is a ranked list (first author, year, title, Zotero key, `zotero://` link, DOI),
a `Ranking signal` line (which scoring scale is live), and a final `CLAIM-CHECK`
line. Below the active threshold the passage is flagged `UNSUPPORTED by your library`.
With `--verify` (default ON), all top-k retrieved papers are checked for *textual support*
of the claim — see **Verified grounding** below.

### Verified grounding (does the paper actually support the claim?)

Retrieval and re-ranking answer *"which library paper is most **similar** to this claim?"* —
but similarity is not support. A paper can score high on the cross-encoder while its abstract
doesn't actually *say* what the claim asserts (the classic
*high-similarity-but-doesn't-really-support* failure). `--verify` (ON by default) adds a second,
independent pass: it asks a local fact-verification model whether **any of the top-k retrieved
papers' text supports the claim**, not just whether they rank near it.

> **Important:** on a specialized personal library, a clean VERIFIED is the exception, not the
> norm. The ⚠ flag means *"go read the source to confirm"* — it does NOT mean *"this claim is
> wrong."* A one-sentence paraphrase often is not stated verbatim in any single abstract; the
> verifier errs toward caution, not false contradiction.

```bash
scholia cite "<passage>"                       # verify ON (default)
scholia cite "<passage>" --no-verify           # similarity verdict only
scholia cite "<passage>" --verify-threshold 0.6
scholia cite "<passage>" --verify-model lytang/MiniCheck-Flan-T5-Large
```

Three outcomes:

- **SUPPORTED** — similarity *and* support agree. Example output:
  `CLAIM-CHECK: SUPPORTED (top=X.XXX >= Y.YYY) | VERIFIED: Author (year) — title supports the claim (best support=X.XXX >= Y.YYY)`
  (with `(+N more)` appended when multiple papers entail the claim).
- **⚠ retrieved but not clearly supported** — papers ranked highly by similarity, but none
  clearly support the claim in text. Scholia prints:
  `⚠ retrieved N paper(s) but none clearly support this claim — verify the source (best support=X.XXX < Y.YYY).`
- **UNSUPPORTED** — nothing cleared the similarity threshold.

**Honest about its limits.** This is deliberately **support-verification only** — it *never*
claims a paper "contradicts" the claim. Scientific stance detection is error-prone, and a false
contradiction flag would be worse than useless, so the only failure mode we surface is the
conservative "verify the source." The model (MiniCheck) checks *literal grounding*: on a
specialized library, a one-sentence paraphrase often is not stated verbatim in any single
abstract, so verification **errs toward "not clearly supported" rather than over-claiming**.
The model runs on CPU, scoring each of the k retrieved document/claim pairs (~k × 0.26 s warm),
and downloads once on first use.

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

### Writing partner (gap/structure suggestions)

`scholia suggest` helps you see what your draft is **missing** — grounded in your *own*
library. For a passage it retrieves your most relevant papers, then asks a language model to
return **structured pointers only** under three headers: *missing-but-library-covered topics*,
*claims that appear to need a citation*, and *suggested next angles*. Each result is a short
pointer plus the library papers that support it.

```bash
scholia suggest "<passage>"                       # local model (default, on-device)
scholia suggest "<passage>" --backend fake        # deterministic, no model (offline)
scholia suggest "<passage>" --local-url http://localhost:11434/v1   # e.g. Ollama
scholia suggest "<passage>" --backend cloud --allow-cloud           # opt-in cloud
```

**Assist, never ghostwrite (hard rule).** `suggest` *suggests and flags*; it **never writes
manuscript prose and never rewrites your sentences**. This is enforced in two places: the system
prompt explicitly instructs the model to *"suggest what to address and which of their papers is
relevant; do not draft text,"* and Scholia's parser only ever extracts short pointer lines —
anything that looks like a drafted sentence is dropped, never shown to you. There is no codepath
that emits drafted text.

**Local by default; cloud is opt-in and off.** Privacy posture for the model that reads your prose:

- **`--backend local` (default)** talks to a local OpenAI-compatible server (LM Studio at
  `http://localhost:1234/v1`, or Ollama) over the standard library — fully on-device, nothing
  leaves your machine.
- **`--backend fake`** is a deterministic, model-free stub for offline/test use.
- **`--backend cloud`** sends your passage text to **Anthropic** and therefore **requires
  `--allow-cloud`**. Without that flag it refuses (and explains that this needs your institution's
  sign-off). When enabled it prints a one-line warning that your prose is leaving the machine. The
  `anthropic` SDK is an optional extra (`pip install "scholia[cloud]"`), not a core dependency.

```bash
pip install "scholia[cloud]"   # only needed for --backend cloud
export ANTHROPIC_API_KEY=...    # the cloud path reads the key from the environment
```

---

## How it works

```
Zotero mirror ─▶ embed ─▶ FAISS (cosine) ─▶ cross-encoder re-rank ─▶ claim-check ─▶ verify support
  (.md notes)   (bi-enc)   top-candidate_k     top-k, joint scoring    SUPPORTED?    (entailment)
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
5. **Verify support.** (default ON) A local fact-verification model checks whether any of
   the top-k retrieved papers' text actually *supports* the claim — catching cases where
   high-similarity hits don't really say it. SUPPORTED if any paper entails; flags "verify the
   source" only when none do. Support-only; never a "contradicts" claim. See **Verified grounding**.

**Local bridge & overlay.** `scholia serve` loads the index and models once and exposes a small
localhost JSON API (`/health`, `/cite`, `/discover`) so UI clients respond fast without reloading
models per query. The `scholia overlay` desktop window is a thin client of that bridge: type or
paste a passage (or click **Ground clipboard** to grab whatever you last copied from any editor —
Word Online, VS Code, Obsidian), then **Ground** or **Discover**. Results are rendered with
**clickable DOI and Zotero links** — click any `https://doi.org/...` or `zotero://select/...`
link to open it directly. If the bridge is not running, the results pane shows a clear message
("Can't reach Scholia server — is `scholia serve` running?") instead of crashing.

**Pluggable by design.** Embedder, Reranker, EntailmentChecker, and DiscoverySource are simple
`Protocol`s. A third-party embedder needs only `dim` and `embed(texts)`; a reranker needs only
`rerank(query, papers, top_k)`; an entailment checker needs only `verify(claim, evidence)`; a
discovery source needs only `search(query, limit)`. Swap in your own without touching the
pipeline.

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
- **The writing partner runs on-device by default.** `scholia suggest` uses a **local** model
  (LM Studio / Ollama) by default, so your prose stays local. The only path that transmits your
  passage off the machine is `--backend cloud`, which is **opt-in, default-off, and gated behind
  `--allow-cloud` + a printed warning** — because sending unpublished manuscript text to a cloud
  provider requires your institution's sign-off.
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

**Support verifier (entailment scale — `--verify`, ON by default):**

| Verifier | Score type | Default | Notes |
|---|---|---|---|
| `lytang/MiniCheck-Flan-T5-Large` *(default)* | supported prob (0–1) | **0.50** | MIT-licensed (from Apache-2.0 Flan-T5-Large), ~780M params, ~0.3 s/query warm on CPU. Grounding/fact-verification, not strict NLI. On the real library, genuine support scores ~0.96–0.98 and non-support ~0.01–0.19 — the 0.50 cutoff sits in that gap. Conservative on a specialized library (see **Verified grounding**). |

> **Note:** the meaning of `--threshold` depends on whether re-ranking is on. With re-rank (default)
> it's a cross-encoder relevance score; with `--no-rerank` it's a cosine similarity. `--verify-threshold`
> is separate and always on the support-probability scale. The threshold calibrations above were
> derived empirically against the real local library.

The embedder and reranker weights are Apache-2.0; the support verifier
(`MiniCheck-Flan-T5-Large`) is MIT-licensed (fine-tuned from Apache-2.0 `google/flan-t5-large`).
All download once on first use (local CPU, no cloud).

---

## Status & roadmap

**v0.2.x** — the core is shipped and tested:

- ✅ Local citation/grounding engine (embed → FAISS → cross-encoder re-rank → claim-check → verify support)
- ✅ Verified grounding — top-k aggregated entailment/support check (honest "verify the source" flag; never "contradicts")
- ✅ Discovery (Semantic Scholar + PubMed) with library de-dup and validated `--add`
- ✅ Localhost JSON bridge (`scholia serve`)
- ✅ Desktop overlay v0 (`scholia overlay`) — clickable DOI/Zotero links, graceful bridge-unreachable errors

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
packages. The embedder/reranker model weights (`nomic-embed-text-v1.5`,
`ms-marco-MiniLM-L-6-v2`, `bge-reranker-v2-m3`) are likewise Apache-2.0. The support-verification
model [`MiniCheck-Flan-T5-Large`](https://huggingface.co/lytang/MiniCheck-Flan-T5-Large) is
**MIT-licensed** (fine-tuned from Apache-2.0 [`google/flan-t5-large`](https://huggingface.co/google/flan-t5-large));
see Tang et al., *MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents* (EMNLP 2024).
It is loaded via `transformers` (already a dependency) using the model's documented scoring — no
extra package is required.
