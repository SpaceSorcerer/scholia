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

## Models

Default embedder: `nomic-ai/nomic-embed-text-v1.5` (CPU). Faster fallback:
`scholia index --model sentence-transformers/all-MiniLM-L6-v2`.

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
