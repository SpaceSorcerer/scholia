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
scholia index            # uses the default literature mirror + .scholia_index
scholia index --corpus "C:\path\to\mirror" --index-dir ".scholia_index"
```

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
