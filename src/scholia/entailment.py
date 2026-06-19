"""Entailment / support-verification backends.

This is the keystone of Scholia's *honest* anti-hallucination story. Retrieval +
re-rank answer "which library paper is most *similar* to this claim?" — but
similarity is not support. A paper can score high on a cross-encoder relevance
logit while its abstract does not actually *say* what the claim asserts (the
classic "high-similarity-but-doesn't-really-support" failure). This module adds a
second, independent check: does the evidence text *textually support* the claim?

We deliberately frame this as **support-verification**, NOT stance detection. The
model emits a calibrated support probability; we keep the existing similarity/
rerank verdict and ADD an entailment check on top. When retrieval says SUPPORTED
but entailment does not clearly support, we surface an honest
"retrieved-but-not-clearly-supported" flag — we never claim a paper *contradicts*
a claim. Scientific stance is error-prone and a false contradiction flag would
erode trust; under-claiming ("verify the source") is the safe, honest failure mode.

Mirrors ``rerank.py``/``embedders.py``: an ``EntailmentChecker`` Protocol, a
deterministic model-free ``FakeEntailmentChecker`` for unit tests, and a lazily
loaded real ``MiniCheckEntailmentChecker`` whose ``transformers`` import lives
INSIDE the load method so the unit-test suite never triggers a download. Local
CPU only; the model downloads once on first real use (same posture as the
embedder and reranker).

Model choice (empirically driven). The real checker uses
``lytang/MiniCheck-Flan-T5-Large`` (MIT-licensed; fine-tuned from Apache-2.0
``google/flan-t5-large``). It is purpose-built for *grounding* — "does this
document support this claim?" — which is the right question here. Generic NLI
cross-encoders (e.g. nli-deberta) were evaluated first and REJECTED: on real
abstracts vs paraphrased one-sentence claims they collapse to "neutral" for
BOTH genuine and off-topic pairs (P(entailment) ~ 0.00 across the board), because
MNLI-style NLI demands strict logical entailment, not topical grounding. MiniCheck
separates the same real pairs cleanly (genuine support ~0.96-0.98, non-support
~0.01-0.04). See .scholia_entailment_calib.py / the entailment report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class EntailmentResult:
    """Outcome of one support-verification.

    ``supported`` is the boolean verdict (``score >= threshold``); ``score`` is
    the model's support/entailment probability in ``[0, 1]`` (for the real NLI
    checker, ``P(entailment)`` from a softmax over the entailment/neutral/
    contradiction logits). Higher = the evidence more clearly supports the claim.
    """

    supported: bool
    score: float


@runtime_checkable
class EntailmentChecker(Protocol):
    """Verifies whether ``evidence`` textually supports ``claim``.

    The required surface is a single method::

        verify(claim: str, evidence: str) -> EntailmentResult

    The returned ``score`` is a support probability in ``[0, 1]`` (NOT a cosine
    or a rerank logit — a third, model-specific scale). ``supported`` is the
    boolean verdict against the checker's own threshold.
    """

    def verify(self, claim: str, evidence: str) -> EntailmentResult:
        ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class FakeEntailmentChecker:
    """Deterministic, model-free support checker for unit tests.

    Scores support by the fraction of the claim's content tokens that appear in
    the evidence (recall of claim tokens). This is a coarse stand-in for genuine
    entailment, but it is monotone in "does the evidence mention what the claim
    asserts," which is exactly the behaviour the verified-claim-check logic and
    CLI tests need to assert. No model, no RNG, no network — stable across
    processes. A claim with no content tokens scores 0.0 (cannot be supported).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def verify(self, claim: str, evidence: str) -> EntailmentResult:
        claim_tokens = set(_tokenize(claim))
        if not claim_tokens:
            return EntailmentResult(supported=False, score=0.0)
        evidence_tokens = set(_tokenize(evidence))
        overlap = len(claim_tokens & evidence_tokens)
        score = overlap / len(claim_tokens)
        return EntailmentResult(supported=score >= self.threshold, score=score)


# Per-model context. lytang/MiniCheck-Flan-T5-Large (MIT; from Apache-2.0
# google/flan-t5-large) is a grounding/fact-verification model that emits a
# binary supported/unsupported label with a calibrated probability. We replicate
# its documented scoring with plain `transformers` (already a dep), so no new
# package is required. Empirically (see .scholia_entailment_calib.py against the
# real library): genuine support -> ~0.96-0.98, off-topic/wrong-paper -> ~0.01-0.04.
# The default support threshold (0.50) sits in that ~0.9-wide gap; calibrated like
# the reranker thresholds in cli.py.
_DEFAULT_ENTAILMENT_MODEL = "lytang/MiniCheck-Flan-T5-Large"

# Flan-T5 decoder token ids for the "0" (unsupported) and "1" (supported) labels,
# per the MiniCheck reference inference. Recorded explicitly so a tokenizer change
# would be a deliberate edit, not a silent mislabel.
_MINICHECK_LABEL_TOKEN_IDS = (3, 209)


class MiniCheckEntailmentChecker:
    """Real CPU support checker backed by MiniCheck-Flan-T5 (via transformers).

    Lazily loads the seq2seq model + tokenizer on first ``verify`` call; the
    ``transformers``/``torch`` imports live inside ``_load_backend`` so importing
    this module (and the unit-test suite) never triggers a model download.
    One-time download on first real use, same posture as ``NomicEmbedder`` /
    ``CrossEncoderReranker``. Local CPU only.

    Scoring follows the MiniCheck reference exactly: the input is
    ``"predict: " + document + </s> + claim``; the decoder is run for a single
    step; the supported probability is ``softmax`` over the label-token logits
    (ids 3 = "0"/unsupported, 209 = "1"/supported), index 1. So "the abstract is
    similar but doesn't actually support the claim" lands low — the whole point of
    this layer. We never expose a "contradicts" label; this is support-only.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_ENTAILMENT_MODEL,
        threshold: float = 0.5,
        device: str = "cpu",
        max_length: int = 2048,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load_backend(self):
        # Imported lazily so unit tests never trigger a model download.
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        model.to(self.device)
        model.eval()
        return model, tokenizer

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model, self._tokenizer = self._load_backend()

    def verify(self, claim: str, evidence: str) -> EntailmentResult:
        # An empty claim carries nothing to verify, and empty evidence cannot
        # support anything — short-circuit to unsupported without loading a model.
        if not claim or not claim.strip() or not evidence or not evidence.strip():
            return EntailmentResult(supported=False, score=0.0)
        self._ensure_loaded()
        import torch

        text = "predict: " + evidence + self._tokenizer.eos_token + claim
        enc = self._tokenizer(
            [text], return_tensors="pt", truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        decoder_input_ids = torch.zeros(
            (enc["input_ids"].size(0), 1), dtype=torch.long
        ).to(self.device)
        with torch.no_grad():
            out = self._model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            )
        logits = out.logits.squeeze(1)
        label_logits = logits[:, torch.tensor(list(_MINICHECK_LABEL_TOKEN_IDS))]
        probs = torch.nn.functional.softmax(label_logits, dim=-1)
        score = float(probs[0, 1].item())
        return EntailmentResult(supported=score >= self.threshold, score=score)
