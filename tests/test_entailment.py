"""Entailment-checker unit tests. FakeEntailmentChecker is deterministic and
model-free; the real MiniCheckEntailmentChecker is exercised only behind
@pytest.mark.integration (test_integration_entailment.py)."""

from scholia.entailment import (
    EntailmentChecker,
    EntailmentResult,
    FakeEntailmentChecker,
    MiniCheckEntailmentChecker,
)


# --- FakeEntailmentChecker: protocol + determinism + overlap behaviour ---


def test_fake_checker_satisfies_protocol():
    assert isinstance(FakeEntailmentChecker(), EntailmentChecker)


def test_fake_checker_returns_entailment_result():
    out = FakeEntailmentChecker().verify("QKI splicing", "QKI controls splicing")
    assert isinstance(out, EntailmentResult)
    assert isinstance(out.supported, bool)
    assert 0.0 <= out.score <= 1.0


def test_fake_checker_full_support_when_claim_tokens_all_present():
    # Every content token of the claim appears in the evidence -> score 1.0.
    out = FakeEntailmentChecker().verify(
        "QKI splicing", "QKI regulates alternative splicing in heart"
    )
    assert out.score == 1.0
    assert out.supported is True


def test_fake_checker_low_support_when_evidence_unrelated():
    out = FakeEntailmentChecker().verify(
        "QKI regulates splicing", "olive oil and the mediterranean diet"
    )
    assert out.score == 0.0
    assert out.supported is False


def test_fake_checker_partial_support_below_default_threshold():
    # 1 of 3 claim tokens present (0.33) is below the 0.5 default -> unsupported.
    out = FakeEntailmentChecker().verify(
        "QKI nuclear retention", "QKI is an interesting molecule"
    )
    assert 0.0 < out.score < 0.5
    assert out.supported is False


def test_fake_checker_is_deterministic():
    a = FakeEntailmentChecker().verify("QKI splicing", "QKI controls splicing")
    b = FakeEntailmentChecker().verify("QKI splicing", "QKI controls splicing")
    assert (a.supported, a.score) == (b.supported, b.score)


def test_fake_checker_empty_claim_is_unsupported():
    out = FakeEntailmentChecker().verify("", "any evidence at all")
    assert out.supported is False
    assert out.score == 0.0


def test_fake_checker_threshold_is_configurable():
    # With a lenient threshold, a 0.5-overlap claim becomes supported.
    out = FakeEntailmentChecker(threshold=0.4).verify(
        "QKI splicing", "QKI is a protein"
    )  # 1 of 2 tokens -> 0.5 >= 0.4
    assert out.score == 0.5
    assert out.supported is True


# --- MiniCheckEntailmentChecker: structure without downloading a model ---


def test_minicheck_checker_satisfies_protocol():
    assert isinstance(MiniCheckEntailmentChecker(), EntailmentChecker)


def test_minicheck_checker_does_not_load_on_construction():
    """Construction must be lazy: no model object until first verify()."""
    chk = MiniCheckEntailmentChecker()
    assert chk._model is None


def test_minicheck_checker_short_circuits_empty_without_loading():
    """Empty claim or evidence returns unsupported without touching the model."""
    chk = MiniCheckEntailmentChecker()
    assert chk.verify("", "evidence").supported is False
    assert chk.verify("claim", "   ").supported is False
    assert chk._model is None  # never loaded


class _FakeTokenizer:
    """Minimal stand-in for a Flan-T5 tokenizer. Returns a dummy torch encoding
    with the .to() interface MiniCheckEntailmentChecker.verify uses."""

    eos_token = "</s>"

    def __call__(self, texts, return_tensors=None, truncation=None, max_length=None):
        import torch

        class _Enc(dict):
            def to(self, device):
                return self

        return _Enc(
            input_ids=torch.ones((len(texts), 4), dtype=torch.long),
            attention_mask=torch.ones((len(texts), 4), dtype=torch.long),
        )


class _FakeSeq2SeqModel:
    """Stand-in for AutoModelForSeq2SeqLM. Emits a logits tensor whose label-token
    positions (ids 3 and 209) carry the chosen [unsupported, supported] logits, so
    the softmax-derived support score is deterministic without a download."""

    def __init__(self, supported_logit, unsupported_logit):
        self._sup = supported_logit
        self._unsup = unsupported_logit

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None, decoder_input_ids=None):
        import torch

        # Vocab wide enough to index token ids 3 (unsupported) and 209 (supported).
        n = input_ids.size(0)
        logits = torch.full((n, 1, 300), -1e9)
        logits[:, 0, 3] = self._unsup
        logits[:, 0, 209] = self._sup

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        return out


def test_minicheck_supported_when_supported_logit_dominates(monkeypatch):
    chk = MiniCheckEntailmentChecker(model_name="dummy-minicheck", threshold=0.5)
    monkeypatch.setattr(
        chk, "_load_backend",
        lambda: (_FakeSeq2SeqModel(supported_logit=6.0, unsupported_logit=-5.0),
                 _FakeTokenizer()),
    )
    out = chk.verify("QKI controls splicing", "QKI is an RNA-binding splicing factor")
    assert out.supported is True
    assert out.score > 0.9
    assert chk._model is not None  # loaded on first use


def test_minicheck_unsupported_when_supported_logit_low(monkeypatch):
    chk = MiniCheckEntailmentChecker(model_name="dummy-minicheck", threshold=0.5)
    monkeypatch.setattr(
        chk, "_load_backend",
        lambda: (_FakeSeq2SeqModel(supported_logit=-5.0, unsupported_logit=6.0),
                 _FakeTokenizer()),
    )
    out = chk.verify("QKI controls splicing", "the mediterranean diet lowers risk")
    assert out.supported is False
    assert out.score < 0.1


def test_minicheck_score_is_probability_in_unit_interval(monkeypatch):
    chk = MiniCheckEntailmentChecker(model_name="dummy-minicheck")
    monkeypatch.setattr(
        chk, "_load_backend",
        lambda: (_FakeSeq2SeqModel(supported_logit=2.0, unsupported_logit=0.0),
                 _FakeTokenizer()),
    )
    out = chk.verify("claim", "evidence")
    assert 0.0 <= out.score <= 1.0
