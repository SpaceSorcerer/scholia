"""Real MiniCheck entailment-checker end-to-end test. Run explicitly:

    pytest -m integration

Downloads the MiniCheck-Flan-T5-Large model on first run; deselected by default
via the addopts filter in pyproject.toml (same posture as the embedder/reranker
integration tests). Local CPU only.
"""

import pytest

from scholia.entailment import MiniCheckEntailmentChecker


@pytest.mark.integration
def test_real_minicheck_supports_on_domain_and_rejects_off_domain():
    chk = MiniCheckEntailmentChecker(
        model_name="lytang/MiniCheck-Flan-T5-Large", device="cpu"
    )
    claim = "QKI is a STAR-family RNA-binding protein that controls pre-mRNA splicing."
    supporting = (
        "QKI is a member of the STAR family of RNA-binding proteins that regulates "
        "pre-mRNA alternative splicing, mRNA stability, and translation during "
        "development and disease."
    )
    unrelated = (
        "The Mediterranean diet rich in olive oil reduces cardiovascular disease "
        "risk in adults in large cohort studies."
    )

    good = chk.verify(claim, supporting)
    bad = chk.verify(claim, unrelated)

    # Genuine support scores high and clears the default threshold; off-topic
    # evidence scores low and is flagged as not clearly supporting. This is the
    # real-abstract-vs-paraphrase case generic NLI fails (it returns ~0 for both).
    assert good.supported is True
    assert good.score > 0.8
    assert bad.supported is False
    assert bad.score < 0.2
    # Wide, clean separation between genuine support and non-support.
    assert good.score - bad.score > 0.5
