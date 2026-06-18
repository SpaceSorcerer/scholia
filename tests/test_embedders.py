import numpy as np

from scholia.embedders import FakeEmbedder


def test_fake_embedder_shape_and_dtype():
    emb = FakeEmbedder(dim=16)
    vecs = emb.embed(["hello world", "another passage"])
    assert vecs.shape == (2, 16)
    assert vecs.dtype == np.float32


def test_fake_embedder_is_deterministic():
    a = FakeEmbedder(dim=16).embed(["repeatable text"])
    b = FakeEmbedder(dim=16).embed(["repeatable text"])
    np.testing.assert_array_equal(a, b)


def test_fake_embedder_distinguishes_texts():
    vecs = FakeEmbedder(dim=16).embed(["alpha", "beta"])
    assert not np.allclose(vecs[0], vecs[1])


def test_fake_embedder_vectors_are_unit_norm():
    vecs = FakeEmbedder(dim=16).embed(["x", "yy", "zzz"])
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


from scholia.embedders import NomicEmbedder, Embedder


class _StubST:
    """Stands in for sentence_transformers.SentenceTransformer."""

    def __init__(self, model_name, device="cpu", trust_remote_code=False):
        self.model_name = model_name

    def get_sentence_embedding_dimension(self):
        return 8

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        # Return a fixed (n, 8) array regardless of input.
        return np.ones((len(texts), 8), dtype=np.float32) * 0.5


def test_nomic_embedder_uses_loaded_model(monkeypatch):
    emb = NomicEmbedder(model_name="dummy-model")
    monkeypatch.setattr(emb, "_load_backend", lambda: _StubST("dummy-model"))
    vecs = emb.embed(["passage one", "passage two"])
    assert vecs.shape == (2, 8)
    assert vecs.dtype == np.float32
    assert emb.dim == 8


def test_nomic_embedder_satisfies_protocol():
    assert isinstance(NomicEmbedder(model_name="dummy-model"), Embedder)
