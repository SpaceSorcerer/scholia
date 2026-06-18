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


# --- Item 3: nomic task prefixes ---

class _RecordingST:
    """Records exactly what text is passed to .encode()."""

    def __init__(self, model_name, device="cpu", trust_remote_code=False):
        self.model_name = model_name
        self.encoded: list[list[str]] = []

    def get_sentence_embedding_dimension(self):
        return 8

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        self.encoded.append(list(texts))
        return np.ones((len(texts), 8), dtype=np.float32) * 0.5


def _recording_nomic(model_name="nomic-ai/nomic-embed-text-v1.5"):
    emb = NomicEmbedder(model_name=model_name)
    rec = _RecordingST(model_name)
    emb._load_backend = lambda: rec  # type: ignore[method-assign]
    return emb, rec


def test_nomic_embed_documents_prepends_search_document_prefix():
    emb, rec = _recording_nomic()
    emb.embed_documents(["title one", "title two"])
    assert rec.encoded[-1] == [
        "search_document: title one",
        "search_document: title two",
    ]


def test_nomic_embed_query_prepends_search_query_prefix():
    emb, rec = _recording_nomic()
    emb.embed_query("a draft passage")
    assert rec.encoded[-1] == ["search_query: a draft passage"]


def test_nomic_embed_query_returns_1d_vector():
    emb, _ = _recording_nomic()
    vec = emb.embed_query("x")
    assert vec.shape == (8,)
    assert vec.dtype == np.float32


def test_nomic_plain_embed_is_not_prefixed():
    """Back-compat: .embed() still embeds bare text (no prefixes)."""
    emb, rec = _recording_nomic()
    emb.embed(["bare text"])
    assert rec.encoded[-1] == ["bare text"]


def test_nomic_minilm_path_is_not_prefixed():
    """A non-nomic model loaded via NomicEmbedder must NOT get nomic prefixes."""
    emb, rec = _recording_nomic(model_name="sentence-transformers/all-MiniLM-L6-v2")
    emb.embed_documents(["doc text"])
    emb.embed_query("query text")
    assert rec.encoded[0] == ["doc text"]
    assert rec.encoded[1] == ["query text"]


# --- FakeEmbedder/default-mixin doc+query helpers are no-op wrappers ---

def test_fake_embedder_doc_and_query_equal_plain_embed():
    emb = FakeEmbedder(dim=16)
    docs = emb.embed_documents(["alpha", "beta"])
    plain = emb.embed(["alpha", "beta"])
    np.testing.assert_array_equal(docs, plain)
    q = emb.embed_query("alpha")
    np.testing.assert_array_equal(q, plain[0])


def test_fake_embedder_satisfies_protocol_with_doc_query():
    assert isinstance(FakeEmbedder(), Embedder)
