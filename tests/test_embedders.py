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
