from __future__ import annotations

import numpy as np

from mlfs.attention import scaled_dot_product_attention


def test_fully_masked_query_does_not_attend() -> None:
    Q = K = V = np.ones((1, 2, 4))
    mask = np.zeros((1, 2, 2), dtype=bool)
    out, weights = scaled_dot_product_attention(Q, K, V, mask)
    assert np.allclose(weights, 0.0)
    assert np.allclose(out, 0.0)


def test_partial_mask_keeps_normalized_visible_weights() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1, 3, 4))
    mask = np.ones((1, 3, 3), dtype=bool)
    mask[0, 0, 1] = False  # query 0 cannot see key 1
    out, weights = scaled_dot_product_attention(X, X, X, mask)
    assert weights[0, 0, 1] == 0.0
    assert np.allclose(weights[0, 0].sum(), 1.0)
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-5)
    assert out.shape == X.shape
