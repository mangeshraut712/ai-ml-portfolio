"""Scaled dot-product attention + tiny encoder block (NumPy)."""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Q,K,V: (batch, seq, d). Returns (output, weights)."""
    d = Q.shape[-1]
    scores = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(d)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


class TinySelfAttention:
    """Single-head self-attention with learned projections."""

    def __init__(self, d_model: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.Wq = rng.normal(0, 0.1, size=(d_model, d_model))
        self.Wk = rng.normal(0, 0.1, size=(d_model, d_model))
        self.Wv = rng.normal(0, 0.1, size=(d_model, d_model))
        self.Wo = rng.normal(0, 0.1, size=(d_model, d_model))

    def __call__(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # X: (batch, seq, d)
        Q, K, V = X @ self.Wq, X @ self.Wk, X @ self.Wv
        ctx, weights = scaled_dot_product_attention(Q, K, V)
        return ctx @ self.Wo, weights
