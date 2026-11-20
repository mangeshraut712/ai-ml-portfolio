"""Scaled dot-product attention + tiny encoder block (NumPy)."""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    # An all-(-inf) slice has max -inf, and subtracting it yields NaN.
    axis_max = np.max(x, axis=axis, keepdims=True)
    axis_max = np.where(np.isfinite(axis_max), axis_max, 0.0)
    e = np.exp(x - axis_max)
    e = np.where(np.isfinite(e), e, 0.0)
    total = e.sum(axis=axis, keepdims=True)
    return np.divide(e, total, out=np.zeros_like(e), where=total > 0)


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Q,K,V: (batch, seq, d). Returns (output, weights)."""
    d = Q.shape[-1]
    scores = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(d)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    # A fully masked query becomes 0/0 after softmax; do not attend to any key.
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
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
