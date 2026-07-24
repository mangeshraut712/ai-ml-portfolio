"""Gradient descent variants on a quadratic bowl (teaching demo)."""

from __future__ import annotations

import numpy as np


def batch_gd(
    grad_fn,
    x0: np.ndarray,
    lr: float = 0.1,
    steps: int = 100,
) -> tuple[np.ndarray, list[float]]:
    x = x0.astype(float).copy()
    hist = []
    for _ in range(steps):
        g = grad_fn(x)
        x -= lr * g
        hist.append(float(np.linalg.norm(g)))
    return x, hist


def sgd(
    grad_fn_i,
    x0: np.ndarray,
    n: int,
    lr: float = 0.1,
    steps: int = 200,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = x0.astype(float).copy()
    for _ in range(steps):
        i = int(rng.integers(0, n))
        x -= lr * grad_fn_i(x, i)
    return x


def minibatch_gd(
    grad_fn_batch,
    x0: np.ndarray,
    n: int,
    batch_size: int = 16,
    lr: float = 0.1,
    steps: int = 100,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = x0.astype(float).copy()
    for _ in range(steps):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        x -= lr * grad_fn_batch(x, idx)
    return x
