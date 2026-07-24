"""Bias–variance demonstration via polynomial regression degree sweep."""

from __future__ import annotations

import numpy as np

from .linear_regression import LinearRegression


def make_toy_regression(n: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.linspace(-1, 1, n)[:, None]
    y = np.sin(2 * np.pi * X[:, 0]) + rng.normal(0, 0.15, size=n)
    return X, y


def poly_features(X: np.ndarray, degree: int) -> np.ndarray:
    cols = [X**d for d in range(1, degree + 1)]
    return np.hstack(cols)


def bias_variance_curve(
    degrees: list[int] | None = None, seed: int = 0
) -> list[dict]:
    """Train/test MSE vs polynomial degree (classic U-shape intuition)."""
    degrees = degrees or [1, 2, 3, 5, 8, 12]
    X, y = make_toy_regression(seed=seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    split = int(0.7 * len(X))
    tr, te = idx[:split], idx[split:]
    rows = []
    for d in degrees:
        Phi_tr = poly_features(X[tr], d)
        Phi_te = poly_features(X[te], d)
        model = LinearRegression(lr=0.05, n_iter=4000, l2=1e-4)
        model.fit(Phi_tr, y[tr])
        train_mse = float(np.mean((model.predict(Phi_tr) - y[tr]) ** 2))
        test_mse = float(np.mean((model.predict(Phi_te) - y[te]) ** 2))
        rows.append({"degree": d, "train_mse": train_mse, "test_mse": test_mse})
    return rows
