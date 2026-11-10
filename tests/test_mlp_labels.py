from __future__ import annotations

import numpy as np

from mlfs.mlp import MLPClassifier


def test_mlp_keeps_string_labels() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal([-1], 0.2, size=(20, 1)), rng.normal([1], 0.2, size=(20, 1))])
    y = np.array(["cat"] * 20 + ["dog"] * 20)
    pred = MLPClassifier(hidden=8, lr=0.2, n_iter=400, seed=0).fit(X, y).predict(X)
    assert set(pred).issubset({"cat", "dog"})
    assert (pred == y).mean() >= 0.8
