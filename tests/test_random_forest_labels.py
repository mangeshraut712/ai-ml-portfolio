from __future__ import annotations

import numpy as np

from mlfs.random_forest import RandomForestClassifier


def test_random_forest_keeps_string_labels() -> None:
    X = np.array([[0.0], [1.0], [0.1], [1.1]])
    y = np.array(["cat", "dog", "cat", "dog"])
    pred = RandomForestClassifier(n_estimators=3, max_depth=1, seed=0).fit(X, y).predict(X)
    assert set(pred).issubset({"cat", "dog"})
    assert list(pred)  # non-empty
