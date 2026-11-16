from __future__ import annotations

import numpy as np

from mlfs.decision_tree import DecisionTreeClassifier


def test_decision_tree_keeps_string_labels() -> None:
    X = np.array([[0.0], [1.0], [0.1], [1.1]])
    y = np.array(["cat", "dog", "cat", "dog"])
    pred = DecisionTreeClassifier(max_depth=1).fit(X, y).predict(X)
    assert list(pred) == ["cat", "dog", "cat", "dog"]
