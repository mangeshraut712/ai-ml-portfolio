"""Decision tree classifier (binary/multiclass via Gini)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gini(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p**2))


@dataclass
class Node:
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    prediction: int | None = None


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root_: Node | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        self.root_ = self._grow(X, y, depth=0)
        return self

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            values, counts = np.unique(y, return_counts=True)
            return Node(prediction=int(values[np.argmax(counts)]))

        feat, thr, best = None, None, 1e18
        parent = gini(y)
        for j in range(X.shape[1]):
            for t in np.unique(X[:, j]):
                left = y[X[:, j] <= t]
                right = y[X[:, j] > t]
                if len(left) == 0 or len(right) == 0:
                    continue
                impurity = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
                if impurity < best:
                    best, feat, thr = impurity, j, float(t)
        if feat is None or best >= parent:
            values, counts = np.unique(y, return_counts=True)
            return Node(prediction=int(values[np.argmax(counts)]))

        mask = X[:, feat] <= thr
        return Node(
            feature=feat,
            threshold=thr,
            left=self._grow(X[mask], y[mask], depth + 1),
            right=self._grow(X[~mask], y[~mask], depth + 1),
        )

    def _predict_one(self, x: np.ndarray, node: Node) -> int:
        if node.prediction is not None:
            return node.prediction
        assert node.feature is not None and node.threshold is not None
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)  # type: ignore[arg-type]
        return self._predict_one(x, node.right)  # type: ignore[arg-type]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        assert self.root_ is not None
        return np.array([self._predict_one(x, self.root_) for x in X])
