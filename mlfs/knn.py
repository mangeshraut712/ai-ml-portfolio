"""k-Nearest Neighbors classifier."""

from __future__ import annotations

import numpy as np


class KNNClassifier:
    def __init__(self, k: int = 5):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y).reshape(-1)
        if len(self.X_) != len(self.y_):
            raise ValueError("X and y must contain the same number of rows")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        k = min(self.k, len(self.X_))
        preds = []
        for x in X:
            dist = np.linalg.norm(self.X_ - x, axis=1)
            nn = np.argsort(dist, kind="stable")[:k]
            vals, counts = np.unique(self.y_[nn], return_counts=True)
            preds.append(vals[np.argmax(counts)])
        return np.asarray(preds)
