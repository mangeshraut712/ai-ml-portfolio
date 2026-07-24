"""k-Nearest Neighbors classifier."""

from __future__ import annotations

import numpy as np


class KNNClassifier:
    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y).reshape(-1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        preds = []
        for x in X:
            dist = np.linalg.norm(self.X_ - x, axis=1)
            nn = np.argsort(dist)[: self.k]
            vals, counts = np.unique(self.y_[nn], return_counts=True)
            preds.append(int(vals[np.argmax(counts)]))
        return np.asarray(preds)
