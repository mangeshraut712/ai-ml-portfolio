"""K-Means clustering (Lloyd)."""

from __future__ import annotations

import numpy as np


class KMeans:
    def __init__(self, n_clusters: int = 3, n_iter: int = 100, seed: int = 0):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, X: np.ndarray) -> "KMeans":
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float)
        idx = rng.choice(len(X), size=self.n_clusters, replace=False)
        self.centroids_ = X[idx].copy()
        for _ in range(self.n_iter):
            labels = self.predict(X)
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members):
                    self.centroids_[k] = members.mean(axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        dist = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        return np.argmin(dist, axis=1)
