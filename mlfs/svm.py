"""Linear soft-margin SVM trained with hinge loss + SGD."""

from __future__ import annotations

import numpy as np


class LinearSVM:
    def __init__(self, lr: float = 0.01, n_iter: int = 1000, C: float = 1.0, seed: int = 0):
        self.lr = lr
        self.n_iter = n_iter
        self.C = C
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVM":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        # map {0,1} -> {-1,+1}
        y = np.where(y <= 0, -1.0, 1.0)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.w_ = np.zeros(d)
        self.b_ = 0.0
        for _ in range(self.n_iter):
            i = int(rng.integers(0, n))
            xi, yi = X[i], y[i]
            margin = yi * (xi @ self.w_ + self.b_)
            if margin >= 1:
                self.w_ -= self.lr * self.w_
            else:
                self.w_ -= self.lr * (self.w_ - self.C * yi * xi)
                self.b_ += self.lr * self.C * yi
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.w_ + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0).astype(int)
