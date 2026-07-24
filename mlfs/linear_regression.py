"""Linear regression with optional L2 regularization (Ridge)."""

from __future__ import annotations

import numpy as np


class LinearRegression:
    def __init__(
        self,
        lr: float = 0.01,
        n_iter: int = 1000,
        l2: float = 0.0,
        fit_intercept: bool = True,
    ):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.fit_intercept = fit_intercept

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        return np.c_[np.ones(len(X)), X]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        Xb = self._add_intercept(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        self.w_ = np.zeros(Xb.shape[1])
        n = len(Xb)
        for _ in range(self.n_iter):
            pred = Xb @ self.w_
            err = pred - y
            grad = (Xb.T @ err) / n
            # Do not regularize intercept
            reg = self.l2 * self.w_
            if self.fit_intercept:
                reg[0] = 0.0
            self.w_ -= self.lr * (grad + reg)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xb = self._add_intercept(np.asarray(X, dtype=float))
        return Xb @ self.w_
