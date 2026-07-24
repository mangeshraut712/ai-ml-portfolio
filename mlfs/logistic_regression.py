"""Binary logistic regression with L1/L2 regularization."""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegression:
    def __init__(
        self,
        lr: float = 0.1,
        n_iter: int = 2000,
        l1: float = 0.0,
        l2: float = 0.0,
        fit_intercept: bool = True,
    ):
        self.lr = lr
        self.n_iter = n_iter
        self.l1 = l1
        self.l2 = l2
        self.fit_intercept = fit_intercept

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        return np.c_[np.ones(len(X)), X]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        Xb = self._add_intercept(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        self.w_ = np.zeros(Xb.shape[1])
        n = len(Xb)
        for _ in range(self.n_iter):
            p = sigmoid(Xb @ self.w_)
            grad = (Xb.T @ (p - y)) / n
            reg = self.l2 * self.w_ + self.l1 * np.sign(self.w_)
            if self.fit_intercept:
                reg[0] = 0.0
            self.w_ -= self.lr * (grad + reg)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xb = self._add_intercept(np.asarray(X, dtype=float))
        return sigmoid(Xb @ self.w_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)
