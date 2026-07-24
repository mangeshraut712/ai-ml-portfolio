"""Gaussian Naive Bayes."""

from __future__ import annotations

import numpy as np


class GaussianNB:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y)
        self.mean_ = {}
        self.var_ = {}
        self.priors_ = {}
        for c in self.classes_:
            Xc = X[y == c]
            self.mean_[int(c)] = Xc.mean(axis=0)
            self.var_[int(c)] = Xc.var(axis=0) + 1e-9
            self.priors_[int(c)] = len(Xc) / len(X)
        return self

    def _log_gauss(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
        return float(
            -0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        preds = []
        for x in X:
            best_c, best_score = None, -1e18
            for c in self.classes_:
                c = int(c)
                score = np.log(self.priors_[c]) + self._log_gauss(
                    x, self.mean_[c], self.var_[c]
                )
                if score > best_score:
                    best_score, best_c = score, c
            preds.append(best_c)
        return np.asarray(preds)
