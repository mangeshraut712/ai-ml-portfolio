"""Gradient boosting for binary classification (stumps + logistic loss)."""

from __future__ import annotations

import numpy as np

from .decision_tree import DecisionTreeClassifier


class GradientBoostingBinary:
    """Simplified GBDT: fit regression residuals with shallow trees via class trick.

    Uses a pragmatic approach for interview demos: successive trees fit on
    pseudo-residuals of logistic loss, then combine with learning rate.
    """

    def __init__(
        self,
        n_estimators: int = 20,
        max_depth: int = 2,
        learning_rate: float = 0.1,
        seed: int = 0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self.trees_: list[DecisionTreeClassifier] = []
        self.init_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingBinary":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        # start from log-odds prior
        p = np.clip(y.mean(), 1e-5, 1 - 1e-5)
        self.init_ = float(np.log(p / (1 - p)))
        F = np.full(len(y), self.init_)
        self.trees_ = []
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_estimators):
            p_hat = 1 / (1 + np.exp(-F))
            residual = y - p_hat
            # Map residual sign to pseudo-labels for a shallow tree direction
            pseudo = (residual > 0).astype(int)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=2)
            # subsample for variance (optional)
            idx = rng.choice(len(X), size=len(X), replace=True)
            tree.fit(X[idx], pseudo[idx])
            direction = np.where(tree.predict(X) == 1, 1.0, -1.0)
            F = F + self.learning_rate * direction * np.abs(residual).mean()
            self.trees_.append(tree)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.init_)
        for tree in self.trees_:
            direction = np.where(tree.predict(X) == 1, 1.0, -1.0)
            F = F + self.learning_rate * direction
        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self.decision_function(X)
        return 1 / (1 + np.exp(-z))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)
