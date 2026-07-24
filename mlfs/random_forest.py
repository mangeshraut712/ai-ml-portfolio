"""Random forest via bootstrap aggregation of decision trees."""

from __future__ import annotations

import numpy as np

from .decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 5,
        max_features: str | int | None = "sqrt",
        seed: int = 0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.seed = seed
        self.trees_: list[DecisionTreeClassifier] = []
        self.feat_idx_: list[np.ndarray] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        n, d = X.shape
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(d)))
        elif self.max_features is None:
            k = d
        else:
            k = int(self.max_features)

        self.trees_, self.feat_idx_ = [], []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            feats = rng.choice(d, size=k, replace=False)
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X[idx][:, feats], y[idx])
            self.trees_.append(tree)
            self.feat_idx_.append(feats)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        votes = []
        for tree, feats in zip(self.trees_, self.feat_idx_):
            votes.append(tree.predict(X[:, feats]))
        votes_arr = np.vstack(votes)
        # majority vote per column
        out = []
        for col in votes_arr.T:
            vals, counts = np.unique(col, return_counts=True)
            out.append(int(vals[np.argmax(counts)]))
        return np.asarray(out)
