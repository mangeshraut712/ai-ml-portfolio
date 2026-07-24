"""K-fold cross-validation."""

from __future__ import annotations

from typing import Callable

import numpy as np


def kfold_indices(n: int, n_splits: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_splits)
    splits = []
    for i in range(n_splits):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        splits.append((train, test))
    return splits


def cross_val_score(
    model_factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    scoring: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> np.ndarray:
    from .metrics import accuracy

    scoring = scoring or accuracy
    scores = []
    for train_idx, test_idx in kfold_indices(len(X), n_splits=n_splits):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        scores.append(scoring(y[test_idx], pred))
    return np.asarray(scores, dtype=float)
