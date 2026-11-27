"""K-fold splits must partition the data with no train/test leakage."""

from __future__ import annotations

import numpy as np

from mlfs.model_selection import kfold_indices


def test_kfold_test_folds_partition_all_indices() -> None:
    # 11 rows, 3 folds -> uneven fold sizes (4/4/3) on purpose.
    n, n_splits = 11, 3
    splits = kfold_indices(n, n_splits=n_splits, seed=42)
    assert len(splits) == n_splits
    seen = np.concatenate([test for _, test in splits])
    # Every row must land in exactly one test fold: no drops, no duplicates.
    assert sorted(seen.tolist()) == list(range(n))


def test_kfold_train_never_overlaps_test() -> None:
    for train, test in kfold_indices(12, n_splits=4, seed=0):
        assert set(train.tolist()).isdisjoint(test.tolist())
        assert len(train) + len(test) == 12


def test_kfold_seed_reproduces_and_changes_splits() -> None:
    a = kfold_indices(20, n_splits=5, seed=7)
    b = kfold_indices(20, n_splits=5, seed=7)
    c = kfold_indices(20, n_splits=5, seed=8)
    assert all(
        np.array_equal(tr1, tr2) and np.array_equal(te1, te2)
        for (tr1, te1), (tr2, te2) in zip(a, b)
    )
    # A different seed must be able to produce a different partition.
    assert any(
        not np.array_equal(te1, te2) for (_, te1), (_, te2) in zip(a, c)
    )
