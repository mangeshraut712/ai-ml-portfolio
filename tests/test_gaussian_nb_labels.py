from __future__ import annotations

import numpy as np

from mlfs.naive_bayes import GaussianNB


def test_gaussian_nb_keeps_string_labels() -> None:
    X = np.array([[0.0], [1.0], [0.1], [1.1]])
    y = np.array(["cat", "dog", "cat", "dog"])
    pred = GaussianNB().fit(X, y).predict(X)
    assert list(pred) == ["cat", "dog", "cat", "dog"]


def test_gaussian_nb_far_query_never_returns_none() -> None:
    # Constant features give a tiny variance, so a far query scores below -1e18.
    X = np.array([[0.0], [0.0], [10.0], [10.0]])
    y = np.array([0, 0, 1, 1])
    pred = GaussianNB().fit(X, y).predict(np.array([[1e6]]))
    assert pred[0] == 1
