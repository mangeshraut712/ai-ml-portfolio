from __future__ import annotations

import numpy as np

from mlfs.naive_bayes import GaussianNB


def test_gaussian_nb_keeps_string_labels() -> None:
    X = np.array([[0.0], [1.0], [0.1], [1.1]])
    y = np.array(["cat", "dog", "cat", "dog"])
    pred = GaussianNB().fit(X, y).predict(X)
    assert list(pred) == ["cat", "dog", "cat", "dog"]
