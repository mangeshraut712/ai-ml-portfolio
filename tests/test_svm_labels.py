from __future__ import annotations

import numpy as np
import pytest

from mlfs.svm import LinearSVM

X = np.array([[-2.0], [-1.5], [1.5], [2.0]])


@pytest.mark.parametrize("labels", [(0, 1), (-1, 1), (1, 2), ("ham", "spam")])
def test_linear_svm_predicts_in_the_training_label_space(labels) -> None:
    low, high = labels
    y = np.array([low, low, high, high])
    pred = LinearSVM(lr=0.05, n_iter=500).fit(X, y).predict(X)
    assert list(pred) == list(y)


def test_linear_svm_rejects_a_single_class() -> None:
    with pytest.raises(ValueError, match="two classes"):
        LinearSVM().fit(X, np.array([1, 1, 1, 1]))
