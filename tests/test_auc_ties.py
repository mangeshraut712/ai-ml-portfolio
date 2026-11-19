from __future__ import annotations

import numpy as np
import pytest

from mlfs.metrics import roc_auc_score, roc_curve


def test_all_tied_scores_give_half_regardless_of_order() -> None:
    scores = np.array([0.5, 0.5])
    assert roc_auc_score(np.array([1, 0]), scores) == pytest.approx(0.5)
    assert roc_auc_score(np.array([0, 1]), scores) == pytest.approx(0.5)


def test_partial_ties_count_half() -> None:
    # pairs: (0.9 vs 0.4) win, (0.9 vs 0.1) win, (0.4 vs 0.4) tie, (0.4 vs 0.1) win -> 3.5 / 4
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.4, 0.4, 0.1])
    assert roc_auc_score(y, scores) == pytest.approx(0.875)
    assert roc_auc_score(y[::-1], scores[::-1]) == pytest.approx(0.875)


def test_roc_curve_has_one_point_per_distinct_score() -> None:
    fpr, tpr = roc_curve(np.array([1, 0, 1, 0]), np.array([0.8, 0.8, 0.3, 0.3]))
    np.testing.assert_allclose(fpr, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(tpr, [0.0, 0.5, 1.0])
