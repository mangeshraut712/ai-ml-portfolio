from __future__ import annotations

import numpy as np
import pytest

from mlfs.metrics import precision_recall_f1


def test_precision_recall_f1_rejects_empty_and_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        precision_recall_f1(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="same length"):
        precision_recall_f1(np.array([1, 0, 1]), np.array([1, 0]))
    precision, recall, f1 = precision_recall_f1(
        np.array([1, 0, 1]), np.array([1, 0, 0])
    )
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(2 / 3)
