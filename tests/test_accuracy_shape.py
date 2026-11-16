from __future__ import annotations

import numpy as np
import pytest

from mlfs.metrics import accuracy


def test_accuracy_rejects_empty_and_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        accuracy(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="same length"):
        accuracy(np.array([1, 0]), np.array([1]))
    assert accuracy(np.array([1, 0, 1]), np.array([1, 0, 0])) == pytest.approx(2 / 3)
