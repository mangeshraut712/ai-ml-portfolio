from __future__ import annotations

import numpy as np
import pytest

from mlfs.calibration import brier_score


def test_brier_score_rejects_empty_and_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        brier_score(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="same length"):
        brier_score(np.array([1.0, 0.0]), np.array([0.8]))
    assert brier_score(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0
