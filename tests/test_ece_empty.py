from __future__ import annotations

import numpy as np
import pytest

from mlfs.calibration import expected_calibration_error


def test_expected_calibration_error_rejects_empty_and_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        expected_calibration_error(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="same length"):
        expected_calibration_error(np.array([1.0, 0.0, 1.0]), np.array([0.9, 0.1]))
    assert expected_calibration_error(
        np.array([1.0, 0.0]), np.array([1.0, 0.0])
    ) == pytest.approx(0.0)
