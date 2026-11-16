from __future__ import annotations

import numpy as np
import pytest

from mlfs.bias_variance import poly_features


def test_poly_features_rejects_non_positive_degree() -> None:
    X = np.array([[1.0], [2.0]])
    with pytest.raises(ValueError, match="positive integer"):
        poly_features(X, 0)
    with pytest.raises(ValueError, match="positive integer"):
        poly_features(X, -1)
    assert poly_features(X, 2).shape == (2, 2)
