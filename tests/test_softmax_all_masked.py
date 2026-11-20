from __future__ import annotations

import warnings

import numpy as np

from mlfs.attention import softmax


def test_softmax_all_negative_infinity_is_zeros() -> None:
    scores = np.full((2, 3), -np.inf)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        weights = softmax(scores)
    assert np.allclose(weights, 0.0)
    assert np.all(np.isfinite(weights))
