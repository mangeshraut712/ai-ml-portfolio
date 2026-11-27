"""Scalers must survive constant columns without dividing by zero."""

from __future__ import annotations

import numpy as np

from mlfs.preprocessing import MinMaxScaler, StandardScaler


def test_standard_scaler_constant_column_maps_to_zero() -> None:
    X = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    scaler = StandardScaler().fit(X)
    # std == 0 is replaced by 1.0 so the transform stays finite.
    assert scaler.scale_[1] == 1.0
    Z = scaler.transform(X)
    assert np.all(np.isfinite(Z))
    assert np.allclose(Z[:, 1], 0.0)
    # The informative column is still standardised.
    assert np.allclose(Z[:, 0].mean(), 0.0)
    assert np.allclose(Z[:, 0].std(), 1.0)


def test_minmax_scaler_constant_column_maps_to_zero() -> None:
    X = np.array([[2.0, 5.0], [4.0, 5.0], [6.0, 5.0]])
    scaler = MinMaxScaler().fit(X)
    # max - min == 0 is replaced by 1.0 so the transform stays finite.
    assert scaler.range_[1] == 1.0
    Z = scaler.transform(X)
    assert np.all(np.isfinite(Z))
    assert np.allclose(Z[:, 1], 0.0)
    assert np.allclose(Z[:, 0], [0.0, 0.5, 1.0])
