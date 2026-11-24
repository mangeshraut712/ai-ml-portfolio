from __future__ import annotations

import numpy as np

from mlfs.preprocessing import MinMaxScaler, StandardScaler


def test_scalers_accept_nested_lists() -> None:
    X = [[1, 2], [3, 4], [5, 6]]
    z = StandardScaler().fit_transform(X)
    assert np.allclose(z.mean(axis=0), 0.0)
    m = MinMaxScaler().fit_transform(X)
    assert np.allclose(m.min(axis=0), 0.0) and np.allclose(m.max(axis=0), 1.0)
