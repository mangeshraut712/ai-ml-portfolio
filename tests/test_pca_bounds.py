"""PCA component-count checks."""
from __future__ import annotations

import numpy as np
import pytest

from mlfs.pca import PCA


def test_pca_rejects_more_components_than_rank():
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="cannot exceed min"):
        PCA(n_components=3).fit(X)


def test_pca_rejects_non_positive_component_count():
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="positive integer"):
        PCA(n_components=0).fit(X)
