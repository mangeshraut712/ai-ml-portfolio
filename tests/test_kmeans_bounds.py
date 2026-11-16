"""KMeans input-size checks."""
from __future__ import annotations

import numpy as np
import pytest

from mlfs.kmeans import KMeans


def test_kmeans_rejects_more_clusters_than_rows():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="cannot exceed the number of samples"):
        KMeans(n_clusters=3, seed=0).fit(X)


def test_kmeans_rejects_non_positive_cluster_count():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="positive integer"):
        KMeans(n_clusters=0, seed=0).fit(X)
