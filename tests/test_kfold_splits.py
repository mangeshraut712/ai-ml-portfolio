from __future__ import annotations

import pytest

from mlfs.model_selection import kfold_indices


def test_kfold_indices_rejects_invalid_split_counts() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        kfold_indices(10, n_splits=1)
    with pytest.raises(ValueError, match="at least 2"):
        kfold_indices(10, n_splits=0)
    with pytest.raises(ValueError, match="<= n"):
        kfold_indices(3, n_splits=5)
    splits = kfold_indices(10, n_splits=5)
    assert len(splits) == 5
    for train, test in splits:
        assert len(train) == 8
        assert len(test) == 2
        assert set(train).isdisjoint(set(test))
