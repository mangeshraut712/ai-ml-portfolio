from __future__ import annotations

import numpy as np
import pytest

from mlfs.metrics import roc_auc_score, roc_curve


def test_roc_auc_requires_both_classes() -> None:
    scores = np.array([0.1, 0.4, 0.9])
    with pytest.raises(ValueError, match="both classes"):
        roc_auc_score(np.array([1, 1, 1]), scores)
    with pytest.raises(ValueError, match="both classes"):
        roc_curve(np.array([0, 0, 0]), scores)
    assert roc_auc_score(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8])) > 0.9
