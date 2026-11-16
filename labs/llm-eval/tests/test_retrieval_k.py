from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

from eval_metrics.retrieval import ndcg_at_k, recall_at_k


def test_recall_and_ndcg_reject_non_positive_k() -> None:
    retrieved = ["d1", "d2", "d3", "d4"]
    relevant = ["d3"]
    with pytest.raises(ValueError, match="positive integer"):
        recall_at_k(relevant, retrieved, -1)
    with pytest.raises(ValueError, match="positive integer"):
        ndcg_at_k(relevant, retrieved, 0)
    assert recall_at_k(relevant, retrieved, 1) == 0.0
    assert recall_at_k(relevant, retrieved, 3) == 1.0
