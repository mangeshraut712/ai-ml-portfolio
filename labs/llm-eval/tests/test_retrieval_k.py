"""Unit tests for retrieval metric argument checks."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

import pytest

from eval_metrics.retrieval import ndcg_at_k, recall_at_k


def test_recall_and_ndcg_reject_non_positive_k():
    rel = ["doc_1"]
    retrieved = ["doc_0", "doc_1", "doc_2"]
    for k in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            recall_at_k(rel, retrieved, k)
        with pytest.raises(ValueError, match="positive integer"):
            ndcg_at_k(rel, retrieved, k)


def test_recall_and_ndcg_still_score_a_valid_k():
    rel = ["doc_1"]
    retrieved = ["doc_0", "doc_1", "doc_2"]
    assert recall_at_k(rel, retrieved, 2) == 1.0
    assert ndcg_at_k(rel, retrieved, 2) > 0
