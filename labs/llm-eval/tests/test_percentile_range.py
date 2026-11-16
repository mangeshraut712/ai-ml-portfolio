from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

from eval_metrics.latency_cost import percentile


def test_percentile_rejects_empty_and_out_of_range_p() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        percentile([], 50)
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        percentile([1, 2, 3, 4, 5], 200)
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        percentile([1, 2, 3, 4, 5], -50)
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 0) == 1
    assert percentile([1, 2, 3, 4, 5], 100) == 5
