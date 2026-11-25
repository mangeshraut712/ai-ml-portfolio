from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

from eval_metrics.latency_cost import estimate_cost_usd


def test_estimate_cost_usd_hand_computed() -> None:
    # 1000 queries averaging 2000 input / 500 output tokens at $3/$15 per
    # million tokens: per query 0.002*3 + 0.0005*15 = 0.0135 USD -> 13.5 total.
    cost = estimate_cost_usd(
        1000, 2000, 500, price_in_per_mtok=3.0, price_out_per_mtok=15.0
    )
    assert cost == pytest.approx(13.5)


def test_estimate_cost_usd_scales_linearly_with_query_count() -> None:
    one_k = estimate_cost_usd(1000, 1500, 250, 1.0, 4.0)
    four_k = estimate_cost_usd(4000, 1500, 250, 1.0, 4.0)
    assert four_k == pytest.approx(4 * one_k)


def test_estimate_cost_usd_uses_per_million_units() -> None:
    # Prices are per million tokens (configs/pricing.yaml) while the report
    # KPI is per 1k; guard against a per-1k/per-million mix-up.
    assert estimate_cost_usd(1, 1_000_000, 0, 2.5, 0.0) == pytest.approx(2.5)
    assert estimate_cost_usd(1, 0, 1_000_000, 0.0, 2.5) == pytest.approx(2.5)


def test_estimate_cost_usd_zero_tokens_or_zero_price_is_zero() -> None:
    assert estimate_cost_usd(500, 0, 0, 5.0, 15.0) == 0.0
    assert estimate_cost_usd(500, 2000, 500, 0.0, 0.0) == 0.0
