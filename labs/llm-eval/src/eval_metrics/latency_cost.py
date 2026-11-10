"""Latency + cost estimation helpers."""

from __future__ import annotations

from typing import Sequence


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    if p < 0 or p > 100:
        raise ValueError(f"percentile p must be in [0, 100], got {p}")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    return float(xs[f] + (xs[c] - xs[f]) * (k - f))


def estimate_cost_usd(
    n_queries: int,
    input_tokens: float,
    output_tokens: float,
    price_in_per_mtok: float,
    price_out_per_mtok: float,
) -> float:
    return n_queries * (
        (input_tokens / 1_000_000.0) * price_in_per_mtok
        + (output_tokens / 1_000_000.0) * price_out_per_mtok
    )
