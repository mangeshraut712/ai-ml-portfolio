"""Calibration metrics (ECE / Brier)."""

from __future__ import annotations

import numpy as np


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(proba, dtype=float).reshape(-1)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(proba, dtype=float).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if not np.any(mask):
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)
