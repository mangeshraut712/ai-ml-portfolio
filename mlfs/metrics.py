"""Classification / ranking metrics including ROC-AUC."""

from __future__ import annotations

import numpy as np


def _as_1d_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    if yt.size == 0 or yp.size == 0:
        raise ValueError("y_true and y_pred must be non-empty")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true and y_pred must have the same length, got {yt.shape} and {yp.shape}")
    return yt, yp


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _as_1d_pair(y_true, y_pred)
    return float(np.mean(yt == yp))


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    return float(precision), float(recall), float(f1)


def roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return FPR, TPR arrays for descending thresholds."""
    y_true = y_true.astype(int)
    order = np.argsort(-scores)
    y = y_true[order]
    P = max(np.sum(y == 1), 1)
    N = max(np.sum(y == 0), 1)
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    tpr = tps / P
    fpr = fps / N
    return np.concatenate([[0.0], fpr]), np.concatenate([[0.0], tpr])


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    # NumPy 2.0 renamed trapz -> trapezoid
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trap(tpr, fpr))


def roc_auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr = roc_curve(y_true, scores)
    return auc(fpr, tpr)
