"""Frame-level classification metrics for labeled VAD evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class FrameMetrics:
    """Binary classification metrics over speech/non-speech frames."""

    frames: int
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    speech_pct_pred: float
    speech_pct_true: float

    def to_dict(self) -> dict:
        return asdict(self)


def frame_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> FrameMetrics:
    """Compute precision/recall/F1 for binary speech flags (1=speech)."""
    yt = np.asarray(y_true, dtype=np.int8).ravel()
    yp = np.asarray(y_pred, dtype=np.int8).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: true={yt.shape} pred={yp.shape}")
    if yt.size == 0:
        return FrameMetrics(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    tp = int(np.sum((yp == 1) & (yt == 1)))
    fp = int(np.sum((yp == 1) & (yt == 0)))
    tn = int(np.sum((yp == 0) & (yt == 0)))
    fn = int(np.sum((yp == 0) & (yt == 1)))

    accuracy = (tp + tn) / yt.size
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )

    return FrameMetrics(
        frames=int(yt.size),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        speech_pct_pred=float(yp.mean() * 100),
        speech_pct_true=float(yt.mean() * 100),
    )


def energy_reference_flags(
    audio: np.ndarray,
    sample_rate: int,
    frame_duration_ms: int = 30,
    energy_percentile: float = 30.0,
    margin: float = 1.5,
) -> np.ndarray:
    """Independent weak labels from frame energy (not WebRTC).

    Used as a secondary reference when human labels are unavailable.
    """
    frame_len = int(sample_rate * (frame_duration_ms / 1000.0))
    if frame_len <= 0:
        return np.zeros(0, dtype=np.int8)

    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return np.zeros(0, dtype=np.int8)

    energies = np.array(
        [
            float(np.sqrt(np.mean(audio[i * frame_len : (i + 1) * frame_len] ** 2)))
            for i in range(n_frames)
        ],
        dtype=np.float64,
    )
    floor = np.percentile(energies, energy_percentile)
    threshold = max(floor * margin, 1e-4)
    return (energies >= threshold).astype(np.int8)
