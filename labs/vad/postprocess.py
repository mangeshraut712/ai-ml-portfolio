"""Frame-level post-processing for VAD timelines (hangover / smoothing)."""

from __future__ import annotations

import numpy as np


def apply_hangover(
    flags: np.ndarray,
    *,
    pad_speech_frames: int = 1,
    fill_gap_frames: int = 2,
    min_speech_frames: int = 2,
) -> np.ndarray:
    """Stabilize raw binary VAD flags for production-like segmentation.

    Steps:
    1. Drop speech islands shorter than ``min_speech_frames`` (noise blips).
    2. Fill non-speech gaps shorter than ``fill_gap_frames`` inside speech.
    3. Pad each speech region by ``pad_speech_frames`` on both sides (hangover)
       to recover clipped onsets/offsets from aggressive GMM decisions.
    """
    if flags.size == 0:
        return flags.astype(np.int8)

    out = np.asarray(flags, dtype=np.int8).copy()
    n = len(out)

    # 1) remove short speech islands
    i = 0
    while i < n:
        if out[i] == 0:
            i += 1
            continue
        j = i
        while j < n and out[j] == 1:
            j += 1
        if (j - i) < min_speech_frames:
            out[i:j] = 0
        i = j

    # 2) fill short gaps
    i = 0
    while i < n:
        if out[i] == 1:
            i += 1
            continue
        j = i
        while j < n and out[j] == 0:
            j += 1
        left_speech = i > 0 and out[i - 1] == 1
        right_speech = j < n and out[j] == 1
        if left_speech and right_speech and (j - i) <= fill_gap_frames:
            out[i:j] = 1
        i = j

    # 3) hangover pad
    if pad_speech_frames > 0:
        padded = out.copy()
        speech_idx = np.flatnonzero(out == 1)
        for idx in speech_idx:
            start = max(0, idx - pad_speech_frames)
            end = min(n, idx + pad_speech_frames + 1)
            padded[start:end] = 1
        out = padded

    return out.astype(np.int8)


def flags_to_segments(
    flags: np.ndarray, frame_duration_ms: int = 30
) -> list[tuple[float, float]]:
    """Convert binary frame flags into contiguous (start_sec, end_sec) segments."""
    frame_sec = frame_duration_ms / 1000.0
    segments: list[tuple[float, float]] = []
    start: float | None = None

    for idx, flag in enumerate(flags):
        t = idx * frame_sec
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            segments.append((start, t))
            start = None

    if start is not None:
        segments.append((start, len(flags) * frame_sec))

    return segments


def segments_to_flags(
    segments: list[tuple[float, float]],
    n_frames: int,
    frame_duration_ms: int = 30,
) -> np.ndarray:
    """Rasterize second-based segments onto a fixed frame grid."""
    flags = np.zeros(n_frames, dtype=np.int8)
    frame_sec = frame_duration_ms / 1000.0
    for start, end in segments:
        i0 = max(0, int(start / frame_sec))
        i1 = min(n_frames, int(np.ceil(end / frame_sec)))
        flags[i0:i1] = 1
    return flags
