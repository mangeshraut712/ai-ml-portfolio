"""Experimental neural / energy VAD stub for A/B vs WebRTC.

Implements the same frame-level interface as ``WebRTCVADEngine`` so pipelines
can swap backends. Default path is a Silero-style energy gate (no weights).
Optional ONNXRuntime + Silero ONNX is used only when both are available.

WebRTC aggressiveness=2 remains the FULL_PASS / challenge_pass gate.
This module is for local A/B comparison only (``--backend neural|compare``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from postprocess import apply_hangover, flags_to_segments


class NeuralVADEngine:
    """Energy / optional-ONNX VAD with WebRTC-compatible method names."""

    VALID_SAMPLE_RATES = {8000, 16000, 32000, 48000}
    VALID_FRAME_MS = {10, 20, 30}

    # Map WebRTC-style aggressiveness → energy percentile threshold
    _AGG_PERCENTILE = {0: 20.0, 1: 30.0, 2: 40.0, 3: 55.0}

    def __init__(self, aggressiveness: int = 2, onnx_model: str | Path | None = None):
        if aggressiveness not in range(4):
            raise ValueError("aggressiveness must be an integer in [0, 3]")
        self.aggressiveness = aggressiveness
        self._onnx_session = None
        self._onnx_model = Path(onnx_model) if onnx_model else None
        self.backend_name = "energy"
        self._try_load_onnx()

    def _try_load_onnx(self) -> None:
        if self._onnx_model is None or not self._onnx_model.exists():
            return
        try:
            import onnxruntime as ort  # optional dependency
        except ImportError:
            return
        self._onnx_session = ort.InferenceSession(
            str(self._onnx_model), providers=["CPUExecutionProvider"]
        )
        self.backend_name = "onnx"

    def frame_generator(
        self, frame_duration_ms: int, audio: np.ndarray, sample_rate: int
    ):
        """Yield float frames (neural path) matching WebRTC frame grid."""
        if sample_rate not in self.VALID_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {sorted(self.VALID_SAMPLE_RATES)}"
            )
        if frame_duration_ms not in self.VALID_FRAME_MS:
            raise ValueError(
                f"frame_duration_ms must be one of {sorted(self.VALID_FRAME_MS)}"
            )
        frame_len = int(sample_rate * (frame_duration_ms / 1000.0))
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        offset = 0
        while offset + frame_len <= len(audio):
            yield audio[offset : offset + frame_len]
            offset += frame_len

    def _energy_flags(
        self, audio: np.ndarray, sample_rate: int, frame_duration_ms: int
    ) -> np.ndarray:
        frames = list(self.frame_generator(frame_duration_ms, audio, sample_rate))
        if not frames:
            return np.asarray([], dtype=np.int8)
        energies = np.asarray([float(np.sqrt(np.mean(f.astype(np.float64) ** 2))) for f in frames])
        # Floor-relative threshold (Silero-ish heuristic without weights)
        pct = self._AGG_PERCENTILE.get(self.aggressiveness, 40.0)
        floor = float(np.percentile(energies, min(pct, 90.0)))
        thr = max(floor * 1.8, 1e-4)
        return (energies >= thr).astype(np.int8)

    def _onnx_flags(
        self, audio: np.ndarray, sample_rate: int, frame_duration_ms: int
    ) -> np.ndarray:
        """Best-effort Silero-ONNX path; falls back to energy on shape mismatch."""
        assert self._onnx_session is not None
        try:
            # Many public Silero ONNX graphs expect (1, N) float32 @ 16 kHz.
            wav = np.asarray(audio, dtype=np.float32).reshape(1, -1)
            inputs = {self._onnx_session.get_inputs()[0].name: wav}
            outs = self._onnx_session.run(None, inputs)
            probs = np.asarray(outs[0]).reshape(-1)
            frame_len = int(sample_rate * (frame_duration_ms / 1000.0))
            n_frames = len(audio) // frame_len if frame_len else 0
            if len(probs) == n_frames:
                thr = {0: 0.3, 1: 0.4, 2: 0.5, 3: 0.65}[self.aggressiveness]
                return (probs >= thr).astype(np.int8)
        except Exception:  # noqa: BLE001
            pass
        return self._energy_flags(audio, sample_rate, frame_duration_ms)

    def detect_speech_raw(
        self,
        audio: np.ndarray,
        sample_rate: int,
        frame_duration_ms: int = 30,
    ) -> np.ndarray:
        if self._onnx_session is not None:
            return self._onnx_flags(audio, sample_rate, frame_duration_ms)
        return self._energy_flags(audio, sample_rate, frame_duration_ms)

    def detect_speech(
        self,
        audio: np.ndarray,
        sample_rate: int,
        frame_duration_ms: int = 30,
        hangover: bool = True,
    ) -> np.ndarray:
        flags = self.detect_speech_raw(audio, sample_rate, frame_duration_ms)
        if hangover:
            flags = apply_hangover(flags)
        return flags

    def speech_segments(
        self,
        audio: np.ndarray,
        sample_rate: int,
        frame_duration_ms: int = 30,
        hangover: bool = True,
    ) -> list[tuple[float, float]]:
        flags = self.detect_speech(
            audio, sample_rate, frame_duration_ms, hangover=hangover
        )
        return flags_to_segments(flags, frame_duration_ms)
