"""WebRTC GMM-based Voice Activity Detector."""

from __future__ import annotations

import numpy as np
import webrtcvad
from postprocess import apply_hangover, flags_to_segments


class WebRTCVADEngine:
    """Frame-level speech detector using WebRTC's GMM classifier.

    WebRTC VAD accepts only 8/16/32/48 kHz mono PCM and frame lengths of
    10, 20, or 30 ms. Aggressiveness ranges from 0 (lax) to 3 (strict).
    """

    VALID_SAMPLE_RATES = {8000, 16000, 32000, 48000}
    VALID_FRAME_MS = {10, 20, 30}

    def __init__(self, aggressiveness: int = 2):
        if aggressiveness not in range(4):
            raise ValueError("aggressiveness must be an integer in [0, 3]")
        self.vad = webrtcvad.Vad(aggressiveness)
        self.aggressiveness = aggressiveness

    def frame_generator(
        self, frame_duration_ms: int, audio: np.ndarray, sample_rate: int
    ):
        """Yield fixed-length 16-bit PCM frames from a float waveform."""
        if sample_rate not in self.VALID_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {sorted(self.VALID_SAMPLE_RATES)}"
            )
        if frame_duration_ms not in self.VALID_FRAME_MS:
            raise ValueError(
                f"frame_duration_ms must be one of {sorted(self.VALID_FRAME_MS)}"
            )

        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        audio_clipped = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16).tobytes()

        offset = 0
        while offset + n <= len(audio_int16):
            yield audio_int16[offset : offset + n]
            offset += n

    def detect_speech_raw(
        self,
        audio: np.ndarray,
        sample_rate: int,
        frame_duration_ms: int = 30,
    ) -> np.ndarray:
        """Raw GMM decisions without hangover smoothing."""
        frames = self.frame_generator(frame_duration_ms, audio, sample_rate)
        speech_timeline = [
            1 if self.vad.is_speech(frame, sample_rate) else 0 for frame in frames
        ]
        return np.asarray(speech_timeline, dtype=np.int8)

    def detect_speech(
        self,
        audio: np.ndarray,
        sample_rate: int,
        frame_duration_ms: int = 30,
        hangover: bool = True,
    ) -> np.ndarray:
        """Return binary speech timeline (1 = speech, 0 = non-speech).

        Every complete frame in the audio is classified — no frames are skipped
        inside the valid PCM window (e.g. 258/258 for a 7.75 s @ 30 ms clip).
        """
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
        """Convert frame flags into contiguous (start_sec, end_sec) segments."""
        flags = self.detect_speech(
            audio, sample_rate, frame_duration_ms, hangover=hangover
        )
        return flags_to_segments(flags, frame_duration_ms)
