"""End-to-end Denoiser + WebRTC VAD pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from denoiser import AudioDenoiser
from postprocess import flags_to_segments
from vad_engine import WebRTCVADEngine


@dataclass
class VADResult:
    """Structured output for a single processed audio file."""

    clean_audio: np.ndarray
    sample_rate: int
    speech_flags: np.ndarray
    speech_segments: list[tuple[float, float]]
    speech_ratio: float
    frame_duration_ms: int = 30
    expected_frames: int = 0
    frames_classified: int = 0
    hangover: bool = True
    meta: dict = field(default_factory=dict)

    @property
    def frame_complete(self) -> bool:
        """True when every expected PCM frame was classified."""
        return (
            self.expected_frames > 0 and self.frames_classified == self.expected_frames
        )


class VADPipeline:
    """Two-stage speech segregation: spectral gating → GMM VAD → hangover."""

    def __init__(self, aggressiveness: int = 2, sr: int = 16000):
        self.denoiser = AudioDenoiser(sr=sr)
        self.vad_engine = WebRTCVADEngine(aggressiveness=aggressiveness)
        self.sr = sr

    def process_array(
        self,
        audio: np.ndarray,
        sample_rate: int | None = None,
        frame_duration_ms: int = 30,
        denoise: bool = True,
        hangover: bool = True,
    ) -> VADResult:
        """Run VAD on an in-memory waveform."""
        sr = sample_rate or self.sr
        if denoise:
            clean_audio = self.denoiser.remove_noise_array(audio)
        else:
            clean_audio = np.asarray(audio, dtype=np.float32)

        frame_len = int(sr * (frame_duration_ms / 1000.0))
        expected_frames = len(clean_audio) // frame_len if frame_len else 0

        speech_flags = self.vad_engine.detect_speech(
            clean_audio,
            sr,
            frame_duration_ms=frame_duration_ms,
            hangover=hangover,
        )
        segments = flags_to_segments(speech_flags, frame_duration_ms)
        speech_ratio = float(speech_flags.mean()) if len(speech_flags) else 0.0

        return VADResult(
            clean_audio=clean_audio,
            sample_rate=sr,
            speech_flags=speech_flags,
            speech_segments=segments,
            speech_ratio=speech_ratio,
            frame_duration_ms=frame_duration_ms,
            expected_frames=expected_frames,
            frames_classified=int(len(speech_flags)),
            hangover=hangover,
            meta={"denoise": denoise},
        )

    def process_file(
        self,
        file_path: str,
        frame_duration_ms: int = 30,
        denoise: bool = True,
        hangover: bool = True,
    ) -> VADResult:
        """Denoise (optional) and detect speech frames for one audio file."""
        import librosa

        if denoise:
            clean_audio, sr = self.denoiser.remove_noise(file_path)
        else:
            clean_audio, sr = librosa.load(file_path, sr=self.sr, mono=True)

        result = self.process_array(
            clean_audio,
            sample_rate=sr,
            frame_duration_ms=frame_duration_ms,
            denoise=False,  # already denoised above if requested
            hangover=hangover,
        )
        result.meta["denoise"] = denoise
        result.meta["source"] = file_path
        return result
