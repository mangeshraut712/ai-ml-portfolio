"""Full test suite for the Sarvam VAD interview challenge pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

CHALLENGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CHALLENGE_DIR.parents[1]
SAMPLE = REPO_ROOT / "sample_data" / "sample_audio" / "sample_001.wav"
GT_PATH = REPO_ROOT / "sample_data" / "vad_labeled" / "ground_truth.json"

sys.path.insert(0, str(CHALLENGE_DIR))

from denoiser import AudioDenoiser  # noqa: E402
from metrics import energy_reference_flags, frame_metrics  # noqa: E402
from pipeline import VADPipeline  # noqa: E402
from postprocess import apply_hangover, segments_to_flags  # noqa: E402
from vad_engine import WebRTCVADEngine  # noqa: E402


@pytest.mark.skipif(not SAMPLE.exists(), reason="Run make fetch-vad-samples first")
def test_pipeline_detects_speech_on_sample():
    result = VADPipeline(aggressiveness=2).process_file(str(SAMPLE))
    assert result.sample_rate == 16000
    assert len(result.speech_flags) > 0
    assert result.speech_ratio > 0.1
    assert result.speech_segments
    assert result.frame_complete
    assert result.frames_classified == result.expected_frames


@pytest.mark.skipif(not SAMPLE.exists(), reason="Run make fetch-vad-samples first")
def test_sample_001_classifies_all_frames():
    """Every expected 30 ms frame is classified (e.g. 258/258)."""
    result = VADPipeline(aggressiveness=2).process_file(str(SAMPLE))
    assert result.expected_frames >= 250
    assert result.frames_classified == result.expected_frames
    assert result.frame_complete is True


def test_vad_on_synthetic_tone_plus_silence(tmp_path: Path):
    sr = 16000
    silence = np.zeros(sr, dtype=np.float32)
    rng = np.random.default_rng(0)
    burst = (rng.normal(0, 0.2, sr).astype(np.float32) * np.hanning(sr)).astype(
        np.float32
    )
    audio = np.concatenate([silence, burst, silence])
    path = tmp_path / "synthetic.wav"
    sf.write(path, audio, sr, subtype="PCM_16")

    clean, out_sr = AudioDenoiser(sr=sr).remove_noise(str(path))
    assert out_sr == sr
    assert float(np.max(np.abs(clean))) > 0.01

    flags = WebRTCVADEngine(aggressiveness=1).detect_speech(clean, sr)
    assert flags.sum() > 0


def test_hangover_fills_short_gaps():
    flags = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 1], dtype=np.int8)
    out = apply_hangover(
        flags, pad_speech_frames=0, fill_gap_frames=2, min_speech_frames=1
    )
    assert out[2] == 1  # gap filled


def test_frame_metrics_perfect():
    y = np.array([1, 1, 0, 0, 1], dtype=np.int8)
    m = frame_metrics(y, y)
    assert m.f1 == 1.0
    assert m.accuracy == 1.0
    assert m.frames == 5


def test_segments_roundtrip():
    segments = [(0.5, 1.5), (2.0, 3.0)]
    flags = segments_to_flags(segments, n_frames=100, frame_duration_ms=30)
    assert flags.sum() > 0
    assert flags[0] == 0


def test_energy_reference_flags_shape():
    sr = 16000
    audio = np.zeros(sr * 2, dtype=np.float32)
    audio[sr : sr + sr // 2] = 0.2
    flags = energy_reference_flags(audio, sr, frame_duration_ms=30)
    assert len(flags) == (2 * sr) // int(sr * 0.03)
    assert flags.sum() > 0


@pytest.mark.skipif(not SAMPLE.exists(), reason="Run make fetch-vad-samples first")
def test_exact_synthetic_f1_gate(tmp_path: Path):
    """Build one exact-labeled clip and require strong F1."""
    y, _ = __import__("librosa").load(SAMPLE, sr=16000, mono=True)
    speech = y[: int(1.5 * 16000)]
    silence = np.zeros(int(0.6 * 16000), dtype=np.float32)
    audio = np.concatenate(
        [silence, speech, silence, speech[: int(1.0 * 16000)], silence]
    )
    path = tmp_path / "exact.wav"
    sf.write(path, audio, 16000, subtype="PCM_16")

    segments = [(0.6, 2.1), (2.7, 3.7)]
    frame_ms = 30
    n_frames = len(audio) // int(16000 * frame_ms / 1000)
    y_true = segments_to_flags(segments, n_frames, frame_ms)

    result = VADPipeline(aggressiveness=2).process_file(str(path), hangover=True)
    n = min(len(y_true), len(result.speech_flags))
    m = frame_metrics(y_true[:n], result.speech_flags[:n])
    assert m.f1 >= 0.85, m
    assert result.frame_complete


@pytest.mark.skipif(not GT_PATH.exists(), reason="Run build_labeled_benchmark.py first")
def test_ground_truth_manifest_valid():
    manifest = json.loads(GT_PATH.read_text())
    assert manifest["n_synthetic_exact"] >= 10
    assert manifest["n_sample_soft"] >= 50
    for rec in manifest["records"][:3]:
        assert len(rec["flags"]) == rec["n_frames"]
