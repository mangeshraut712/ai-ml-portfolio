#!/usr/bin/env python3
"""Build a labeled VAD benchmark with exact ground-truth frame labels.

Creates synthetic clips by concatenating silence with real speech excerpts so
every frame has an objective label. Also writes energy-based soft labels for
the full sample_audio set (secondary reference when human labels are absent).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from metrics import energy_reference_flags  # noqa: E402
from postprocess import segments_to_flags  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "sample_data" / "vad_labeled"
SAMPLE_DIR = REPO_ROOT / "sample_data" / "sample_audio"
FRAME_MS = 30
SR = 16000


def _speech_excerpt(
    path: Path, duration_sec: float, rng: np.random.Generator
) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    need = int(duration_sec * SR)
    if len(y) < need:
        reps = int(np.ceil(need / max(len(y), 1)))
        y = np.tile(y, reps)
    start = int(rng.integers(0, max(1, len(y) - need)))
    clip = y[start : start + need].astype(np.float32)
    # Light fade to avoid clicks at segment edges
    fade = min(int(0.02 * SR), need // 4)
    if fade > 0:
        clip[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        clip[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return clip


def _silence(duration_sec: float) -> np.ndarray:
    return np.zeros(int(duration_sec * SR), dtype=np.float32)


def _add_noise(
    audio: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    power = np.mean(audio**2) + 1e-12
    noise_power = power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)


def build_synthetic_clip(
    speech_sources: list[Path],
    rng: np.random.Generator,
    snr_db: float | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Return audio + exact speech segments in seconds."""
    src = speech_sources[int(rng.integers(0, len(speech_sources)))]
    pattern = [
        ("silence", 0.9),
        ("speech", 2.0),
        ("silence", 0.8),
        ("speech", 1.6),
        ("silence", 0.7),
        ("speech", 1.2),
        ("silence", 0.8),
    ]
    parts: list[np.ndarray] = []
    segments: list[tuple[float, float]] = []
    t = 0.0
    for kind, dur in pattern:
        if kind == "silence":
            parts.append(_silence(dur))
        else:
            parts.append(_speech_excerpt(src, dur, rng))
            segments.append((t, t + dur))
        t += dur
    audio = np.concatenate(parts)
    if snr_db is not None:
        audio = _add_noise(audio, snr_db, rng)
    return audio, segments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-synthetic", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-ms", type=int, default=FRAME_MS)
    args = parser.parse_args()

    samples = sorted(SAMPLE_DIR.glob("*.wav"))
    if len(samples) < 5:
        print(
            f"Need sample WAVs in {SAMPLE_DIR}. Run make fetch-vad-samples.",
            file=sys.stderr,
        )
        return 1

    out_dir = args.out_dir
    synth_dir = out_dir / "synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    records = []

    # Clean synthetic + noisy variants for robustness
    for i in range(args.n_synthetic):
        snr = None if i < args.n_synthetic // 2 else float(rng.choice([5, 10, 15]))
        audio, segments = build_synthetic_clip(samples, rng, snr_db=snr)
        name = f"synth_{i + 1:03d}.wav"
        path = synth_dir / name
        sf.write(path, audio, SR, subtype="PCM_16")
        frame_len = int(SR * (args.frame_ms / 1000.0))
        n_frames = len(audio) // frame_len
        flags = segments_to_flags(segments, n_frames, args.frame_ms)
        records.append(
            {
                "id": name,
                "path": str(path.relative_to(REPO_ROOT)),
                "split": "synthetic",
                "snr_db": snr,
                "sample_rate": SR,
                "frame_ms": args.frame_ms,
                "n_frames": n_frames,
                "duration_sec": round(len(audio) / SR, 3),
                "segments": [{"start": s, "end": e} for s, e in segments],
                "flags": flags.tolist(),
                "label_type": "exact",
            }
        )

    # Soft energy labels for all real samples (secondary)
    for path in samples:
        y, _ = librosa.load(path, sr=SR, mono=True)
        flags = energy_reference_flags(y, SR, frame_duration_ms=args.frame_ms)
        records.append(
            {
                "id": path.name,
                "path": str(path.relative_to(REPO_ROOT)),
                "split": "sample_soft",
                "snr_db": None,
                "sample_rate": SR,
                "frame_ms": args.frame_ms,
                "n_frames": int(len(flags)),
                "duration_sec": round(len(y) / SR, 3),
                "segments": [],
                "flags": flags.tolist(),
                "label_type": "energy_reference",
            }
        )

    manifest = {
        "frame_ms": args.frame_ms,
        "sample_rate": SR,
        "n_records": len(records),
        "n_synthetic_exact": sum(1 for r in records if r["label_type"] == "exact"),
        "n_sample_soft": sum(
            1 for r in records if r["label_type"] == "energy_reference"
        ),
        "records": records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "ground_truth.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[+] Wrote {manifest_path}")
    print(
        f"    exact synthetic={manifest['n_synthetic_exact']} "
        f"soft sample labels={manifest['n_sample_soft']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
