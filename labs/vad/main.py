#!/usr/bin/env python3
"""CLI entry point for the Sarvam AI VAD interview challenge pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python main.py ...` when run from this directory
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pipeline import VADPipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Denoiser + VAD (WebRTC GMM default; neural experimental A/B)."
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to a WAV/audio file (preferably 16 kHz mono PCM).",
    )
    parser.add_argument(
        "--aggressiveness",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="VAD aggressiveness (0=lax, 3=strict). Default: 2",
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=30,
        choices=[10, 20, 30],
        help="Frame duration in milliseconds. Default: 30",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Skip spectral gating and run VAD on raw audio.",
    )
    parser.add_argument(
        "--backend",
        choices=["webrtc", "neural", "compare"],
        default="webrtc",
        help="VAD backend. webrtc=FULL_PASS gate; neural=experimental; compare=A/B.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human text.",
    )
    return parser


def _run_one(args, backend: str) -> dict:
    pipeline = VADPipeline(aggressiveness=args.aggressiveness, backend=backend)
    result = pipeline.process_file(
        str(args.audio_file),
        frame_duration_ms=args.frame_ms,
        denoise=not args.no_denoise,
    )
    total_frames = len(result.speech_flags)
    speech_frames = int(result.speech_flags.sum())
    return {
        "backend": backend,
        "file": str(args.audio_file),
        "sample_rate": result.sample_rate,
        "total_frames": total_frames,
        "speech_frames": speech_frames,
        "speech_percentage": round(result.speech_ratio * 100, 2),
        "expected_frames": result.expected_frames,
        "frames_classified": result.frames_classified,
        "frame_complete": result.frame_complete,
        "hangover": result.hangover,
        "segments": [
            {"start": round(s, 3), "end": round(e, 3)}
            for s, e in result.speech_segments
        ],
    }


def main() -> int:
    args = build_parser().parse_args()
    audio_file = args.audio_file

    if not audio_file.exists():
        print(f"Error: File {audio_file} not found.", file=sys.stderr)
        return 1

    print(f"[*] Processing: {audio_file} (backend={args.backend})...", file=sys.stderr)

    if args.backend == "compare":
        payloads = {
            "webrtc": _run_one(args, "webrtc"),
            "neural": _run_one(args, "neural"),
        }
        # Frame agreement for quick A/B
        from pipeline import VADPipeline

        w = VADPipeline(aggressiveness=args.aggressiveness, backend="webrtc")
        n = VADPipeline(aggressiveness=args.aggressiveness, backend="neural")
        wr = w.process_file(
            str(audio_file),
            frame_duration_ms=args.frame_ms,
            denoise=not args.no_denoise,
        )
        nr = n.process_file(
            str(audio_file),
            frame_duration_ms=args.frame_ms,
            denoise=not args.no_denoise,
        )
        n_frames = min(len(wr.speech_flags), len(nr.speech_flags))
        agree = (
            float((wr.speech_flags[:n_frames] == nr.speech_flags[:n_frames]).mean())
            if n_frames
            else 0.0
        )
        payloads["frame_agreement"] = round(agree, 4)
        payloads["note"] = (
            "neural is experimental A/B; WebRTC aggressiveness=2 remains FULL_PASS"
        )
        if args.json:
            print(json.dumps(payloads, indent=2))
        else:
            print(
                f"[+] WebRTC speech: {payloads['webrtc']['speech_percentage']:.2f}% | "
                f"Neural speech: {payloads['neural']['speech_percentage']:.2f}% | "
                f"frame agreement: {agree:.3f}"
            )
        return 0

    payload = _run_one(args, args.backend)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"[+] Done ({args.backend}). Detected speech in "
            f"{payload['speech_percentage']:.2f}% of the audio timeline "
            f"({payload['speech_frames']}/{payload['total_frames']} frames)."
        )
        if payload["expected_frames"]:
            print(
                f"[+] Frame coverage: {payload['frames_classified']}/"
                f"{payload['expected_frames']} "
                f"({'complete' if payload['frame_complete'] else 'INCOMPLETE'})"
            )
        if payload["segments"]:
            print("[+] Speech segments (sec):")
            for seg in payload["segments"]:
                print(f"    {seg['start']:.2f} → {seg['end']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
