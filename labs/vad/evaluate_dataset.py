#!/usr/bin/env python3
"""Batch evaluation helper for the Sarvam VAD challenge dataset."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pipeline import VADPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run VAD across a folder and report aggregate metrics."
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("sample_data/sample_audio"),
        help="Directory containing WAV files.",
    )
    parser.add_argument("--glob", default="*.wav", help="Glob for audio files.")
    parser.add_argument("--aggressiveness", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--frame-ms", type=int, default=30, choices=[10, 20, 30])
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    files = sorted(args.audio_dir.glob(args.glob))
    if not files:
        print(f"No files found at {args.audio_dir}/{args.glob}", file=sys.stderr)
        return 1

    pipeline = VADPipeline(aggressiveness=args.aggressiveness)
    rows = []
    for path in files:
        t0 = time.perf_counter()
        result = pipeline.process_file(
            str(path),
            frame_duration_ms=args.frame_ms,
            denoise=not args.no_denoise,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        duration_sec = len(result.clean_audio) / result.sample_rate
        rtf = elapsed_ms / max(duration_sec * 1000, 1e-9)
        rows.append(
            {
                "file": path.name,
                "speech_percentage": result.speech_ratio * 100,
                "segments": len(result.speech_segments),
                "latency_ms": elapsed_ms,
                "duration_sec": duration_sec,
                "rtf": rtf,
            }
        )

    speech = [r["speech_percentage"] for r in rows]
    lat = [r["latency_ms"] for r in rows]
    rtf = [r["rtf"] for r in rows]

    warm_rows = rows[1:] if len(rows) > 1 else rows
    warm_lat = [r["latency_ms"] for r in warm_rows]
    warm_rtf = [r["rtf"] for r in warm_rows]

    summary = {
        "files": len(rows),
        "mean_speech_percentage": round(statistics.fmean(speech), 2),
        "p10_speech_percentage": round(float(np.percentile(speech, 10)), 2),
        "p90_speech_percentage": round(float(np.percentile(speech, 90)), 2),
        "mean_latency_ms": round(statistics.fmean(lat), 2),
        "p95_latency_ms": round(float(np.percentile(lat, 95)), 2),
        "mean_rtf": round(statistics.fmean(rtf), 4),
        "max_rtf": round(max(rtf), 4),
        "steady_state_mean_latency_ms": round(statistics.fmean(warm_lat), 2),
        "steady_state_p95_latency_ms": round(float(np.percentile(warm_lat, 95)), 2),
        "steady_state_mean_rtf": round(statistics.fmean(warm_rtf), 4),
    }

    sample_001 = next((r for r in rows if r["file"] == "sample_001.wav"), None)
    if sample_001:
        summary["sample_001_speech_percentage"] = round(
            sample_001["speech_percentage"], 2
        )

    if args.json:
        print(json.dumps({"summary": summary, "files": rows}, indent=2))
    else:
        print("[+] Batch Evaluation Summary")
        for key, value in summary.items():
            print(f"    {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
