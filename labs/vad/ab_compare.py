#!/usr/bin/env python3
"""Quick WebRTC vs neural VAD A/B on one or more WAVs.

FULL_PASS / challenge_pass.py is unchanged and still uses WebRTC only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pipeline import VADPipeline  # noqa: E402


def compare_file(
    path: Path, aggressiveness: int = 2, frame_ms: int = 30, denoise: bool = True
) -> dict:
    webrtc = VADPipeline(aggressiveness=aggressiveness, backend="webrtc")
    neural = VADPipeline(aggressiveness=aggressiveness, backend="neural")
    wr = webrtc.process_file(str(path), frame_duration_ms=frame_ms, denoise=denoise)
    nr = neural.process_file(str(path), frame_duration_ms=frame_ms, denoise=denoise)
    n = min(len(wr.speech_flags), len(nr.speech_flags))
    agree = (
        float((wr.speech_flags[:n] == nr.speech_flags[:n]).mean()) if n else 0.0
    )
    return {
        "file": str(path),
        "webrtc_speech_pct": round(wr.speech_ratio * 100, 2),
        "neural_speech_pct": round(nr.speech_ratio * 100, 2),
        "neural_backend": getattr(neural.vad_engine, "backend_name", "energy"),
        "frame_agreement": round(agree, 4),
        "frames": n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="WebRTC vs neural VAD A/B")
    parser.add_argument("audio_files", nargs="+", type=Path)
    parser.add_argument("--aggressiveness", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--frame-ms", type=int, default=30, choices=[10, 20, 30])
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = []
    for path in args.audio_files:
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 1
        rows.append(
            compare_file(
                path,
                aggressiveness=args.aggressiveness,
                frame_ms=args.frame_ms,
                denoise=not args.no_denoise,
            )
        )

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print("file | webrtc% | neural% | agreement")
        for r in rows:
            print(
                f"{Path(r['file']).name} | {r['webrtc_speech_pct']:.1f} | "
                f"{r['neural_speech_pct']:.1f} | {r['frame_agreement']:.3f}"
            )
        print(
            "\nNote: neural is experimental A/B; "
            "WebRTC aggressiveness=2 remains the FULL_PASS gate."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
