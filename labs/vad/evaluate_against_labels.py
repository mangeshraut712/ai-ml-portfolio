#!/usr/bin/env python3
"""Evaluate the VAD pipeline against labeled ground truth (exact + soft)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from metrics import frame_metrics  # noqa: E402
from pipeline import VADPipeline  # noqa: E402

DEFAULT_GT = REPO_ROOT / "sample_data" / "vad_labeled" / "ground_truth.json"


def evaluate_record(pipeline: VADPipeline, record: dict, hangover: bool) -> dict:
    path = REPO_ROOT / record["path"]
    result = pipeline.process_file(
        str(path),
        frame_duration_ms=record["frame_ms"],
        hangover=hangover,
    )
    y_true = np.asarray(record["flags"], dtype=np.int8)
    y_pred = result.speech_flags

    # Align lengths (tail residual samples may differ by 0–1 frame)
    n = min(len(y_true), len(y_pred))
    metrics = frame_metrics(y_true[:n], y_pred[:n])
    return {
        "id": record["id"],
        "split": record["split"],
        "label_type": record["label_type"],
        "snr_db": record.get("snr_db"),
        "frame_complete": result.frame_complete,
        "expected_frames": result.expected_frames,
        "frames_classified": result.frames_classified,
        "frames_scored": n,
        **metrics.to_dict(),
    }


def summarize(rows: list[dict], label_type: str | None = None) -> dict:
    subset = [r for r in rows if label_type is None or r["label_type"] == label_type]
    if not subset:
        return {}
    return {
        "n": len(subset),
        "mean_f1": round(float(np.mean([r["f1"] for r in subset])), 4),
        "mean_accuracy": round(float(np.mean([r["accuracy"] for r in subset])), 4),
        "mean_precision": round(float(np.mean([r["precision"] for r in subset])), 4),
        "mean_recall": round(float(np.mean([r["recall"] for r in subset])), 4),
        "min_f1": round(float(np.min([r["f1"] for r in subset])), 4),
        "frame_complete_rate": round(
            float(np.mean([1.0 if r["frame_complete"] else 0.0 for r in subset])), 4
        ),
        "total_frames_scored": int(sum(r["frames_scored"] for r in subset)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--aggressiveness", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--no-hangover", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--split",
        choices=["all", "synthetic", "sample_soft"],
        default="all",
    )
    args = parser.parse_args()

    if not args.ground_truth.exists():
        print(
            f"Missing {args.ground_truth}. Run: "
            "python examples/sarvam-vad-challenge/build_labeled_benchmark.py",
            file=sys.stderr,
        )
        return 1

    manifest = json.loads(args.ground_truth.read_text())
    records = manifest["records"]
    if args.split != "all":
        records = [r for r in records if r["split"] == args.split]

    pipeline = VADPipeline(aggressiveness=args.aggressiveness)
    rows = [
        evaluate_record(pipeline, rec, hangover=not args.no_hangover) for rec in records
    ]

    report = {
        "aggressiveness": args.aggressiveness,
        "hangover": not args.no_hangover,
        "summary_exact": summarize(rows, "exact"),
        "summary_soft": summarize(rows, "energy_reference"),
        "summary_all": summarize(rows),
        "files": rows,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("[+] Labeled evaluation")
        for key in ("summary_exact", "summary_soft", "summary_all"):
            block = report[key]
            if not block:
                continue
            print(f"  {key}:")
            for k, v in block.items():
                print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
