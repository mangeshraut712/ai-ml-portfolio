#!/usr/bin/env python3
"""Single gate: fully pass / fail the Sarvam VAD interview challenge criteria."""

from __future__ import annotations

import argparse
import json
import subprocess  # nosec B404
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from evaluate_against_labels import evaluate_record, summarize  # noqa: E402
from pipeline import VADPipeline  # noqa: E402

SAMPLE_DIR = REPO_ROOT / "sample_data" / "sample_audio"
GT_PATH = REPO_ROOT / "sample_data" / "vad_labeled" / "ground_truth.json"
SAMPLE_001 = SAMPLE_DIR / "sample_001.wav"

# Acceptance gates (interview-grade, evidence-backed)
GATES = {
    "min_exact_clean_mean_f1": 0.92,
    "min_exact_clean_mean_accuracy": 0.90,
    "min_exact_clean_min_f1": 0.90,
    "min_exact_noisy_mean_f1": 0.75,
    "min_soft_mean_f1": 0.78,
    "min_batch_files": 50,
    "min_batch_mean_speech_pct": 70.0,
    "max_steady_p95_latency_ms": 100.0,
    "max_steady_mean_rtf": 0.05,
    "require_sample_001_frame_complete": True,
    "require_pytest": True,
}


def run_pytest() -> tuple[bool, str]:
    # Fixed argv: interpreter + local test path only (no user-controlled input).
    proc = subprocess.run(  # nosec B603
        [
            sys.executable,
            "-m",
            "pytest",
            str(_HERE / "test_vad_pipeline.py"),
            "-q",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    ok = proc.returncode == 0
    tail = (proc.stdout or "") + (proc.stderr or "")
    return ok, tail.strip().splitlines()[-5:] and "\n".join(
        tail.strip().splitlines()[-8:]
    )


def ensure_ground_truth() -> None:
    if GT_PATH.exists():
        return
    # Fixed argv: interpreter + local script path only.
    proc = subprocess.run(  # nosec B603
        [sys.executable, str(_HERE / "build_labeled_benchmark.py")],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "benchmark build failed")


def batch_sanity(pipeline: VADPipeline) -> dict:
    files = sorted(SAMPLE_DIR.glob("*.wav"))
    rows = []
    for path in files:
        t0 = time.perf_counter()
        result = pipeline.process_file(str(path))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        duration = len(result.clean_audio) / result.sample_rate
        rows.append(
            {
                "file": path.name,
                "speech_pct": result.speech_ratio * 100,
                "latency_ms": elapsed_ms,
                "rtf": elapsed_ms / max(duration * 1000, 1e-9),
                "frame_complete": result.frame_complete,
                "expected_frames": result.expected_frames,
                "frames_classified": result.frames_classified,
            }
        )
    steady = rows[1:] if len(rows) > 1 else rows
    sample_001 = next((r for r in rows if r["file"] == "sample_001.wav"), None)
    return {
        "files": len(rows),
        "mean_speech_pct": (
            float(np.mean([r["speech_pct"] for r in rows])) if rows else 0.0
        ),
        "steady_p95_latency_ms": (
            float(np.percentile([r["latency_ms"] for r in steady], 95))
            if steady
            else 0.0
        ),
        "steady_mean_rtf": (
            float(np.mean([r["rtf"] for r in steady])) if steady else 0.0
        ),
        "frame_complete_rate": (
            float(np.mean([1.0 if r["frame_complete"] else 0.0 for r in rows]))
            if rows
            else 0.0
        ),
        "sample_001": sample_001,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggressiveness", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()

    checks: list[dict] = []
    ensure_ground_truth()
    pipeline = VADPipeline(aggressiveness=args.aggressiveness)

    # 1) Labeled F1 (exact synthetic + soft sample labels)
    manifest = json.loads(GT_PATH.read_text())
    labeled_rows = [
        evaluate_record(pipeline, rec, hangover=True) for rec in manifest["records"]
    ]
    exact = summarize(labeled_rows, "exact")
    soft = summarize(labeled_rows, "energy_reference")
    exact_clean_rows = [
        r
        for r, rec in zip(labeled_rows, manifest["records"])
        if rec.get("label_type") == "exact" and rec.get("snr_db") is None
    ]
    exact_noisy_rows = [
        r
        for r, rec in zip(labeled_rows, manifest["records"])
        if rec.get("label_type") == "exact" and rec.get("snr_db") is not None
    ]
    exact_clean = {
        "n": len(exact_clean_rows),
        "mean_f1": (
            round(float(np.mean([r["f1"] for r in exact_clean_rows])), 4)
            if exact_clean_rows
            else 0.0
        ),
        "mean_accuracy": (
            round(float(np.mean([r["accuracy"] for r in exact_clean_rows])), 4)
            if exact_clean_rows
            else 0.0
        ),
        "min_f1": (
            round(float(np.min([r["f1"] for r in exact_clean_rows])), 4)
            if exact_clean_rows
            else 0.0
        ),
    }
    exact_noisy = {
        "n": len(exact_noisy_rows),
        "mean_f1": (
            round(float(np.mean([r["f1"] for r in exact_noisy_rows])), 4)
            if exact_noisy_rows
            else 0.0
        ),
        "mean_accuracy": (
            round(float(np.mean([r["accuracy"] for r in exact_noisy_rows])), 4)
            if exact_noisy_rows
            else 0.0
        ),
        "min_f1": (
            round(float(np.min([r["f1"] for r in exact_noisy_rows])), 4)
            if exact_noisy_rows
            else 0.0
        ),
    }

    checks.append(
        {
            "name": "exact_clean_mean_f1",
            "value": exact_clean["mean_f1"],
            "threshold": GATES["min_exact_clean_mean_f1"],
            "op": ">=",
            "pass": exact_clean["mean_f1"] >= GATES["min_exact_clean_mean_f1"],
        }
    )
    checks.append(
        {
            "name": "exact_clean_mean_accuracy",
            "value": exact_clean["mean_accuracy"],
            "threshold": GATES["min_exact_clean_mean_accuracy"],
            "op": ">=",
            "pass": exact_clean["mean_accuracy"]
            >= GATES["min_exact_clean_mean_accuracy"],
        }
    )
    checks.append(
        {
            "name": "exact_clean_min_f1",
            "value": exact_clean["min_f1"],
            "threshold": GATES["min_exact_clean_min_f1"],
            "op": ">=",
            "pass": exact_clean["min_f1"] >= GATES["min_exact_clean_min_f1"],
        }
    )
    checks.append(
        {
            "name": "exact_noisy_mean_f1",
            "value": exact_noisy["mean_f1"],
            "threshold": GATES["min_exact_noisy_mean_f1"],
            "op": ">=",
            "pass": exact_noisy["mean_f1"] >= GATES["min_exact_noisy_mean_f1"],
        }
    )
    checks.append(
        {
            "name": "soft_mean_f1",
            "value": soft.get("mean_f1", 0.0),
            "threshold": GATES["min_soft_mean_f1"],
            "op": ">=",
            "pass": soft.get("mean_f1", 0.0) >= GATES["min_soft_mean_f1"],
        }
    )

    # 2) Batch + frame completeness (all frames classified, e.g. 258/258)
    batch = batch_sanity(pipeline)
    checks.append(
        {
            "name": "batch_files",
            "value": batch["files"],
            "threshold": GATES["min_batch_files"],
            "op": ">=",
            "pass": batch["files"] >= GATES["min_batch_files"],
        }
    )
    checks.append(
        {
            "name": "batch_mean_speech_pct",
            "value": round(batch["mean_speech_pct"], 2),
            "threshold": GATES["min_batch_mean_speech_pct"],
            "op": ">=",
            "pass": batch["mean_speech_pct"] >= GATES["min_batch_mean_speech_pct"],
        }
    )
    checks.append(
        {
            "name": "steady_p95_latency_ms",
            "value": round(batch["steady_p95_latency_ms"], 2),
            "threshold": GATES["max_steady_p95_latency_ms"],
            "op": "<=",
            "pass": batch["steady_p95_latency_ms"]
            <= GATES["max_steady_p95_latency_ms"],
        }
    )
    checks.append(
        {
            "name": "steady_mean_rtf",
            "value": round(batch["steady_mean_rtf"], 4),
            "threshold": GATES["max_steady_mean_rtf"],
            "op": "<=",
            "pass": batch["steady_mean_rtf"] <= GATES["max_steady_mean_rtf"],
        }
    )
    checks.append(
        {
            "name": "batch_frame_complete_rate",
            "value": batch["frame_complete_rate"],
            "threshold": 1.0,
            "op": "==",
            "pass": batch["frame_complete_rate"] == 1.0,
        }
    )

    s001 = batch.get("sample_001")
    if GATES["require_sample_001_frame_complete"]:
        ok = bool(s001 and s001["frame_complete"])
        checks.append(
            {
                "name": "sample_001_frame_complete",
                "value": {
                    "expected_frames": s001["expected_frames"] if s001 else None,
                    "frames_classified": s001["frames_classified"] if s001 else None,
                    "speech_pct": round(s001["speech_pct"], 2) if s001 else None,
                },
                "threshold": "expected_frames == frames_classified (e.g. 258/258)",
                "op": "==",
                "pass": ok,
            }
        )

    # 3) Pytest
    if GATES["require_pytest"] and not args.skip_pytest:
        pytest_ok, pytest_tail = run_pytest()
        checks.append(
            {
                "name": "pytest",
                "value": pytest_tail,
                "threshold": "exit 0",
                "op": "==",
                "pass": pytest_ok,
            }
        )

    passed = all(c["pass"] for c in checks)
    report = {
        "status": "FULL_PASS" if passed else "FAIL",
        "challenge_goal": (
            "Build VAD from scratch (no external APIs), prove accuracy with "
            "labeled F1, code quality, latency awareness, and defendable trade-offs."
        ),
        "aggressiveness": args.aggressiveness,
        "gates": GATES,
        "exact_summary": exact,
        "exact_clean_summary": exact_clean,
        "exact_noisy_summary": exact_noisy,
        "soft_summary": soft,
        "batch_summary": {k: v for k, v in batch.items() if k != "rows"},
        "checks": checks,
        "missing_closed": [
            "Exact ground-truth synthetic benchmark + F1/precision/recall",
            "Energy soft labels for all 50 sample WAVs",
            "Hangover post-processing for onset/offset recovery",
            "Frame-complete classification (all expected frames scored)",
            "Acceptance gates in challenge_pass.py",
            "Noise-robust synthetic variants in labeled set",
        ],
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 60)
        print(f"SARVAM VAD CHALLENGE: {report['status']}")
        print("=" * 60)
        for c in checks:
            mark = "PASS" if c["pass"] else "FAIL"
            print(f"  [{mark}] {c['name']}: {c['value']} ({c['op']} {c['threshold']})")
        print("-" * 60)
        print("Exact labels:", exact)
        print("Exact clean labels:", exact_clean)
        print("Exact noisy labels:", exact_noisy)
        print("Soft labels:", soft)
        if s001:
            print(
                f"sample_001: {s001['frames_classified']}/{s001['expected_frames']} "
                f"frames classified | speech={s001['speech_pct']:.2f}%"
            )
        print("=" * 60)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
