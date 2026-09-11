#!/usr/bin/env python3
"""Render README gallery PNGs from live lab outputs (no mock UI chrome)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
VAD_DIR = REPO / "labs" / "vad"
SAMPLE = REPO / "sample_data" / "sample_audio" / "sample_001.wav"
REPORT = REPO / "labs" / "llm-eval" / "reports" / "latest_report.json"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(VAD_DIR))

plt.rcParams.update(
    {
        "figure.facecolor": "#f6f4ef",
        "axes.facecolor": "#fbfaf7",
        "axes.edgecolor": "#2c3e50",
        "axes.labelcolor": "#1f2a37",
        "xtick.color": "#1f2a37",
        "ytick.color": "#1f2a37",
        "text.color": "#1f2a37",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def render_vad() -> None:
    from pipeline import VADPipeline

    pipe = VADPipeline(aggressiveness=2)
    result = pipe.process_file(str(SAMPLE), frame_duration_ms=30, denoise=True)
    flags = result.speech_flags
    audio = result.clean_audio
    sr = result.sample_rate
    times = np.arange(len(flags)) * (result.frame_duration_ms / 1000.0)
    audio_t = np.arange(len(audio)) / sr

    fig, axes = plt.subplots(2, 1, figsize=(11.2, 5.2), sharex=True)
    axes[0].plot(audio_t, audio, linewidth=0.55, color="#1f4e5f")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(
        f"VAD · {SAMPLE.name} · speech {result.speech_ratio * 100:.1f}% · "
        f"{len(result.speech_segments)} segments · agg=2 · denoise on"
    )
    axes[1].fill_between(times, flags, step="pre", alpha=0.8, color="#c45c26")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_ylabel("Speech")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    dest = OUT / "vad-timeline.png"
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    print(f"wrote {dest}")


def render_llm_eval() -> None:
    report = json.loads(REPORT.read_text())
    retrieval = report["retrieval"]
    rerank = report["rerank"]
    models = report["model_comparison"]
    names = [row["provider"] for row in models]
    faith = [row["mean_faithfulness"] for row in models]
    f1 = [row["mean_token_f1"] for row in models]
    cost = [row["est_cost_usd_per_1k_queries"] for row in models]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios": [1.1, 1.4]})
    fig.suptitle(
        "LLM eval (offline stub) · "
        f"Recall@{retrieval['k']}={retrieval['recall_at_k']:.3f} · "
        f"MRR={retrieval['mrr']:.3f} · "
        f"hallucination={report['hallucination_rate_adversarial']:.3f}",
        fontsize=12,
    )

    axes[0].barh(
        ["Recall@k", "MRR", "nDCG@k", "Rerank MRR"],
        [
            retrieval["recall_at_k"],
            retrieval["mrr"],
            retrieval["ndcg_at_k"],
            rerank["mrr"],
        ],
        color="#1f4e5f",
    )
    axes[0].set_xlim(0, 1.05)
    axes[0].set_title("Retrieval quality")
    axes[0].invert_yaxis()

    x = np.arange(len(names))
    w = 0.36
    axes[1].bar(x - w / 2, faith, w, label="faithfulness", color="#1f4e5f")
    axes[1].bar(x + w / 2, f1, w, label="token F1", color="#c45c26")
    axes[1].set_xticks(x, names, rotation=35, ha="right")
    axes[1].set_ylim(0, 1.15)
    axes[1].set_title("Provider stubs (faithfulness vs F1)")
    ax2 = axes[1].twinx()
    ax2.plot(x, cost, color="#6b4c9a", marker="o", linewidth=1.4, label="$/1k q")
    ax2.set_ylabel("$ / 1k queries")
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    dest = OUT / "llm-eval-dashboard.png"
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    print(f"wrote {dest}")


def render_mlfs() -> None:
    from mlfs.bias_variance import bias_variance_curve
    from mlfs.demos.run_all import _blob
    from mlfs.logistic_regression import LogisticRegression
    from mlfs.metrics import roc_auc_score, roc_curve
    from mlfs.preprocessing import StandardScaler

    X, y = _blob()
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(lr=0.5, n_iter=1500, l2=0.01).fit(Xs, y)
    scores = clf.predict_proba(Xs)
    fpr, tpr = roc_curve(y, scores)
    auc = roc_auc_score(y, scores)
    curve = bias_variance_curve()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    fig.suptitle("ML from scratch · logistic ROC + polynomial bias–variance", fontsize=12)
    axes[0].plot(fpr, tpr, color="#c45c26", linewidth=2.0, label=f"AUC={auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#8a8a8a", linewidth=1)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("Logistic regression ROC")
    axes[0].legend(frameon=False)

    degrees = [row["degree"] for row in curve]
    axes[1].plot(degrees, [row["train_mse"] for row in curve], marker="o", color="#1f4e5f", label="train MSE")
    axes[1].plot(degrees, [row["test_mse"] for row in curve], marker="s", color="#c45c26", label="test MSE")
    axes[1].set_xlabel("Polynomial degree")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Bias–variance curve")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    dest = OUT / "mlfs-roc-bias-variance.png"
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    print(f"wrote {dest}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not SAMPLE.exists():
        raise SystemExit(f"missing sample WAV: {SAMPLE}")
    if not REPORT.exists():
        raise SystemExit(f"missing LLM report: {REPORT}")
    render_vad()
    render_llm_eval()
    render_mlfs()


if __name__ == "__main__":
    main()
