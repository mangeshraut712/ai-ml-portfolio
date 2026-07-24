"""Shared paths for the LLM Evaluation Lab."""

from __future__ import annotations

from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parents[1]  # labs/llm-eval → monorepo root
DATA_DIR = LAB_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
GOLD_DIR = DATA_DIR / "gold"
CONFIG_DIR = LAB_ROOT / "configs"
REPORT_DIR = LAB_ROOT / "reports"
