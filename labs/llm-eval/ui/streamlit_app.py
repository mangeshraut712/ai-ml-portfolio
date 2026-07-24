"""Minimal Streamlit dashboard for LLM Evaluation Lab reports."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

REPORT = Path(__file__).resolve().parents[1] / "reports" / "latest_report.json"

st.set_page_config(page_title="LLM Evaluation Lab", layout="wide")
st.title("LLM Evaluation Lab")
st.caption("Retrieval · faithfulness · latency/cost · model comparison")

if not REPORT.exists():
    st.warning("No report yet. Run: `make lab-llm-eval`")
    st.stop()

report = json.loads(REPORT.read_text())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Recall@k", f"{report['retrieval']['recall_at_k']:.3f}")
c2.metric("MRR", f"{report['retrieval']['mrr']:.3f}")
c3.metric("Rerank MRR", f"{report['rerank']['mrr']:.3f}")
c4.metric("Hallucination rate", f"{report['hallucination_rate_adversarial']:.3f}")

st.subheader("Prompt A/B")
st.json(report["prompt_ab"])

st.subheader("Model comparison")
st.dataframe(report["model_comparison"], use_container_width=True)
