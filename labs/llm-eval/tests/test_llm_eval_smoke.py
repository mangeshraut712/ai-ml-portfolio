"""Smoke tests for LLM Evaluation Lab metrics + pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

from eval_metrics.faithfulness import faithfulness_score, hallucination_on_unanswerable
from eval_metrics.retrieval import mrr, ndcg_at_k, recall_at_k
from ingest import load_corpus
from paths import CORPUS_DIR
from retrieve import TfidfRetriever


def test_recall_mrr_ndcg():
    rel = ["doc_1"]
    retrieved = ["doc_0", "doc_1", "doc_2"]
    assert recall_at_k(rel, retrieved, 3) == 1.0
    assert mrr(rel, retrieved) == 0.5
    assert ndcg_at_k(rel, retrieved, 3) > 0


def test_faithfulness_and_hallucination():
    assert faithfulness_score("I don't know", []) == 1.0
    assert hallucination_on_unanswerable("Paris is the capital.") is True
    assert hallucination_on_unanswerable("I don't know") is False


def test_retriever_finds_sarvam_doc():
    docs = load_corpus(CORPUS_DIR / "indic_ai_faq.md")
    retriever = TfidfRetriever(docs)
    hits = retriever.search("What is Sarvam AI?", top_k=3)
    assert hits
    assert any("Sarvam" in h.title or "Sarvam" in h.text for h in hits)
