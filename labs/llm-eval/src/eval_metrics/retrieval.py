"""Retrieval metrics: Recall@k, MRR, nDCG@k."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def recall_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    hits = sum(1 for r in relevant if r in top)
    return hits / len(relevant)


def mrr(relevant: Sequence[str], retrieved: Sequence[str]) -> float:
    rel = set(relevant)
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    rel = set(relevant)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        gain = 1.0 if doc_id in rel else 0.0
        dcg += gain / math.log2(i + 1)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg else 0.0


def aggregate_retrieval(rows: Iterable[dict], k: int = 5) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {"recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0, "n": 0}
    return {
        "n": len(rows),
        "recall_at_k": sum(r["recall_at_k"] for r in rows) / len(rows),
        "mrr": sum(r["mrr"] for r in rows) / len(rows),
        "ndcg_at_k": sum(r["ndcg_at_k"] for r in rows) / len(rows),
        "k": k,
    }
