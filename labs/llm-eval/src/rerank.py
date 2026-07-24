"""Simple reranking via BM25 fusion with dense/TF-IDF ranks."""

from __future__ import annotations

from ingest import Document
from rank_bm25 import BM25Okapi
from retrieve import SearchHit


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


class BM25Reranker:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.id_to_doc = {d.doc_id: d for d in documents}
        self.corpus_tokens = [_tokenize(d.text) for d in documents]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.doc_ids = [d.doc_id for d in documents]

    def rerank(
        self, query: str, hits: list[SearchHit], top_k: int = 3
    ) -> list[SearchHit]:
        if not hits:
            return []
        q = _tokenize(query)
        bm_scores = self.bm25.get_scores(q)
        bm_map = {doc_id: float(bm_scores[i]) for i, doc_id in enumerate(self.doc_ids)}

        # Rank fusion: reciprocal rank of TF-IDF list + BM25 score normalization
        fused: list[tuple[float, SearchHit]] = []
        max_bm = max((bm_map.get(h.doc_id, 0.0) for h in hits), default=1.0) or 1.0
        for rank, hit in enumerate(hits, start=1):
            rr = 1.0 / rank
            bm = bm_map.get(hit.doc_id, 0.0) / max_bm
            score = 0.6 * rr + 0.4 * bm
            fused.append((score, hit))
        fused.sort(key=lambda x: -x[0])
        out: list[SearchHit] = []
        for score, hit in fused[:top_k]:
            out.append(
                SearchHit(
                    doc_id=hit.doc_id,
                    score=float(score),
                    text=hit.text,
                    title=hit.title,
                )
            )
        return out
