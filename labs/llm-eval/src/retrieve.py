"""TF-IDF dense-ish retrieval (offline embedding benchmark baseline)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ingest import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SearchHit:
    doc_id: str
    score: float
    text: str
    title: str


class TfidfRetriever:
    """Acts as the embedding/index baseline without external API calls."""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform([d.text for d in documents])

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        order = np.argsort(-sims)[:top_k]
        hits: list[SearchHit] = []
        for i in order:
            doc = self.documents[int(i)]
            hits.append(
                SearchHit(
                    doc_id=doc.doc_id,
                    score=float(sims[int(i)]),
                    text=doc.text,
                    title=doc.title,
                )
            )
        return hits
