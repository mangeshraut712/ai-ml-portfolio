"""Corpus ingest: chunk markdown FAQ into numbered documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


def load_corpus(corpus_path: Path) -> list[Document]:
    text = corpus_path.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^# ", text)
    docs: list[Document] = []
    idx = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        docs.append(Document(doc_id=f"doc_{idx}", title=title, text=f"{title}. {body}"))
        idx += 1
    return docs
