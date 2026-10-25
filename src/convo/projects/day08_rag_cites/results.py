"""Loads the index, rebuilding chunk text from offsets rather than storing it.

The artifact keeps only `(doc_id, start, end)` per chunk and the text is sliced
back out of the corpus at load time. That is deliberate: it halves the
artifact, and more usefully it means a wrong offset cannot hide. If the
arithmetic were off, the reconstructed chunk would be visibly wrong rather than
quietly disagreeing with a copy stored beside it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from convo.projects.day08_rag_cites.chunking import Chunk, Document, load_corpus
from convo.projects.day08_rag_cites.index import BM25, DenseIndex
from convo.projects.day08_rag_cites.questions import Question

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
INDEX_FILE = ARTIFACT_DIR / "index.npz"
META_FILE = ARTIFACT_DIR / "meta.json"


@dataclass(frozen=True)
class StrategyIndex:
    name: str
    describe: str
    chunks: list[Chunk]
    dense: DenseIndex
    bm25: BM25
    stats: dict[str, Any]


@dataclass(frozen=True)
class RagIndex:
    documents: list[Document]
    strategies: dict[str, StrategyIndex]
    questions: list[Question]
    question_vectors: np.ndarray
    encoder: str

    @property
    def by_id(self) -> dict[str, Document]:
        return {d.doc_id: d for d in self.documents}

    def vector_for(self, text: str) -> np.ndarray | None:
        """The stored embedding for a known question, or None if unseen."""
        for index, question in enumerate(self.questions):
            if question.text == text:
                return self.question_vectors[index]
        return None


@lru_cache(maxsize=1)
def load() -> RagIndex | None:
    """Return the built index, or None if the build has not been run."""
    if not INDEX_FILE.exists() or not META_FILE.exists():
        return None

    arrays = np.load(INDEX_FILE)
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    documents = load_corpus()
    by_id = {d.doc_id: d for d in documents}

    strategies: dict[str, StrategyIndex] = {}
    for name, block in meta["strategies"].items():
        chunks: list[Chunk] = []
        for item in block["items"]:
            document = by_id[item["doc_id"]]
            text = document.span(item["start"], item["end"])
            if not text:
                raise ValueError(
                    f"{item['doc_id']} {item['start']}:{item['end']} slices to nothing "
                    "— the corpus and the artifact disagree, rebuild the index"
                )
            chunks.append(
                Chunk(
                    doc_id=item["doc_id"],
                    text=text,
                    start=item["start"],
                    end=item["end"],
                    strategy=item["strategy"],
                    heading=item.get("heading", ""),
                    index=item["index"],
                )
            )
        embeddings = arrays[f"emb_{name}"]
        strategies[name] = StrategyIndex(
            name=name,
            describe=block["describe"],
            chunks=chunks,
            dense=DenseIndex(chunks, embeddings),
            bm25=BM25(chunks),
            stats={k: v for k, v in block.items() if k != "items"},
        )

    questions = [
        Question(q["text"], q["expect_doc"], q.get("probe", "")) for q in meta["questions"]
    ]
    return RagIndex(
        documents=documents,
        strategies=strategies,
        questions=questions,
        question_vectors=arrays["emb_questions"],
        encoder=meta["encoder"],
    )
