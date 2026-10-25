"""Two retrievers: BM25, which needs no model, and dense, which needs an artifact.

BM25 is here because a demo you cannot type into is a poor demo, and the hub
carries no ML framework — so free-text questions are served lexically while the
dense side runs over embeddings computed offline. Day 9 turns this pair into a
hybrid and adds a reranker; having both already is the point.

It also makes a useful contrast visible for free. BM25 matches words, so it
cannot connect "how much notice to cancel" to a passage that says "four hours
before the reservation". Dense retrieval can, and misses in different ways.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from convo.projects.day08_rag_cites.chunking import Chunk
from convo.projects.day08_rag_cites.stemming import stem

WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "from",
        "by",
        "as",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "so",
        "such",
        "not",
        "no",
        "nor",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "about",
        "into",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "only",
        "own",
        "same",
        "too",
        "very",
        "s",
        "t",
        "just",
    ]
)


def tokenise(text: str) -> list[str]:
    """Words, minus stopwords, reduced to stems.

    Stemming is applied exactly once per side. The stemmer is not idempotent
    for stems ending in -s, so a second pass anywhere would quietly break
    matching; there is a test pinning that.
    """
    return [stem(w) for w in WORD.findall(text.lower()) if w not in STOPWORDS]


def surface_forms(query: str) -> dict[str, str]:
    """Map each query stem back to the word the reader actually typed.

    Stems are matching keys, not labels. A citation that reports it matched
    "dunda" and "park" tells a reader nothing; one that reports "Dundas" and
    "parking" tells them why the span was chosen. First spelling wins, which
    is the one in the question.
    """
    forms: dict[str, str] = {}
    for word in WORD.findall(query.lower()):
        if word not in STOPWORDS:
            forms.setdefault(stem(word), word)
    return forms


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int
    retriever: str

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "heading": self.chunk.heading,
            "score": round(self.score, 4),
            "rank": self.rank,
            "retriever": self.retriever,
            "start": self.chunk.start,
            "end": self.chunk.end,
        }


@dataclass
class BM25:
    """Okapi BM25 over the chunks. No model, no artifact, works on any query."""

    chunks: list[Chunk]
    k1: float = 1.5
    b: float = 0.75
    _docs: list[list[str]] = field(default_factory=list)
    _freqs: list[Counter] = field(default_factory=list)
    _df: Counter = field(default_factory=Counter)
    _avg_len: float = 0.0

    def __post_init__(self) -> None:
        self._docs = [tokenise(c.text) for c in self.chunks]
        self._freqs = [Counter(d) for d in self._docs]
        self._df = Counter()
        for freq in self._freqs:
            self._df.update(freq.keys())
        lengths = [len(d) for d in self._docs]
        self._avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = self._df.get(term, 0)
        # The +1 keeps a term appearing in every chunk from going negative.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> np.ndarray:
        terms = tokenise(query)
        scores = np.zeros(len(self._docs), dtype=np.float32)
        if not terms or not self._avg_len:
            return scores
        for index, freq in enumerate(self._freqs):
            length = len(self._docs[index])
            total = 0.0
            for term in terms:
                tf = freq.get(term, 0)
                if not tf:
                    continue
                denominator = tf + self.k1 * (1 - self.b + self.b * length / self._avg_len)
                total += self._idf(term) * tf * (self.k1 + 1) / denominator
            scores[index] = total
        return scores

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores = self.score(query)
        # Stable: stemming collapses vocabulary, so exact ties are common now.
        # A tie broken by quicksort's pivot is not reproducible; corpus order is.
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            Hit(self.chunks[i], float(scores[i]), rank, "bm25")
            for rank, i in enumerate(order, start=1)
            if scores[i] > 0
        ]


@dataclass
class DenseIndex:
    """Cosine search over chunk embeddings computed offline.

    Queries must already be embedded, which is why the hub ships a fixed set of
    embedded questions rather than an encoder. The alternative is putting torch
    in the image, which costs 480 MB to answer questions nobody has asked yet.
    """

    chunks: list[Chunk]
    embeddings: np.ndarray
    """(n_chunks, dim), L2-normalised."""

    def __post_init__(self) -> None:
        if len(self.chunks) != len(self.embeddings):
            raise ValueError(f"{len(self.chunks)} chunks but {len(self.embeddings)} embeddings")

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[Hit]:
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm:
            vector = vector / norm
        scores = self.embeddings @ vector
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            Hit(self.chunks[i], float(scores[i]), rank, "dense")
            for rank, i in enumerate(order, start=1)
        ]
