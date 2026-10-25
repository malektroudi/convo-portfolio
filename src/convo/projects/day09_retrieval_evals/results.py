"""Loads the stored rankings and recomputes metrics from them.

Metrics live here rather than in the artifact on purpose. Storing rankings and
deriving scores means the cutoffs can change without a rebuild, and the number
in the README is produced by the same code path the tab renders — so a figure
cannot drift from the thing it claims to describe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from convo.core.retrieval import RetrievalScore, score_runs

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
RANKINGS_FILE = ARTIFACT_DIR / "rankings.json"

RETRIEVERS = ("bm25", "dense", "hybrid", "hybrid+rerank")
RETRIEVER_NOTE = {
    "bm25": "Lexical only. No model; matches words.",
    "dense": "Cosine over MiniLM embeddings. Matches meaning, misses exact terms.",
    "hybrid": "Reciprocal rank fusion of the two, positions only, no score blending.",
    "hybrid+rerank": (
        "A cross-encoder reorders the fused top 10, reading query and passage "
        "together rather than comparing two vectors."
    ),
}


@dataclass(frozen=True)
class QuestionResult:
    question: str
    relevant: set[str]
    rankings: dict[str, list[str]]
    hard: bool = False


@dataclass(frozen=True)
class StrategyResults:
    name: str
    describe: str
    chunks: int
    questions: list[QuestionResult]

    def runs_for(self, retriever: str) -> dict[str, tuple[list[str], set[str]]]:
        return {q.question: (q.rankings.get(retriever, []), q.relevant) for q in self.questions}

    def score(self, retriever: str, k: int = 5) -> RetrievalScore:
        return score_runs(retriever, self.runs_for(retriever), k)

    def score_subset(self, retriever: str, hard: bool) -> RetrievalScore:
        runs = {
            q.question: (q.rankings.get(retriever, []), q.relevant)
            for q in self.questions
            if q.hard == hard
        }
        return score_runs(retriever, runs)


@dataclass(frozen=True)
class Day9Results:
    strategies: dict[str, StrategyResults]
    encoder: str
    reranker: str
    depth: int
    rerank_ms_median: float
    meta: dict[str, Any]

    def table(self, strategy: str) -> list[RetrievalScore]:
        return [self.strategies[strategy].score(r) for r in RETRIEVERS]


@lru_cache(maxsize=1)
def load() -> Day9Results | None:
    """Return the stored rankings, or None if the build has not been run."""
    if not RANKINGS_FILE.exists():
        return None
    payload = json.loads(RANKINGS_FILE.read_text(encoding="utf-8"))
    hard_by_question = {q["question"]: q["hard"] for q in payload["questions"]}

    strategies: dict[str, StrategyResults] = {}
    for name, block in payload["strategies"].items():
        questions = [
            QuestionResult(
                question=row["question"],
                relevant=set(row["relevant"]),
                rankings={k: list(v) for k, v in row["rankings"].items()},
                hard=hard_by_question.get(row["question"], False),
            )
            for row in block["results"]
        ]
        if any(not q.relevant for q in questions):
            raise ValueError(
                f"{name}: a question has no relevant chunk, so its score is "
                "meaningless — rebuild the rankings"
            )
        strategies[name] = StrategyResults(
            name=name,
            describe=block["describe"],
            chunks=block["chunks"],
            questions=questions,
        )

    return Day9Results(
        strategies=strategies,
        encoder=payload["encoder"],
        reranker=payload["reranker"],
        depth=payload["depth"],
        rerank_ms_median=payload["rerank_ms_median"],
        meta={"questions": payload["questions"]},
    )
