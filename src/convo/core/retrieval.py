"""Ranking metrics: hit@k, MRR and nDCG@k over binary relevance.

In `core` from the start rather than promoted later, unlike the day 2 maths —
these are textbook definitions with nothing day-9-specific in them, and day 20
scores a managed retriever with the same functions so the comparison is one
implementation rather than two that agree by luck.

Binary relevance throughout: a chunk either carries the answer or it does not.
Graded relevance would need a judgement about how partially an almost-right
passage answers, which is exactly the kind of thing that quietly becomes
whatever makes the chart look best.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> bool:
    """Did anything relevant appear in the top k?"""
    if k <= 0:
        raise ValueError("k must be positive")
    return any(item in relevant for item in ranked[:k])


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    """1/rank of the first relevant result, or 0 if none appears.

    The metric that cares where the right answer landed, not merely whether it
    was somewhere in the list. Rank 1 scores 1.0, rank 5 scores 0.2.
    """
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def dcg(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return sum(
        1.0 / math.log2(position + 1)
        for position, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Discounted gain against the best possible ordering.

    Rewards putting *several* relevant chunks high, which hit@k cannot see and
    MRR only half sees — it stops caring once the first one is found.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return round(dcg(ranked, relevant, k) / ideal, 6) if ideal else 0.0


@dataclass(frozen=True)
class RetrievalScore:
    """One retriever's scores over a whole query set."""

    retriever: str
    queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    ndcg_at_5: float
    misses: tuple[str, ...] = ()
    """Queries where nothing relevant appeared at any rank."""

    def as_dict(self) -> dict:
        return {**asdict(self), "misses": list(self.misses)}


def score_runs(
    retriever: str,
    runs: dict[str, tuple[Sequence[str], set[str]]],
    k: int = 5,
) -> RetrievalScore:
    """Aggregate one retriever over `{query: (ranked_ids, relevant_ids)}`."""
    if not runs:
        return RetrievalScore(retriever, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    totals = {1: 0, 3: 0, 5: 0}
    rr = 0.0
    gain = 0.0
    misses: list[str] = []

    for query, (ranked, relevant) in runs.items():
        for cutoff in totals:
            totals[cutoff] += hit_at_k(ranked, relevant, cutoff)
        this_rr = reciprocal_rank(ranked, relevant)
        rr += this_rr
        gain += ndcg_at_k(ranked, relevant, k)
        if this_rr == 0.0:
            misses.append(query)

    n = len(runs)
    return RetrievalScore(
        retriever=retriever,
        queries=n,
        hit_at_1=round(totals[1] / n, 4),
        hit_at_3=round(totals[3] / n, 4),
        hit_at_5=round(totals[5] / n, 4),
        mrr=round(rr / n, 4),
        ndcg_at_5=round(gain / n, 4),
        misses=tuple(sorted(misses)),
    )


def reciprocal_rank_fusion(rankings: dict[str, Sequence[str]], k: int = 60) -> list[str]:
    """Fuse several rankings by 1/(k + rank), highest total first.

    Chosen over blending the raw scores because BM25 and cosine live on
    different scales with no principled conversion between them — any weighting
    would be a knob tuned until the chart improved. RRF only reads positions,
    so it cannot be gamed that way. k=60 is the value from the original paper.
    """
    totals: dict[str, float] = {}
    for ranked in rankings.values():
        for position, item in enumerate(ranked, start=1):
            totals[item] = totals.get(item, 0.0) + 1.0 / (k + position)
    # Sort by score, then by id so ties are broken deterministically.
    return [item for item, _ in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))]
