"""Decide whether the retrieved evidence supports answering at all.

This is day 2's rejection threshold wearing different clothes. There, a
classifier's top softmax probability was compared against a bar and anything
below it became "I didn't catch that". Here a reranker's relevance score is
compared against a bar and anything below it becomes "I don't know". The
tradeoff has the same shape and the same uncomfortable property: every point of
refusal accuracy is paid for by refusing something you could have answered.

Two gates, not one, and the order matters:

  1. EVIDENCE   the reranker must score the best passage above the threshold
  2. CITATION   a span inside that passage must verify against the source

The second gate is structural rather than advisory. An answer with no
verifiable citation is not emitted at a lower confidence, it is not emitted —
which is the difference between a system that cites and a system that is asked
nicely to cite.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from convo.projects.day08_rag_cites.answer import Citation

ANSWER = "answer"
REFUSE_NO_EVIDENCE = "refuse: evidence below threshold"
REFUSE_NO_CITATION = "refuse: no span verifies"

REFUSAL_TEXT = (
    "I don't have anything in the help centre that answers that. "
    "The restaurant can tell you directly."
)


@dataclass(frozen=True)
class Evidence:
    """What retrieval produced for one question, before any decision."""

    question: str
    doc_id: str
    chunk_id: str
    score: float
    """Reranker relevance. Higher is better; not a probability."""
    citation: Citation | None

    @property
    def has_citation(self) -> bool:
        return self.citation is not None


@dataclass(frozen=True)
class Decision:
    verdict: str
    reason: str
    answered: bool
    citation: Citation | None = None

    @property
    def text(self) -> str:
        if not self.answered or self.citation is None:
            return REFUSAL_TEXT
        return self.citation.text


def decide(evidence: Evidence, threshold: float) -> Decision:
    """Answer only if the evidence clears the bar AND a span verifies."""
    if evidence.score < threshold:
        return Decision(REFUSE_NO_EVIDENCE, "evidence below threshold", False)
    if not evidence.has_citation:
        return Decision(REFUSE_NO_CITATION, "retrieved but nothing cites", False)
    return Decision(ANSWER, "cited", True, evidence.citation)


@dataclass(frozen=True)
class GroundingMetrics:
    """What one threshold costs and buys, in day 2's vocabulary."""

    threshold: float
    answer_accuracy: float
    """Of answerable questions, the share answered from the right article."""
    refusal_recall: float
    """Of unanswerable questions, the share correctly refused."""
    false_refusal: float
    """Of answerable questions, the share refused anyway."""
    false_answer: float
    """Of unanswerable questions, the share answered regardless."""
    answerable_n: int
    unanswerable_n: int

    @property
    def balanced(self) -> float:
        return round((self.answer_accuracy + self.refusal_recall) / 2, 4)

    def as_dict(self) -> dict:
        return {**asdict(self), "balanced": self.balanced}


def score_threshold(
    answerable: list[tuple[Evidence, str]],
    unanswerable: list[Evidence],
    threshold: float,
) -> GroundingMetrics:
    """Score one threshold.

    `answerable` pairs each evidence with the article that should have been
    cited, so answering from the wrong one counts as wrong rather than as a
    success for having said something.
    """
    answered_right = refused_answerable = 0
    for evidence, expected_doc in answerable:
        decision = decide(evidence, threshold)
        if not decision.answered:
            refused_answerable += 1
        elif decision.citation and decision.citation.doc_id == expected_doc:
            answered_right += 1

    refused_unanswerable = sum(
        1 for evidence in unanswerable if not decide(evidence, threshold).answered
    )

    n_answerable = len(answerable) or 1
    n_unanswerable = len(unanswerable) or 1
    return GroundingMetrics(
        threshold=round(threshold, 4),
        answer_accuracy=round(answered_right / n_answerable, 4),
        refusal_recall=round(refused_unanswerable / n_unanswerable, 4),
        false_refusal=round(refused_answerable / n_answerable, 4),
        false_answer=round((len(unanswerable) - refused_unanswerable) / n_unanswerable, 4),
        answerable_n=len(answerable),
        unanswerable_n=len(unanswerable),
    )


def sweep(
    answerable: list[tuple[Evidence, str]],
    unanswerable: list[Evidence],
    thresholds: list[float] | None = None,
) -> list[GroundingMetrics]:
    """Score a range of thresholds over the reranker's actual score range."""
    if thresholds is None:
        scores = [e.score for e, _ in answerable] + [e.score for e in unanswerable]
        low, high = (min(scores), max(scores)) if scores else (-10.0, 10.0)
        step = (high - low) / 60 if high > low else 1.0
        thresholds = [low + step * i for i in range(61)]
    return [score_threshold(answerable, unanswerable, t) for t in thresholds]


def best_threshold(
    answerable: list[tuple[Evidence, str]],
    unanswerable: list[Evidence],
) -> GroundingMetrics:
    """The threshold with the best balance of answering and refusing.

    Weighted equally, which is a starting point for a conversation with a
    client rather than an answer. A restaurant booking line and a medical
    triage line do not want the same point on this curve.
    """
    return max(sweep(answerable, unanswerable), key=lambda m: m.balanced)
