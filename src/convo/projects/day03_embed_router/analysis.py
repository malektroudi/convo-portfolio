"""Segment-wise scoring: the comparison only means anything split three ways.

An overall accuracy number hides the whole result. The classifier is better on
the intents it was trained on and scores exactly zero on the ones it was not,
so a single average just blends a strength and a structural gap into a
meaningless middle. These three segments keep them apart:

  known         the 140 intents both approaches have full data for
  new           the 10 intents withheld from the classifier, few-shot for the router
  out-of-scope  the 1,000 turns both should refuse

Scores come in as (n, 150) over the same canonical class space for both
approaches, so everything here is approach-agnostic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from convo.core.classification import OOS, predict

KNOWN = "known"
NEW = "new"
OUT_OF_SCOPE = "out-of-scope"


@dataclass(frozen=True)
class SegmentMetrics:
    """How one approach does on one slice of the test set."""

    segment: str
    n: int
    correct: float
    """For in-scope segments, the share routed to the right intent. For
    out-of-scope, the share correctly refused."""
    rejected: float
    """Share of the segment refused. Good for out-of-scope, costly elsewhere."""

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def segment_masks(labels: np.ndarray, held_out: list[int]) -> dict[str, np.ndarray]:
    """Boolean masks for the three slices."""
    labels = np.asarray(labels)
    withheld = np.isin(labels, np.asarray(held_out, dtype=labels.dtype))
    in_scope = labels >= 0
    return {
        KNOWN: in_scope & ~withheld,
        NEW: in_scope & withheld,
        OUT_OF_SCOPE: ~in_scope,
    }


def segment_report(
    scores: np.ndarray,
    labels: np.ndarray,
    held_out: list[int],
    threshold: float,
) -> list[SegmentMetrics]:
    """Score all three segments at one threshold."""
    labels = np.asarray(labels)
    predicted, _ = predict(scores, threshold)
    masks = segment_masks(labels, held_out)

    rows: list[SegmentMetrics] = []
    for segment in (KNOWN, NEW, OUT_OF_SCOPE):
        mask = masks[segment]
        n = int(mask.sum())
        if not n:
            rows.append(SegmentMetrics(segment, 0, 0.0, 0.0))
            continue
        rejected = float((predicted[mask] == OOS).sum()) / n
        if segment == OUT_OF_SCOPE:
            correct = rejected
        else:
            correct = float((predicted[mask] == labels[mask]).sum()) / n
        rows.append(SegmentMetrics(segment, n, round(correct, 4), round(rejected, 4)))
    return rows


def best_threshold_for_segments(
    scores: np.ndarray,
    labels: np.ndarray,
    held_out: list[int],
    thresholds: np.ndarray | None = None,
) -> float:
    """The threshold maximising the unweighted mean of the three segment scores.

    Weighting the segments equally is a choice, not a law: it stops the 4,200
    known-intent turns from drowning out the 300 new ones, which are the point
    of the experiment.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.0, 1.001, 0.01), 4)
    best, best_score = 0.0, -1.0
    for threshold in thresholds:
        rows = segment_report(scores, labels, held_out, float(threshold))
        score = sum(r.correct for r in rows) / len(rows)
        if score > best_score:
            best, best_score = float(threshold), score
    return best


def per_new_intent_recall(
    scores: np.ndarray,
    labels: np.ndarray,
    held_out: list[int],
    class_names: list[str],
    threshold: float,
) -> list[tuple[str, float, int]]:
    """(intent name, recall, support) for each withheld intent, worst first.

    Few-shot routing is uneven in a way an average hides: some new intents are
    semantically distinctive and route almost perfectly from five examples,
    while others sit on top of an existing intent and barely route at all.
    """
    labels = np.asarray(labels)
    predicted, _ = predict(scores, threshold)

    rows: list[tuple[str, float, int]] = []
    for canonical in held_out:
        mask = labels == canonical
        support = int(mask.sum())
        if not support:
            continue
        recall = float((predicted[mask] == canonical).sum()) / support
        rows.append((class_names[canonical], round(recall, 4), support))

    rows.sort(key=lambda row: (row[1], row[0]))
    return rows
