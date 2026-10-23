"""Threshold and confusion analysis. Pure numpy — no torch, no model.

This is the whole reason day 2 is shaped the way it is. The classifier is
trained on the 150 in-scope intents ONLY, because in production you never get
representative out-of-scope training data. Out-of-scope is then rejected at
inference by thresholding the max softmax probability.

That makes the threshold a product decision, not a hyperparameter: every point
you gain in out-of-scope recall costs you in-scope accuracy. Sweeping it over
cached test logits is instant, which is what lets the UI put a slider on it.

Labels use OOS (-1) for an out-of-scope example; anything >= 0 is a model class
index.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

OOS = -1
"""Sentinel label for an out-of-scope example."""


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, shifted for numerical stability."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def predict(logits: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted class or OOS, confidence) per row.

    A row whose top probability falls below `threshold` is rejected: its
    prediction becomes OOS, which is exactly what a bot's fallback does.
    """
    probs = softmax(logits)
    top = probs.argmax(axis=1)
    confidence = probs[np.arange(len(probs)), top]
    predicted = np.where(confidence >= threshold, top, OOS)
    return predicted, confidence


@dataclass(frozen=True)
class Metrics:
    """What one threshold costs and buys. Rates are 0..1."""

    threshold: float
    in_scope_accuracy: float
    oos_recall: float
    false_rejection_rate: float
    overall_accuracy: float
    in_scope_n: int
    oos_n: int

    @property
    def balanced(self) -> float:
        """Unweighted mean of the two rates that trade against each other."""
        return round((self.in_scope_accuracy + self.oos_recall) / 2, 4)

    def as_dict(self) -> dict[str, float | int]:
        return {**asdict(self), "balanced": self.balanced}


def metrics_at(logits: np.ndarray, labels: np.ndarray, threshold: float) -> Metrics:
    """Score one threshold.

    A rejected in-scope example counts as wrong, not as abstained. Scoring it
    any other way lets a bot look good by refusing to answer anything.
    """
    labels = np.asarray(labels)
    predicted, _ = predict(logits, threshold)

    in_scope = labels >= 0
    oos = ~in_scope
    n_in, n_oos = int(in_scope.sum()), int(oos.sum())

    correct_in = int((predicted[in_scope] == labels[in_scope]).sum()) if n_in else 0
    rejected_in = int((predicted[in_scope] == OOS).sum()) if n_in else 0
    rejected_oos = int((predicted[oos] == OOS).sum()) if n_oos else 0

    total = n_in + n_oos
    return Metrics(
        threshold=round(float(threshold), 4),
        in_scope_accuracy=round(correct_in / n_in, 4) if n_in else 0.0,
        oos_recall=round(rejected_oos / n_oos, 4) if n_oos else 0.0,
        false_rejection_rate=round(rejected_in / n_in, 4) if n_in else 0.0,
        overall_accuracy=round((correct_in + rejected_oos) / total, 4) if total else 0.0,
        in_scope_n=n_in,
        oos_n=n_oos,
    )


def sweep(
    logits: np.ndarray, labels: np.ndarray, thresholds: np.ndarray | None = None
) -> list[Metrics]:
    """Score a range of thresholds. Cheap enough to redo on every slider move."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.0, 1.001, 0.01), 4)
    return [metrics_at(logits, labels, float(t)) for t in thresholds]


def best_threshold(logits: np.ndarray, labels: np.ndarray, objective: str = "balanced") -> Metrics:
    """The threshold that maximises `objective`.

    "balanced" weights in-scope accuracy and out-of-scope recall equally, which
    is a starting point for a conversation with a client, not an answer.
    """
    rows = sweep(logits, labels)
    if objective == "balanced":
        key = lambda m: m.balanced  # noqa: E731
    elif objective == "overall":
        key = lambda m: m.overall_accuracy  # noqa: E731
    else:
        raise ValueError(f"unknown objective: {objective!r}")
    return max(rows, key=key)


@dataclass(frozen=True)
class ConfidenceStats:
    """How sharp the model's probabilities are.

    Worth surfacing because an undertrained 150-way classifier spreads mass
    across every class: the first run of this project reached a max softmax of
    0.09 over the whole test set, which makes every threshold above 0.1 reject
    everything. The accuracy number alone does not show that, and a threshold
    slider over such a model is a dead control.
    """

    mean: float
    p50: float
    p95: float
    maximum: float

    @property
    def is_degenerate(self) -> bool:
        """True when the threshold has no usable range worth exposing."""
        return self.maximum < 0.5

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def confidence_stats(logits: np.ndarray) -> ConfidenceStats:
    """Summarise the top-probability distribution."""
    confidence = softmax(logits).max(axis=1)
    return ConfidenceStats(
        mean=round(float(confidence.mean()), 4),
        p50=round(float(np.percentile(confidence, 50)), 4),
        p95=round(float(np.percentile(confidence, 95)), 4),
        maximum=round(float(confidence.max()), 4),
    )


@dataclass(frozen=True)
class ConfusionPair:
    """One recurring mistake, named. This is what you actually act on."""

    true_intent: str
    predicted_intent: str
    count: int

    @property
    def is_rejection(self) -> bool:
        return self.predicted_intent == "(rejected)"


def confusion_pairs(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    threshold: float,
    top_k: int = 15,
    include_rejections: bool = False,
) -> list[ConfusionPair]:
    """The most frequent mistakes at this threshold, worst first.

    A 150x150 confusion matrix is unreadable, so the actionable view is a list
    of the pairs the model actually mixes up.
    """
    labels = np.asarray(labels)
    predicted, _ = predict(logits, threshold)
    in_scope = labels >= 0

    counts: dict[tuple[int, int], int] = {}
    for true_label, pred_label in zip(labels[in_scope], predicted[in_scope], strict=True):
        if true_label == pred_label:
            continue
        if pred_label == OOS and not include_rejections:
            continue
        key = (int(true_label), int(pred_label))
        counts[key] = counts.get(key, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        ConfusionPair(
            true_intent=class_names[true_label],
            predicted_intent="(rejected)" if pred_label == OOS else class_names[pred_label],
            count=count,
        )
        for (true_label, pred_label), count in ranked[:top_k]
    ]


def worst_intents(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    threshold: float,
    top_k: int = 12,
) -> list[tuple[int, str, float, int]]:
    """(class index, name, per-intent recall, support), worst recall first."""
    labels = np.asarray(labels)
    predicted, _ = predict(logits, threshold)

    rows: list[tuple[int, str, float, int]] = []
    for index in range(len(class_names)):
        mask = labels == index
        support = int(mask.sum())
        if not support:
            continue
        recall = float((predicted[mask] == index).sum()) / support
        rows.append((index, class_names[index], round(recall, 4), support))

    rows.sort(key=lambda row: (row[2], row[1]))
    return rows[:top_k]


def confusion_submatrix(
    logits: np.ndarray,
    labels: np.ndarray,
    class_indices: list[int],
    threshold: float,
) -> np.ndarray:
    """A readable confusion matrix over a chosen subset of intents.

    Shape is (len(class_indices), len(class_indices) + 1); the extra final
    column counts rejections, so every row sums to that intent's support.
    """
    labels = np.asarray(labels)
    predicted, _ = predict(logits, threshold)

    size = len(class_indices)
    position = {index: slot for slot, index in enumerate(class_indices)}
    matrix = np.zeros((size, size + 1), dtype=int)

    for slot, index in enumerate(class_indices):
        mask = labels == index
        for pred_label in predicted[mask]:
            pred_label = int(pred_label)
            if pred_label == OOS:
                matrix[slot, size] += 1
            elif pred_label in position:
                matrix[slot, position[pred_label]] += 1
            # A prediction outside the subset is deliberately not counted here;
            # confusion_pairs() is the view that catches those.
    return matrix
