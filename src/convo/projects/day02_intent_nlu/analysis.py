"""Day 2's view of the shared scoring layer: logits in, metrics out.

The maths lives in `convo.core.classification`, which works on any score array
whose row maximum is a comparable confidence. This module is the adapter that
turns raw classifier logits into such an array by softmaxing them, so day 2 and
day 3 are measured by exactly the same code.

The design decision this encodes: the classifier is trained on the 150 in-scope
intents ONLY, because in production you never get representative out-of-scope
training data. Out-of-scope is rejected at inference by thresholding the max
softmax probability, which makes the threshold a product decision rather than a
hyperparameter — every point of out-of-scope recall is paid for in in-scope
accuracy.
"""

from __future__ import annotations

import numpy as np

from convo.core import classification as cls
from convo.core.classification import (
    OOS,
    ConfidenceStats,
    ConfusionPair,
    Metrics,
    softmax,
)

__all__ = [
    "OOS",
    "ConfidenceStats",
    "ConfusionPair",
    "Metrics",
    "best_threshold",
    "confidence_stats",
    "confusion_pairs",
    "confusion_submatrix",
    "metrics_at",
    "predict",
    "softmax",
    "sweep",
    "worst_intents",
]


def predict(logits: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted class or OOS, softmax confidence) per row."""
    return cls.predict(softmax(logits), threshold)


def metrics_at(logits: np.ndarray, labels: np.ndarray, threshold: float) -> Metrics:
    """Score one threshold against the test labels."""
    return cls.metrics_at(softmax(logits), labels, threshold)


def sweep(
    logits: np.ndarray, labels: np.ndarray, thresholds: np.ndarray | None = None
) -> list[Metrics]:
    """Score a range of thresholds; instant over cached logits."""
    return cls.sweep(softmax(logits), labels, thresholds)


def best_threshold(logits: np.ndarray, labels: np.ndarray, objective: str = "balanced") -> Metrics:
    """The threshold maximising `objective` ("balanced" or "overall")."""
    return cls.best_threshold(softmax(logits), labels, objective)


def confidence_stats(logits: np.ndarray) -> ConfidenceStats:
    """Summarise the max-softmax distribution; flags an unsharpened model."""
    return cls.confidence_stats(softmax(logits))


def confusion_pairs(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    threshold: float,
    top_k: int = 15,
    include_rejections: bool = False,
) -> list[ConfusionPair]:
    """The most frequent mistakes at this threshold, worst first."""
    return cls.confusion_pairs(
        softmax(logits), labels, class_names, threshold, top_k, include_rejections
    )


def worst_intents(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    threshold: float,
    top_k: int = 12,
) -> list[tuple[int, str, float, int]]:
    """(class index, name, per-intent recall, support), worst recall first."""
    return cls.worst_intents(softmax(logits), labels, class_names, threshold, top_k)


def confusion_submatrix(
    logits: np.ndarray,
    labels: np.ndarray,
    class_indices: list[int],
    threshold: float,
) -> np.ndarray:
    """A readable confusion matrix over a chosen subset of intents."""
    return cls.confusion_submatrix(softmax(logits), labels, class_indices, threshold)
