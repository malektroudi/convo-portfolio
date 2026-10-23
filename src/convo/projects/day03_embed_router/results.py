"""Loads the cached comparison the day 3 tab runs on.

Same pattern as day 2: the benchmark does the expensive work once and commits
score matrices, so the tab is numpy over a 150-column array and the Cloud Run
image still needs no torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
COMPARISON_FILE = ARTIFACT_DIR / "comparison.npz"
METADATA_FILE = ARTIFACT_DIR / "metadata.json"


@dataclass(frozen=True)
class Comparison:
    router_scores: np.ndarray  # (n, 150) cosine similarity
    classifier_scores: np.ndarray  # (n, 150) softmax, zero in held-out columns
    labels: np.ndarray  # (n,) canonical class index, OOS == -1
    held_out: list[int]
    class_names: list[str]
    metadata: dict

    @property
    def n_test(self) -> int:
        return len(self.labels)

    @property
    def held_out_names(self) -> list[str]:
        return [self.class_names[i] for i in self.held_out]


@lru_cache(maxsize=1)
def load() -> Comparison | None:
    """Return the cached comparison, or None if the benchmark has not been run."""
    if not COMPARISON_FILE.exists() or not METADATA_FILE.exists():
        return None

    bundle = np.load(COMPARISON_FILE)
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    class_names = metadata["class_names"]

    router = bundle["router_scores"]
    classifier = bundle["classifier_scores"]
    if router.shape != classifier.shape:
        raise ValueError(
            f"score matrices disagree: router {router.shape}, "
            f"classifier {classifier.shape} — they must share one class space"
        )
    if router.shape[1] != len(class_names):
        raise ValueError(
            f"artifact mismatch: scores have {router.shape[1]} classes, "
            f"metadata names {len(class_names)}"
        )
    return Comparison(
        router_scores=router,
        classifier_scores=classifier,
        labels=bundle["labels"],
        held_out=[int(i) for i in bundle["held_out"]],
        class_names=class_names,
        metadata=metadata,
    )
