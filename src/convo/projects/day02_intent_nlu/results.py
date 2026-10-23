"""Loads the cached test-set logits the UI runs on.

The artifact lives inside the package so it installs with the wheel. That is
what lets the hub do the whole threshold analysis with numpy alone — no torch
in the Cloud Run image, which is the difference between fitting the free tier
and not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
LOGITS_FILE = ARTIFACT_DIR / "test_logits.npz"
METADATA_FILE = ARTIFACT_DIR / "metadata.json"


@dataclass(frozen=True)
class Results:
    logits: np.ndarray  # (n_test, 150) float32
    labels: np.ndarray  # (n_test,) int64, OOS == -1
    class_names: list[str]
    metadata: dict

    @property
    def n_test(self) -> int:
        return len(self.labels)


@lru_cache(maxsize=1)
def load() -> Results | None:
    """Return the cached results, or None if training has not been run yet.

    None is a normal state, not an error: CI builds the hub without artifacts,
    and the tab renders an instruction instead of a chart.
    """
    if not LOGITS_FILE.exists() or not METADATA_FILE.exists():
        return None

    bundle = np.load(LOGITS_FILE)
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    class_names = metadata["class_names"]

    logits = bundle["logits"]
    if logits.shape[1] != len(class_names):
        raise ValueError(
            f"artifact mismatch: logits have {logits.shape[1]} classes, "
            f"metadata names {len(class_names)}"
        )
    return Results(
        logits=logits,
        labels=bundle["labels"],
        class_names=class_names,
        metadata=metadata,
    )
