"""An embedding router: no training, no output layer, no retraining to extend.

Pure numpy over precomputed embeddings. Intents are represented by the vectors
of their example utterances, and routing is nearest-neighbour by cosine
similarity. Because the similarity is already a bounded confidence, it drops
straight into `convo.core.classification` and is scored by the same code as day
2's fine-tuned classifier.

The point of day 3 is the asymmetry this creates. A classifier's output layer
has a fixed width: a 151st intent means a new head and a full retrain. A router
has no output layer, so `add_intent` is an array append — which is the whole
argument for not fine-tuning until you have to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

QUERY_CHUNK = 512
"""Rows scored per batch. 5,500 queries x 15,000 exemplars would be 330 MB at
float32 in one go, so the similarity matrix is built in slices."""


def l2_normalize(vectors: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Scale rows to unit length, so a dot product IS the cosine similarity.

    A zero vector stays zero rather than producing NaN, which matters because a
    degenerate embedding should score 0 against everything, not poison the row.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, epsilon)


@dataclass
class EmbeddingRouter:
    """Routes an utterance to an intent by similarity to that intent's examples.

    `strategy` picks how a class's many exemplars collapse to one score:
      "knn"      mean of the top `k` exemplar similarities (k=1 is plain max)
      "centroid" similarity to the mean vector, one per class — cheaper, blunter
    """

    class_names: list[str]
    strategy: str = "knn"
    k: int = 1
    exemplars: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    owners: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    def __post_init__(self) -> None:
        if self.strategy not in {"knn", "centroid"}:
            raise ValueError(f"unknown strategy: {self.strategy!r}")
        if self.k < 1:
            raise ValueError("k must be at least 1")

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        labels: np.ndarray,
        class_names: list[str],
        strategy: str = "knn",
        k: int = 1,
    ) -> EmbeddingRouter:
        """Index labelled examples. There is no optimisation step — this is a copy."""
        labels = np.asarray(labels, dtype=np.int64)
        if len(labels) != len(embeddings):
            raise ValueError("embeddings and labels differ in length")
        if len(labels) and (labels < 0).any():
            raise ValueError("router exemplars must all be in-scope (no OOS labels)")

        router = cls(class_names=list(class_names), strategy=strategy, k=k)
        router.exemplars = l2_normalize(embeddings)
        router.owners = labels
        return router

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def n_exemplars(self) -> int:
        return len(self.owners)

    def examples_per_class(self) -> np.ndarray:
        return np.bincount(self.owners, minlength=self.n_classes)

    def add_intent(self, name: str, embeddings: np.ndarray) -> int:
        """Add a brand-new intent from a handful of examples. No retraining.

        This is the cold-start operation the fine-tuned classifier cannot
        perform at all: its output layer has no slot for this class.
        """
        if name in self.class_names:
            raise ValueError(f"intent already present: {name!r}")
        vectors = l2_normalize(embeddings)
        if self.n_exemplars and vectors.shape[1] != self.exemplars.shape[1]:
            raise ValueError("embedding dimension does not match the index")

        index = self.n_classes
        self.class_names.append(name)
        self.exemplars = vectors if not self.n_exemplars else np.vstack([self.exemplars, vectors])
        self.owners = np.concatenate([self.owners, np.full(len(vectors), index, dtype=np.int64)])
        return index

    def _centroids(self) -> np.ndarray:
        """One unit vector per class; classes with no examples stay zero."""
        sums = np.zeros((self.n_classes, self.exemplars.shape[1]), dtype=np.float32)
        np.add.at(sums, self.owners, self.exemplars)
        return l2_normalize(sums)

    def scores(self, queries: np.ndarray) -> np.ndarray:
        """Cosine similarity of each query to each class, shape (n, n_classes).

        A class with no exemplars scores 0 rather than -inf, so an unpopulated
        intent is merely never the argmax instead of breaking the row.
        """
        queries = l2_normalize(queries)
        if not self.n_exemplars:
            return np.zeros((len(queries), self.n_classes), dtype=np.float32)

        if self.strategy == "centroid":
            return (queries @ self._centroids().T).astype(np.float32)

        # knn: per class, the mean of its top-k exemplar similarities.
        masks = [self.owners == index for index in range(self.n_classes)]
        out = np.zeros((len(queries), self.n_classes), dtype=np.float32)
        for start in range(0, len(queries), QUERY_CHUNK):
            block = queries[start : start + QUERY_CHUNK]
            sims = block @ self.exemplars.T
            for index, mask in enumerate(masks):
                if not mask.any():
                    continue
                owned = sims[:, mask]
                take = min(self.k, owned.shape[1])
                if take == 1:
                    out[start : start + len(block), index] = owned.max(axis=1)
                else:
                    top = np.partition(owned, -take, axis=1)[:, -take:]
                    out[start : start + len(block), index] = top.mean(axis=1)
        return out
