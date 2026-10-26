"""Load the benchmark run. No models, no corpus download, no network."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
RETRIEVERS = ("bm25", "dense", "hybrid", "hybrid+rerank")


@dataclass(frozen=True)
class BenchmarkResults:
    name: str
    documents: int
    queries: int
    judgements: int
    mean_relevant: float
    encoder: str
    reranker: str
    depths: list[int]
    by_depth: dict[str, dict]
    restaurant: dict[str, dict]
    truncated: bool

    def scores(self, depth: int, retriever: str) -> dict:
        return self.by_depth[str(depth)]["scores"][retriever]

    def rerank_ms(self, depth: int) -> float:
        return self.by_depth[str(depth)].get("rerank_ms_median", 0.0)

    def best_at(self, depth: int, metric: str = "mrr") -> str:
        """Which retriever wins at this shortlist depth."""
        return max(RETRIEVERS, key=lambda r: self.scores(depth, r)[metric])

    def rerank_delta(self, depth: int, metric: str = "mrr") -> float:
        """What the cross-encoder adds to hybrid fusion. Negative means it hurt."""
        return round(
            self.scores(depth, "hybrid+rerank")[metric] - self.scores(depth, "hybrid")[metric], 4
        )

    def restaurant_rerank_delta(self, metric: str = "mrr") -> float:
        if not self.restaurant:
            return 0.0
        return round(
            self.restaurant["hybrid+rerank"][metric] - self.restaurant["hybrid"][metric], 4
        )

    @property
    def deepest(self) -> int:
        return max(self.depths)


@lru_cache(maxsize=1)
def load(name: str = "scifact") -> BenchmarkResults | None:
    """Return the stored run, or None if the benchmark has not been run."""
    path = ARTIFACT_DIR / f"{name}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    described = payload["benchmark"]

    # An artifact from before the depth sweep stores one set of scores under
    # `scores` and the depth under `depth`. Adapt rather than raise: the data
    # is still valid, just narrower, and a tab that throws on load takes the
    # whole hub down with it.
    by_depth = payload.get("by_depth")
    if by_depth is None:
        depth = payload.get("depth", 10)
        by_depth = {
            str(depth): {
                "scores": payload["scores"],
                "rerank_ms_median": payload.get("rerank_ms_median", 0.0),
            }
        }
        depths = [depth]
    else:
        depths = payload.get("depths", sorted(int(k) for k in by_depth))
    return BenchmarkResults(
        name=described["name"],
        documents=described["documents"],
        queries=described["queries"],
        judgements=described["judgements"],
        mean_relevant=described["mean_relevant_per_query"],
        encoder=payload["encoder"],
        reranker=payload["reranker"],
        depths=depths,
        by_depth=by_depth,
        restaurant=payload.get("restaurant", {}),
        truncated=payload.get("truncated", False),
    )
