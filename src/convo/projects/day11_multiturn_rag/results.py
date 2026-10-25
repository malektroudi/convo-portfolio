"""Load the stored rankings and score them, in pure Python.

Metrics are recomputed from the artifact rather than stored, exactly as day 9
does, so the tab can slice by condition, retriever or turn position without a
model present — and so every figure in the README can be regenerated instead
of transcribed.

The slicing that matters is **follow-ups only**. Turn 1 of every conversation
is standalone by construction, so including it would mix in twelve questions
no rewriter touches and shrink every difference by a third.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from convo.core.retrieval import RetrievalScore, score_runs

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
TURNS_FILE = ARTIFACT_DIR / "turns.json"

CONDITIONS = ("raw", "concat", "rewrite", "oracle")
RETRIEVERS = ("bm25", "dense", "hybrid+rerank")


@dataclass(frozen=True)
class TurnRow:
    conversation: str
    index: int
    text: str
    query: str
    strategy: str
    detail: str
    ranked: tuple[str, ...]
    doc_id: str
    standalone: bool

    @property
    def key(self) -> str:
        return f"{self.conversation}:{self.index}"

    @property
    def rewritten(self) -> bool:
        return self.query.strip().lower() != self.text.strip().lower()


@dataclass(frozen=True)
class Day11Results:
    runs: dict[str, tuple[TurnRow, ...]]
    ablations: dict[str, tuple[TurnRow, ...]]
    depths: dict[str, tuple[TurnRow, ...]]
    relevance: dict[str, set[str]]
    conversations: list[dict]
    meta: dict[str, Any]

    def _runs_for(self, rows: tuple[TurnRow, ...], follow_ups_only: bool):
        chosen = [r for r in rows if not (follow_ups_only and r.index == 0)]
        return {r.key: (list(r.ranked), self.relevance[r.key]) for r in chosen}

    def score(
        self,
        condition: str,
        retriever: str = "bm25",
        follow_ups_only: bool = True,
    ) -> RetrievalScore:
        rows = self.runs[f"{condition}|{retriever}"]
        return score_runs(condition, self._runs_for(rows, follow_ups_only))

    def score_rows(self, label: str, rows: tuple[TurnRow, ...]) -> RetrievalScore:
        return score_runs(label, self._runs_for(rows, follow_ups_only=True))

    def ablation_table(self) -> list[RetrievalScore]:
        return [self.score_rows(name, rows) for name, rows in self.ablations.items()]

    def depth_table(self) -> list[tuple[int, RetrievalScore]]:
        return [
            (int(depth), self.score_rows(f"depth {depth}", rows))
            for depth, rows in sorted(self.depths.items(), key=lambda kv: int(kv[0]))
        ]

    def gap_closed(self, retriever: str = "bm25") -> float | None:
        """Share of the raw-to-oracle gap that rewriting captures, by MRR.

        None when raw already matches the oracle, because a share of nothing
        is not zero — it is undefined, and reporting it as 0% would read as a
        failure rather than as an absent opportunity.
        """
        raw = self.score("raw", retriever).mrr
        oracle = self.score("oracle", retriever).mrr
        rewritten = self.score("rewrite", retriever).mrr
        if abs(oracle - raw) < 1e-9:
            return None
        return round((rewritten - raw) / (oracle - raw), 4)

    def turns(self, condition: str, retriever: str = "bm25") -> tuple[TurnRow, ...]:
        return self.runs[f"{condition}|{retriever}"]

    def comparison(self, retriever: str = "bm25") -> list[dict]:
        """Per follow-up, what each condition retrieved — for the tab's table."""
        raw = {r.key: r for r in self.turns("raw", retriever)}
        rewritten = {r.key: r for r in self.turns("rewrite", retriever)}
        oracle = {r.key: r for r in self.turns("oracle", retriever)}

        rows = []
        for key, row in rewritten.items():
            if row.index == 0:
                continue
            want = self.relevance[key]
            rows.append(
                {
                    "turn": row.text,
                    "rewritten as": row.query if row.rewritten else "— unchanged —",
                    "rule": row.strategy,
                    "raw": _hit(raw[key], want),
                    "rewrite": _hit(row, want),
                    "oracle": _hit(oracle[key], want),
                }
            )
        return rows


def _hit(row: TurnRow, want: set[str]) -> str:
    """Where the first relevant chunk landed, as a reader-facing label."""
    for position, chunk_id in enumerate(row.ranked, start=1):
        if chunk_id in want:
            return "1" if position == 1 else f"{position}"
    return "miss"


def _rows(payload: list[dict]) -> tuple[TurnRow, ...]:
    return tuple(
        TurnRow(
            conversation=r["conversation"],
            index=r["index"],
            text=r["text"],
            query=r["query"],
            strategy=r["strategy"],
            detail=r["detail"],
            ranked=tuple(r["ranked"]),
            doc_id=r["doc_id"],
            standalone=r["standalone"],
        )
        for r in payload
    )


@lru_cache(maxsize=1)
def load() -> Day11Results | None:
    """Return the stored rankings, or None if the build has not been run."""
    if not TURNS_FILE.exists():
        return None
    payload = json.loads(TURNS_FILE.read_text(encoding="utf-8"))
    return Day11Results(
        runs={k: _rows(v) for k, v in payload["runs"].items()},
        ablations={k: _rows(v) for k, v in payload["ablations"].items()},
        depths={k: _rows(v) for k, v in payload["depths"].items()},
        relevance={k: set(v) for k, v in payload["relevance"].items()},
        conversations=payload["conversations"],
        meta={
            "strategy": payload["strategy"],
            "chunks": payload["chunks"],
            "encoder": payload["encoder"],
            "reranker": payload["reranker"],
            "depth": payload["depth"],
        },
    )
