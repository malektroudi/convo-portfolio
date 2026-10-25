"""Loads the stored evidence and replays the guarded pipeline over it.

Decisions are recomputed rather than stored, so moving the threshold in the tab
moves every number on the page — including the red-team scoreboard. A guardrail
report that cannot be re-run at a different setting is a screenshot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from convo.projects.day08_rag_cites.answer import Citation
from convo.projects.day10_guardrails.groundedness import (
    Evidence,
    GroundingMetrics,
    best_threshold,
    score_threshold,
    sweep,
)
from convo.projects.day10_guardrails.pipeline import (
    GuardedReply,
    RedTeamOutcome,
    guarded_reply,
    score_red_team,
)
from convo.projects.day10_guardrails.probes import RED_TEAM

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
EVIDENCE_FILE = ARTIFACT_DIR / "evidence.json"


def _citation(payload: dict | None) -> Citation | None:
    if not payload:
        return None
    return Citation(
        doc_id=payload["doc_id"],
        text=payload["text"],
        start=payload["start"],
        end=payload["end"],
        score=payload["score"],
        chunk_id=payload["chunk_id"],
        heading=payload.get("heading", ""),
        matched_terms=tuple(payload.get("matched_terms", [])),
    )


def _evidence(record: dict) -> Evidence | None:
    if record["chunk_id"] is None:
        return None
    return Evidence(
        question=record["question"],
        doc_id=record["doc_id"],
        chunk_id=record["chunk_id"],
        score=record["score"],
        citation=_citation(record["citation"]),
    )


@dataclass(frozen=True)
class GuardResults:
    answerable: list[tuple[Evidence, str]]
    unanswerable: list[Evidence]
    red_team: list[tuple[object, Evidence | None]]
    strategy: str
    chunks: int
    unanswerable_why: dict[str, str]

    def sweep(self) -> list[GroundingMetrics]:
        return sweep(self.answerable, self.unanswerable)

    def best(self) -> GroundingMetrics:
        return best_threshold(self.answerable, self.unanswerable)

    def at(self, threshold: float) -> GroundingMetrics:
        return score_threshold(self.answerable, self.unanswerable, threshold)

    def replies(self, threshold: float) -> list[GuardedReply]:
        out = []
        for case, evidence in self.red_team:
            out.append(guarded_reply(case.prompt, evidence, threshold))
        return out

    def red_team_outcomes(self, threshold: float) -> list[RedTeamOutcome]:
        return [
            score_red_team(case, reply)
            for (case, _), reply in zip(self.red_team, self.replies(threshold), strict=True)
        ]

    def red_team_rate(self, threshold: float) -> float:
        outcomes = self.red_team_outcomes(threshold)
        return round(sum(o.passed for o in outcomes) / len(outcomes), 4) if outcomes else 0.0


@lru_cache(maxsize=1)
def load() -> GuardResults | None:
    """Return the stored evidence, or None if the build has not been run."""
    if not EVIDENCE_FILE.exists():
        return None
    payload = json.loads(EVIDENCE_FILE.read_text(encoding="utf-8"))

    answerable: list[tuple[Evidence, str]] = []
    unanswerable: list[Evidence] = []
    by_prompt: dict[str, Evidence | None] = {}

    for record in payload["records"]:
        evidence = _evidence(record)
        if record["kind"] == "answerable":
            if evidence is None:
                raise ValueError(
                    f"{record['question']!r} retrieved nothing — the evidence "
                    "artifact is stale, rebuild it"
                )
            answerable.append((evidence, record["expect_doc"]))
        elif record["kind"] == "unanswerable":
            if evidence is not None:
                unanswerable.append(evidence)
        else:
            by_prompt[record["question"]] = evidence

    return GuardResults(
        answerable=answerable,
        unanswerable=unanswerable,
        red_team=[(case, by_prompt.get(case.prompt)) for case in RED_TEAM],
        strategy=payload["strategy"],
        chunks=payload["chunks"],
        unanswerable_why={u["question"]: u["why"] for u in payload["unanswerable"]},
    )
