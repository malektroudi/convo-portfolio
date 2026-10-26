"""Load day 18's stored comparison. The only module the gate and the tab share.

Dependency-light on purpose: this imports day 6's suite (for the utterances)
and nothing from ADK. The gate needs ADK to *re-score*, but it needs this to
know what to re-score against — so if the two were tangled, the half of the
gate that checks thresholds and regressions could not run without a framework
that cannot be installed alongside Gradio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from convo.core.evals import CaseResult, EvalRun
from convo.projects.day06_eval_harness.suite import SUITE
from convo.projects.day18_eval_gate.compare import Adjudication, Agreement, Row

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_FILE = ARTIFACT_DIR / "evaluators.json"

UTTERANCES = {scenario.id: scenario.utterances for scenario in SUITE}
"""Read from the suite rather than stored in the artifact.

Storing them would duplicate day 6's suite into a file that then has to be
kept in step with it, and a trajectory scored against stale utterances would
be wrong in a way nothing would notice.
"""


@dataclass(frozen=True)
class StoredTrajectory:
    scenario_id: str
    actual: tuple[tuple[str, ...], ...]
    expected: tuple[tuple[str, ...], ...]
    score: float
    passed: bool
    flat_ok: bool
    only_timing: bool
    detail: str

    @property
    def utterances(self) -> tuple[str, ...]:
        return UTTERANCES.get(self.scenario_id, ())

    @property
    def tools_called(self) -> int:
        return sum(len(t) for t in self.actual)

    @property
    def tools_expected(self) -> int:
        return sum(len(t) for t in self.expected)


@dataclass(frozen=True)
class Comparison:
    model: str
    scenarios: int
    of_suite: int
    partial: bool
    quota_note: str
    threshold: float
    harness: EvalRun
    rows: tuple[Row, ...]
    trajectories: tuple[StoredTrajectory, ...]
    agreement_rate: float
    trajectory_pass_rate: float
    flat_pass_rate: float
    only_timing: int
    spec_agreement: dict[str, float]
    gate_choice: str
    day16_pass_rate: float | None
    rescorable: bool = False
    """Whether the artifact kept the raw outcomes, not just the scored verdicts.

    False for the run on disk, which predates the fix it caused. Day 18 found
    day 6's `score_safety` convicting a bot for offering to reserve a table,
    and then could not apply the correction to its own figures, because only
    the verdicts had been stored. `Outcome.as_dict` and `build --rescore` now
    exist; the tab keys its correction note off this flag so a re-run clears
    the note by itself.
    """

    @property
    def disagreements(self) -> tuple[Row, ...]:
        return tuple(r for r in self.rows if not r.agreement.agree)

    def count(self, agreement: Agreement) -> int:
        return sum(1 for r in self.rows if r.agreement is agreement)

    def adjudged(self, call: Adjudication) -> tuple[Row, ...]:
        return tuple(r for r in self.rows if r.adjudication is call)

    def trajectory(self, scenario_id: str) -> StoredTrajectory | None:
        return next((t for t in self.trajectories if t.scenario_id == scenario_id), None)

    def row(self, scenario_id: str) -> Row | None:
        return next((r for r in self.rows if r.scenario_id == scenario_id), None)

    @property
    def variance(self) -> float | None:
        """How far this run's pass rate sits from day 16's, on the same suite.

        Same agent, same model, same scenarios, a few hours apart. Whatever
        this number is, it is the precision of every single-run pass rate in
        this repository that involved a model — including day 16's.
        """
        if self.day16_pass_rate is None:
            return None
        return round(self.harness.pass_rate - self.day16_pass_rate, 4)

    @property
    def redundant_writes(self) -> tuple[StoredTrajectory, ...]:
        """Conversations that called `book_table` more often than once.

        The finding with actual consequences. Day 4's engine books once, on
        the turn the guest confirms. These conversations book, get corrected,
        and book again — so a reservation system sees two writes where it
        should see one. Every one of them passes day 6's assertions, because
        the *final* state is right and the final state is all they read.
        """
        from convo.projects.day18_eval_gate.trajectory import BOOK_TOOL

        found = []
        for stored in self.trajectories:
            got = sum(turn.count(BOOK_TOOL) for turn in stored.actual)
            want = sum(turn.count(BOOK_TOOL) for turn in stored.expected)
            if got > want:
                found.append(stored)
        return tuple(found)

    @property
    def timing_only_share(self) -> float:
        """Of the trajectory failures, how many are the confirmation step.

        The number that turns "ADK's evaluator rejects most of the suite" into
        one actionable behavioural difference.
        """
        failures = sum(1 for t in self.trajectories if not t.passed)
        return round(self.only_timing / failures, 4) if failures else 0.0


def _run_from(payload: dict[str, Any]) -> EvalRun:
    return EvalRun(
        suite=payload["suite"],
        run_id=payload["run_id"],
        started_at=payload["started_at"],
        cases=[CaseResult.from_dict(c) for c in payload.get("cases", [])],
        meta=payload.get("meta", {}),
    )


@lru_cache(maxsize=1)
def load() -> Comparison | None:
    if not ARTIFACT_FILE.exists():
        return None
    payload = json.loads(ARTIFACT_FILE.read_text(encoding="utf-8"))

    rows = tuple(
        Row(
            scenario_id=raw["scenario_id"],
            spec_ok=raw["spec_ok"],
            harness_ok=raw["harness_ok"],
            adk_ok=raw["adk_ok"],
            adk_score=raw["adk_score"],
            adk_detail=raw["adk_detail"],
            harness_detail=raw["harness_detail"],
        )
        for raw in payload.get("comparison", [])
    )
    trajectories = tuple(
        StoredTrajectory(
            scenario_id=scenario_id,
            actual=tuple(tuple(t) for t in raw["actual"]),
            expected=tuple(tuple(t) for t in raw["expected"]),
            score=raw["score"],
            passed=raw["passed"],
            flat_ok=raw.get("flat_ok", False),
            only_timing=raw.get("only_timing", False),
            detail=raw.get("detail", ""),
        )
        for scenario_id, raw in payload.get("trajectories", {}).items()
    )

    return Comparison(
        model=payload["model"],
        scenarios=payload["scenarios"],
        of_suite=payload["of_suite"],
        partial=bool(payload.get("partial")),
        quota_note=payload.get("quota_note") or "",
        threshold=payload.get("threshold", 1.0),
        harness=_run_from(payload["harness"]),
        rows=rows,
        trajectories=trajectories,
        agreement_rate=payload["agreement_rate"],
        trajectory_pass_rate=payload.get("trajectory_pass_rate", 0.0),
        flat_pass_rate=payload.get("flat_pass_rate", 0.0),
        only_timing=payload.get("only_timing", 0),
        spec_agreement=dict(payload.get("spec_agreement", {})),
        gate_choice=payload.get("gate_choice", "harness"),
        day16_pass_rate=payload.get("day16_pass_rate"),
        rescorable=bool(payload.get("outcomes")),
    )
