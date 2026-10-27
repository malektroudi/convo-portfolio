"""Load day 19's artifact, or report honestly that the run has not happened.

The artifact does not exist yet, and that is a state this module is built to
represent rather than crash on. Day 19's free half — the topology proof, both
oracles, and day 16's pre-registered baseline — was written in the ten hours
between the daily quota running out and resetting. The tab renders all of it
and says plainly that the measurement is outstanding.

A day that shows a proof and a stated prediction with the number still missing
is more honest than one that shows a number from a run nobody can reproduce,
and it is the one state this repository's rule actually permits: every
published figure must be re-derivable from a committed artifact, so a figure
with no artifact is not published.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from convo.core.evals import CaseResult, EvalRun
from convo.projects.day06_eval_harness.suite import SUITE
from convo.projects.day19_multi_agent.agents import HANDOFF
from convo.projects.day19_multi_agent.oracle import route_for

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_FILE = ARTIFACT_DIR / "multi_agent.json"


@dataclass(frozen=True)
class RouteRow:
    """One scenario's routing verdict, as stored."""

    scenario_id: str
    expected: tuple[str, ...]
    actual: tuple[tuple[str, ...], ...]
    tools: tuple[tuple[str, ...], ...]
    misroutes: tuple[int, ...]
    late_booking_turn: int | None
    scope_breaches: tuple[tuple[str, str], ...]
    routing_accuracy: float | None

    @property
    def late_booking(self) -> bool:
        return self.late_booking_turn is not None

    @property
    def is_handoff(self) -> bool:
        return HANDOFF in self.expected

    @property
    def utterances(self) -> tuple[str, ...]:
        found = next((s for s in SUITE if s.id == self.scenario_id), None)
        return found.utterances if found else ()


@dataclass(frozen=True)
class Results:
    model: str
    scenarios: int
    of_suite: int
    partial: bool
    quota_note: str
    unavailable: tuple[str, ...]
    topology: dict[str, Any]
    run: EvalRun
    engine: EvalRun
    rows: tuple[RouteRow, ...]
    routing_accuracy: float | None
    handoff_held: int
    transfers: dict[str, int]
    day16_stored: dict[str, str]
    day16_corrected: dict[str, str]
    day16_stale: bool

    @property
    def handoff_rows(self) -> tuple[RouteRow, ...]:
        return tuple(r for r in self.rows if r.is_handoff)

    @property
    def late_bookings(self) -> tuple[RouteRow, ...]:
        return tuple(r for r in self.rows if r.late_booking)

    @property
    def scope_breaches(self) -> tuple[RouteRow, ...]:
        return tuple(r for r in self.rows if r.scope_breaches)

    @property
    def total_transfers(self) -> int:
        return sum(self.transfers.values())

    @property
    def verdicts(self) -> dict[str, bool]:
        """Day 19's pass/fail per scenario, for comparing with day 16's."""
        return {case.case_id: case.passed for case in self.run.cases}

    def fixed_what_day16_broke(self) -> tuple[str, ...]:
        """Handoff scenarios day 16 failed and day 19 passes.

        Compared against day 16's *corrected* verdicts, so the router is not
        credited with a conversation the single agent only appeared to fail.
        """
        mine = self.verdicts
        return tuple(
            scenario
            for scenario, verdict in sorted(self.day16_corrected.items())
            if verdict == "fail" and mine.get(scenario) is True
        )

    def broke_what_day16_held(self) -> tuple[str, ...]:
        """The other direction, reported with equal prominence."""
        mine = self.verdicts
        return tuple(
            scenario
            for scenario, verdict in sorted(self.day16_corrected.items())
            if verdict == "pass" and mine.get(scenario) is False
        )


PENDING_TARGETS = ("does_not_book_after_abandonment", "does_not_book_on_escalation")
"""The two scenarios day 19 was built to re-test, named before the run.

Day 16's single agent failed both on both of its runs, and they are the only
failures in this repository that have reproduced across every run of every
model. Written down here so the comparison cannot be chosen after the fact.
"""


def handoff_scenarios() -> tuple[str, ...]:
    """The five scenarios whose derived route reaches the handoff desk.

    Computed, not stored — this is free, deterministic and available with no
    artifact, which is what lets the tab describe the experiment before the
    run has happened.
    """
    return tuple(s.id for s in SUITE if route_for(s).handoff_turn is not None)


def _run_from(payload: dict[str, Any]) -> EvalRun:
    return EvalRun(
        suite=payload["suite"],
        run_id=payload["run_id"],
        started_at=payload["started_at"],
        cases=[CaseResult.from_dict(c) for c in payload.get("cases", [])],
        meta=payload.get("meta", {}),
    )


@lru_cache(maxsize=1)
def load() -> Results | None:
    """The stored run, or None when it has not been paid for yet."""
    if not ARTIFACT_FILE.exists():
        return None
    payload = json.loads(ARTIFACT_FILE.read_text(encoding="utf-8"))
    routing = payload.get("routing", {})

    rows = tuple(
        RouteRow(
            scenario_id=raw["scenario_id"],
            expected=tuple(raw["expected"]),
            actual=tuple(tuple(a) for a in raw["actual"]),
            tools=tuple(tuple(t) for t in raw["tools"]),
            misroutes=tuple(raw["misroutes"]),
            late_booking_turn=raw["late_booking_turn"],
            scope_breaches=tuple(tuple(b) for b in raw["scope_breaches"]),
            routing_accuracy=raw.get("routing_accuracy"),
        )
        for raw in routing.get("verdicts", [])
    )

    return Results(
        model=payload["model"],
        scenarios=payload["scenarios"],
        of_suite=payload["of_suite"],
        partial=bool(payload.get("partial")),
        quota_note=payload.get("quota_note") or "",
        unavailable=tuple(payload.get("unavailable") or ()),
        topology=dict(payload.get("topology", {})),
        run=_run_from(payload["run"]),
        engine=_run_from(payload["engine"]),
        rows=rows,
        routing_accuracy=routing.get("routing_accuracy"),
        handoff_held=routing.get("handoff_held", 0),
        transfers=dict(payload.get("transfers", {})),
        day16_stored=dict(payload.get("day16_handoff_verdicts", {})),
        day16_corrected=dict(payload.get("day16_handoff_corrected", {})),
        day16_stale=bool(payload.get("day16_stale")),
    )
