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
    baseline_model: str
    model_drift: bool
    """Whether this run used a different model from day 16's.

    Stored by builds that know about the pin, and derived by comparing the two
    artifacts for one that predates it — so a tab or a page can state whether
    the comparison beside it is cross-model without caring which kind of
    artifact it is reading.
    """

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

    @property
    def scenario_ids(self) -> frozenset[str]:
        return frozenset(case.case_id for case in self.run.cases)

    def too_thin_to_compare(self, other: Results) -> str:
        """Why these two runs cannot be compared, or "" if they can.

        A rate is not a result just because it was computed. The cross-model
        run that prompted this reached **one** scenario before its daily
        allowance ran out, and `gemini-3.5-flash` turns out to allow 20
        requests a day against `flash-lite`'s 500 — so it scored 100% on
        `happy_one_shot` and nothing else.

        Rendering that beside the controlled run as "100% vs 91.7%" would be
        the most misleading honest number in the repository. Two conditions,
        both about whether the comparison could possibly answer its question:

        * enough shared scenarios to carry a percentage at all;
        * at least one of the five handoff conversations, since the structural
          claim is the only thing worth reading across models and those are
          where it is tested.
        """
        shared = self.scenario_ids & other.scenario_ids
        if len(shared) < MIN_SHARED_SCENARIOS:
            return (
                f"only {len(shared)} of {self.of_suite} scenarios are common to both "
                f"runs, below the {MIN_SHARED_SCENARIOS} this reports a rate on"
            )
        if not shared & frozenset(handoff_scenarios()):
            return (
                "no handoff conversation is common to both runs, so the structural "
                "claim — the only one worth reading across models — is untested"
            )
        return ""

    def rates_on(self, ids: frozenset[str]) -> dict[str, float | None]:
        """Pass rate and each dimension, restricted to `ids`.

        Needed because two runs rarely cover the same scenarios — the free
        tier stops them in different places — and comparing a 32-scenario rate
        against a 24-scenario one is the denominator mistake day 5 made. `None`
        for a dimension nothing in the subset claims, never 0.0.
        """
        cases = [case for case in self.run.cases if case.case_id in ids]
        if not cases:
            return {"pass_rate": None, "task": None, "safety": None, "efficiency": None}

        out: dict[str, float | None] = {
            "pass_rate": round(sum(1 for c in cases if c.passed) / len(cases), 4)
        }
        for dimension in ("task", "safety", "efficiency"):
            scored = [c for c in cases if c.scored_in(dimension)]
            out[dimension] = (
                round(sum(1 for c in scored if c.passed_in(dimension)) / len(scored), 4)
                if scored
                else None
            )
        return out

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


MIN_SHARED_SCENARIOS = 8
"""Fewest scenarios two runs must share before a rate is shown for them.

Eight is a judgement, not a calculation: a quarter of the suite, below which a
single conversation moves the number by more than the run-to-run spread day 18
measured at 3.1 points. The guard it serves matters more than the exact figure
— see `Results.too_thin_to_compare`.
"""

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
    """The **controlled** run — day 16's model — or None if it has not been run.

    Cross-model runs live in their own files and are read by `load_others`, so
    that nothing which renders "router against one agent" can quietly be
    showing a different model's numbers.
    """
    return _load_path(ARTIFACT_FILE)


@lru_cache(maxsize=1)
def load_others() -> tuple[Results, ...]:
    """Runs on a model other than day 16's, newest filename order.

    Separate from `load` rather than merged into it: these answer a different
    question — whether day 19's result is architectural or particular to one
    small model — and a reader has to be told which they are looking at.
    """
    found = []
    for path in sorted(ARTIFACT_DIR.glob("multi_agent.*.json")):
        loaded = _load_path(path)
        if loaded is not None:
            found.append(loaded)
    return tuple(found)


def _load_path(path: Path) -> Results | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
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

    from convo.projects.day19_multi_agent.build import baseline_model

    stored_baseline = payload.get("baseline_model") or baseline_model()
    return Results(
        model=payload["model"],
        baseline_model=stored_baseline,
        model_drift=bool(
            payload["model_drift"]
            if "model_drift" in payload
            else (stored_baseline and payload["model"] != stored_baseline)
        ),
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
