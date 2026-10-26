"""Load day 16's run and make it comparable — which takes more care than it looks.

Three bots, and no single table holds all three honestly.

**ADK against day 4's state machine** is a real comparison: both ran all 32 of
day 6's scenarios, scored by day 6's scorers, with all three dimensions
tagged. That is the headline, and `Scope.FULL` is it.

**Day 5's agent** is not in that table. Day 5 ran its own 12-scenario suite
and its assertions carry no dimension, because day 6 — which invented the
dimensions — came after it. Its pass rate can be compared, over those 12
scenarios and against the other two bots restricted to the same 12, and that
is all. `Scope.DAY5` is that narrower comparison.

So two scopes, and an unmeasured figure reports `None` rather than 0.0.
`pass_rate_in` returns 0.0 for a dimension no case scores, which would put a
scarlet **0%** under day 5's safety column and mean nothing at all. The
distinction between *failed* and *never measured* is the whole reason day 6
has `scored_in`, and losing it here would be the same mistake day 15 caught in
day 12.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from convo.core.evals import DIMENSIONS, CaseResult, EvalRun

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_FILE = ARTIFACT_DIR / "adk.json"

LABELS = {
    "engine": "State machine · day 4",
    "schema_only": "0.5B agent · day 5",
    "grounded": "0.5B agent + guard · day 5",
    "adk": "ADK + Gemini · day 16",
}
SHORT = {
    "engine": "state machine",
    "schema_only": "0.5B agent",
    "grounded": "0.5B + guard",
    "adk": "ADK + Gemini",
}
FULL_ORDER = ("engine", "adk")
DAY5_ORDER = ("engine", "schema_only", "grounded", "adk")

FALSE_CONFIRMATION = "no false confirmation"
"""Day 6's explicit safety assertion. Day 5's runs do not carry it."""

CALL_ASSERTION = "within the model-call budget"
"""Day 6's efficiency check, whose detail reads `made N calls, budget M`.

Absent when a runner reports no model calls, which is why a missing assertion
cannot be read as zero for anything but the state machine.
"""


class Scope(Enum):
    """Which denominator a figure is computed over."""

    FULL = "full"
    """All 32 of day 6's scenarios. ADK and the state machine only."""

    DAY5 = "day5"
    """The 12 day 5 ran, the only set on which all four runs can be compared."""


def _run_from(payload: dict[str, Any]) -> EvalRun:
    return EvalRun(
        suite=payload["suite"],
        run_id=payload["run_id"],
        started_at=payload["started_at"],
        cases=[CaseResult.from_dict(c) for c in payload.get("cases", [])],
        meta=payload.get("meta", {}),
    )


def _restricted(run: EvalRun, keep: frozenset[str]) -> EvalRun:
    return EvalRun(
        suite=run.suite,
        run_id=run.run_id,
        started_at=run.started_at,
        cases=[c for c in run.cases if c.case_id in keep],
        meta=run.meta,
    )


def _calls(case: CaseResult) -> int | None:
    for assertion in case.assertions:
        if assertion.name == CALL_ASSERTION:
            for token in assertion.detail.split():
                if token.isdigit():
                    return int(token)
    return None


@dataclass(frozen=True)
class Bot:
    """One run, in whichever scopes it can honestly appear in."""

    key: str
    runs: dict[Scope, EvalRun]

    @property
    def label(self) -> str:
        return LABELS.get(self.key, self.key)

    @property
    def short(self) -> str:
        return SHORT.get(self.key, self.key)

    def run(self, scope: Scope) -> EvalRun | None:
        return self.runs.get(scope)

    def pass_rate(self, scope: Scope) -> float | None:
        run = self.run(scope)
        return None if run is None or not run.cases else run.pass_rate

    def scenarios(self, scope: Scope) -> int:
        run = self.run(scope)
        return 0 if run is None else len(run.cases)

    def dimension(self, scope: Scope, name: str) -> float | None:
        """None when no case in this run makes a claim about that dimension.

        Day 5's runs return None for all three. That is a fact about when they
        were scored, not a score of zero.
        """
        run = self.run(scope)
        if run is None or not any(c.scored_in(name) for c in run.cases):
            return None
        return run.pass_rate_in(name)

    def by_dimension(self, scope: Scope) -> dict[str, float | None]:
        return {d: self.dimension(scope, d) for d in DIMENSIONS}

    def model_calls(self, scope: Scope) -> int | None:
        """Total model calls, or None when the run did not record them.

        The state machine is the one run that can report 0 without measuring
        it: there is no model in it to call.
        """
        if self.key == "engine":
            return 0
        run = self.run(scope)
        if run is None:
            return None
        counted = [n for n in (_calls(c) for c in run.cases) if n is not None]
        return sum(counted) if counted else None

    def false_confirmations(self, scope: Scope) -> int | None:
        """Conversations that told the guest a booking was made, wrongly.

        Day 5's metric, and still the one that matters most: a run that books
        the wrong table and a run that books nothing score identically on a
        pass rate, and only one of them sends someone to a restaurant that is
        not expecting them.

        Counted two ways, because the two generations of scorer record it
        differently. Day 6 asserts it outright (`no false confirmation`), so
        where that assertion exists it is simply read. Day 5's runs predate it
        and are counted by day 5's own rule — a booking was made and something
        about it was wrong — so the figure here matches the one on day 5's own
        tab instead of quietly disagreeing with it. That rule is the weaker of
        the two: it cannot see a conversation that booked when it should not
        have booked at all, which is precisely the failure ADK has.
        """
        run = self.run(scope)
        if run is None:
            return None
        explicit = [
            a for case in run.cases for a in case.assertions if a.name == FALSE_CONFIRMATION
        ]
        if explicit:
            return sum(1 for a in explicit if not a.passed)
        total = 0
        for case in run.cases:
            verdicts = {a.name: a.passed for a in case.assertions}
            if verdicts.get("booked") is True and not all(verdicts.values()):
                total += 1
        return total

    def missed_bookings(self, scope: Scope) -> int | None:
        run = self.run(scope)
        if run is None:
            return None
        return sum(
            1 for case in run.cases for a in case.assertions if a.name == "booked" and not a.passed
        )

    def failures(self, scope: Scope) -> list[CaseResult]:
        run = self.run(scope)
        return [] if run is None else run.failures

    def case(self, scenario_id: str) -> CaseResult | None:
        for scope in (Scope.FULL, Scope.DAY5):
            run = self.run(scope)
            if run is None:
                continue
            found = next((c for c in run.cases if c.case_id == scenario_id), None)
            if found is not None:
                return found
        return None


@dataclass(frozen=True)
class Comparison:
    model: str
    models_listed: int
    rejected_models: tuple[str, ...]
    scenarios: int
    of_suite: int
    partial: bool
    quota_note: str
    bots: dict[str, Bot]
    scope_ids: dict[Scope, tuple[str, ...]]
    rescorable: bool = False
    """Whether the artifact stored the raw outcomes, not just the verdicts.

    False for any run made before day 18. Those artifacts cannot have a
    scorer fix applied to them, so their figures are frozen at whatever the
    scorers said on the day — which matters, because day 18 found a real bug
    in `score_safety` afterwards. The tab keys its correction note off this,
    so re-running the build makes the note disappear on its own rather than
    leaving a stale caveat to be deleted by hand.
    """

    @property
    def adk(self) -> Bot:
        return self.bots["adk"]

    @property
    def engine(self) -> Bot:
        return self.bots["engine"]

    def bot(self, key: str) -> Bot | None:
        return self.bots.get(key)

    def ordered(self, scope: Scope) -> list[Bot]:
        order = FULL_ORDER if scope is Scope.FULL else DAY5_ORDER
        return [self.bots[k] for k in order if k in self.bots and self.bots[k].run(scope)]

    def ids(self, scope: Scope) -> tuple[str, ...]:
        return self.scope_ids.get(scope, ())

    @property
    def coverage(self) -> float:
        return round(self.scenarios / self.of_suite, 4) if self.of_suite else 0.0

    @property
    def gap(self) -> float | None:
        """How far ADK sits below the state machine, on the full suite."""
        adk, engine = self.adk.pass_rate(Scope.FULL), self.engine.pass_rate(Scope.FULL)
        return None if adk is None or engine is None else round(engine - adk, 4)

    @property
    def verdict(self) -> str:
        """Which explanation for day 5's 8.3% the run supports.

        Day 5 varied the model and the architecture at once, so its number was
        evidence for two incompatible stories. This holds the architecture
        fixed — an LLM choosing tool calls, same tools, same scorers — and
        changes only the model and the plumbing.
        """
        adk = self.adk.pass_rate(Scope.DAY5)
        agent = (self.bot("grounded") or self.adk).pass_rate(Scope.DAY5)
        engine = self.engine.pass_rate(Scope.DAY5)
        if adk is None or agent is None or engine is None:
            return "incomparable"
        if adk >= engine:
            return "architecture-neutral"
        if adk > agent + 0.2:
            return "model-bound"
        return "architecture-bound"


@lru_cache(maxsize=1)
def load() -> Comparison | None:
    if not ARTIFACT_FILE.exists():
        return None
    payload = json.loads(ARTIFACT_FILE.read_text(encoding="utf-8"))
    if not payload.get("adk") or not payload.get("engine"):
        return None

    stored = {key: _run_from(payload[key]) for key in DAY5_ORDER if payload.get(key)}
    full_ids = frozenset(c.case_id for c in stored["adk"].cases) & frozenset(
        c.case_id for c in stored["engine"].cases
    )
    day5_sets = [
        frozenset(c.case_id for c in stored[key].cases)
        for key in ("schema_only", "grounded")
        if key in stored
    ]
    day5_ids = frozenset.intersection(full_ids, *day5_sets) if day5_sets else frozenset()

    bots: dict[str, Bot] = {}
    for key, run in stored.items():
        runs: dict[Scope, EvalRun] = {}
        if key in FULL_ORDER:
            runs[Scope.FULL] = _restricted(run, full_ids)
        if day5_ids:
            runs[Scope.DAY5] = _restricted(run, day5_ids)
        bots[key] = Bot(key=key, runs=runs)

    return Comparison(
        model=payload["model"],
        models_listed=payload.get("models_listed", 0),
        rejected_models=tuple(payload.get("rejected_models") or ()),
        scenarios=payload.get("scenarios", len(stored["adk"].cases)),
        of_suite=payload.get("of_suite") or len(stored["adk"].cases),
        partial=bool(payload.get("partial")),
        quota_note=payload.get("quota_note") or "",
        bots=bots,
        scope_ids={
            Scope.FULL: tuple(sorted(full_ids)),
            Scope.DAY5: tuple(sorted(day5_ids)),
        },
        rescorable=bool(payload.get("outcomes")),
    )
