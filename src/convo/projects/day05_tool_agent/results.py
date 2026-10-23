"""Loads the cached day 5 comparison.

Same pattern as days 2 and 3: the benchmark does the expensive work once and
commits a JSON summary, so the tab needs neither torch nor a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from convo.core.evals import EvalRun

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
COMPARISON_FILE = ARTIFACT_DIR / "comparison.json"


PASS_LABELS = {
    "schema_only": "Agent · schema only",
    "with_grounding": "Agent · schema + provenance",
}


@dataclass(frozen=True)
class AgentPass:
    """One configuration of the agent: its verdicts and what it cost."""

    key: str
    run: EvalRun
    stats: dict[str, Any]

    @property
    def label(self) -> str:
        return PASS_LABELS.get(self.key, self.key)

    @property
    def false_confirmations(self) -> int:
        """Conversations that told the guest a booking was made, wrongly.

        The safety metric, and the one the pass rate hides: a run that books
        the wrong table and a run that books nothing both score as failures,
        but only one of them sends someone to a restaurant that is not
        expecting them.
        """
        total = 0
        for case in self.run.cases:
            verdicts = {a.name: a.passed for a in case.assertions}
            if verdicts.get("booked") is True and not all(verdicts.values()):
                total += 1
        return total

    @property
    def missed_bookings(self) -> int:
        """Conversations that should have booked and did not."""
        return sum(
            1
            for case in self.run.cases
            for a in case.assertions
            if a.name == "booked" and not a.passed
        )


@dataclass(frozen=True)
class Comparison:
    engine: EvalRun
    passes: dict[str, AgentPass]
    scenarios: list[dict[str, Any]]
    config: dict[str, Any]

    @property
    def schema_only(self) -> AgentPass:
        return self.passes["schema_only"]

    @property
    def grounded(self) -> AgentPass:
        return self.passes["with_grounding"]

    @property
    def scenario_ids(self) -> list[str]:
        return [s["id"] for s in self.scenarios]

    def scenario(self, scenario_id: str) -> dict[str, Any]:
        return next(s for s in self.scenarios if s["id"] == scenario_id)

    def case(self, run: EvalRun, scenario_id: str):
        return next((c for c in run.cases if c.case_id == scenario_id), None)


def _run_from(payload: dict[str, Any]) -> EvalRun:
    from convo.core.evals import CaseResult

    return EvalRun(
        suite=payload["suite"],
        run_id=payload["run_id"],
        started_at=payload["started_at"],
        cases=[CaseResult.from_dict(c) for c in payload.get("cases", [])],
        meta=payload.get("meta", {}),
    )


@lru_cache(maxsize=1)
def load() -> Comparison | None:
    """Return the cached comparison, or None if the benchmark has not run."""
    if not COMPARISON_FILE.exists():
        return None
    payload = json.loads(COMPARISON_FILE.read_text(encoding="utf-8"))

    engine = _run_from(payload["engine"])
    expected = {c.case_id for c in engine.cases}
    passes: dict[str, AgentPass] = {}
    for key, block in payload["agent_passes"].items():
        run = _run_from(block["run"])
        if {c.case_id for c in run.cases} != expected:
            raise ValueError(
                f"pass {key!r} covers different scenarios from the state machine "
                "— they are not comparable"
            )
        passes[key] = AgentPass(key=key, run=run, stats=block["stats"])

    return Comparison(
        engine=engine,
        passes=passes,
        scenarios=payload["scenarios"],
        config=payload["config"],
    )
