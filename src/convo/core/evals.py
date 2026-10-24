"""Eval result types shared by every project.

Written on day 1 and then left alone, because day 6 (the conversational eval
harness) and day 18 (the pipeline's deploy gate) both read these shapes. A run
serialises to JSON so CI can diff today's run against the last one.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Verdict = Literal["pass", "fail", "error"]


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# Scoring dimensions. Day 5 is why these exist: a single pass rate scored
# "booked the wrong table" and "booked nothing" identically, so a guard that
# eliminated ten false confirmations looked worthless. Different kinds of
# wrong have different costs, and a harness that cannot say which kind got
# worse cannot be used to make a decision.
TASK = "task"
"""Did it achieve what the user asked for?"""
SAFETY = "safety"
"""Did it avoid doing harm — inventing facts, leaking data, lying about state?"""
EFFICIENCY = "efficiency"
"""Did it get there without wasting the user's time or the budget?"""

DIMENSIONS: tuple[str, ...] = (TASK, SAFETY, EFFICIENCY)


@dataclass
class Assertion:
    """One checkable claim about a conversation. Keep these mechanical."""

    name: str
    passed: bool
    detail: str = ""
    dimension: str = ""
    """Which dimension this claim belongs to. Empty means unclassified, which
    still counts toward the overall verdict but not toward any dimension."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "dimension": self.dimension,
        }


@dataclass
class CaseResult:
    """The outcome of one scripted conversation."""

    case_id: str
    verdict: Verdict
    assertions: list[Assertion] = field(default_factory=list)
    judge_score: float | None = None
    judge_rationale: str = ""
    transcript: list[dict[str, str]] = field(default_factory=list)
    latency_ms: float | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    def dimensions(self) -> set[str]:
        return {a.dimension for a in self.assertions if a.dimension}

    def scored_in(self, dimension: str) -> bool:
        """Whether this case makes any claim about that dimension."""
        return any(a.dimension == dimension for a in self.assertions)

    def passed_in(self, dimension: str) -> bool:
        """True when every claim in that dimension held.

        A case that says nothing about a dimension passes it vacuously; use
        `scored_in` to tell "fine" from "not measured".
        """
        return all(a.passed for a in self.assertions if a.dimension == dimension)

    def failures_in(self, dimension: str) -> list[Assertion]:
        return [a for a in self.assertions if a.dimension == dimension and not a.passed]

    @classmethod
    def from_assertions(
        cls, case_id: str, assertions: list[Assertion], **kwargs: Any
    ) -> CaseResult:
        verdict: Verdict = "pass" if all(a.passed for a in assertions) else "fail"
        return cls(case_id=case_id, verdict=verdict, assertions=assertions, **kwargs)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "verdict": self.verdict,
            "assertions": [a.as_dict() for a in self.assertions],
            "judge_score": self.judge_score,
            "judge_rationale": self.judge_rationale,
            "transcript": self.transcript,
            "latency_ms": self.latency_ms,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CaseResult:
        return cls(
            case_id=raw["case_id"],
            verdict=raw["verdict"],
            assertions=[Assertion(**a) for a in raw.get("assertions", [])],
            judge_score=raw.get("judge_score"),
            judge_rationale=raw.get("judge_rationale", ""),
            transcript=raw.get("transcript", []),
            latency_ms=raw.get("latency_ms"),
            tags=raw.get("tags", []),
        )


@dataclass
class Comparison:
    """What changed between two runs. This is what a CI gate actually needs."""

    regressions: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    by_dimension: dict[str, list[str]] = field(default_factory=dict)
    """Cases that regressed, per dimension. The overall list can be empty while
    this is not: a change can trade safety for task success and leave the
    headline verdict untouched, which is exactly the failure day 5 hit."""

    @property
    def clean(self) -> bool:
        """No case regressed overall AND no dimension regressed."""
        return not self.regressions and not any(self.by_dimension.values())

    def regressed_dimensions(self) -> list[str]:
        return sorted(d for d, cases in self.by_dimension.items() if cases)


@dataclass
class EvalRun:
    """One execution of a suite."""

    suite: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: str = field(default_factory=_utc_now)
    cases: list[CaseResult] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def add(self, case: CaseResult) -> CaseResult:
        self.cases.append(case)
        return case

    @property
    def pass_rate(self) -> float:
        """Share of cases that passed. An empty run is 0.0, never a crash."""
        if not self.cases:
            return 0.0
        return round(sum(c.passed for c in self.cases) / len(self.cases), 4)

    @property
    def failures(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.passed]

    def gate(self, threshold: float) -> bool:
        """The day-18 deploy gate: does this run clear the bar?"""
        return self.pass_rate >= threshold

    def dimensions(self) -> list[str]:
        return sorted({d for c in self.cases for d in c.dimensions()})

    def pass_rate_in(self, dimension: str) -> float:
        """Share of the cases that measure this dimension which passed it.

        Scoped to the cases that actually make a claim about it, so adding
        scenarios that ignore a dimension cannot inflate its score.
        """
        scored = [c for c in self.cases if c.scored_in(dimension)]
        if not scored:
            return 0.0
        return round(sum(c.passed_in(dimension) for c in scored) / len(scored), 4)

    def pass_rate_by_dimension(self) -> dict[str, float]:
        return {d: self.pass_rate_in(d) for d in self.dimensions()}

    def failures_in(self, dimension: str) -> list[CaseResult]:
        return [c for c in self.cases if c.scored_in(dimension) and not c.passed_in(dimension)]

    def gate_by_dimension(self, thresholds: dict[str, float]) -> dict[str, bool]:
        """Each dimension against its own bar.

        Safety deserves a higher bar than efficiency, and expressing that is
        the whole reason dimensions exist.
        """
        return {d: self.pass_rate_in(d) >= bar for d, bar in thresholds.items()}

    def compare(self, previous: EvalRun) -> Comparison:
        now = {c.case_id: c.passed for c in self.cases}
        before = {c.case_id: c.passed for c in previous.cases}
        shared = now.keys() & before.keys()

        now_cases = {c.case_id: c for c in self.cases}
        before_cases = {c.case_id: c for c in previous.cases}
        by_dimension: dict[str, list[str]] = {}
        for dimension in sorted(set(self.dimensions()) | set(previous.dimensions())):
            dropped = [
                case_id
                for case_id in sorted(shared)
                if before_cases[case_id].scored_in(dimension)
                and now_cases[case_id].scored_in(dimension)
                and before_cases[case_id].passed_in(dimension)
                and not now_cases[case_id].passed_in(dimension)
            ]
            by_dimension[dimension] = dropped

        return Comparison(
            regressions=sorted(c for c in shared if before[c] and not now[c]),
            fixes=sorted(c for c in shared if not before[c] and now[c]),
            unchanged=sorted(c for c in shared if before[c] == now[c]),
            added=sorted(now.keys() - before.keys()),
            removed=sorted(before.keys() - now.keys()),
            by_dimension=by_dimension,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "pass_rate": self.pass_rate,
            "pass_rate_by_dimension": self.pass_rate_by_dimension(),
            "cases": [c.as_dict() for c in self.cases],
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> EvalRun:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            suite=raw["suite"],
            run_id=raw["run_id"],
            started_at=raw["started_at"],
            cases=[CaseResult.from_dict(c) for c in raw.get("cases", [])],
            meta=raw.get("meta", {}),
        )
