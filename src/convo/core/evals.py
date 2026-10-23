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


@dataclass
class Assertion:
    """One checkable claim about a conversation. Keep these mechanical."""

    name: str
    passed: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


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

    @property
    def clean(self) -> bool:
        return not self.regressions


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

    def compare(self, previous: EvalRun) -> Comparison:
        now = {c.case_id: c.passed for c in self.cases}
        before = {c.case_id: c.passed for c in previous.cases}
        shared = now.keys() & before.keys()
        return Comparison(
            regressions=sorted(c for c in shared if before[c] and not now[c]),
            fixes=sorted(c for c in shared if not before[c] and now[c]),
            unchanged=sorted(c for c in shared if before[c] == now[c]),
            added=sorted(now.keys() - before.keys()),
            removed=sorted(before.keys() - now.keys()),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "pass_rate": self.pass_rate,
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
