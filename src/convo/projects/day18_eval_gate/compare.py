"""Two evaluators on the same conversations, judged against something neither owns.

The obvious way to compare two evaluators is to treat one as correct and count
where the other differs. That measures nothing: it assumes the answer.

So both are scored against a third thing — the **spec verdict**, read straight
off the scenario's own declared expectations:

    booked == expect_booked     and     captured == expect

Those fields are hand-written in day 6's suite as a statement of intent, before
any bot existed. Day 6's *scorers* read them; day 6's verdict then adds safety
and efficiency assertions on top. ADK's trajectory metric never sees them at
all — it only compares tool names against the trajectory day 4's engine
derives. So the spec verdict is independent of both, and asking each evaluator
how often it agrees with the spec is a fair question put to both.

What this cannot settle is which evaluator is *better*, because the spec
verdict only covers the outcome. A conversation can reach the right booking by
a route nobody would ship — six failed tool calls, a confident lie in the
middle — and the spec verdict will call it correct. That is precisely the gap
the trajectory metric is supposed to fill, so counting its disagreements with
the spec as errors would be unfair to it. Hence `UNDECIDED`: the cases where
the two evaluators differ and the spec cannot say who is right are reported as
a question, not scored as a loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from convo.core.evals import CaseResult
from convo.projects.day06_eval_harness.suite import Scenario
from convo.projects.day18_eval_gate.adk_eval import TrajectoryVerdict


class Agreement(Enum):
    BOTH_PASS = "both pass"
    BOTH_FAIL = "both fail"
    HARNESS_ONLY = "day 6 passes, ADK fails"
    ADK_ONLY = "ADK passes, day 6 fails"

    @property
    def agree(self) -> bool:
        return self in (Agreement.BOTH_PASS, Agreement.BOTH_FAIL)


class Adjudication(Enum):
    """Who the spec sides with, where it can speak at all."""

    AGREED = "both evaluators agree"
    HARNESS_RIGHT = "the spec backs day 6"
    ADK_RIGHT = "the spec backs ADK"
    UNDECIDED = "the spec cannot say"


@dataclass(frozen=True)
class Row:
    scenario_id: str
    spec_ok: bool
    """Did the conversation end in the state the scenario declares?"""
    harness_ok: bool
    adk_ok: bool
    adk_score: float
    adk_detail: str
    harness_detail: str

    @property
    def agreement(self) -> Agreement:
        if self.harness_ok and self.adk_ok:
            return Agreement.BOTH_PASS
        if not self.harness_ok and not self.adk_ok:
            return Agreement.BOTH_FAIL
        return Agreement.HARNESS_ONLY if self.harness_ok else Agreement.ADK_ONLY

    @property
    def adjudication(self) -> Adjudication:
        if self.agreement.agree:
            return Adjudication.AGREED
        # The spec only speaks about the outcome. It can convict an evaluator
        # that passed a conversation whose outcome was wrong, or one that
        # failed a conversation whose outcome was right *and* whose route the
        # spec covers. Where the disagreement is about the route, it abstains.
        if self.adk_ok and not self.spec_ok:
            return Adjudication.HARNESS_RIGHT
        if self.harness_ok and not self.spec_ok:
            return Adjudication.ADK_RIGHT
        return Adjudication.UNDECIDED


def spec_verdict(scenario: Scenario, case: CaseResult) -> bool:
    """Did the conversation end as the scenario says it must?

    Read from day 6's own task assertions rather than recomputed, because they
    already encode `expect_booked` and every field of `expect` — and
    recomputing it here from the transcript would be a second implementation
    of the same rule, free to drift from the one that produced the published
    figures. Only the task dimension is used: safety and efficiency are
    judgements layered on top of the spec, not the spec itself.
    """
    task = [a for a in case.assertions if a.dimension == "task"]
    return bool(task) and all(a.passed for a in task)


def _detail(case: CaseResult) -> str:
    broken = [f"{a.name}: {a.detail}" for a in case.assertions if not a.passed]
    return "; ".join(broken)


def compare(
    scenarios: tuple[Scenario, ...],
    cases: dict[str, CaseResult],
    verdicts: dict[str, TrajectoryVerdict],
) -> list[Row]:
    rows: list[Row] = []
    for scenario in scenarios:
        case, verdict = cases.get(scenario.id), verdicts.get(scenario.id)
        if case is None or verdict is None:
            continue
        rows.append(
            Row(
                scenario_id=scenario.id,
                spec_ok=spec_verdict(scenario, case),
                harness_ok=case.passed,
                adk_ok=verdict.passed,
                adk_score=verdict.score,
                adk_detail=verdict.explain(),
                harness_detail=_detail(case),
            )
        )
    return rows


@dataclass(frozen=True)
class Summary:
    rows: tuple[Row, ...]

    @property
    def total(self) -> int:
        return len(self.rows)

    def count(self, agreement: Agreement) -> int:
        return sum(1 for r in self.rows if r.agreement is agreement)

    def adjudged(self, call: Adjudication) -> tuple[Row, ...]:
        return tuple(r for r in self.rows if r.adjudication is call)

    @property
    def agreed(self) -> int:
        return sum(1 for r in self.rows if r.agreement.agree)

    @property
    def agreement_rate(self) -> float:
        return round(self.agreed / self.total, 4) if self.total else 0.0

    @property
    def disagreements(self) -> tuple[Row, ...]:
        return tuple(r for r in self.rows if not r.agreement.agree)

    def spec_agreement(self, which: str) -> float:
        """How often an evaluator's verdict matches the spec's.

        Reported, but **not** a quality score and not what picks the gate. The
        spec speaks only about outcomes, so an evaluator that measures the
        route is penalised here for doing its job. See `gate_choice`.
        """
        if not self.rows:
            return 0.0
        hits = sum(
            1 for r in self.rows if (r.harness_ok if which == "harness" else r.adk_ok) == r.spec_ok
        )
        return round(hits / self.total, 4)

    @property
    def gate_choice(self) -> str:
        """Which evaluator gates a live bot. Decided on convictions, not rates.

        The first version of this compared `spec_agreement` for each
        evaluator and took the higher. That was wrong, and wrong in a way that
        flattered my own harness: the spec verdict is about the outcome, day
        6's assertions are about the outcome, and ADK's trajectory metric is
        deliberately about something else. Scoring it on outcome agreement
        guaranteed it would lose — this run has it at 19.4% against day 6's
        96.8%, which measures the question it was asked, not its quality.

        So the decision rests on adjudications instead, where the spec can
        only convict an evaluator it actually contradicts. A tie — including
        the common case where the spec cannot settle anything, because every
        disagreement is about the route — goes to day 6's harness. Not
        because it is mine, but because it is the one that can score a live
        bot in CI with no key, no quota and no network, and a gate that
        cannot run is not a gate.
        """
        harness_wrong = len(self.adjudged(Adjudication.ADK_RIGHT))
        adk_wrong = len(self.adjudged(Adjudication.HARNESS_RIGHT))
        if harness_wrong > adk_wrong:
            return "adk"
        return "harness"
