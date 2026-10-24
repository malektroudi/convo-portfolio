"""Adapters and the run loop.

An adapter is the only place that knows how a particular bot works. It drives
one scenario and flattens the result into an `Outcome`; everything after that
is shared. Adding a third bot later means writing one adapter, not a second
copy of the suite.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from convo.core.evals import CaseResult, EvalRun
from convo.projects.day04_dialog_engine import flow as fl
from convo.projects.day04_dialog_engine.engine import new_session as new_engine
from convo.projects.day04_dialog_engine.engine import turn as engine_turn
from convo.projects.day05_tool_agent.scenarios import normalise_booking, normalise_engine
from convo.projects.day06_eval_harness.outcome import Outcome
from convo.projects.day06_eval_harness.scorers import score_all
from convo.projects.day06_eval_harness.suite import REF, SUITE, Scenario

Runner = Callable[[Scenario], Outcome]


def engine_runner(scenario: Scenario) -> Outcome:
    """Day 4's state machine."""
    session = new_engine(REF)
    transcript: list[dict[str, str]] = []
    for utterance in scenario.utterances:
        result = engine_turn(session, utterance)
        transcript.append({"role": "user", "content": utterance})
        transcript.append({"role": "assistant", "content": result.reply})

    return Outcome(
        scenario_id=scenario.id,
        booked=session.page == fl.BOOKED,
        captured=normalise_engine(session),
        transcript=transcript,
        escalated=session.page == fl.HANDOFF,
        user_turns=len(session.history),
        latency_ms=round(sum(t.latency_ms for t in session.history), 3),
    )


def agent_runner(backend, config=None) -> Runner:
    """Day 5's tool-calling agent, bound to a backend."""
    from convo.projects.day05_tool_agent.agent import new_session as new_agent
    from convo.projects.day05_tool_agent.agent import turn as agent_turn

    def run(scenario: Scenario) -> Outcome:
        session = new_agent(backend, REF, config)
        transcript: list[dict[str, str]] = []
        for utterance in scenario.utterances:
            result = agent_turn(session, utterance)
            transcript.append({"role": "user", "content": utterance})
            transcript.append({"role": "assistant", "content": result.reply})

        return Outcome(
            scenario_id=scenario.id,
            booked=session.booking is not None,
            captured=normalise_booking(session.booking),
            transcript=transcript,
            escalated=session.exhausted,
            user_turns=len(scenario.utterances),
            model_calls=session.model_calls,
            failed_steps=session.failed_steps,
            latency_ms=round(sum(s.latency_ms for s in session.steps), 3),
        )

    return run


def run_suite(
    suite_name: str,
    runner: Runner,
    scenarios: tuple[Scenario, ...] = SUITE,
    meta: dict | None = None,
) -> EvalRun:
    """Drive every scenario and score it on all three dimensions."""
    run = EvalRun(suite=suite_name, meta=meta or {})
    started = time.perf_counter()

    for scenario in scenarios:
        outcome = runner(scenario)
        run.add(
            CaseResult.from_assertions(
                scenario.id,
                score_all(scenario, outcome),
                transcript=outcome.transcript,
                latency_ms=outcome.latency_ms,
                tags=list(scenario.tags),
                judge_rationale=(
                    f"{outcome.user_turns} turns"
                    + (f", {outcome.model_calls} model calls" if outcome.model_calls else "")
                    + (", escalated" if outcome.escalated else "")
                ),
            )
        )

    run.meta["wall_seconds"] = round(time.perf_counter() - started, 2)
    run.meta["scenarios"] = len(scenarios)
    return run
