"""Run the suite once, score it twice, and store enough that CI never re-runs it.

    uv run python -m convo.projects.day18_eval_gate.build

This is the only part of day 18 that costs anything. It drives day 6's suite
through day 16's ADK agent, recording the tool trajectory per turn — which day
16 did not store, and which is the one thing ADK's evaluator needs. Everything
downstream (the second evaluator, the comparison, the gate) reads the artifact
and calls no model, which is what lets the gate run in CI on a repository with
no API key.

It is a **second** run of the same suite, not a re-score of day 16's. That is
unavoidable — the trajectories were not recorded the first time — and it buys
something day 16 could not report: the same agent, same model, same scenarios,
twice. Any difference between the two pass rates is run-to-run variance, and a
single eval run that claims a pass rate to one decimal place is pretending that
number is stable. Day 12 learned this about a TTS model; there is no reason an
agent should be different.

Day 16's quota discipline is imported rather than re-implemented: exhaustion
raises instead of scoring, each scenario runs exactly once, and a partial run
is recorded as partial.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from convo.core import get_logger, setup
from convo.core.evals import EvalRun
from convo.projects.day05_tool_agent.scenarios import normalise_booking
from convo.projects.day06_eval_harness.harness import run_suite
from convo.projects.day06_eval_harness.outcome import Outcome
from convo.projects.day06_eval_harness.suite import REF, SUITE, Scenario
from convo.projects.day16_adk_agent.agent import run_conversation
from convo.projects.day16_adk_agent.build import (
    BACKOFF_S,
    PAUSE_S,
    RETRIES,
    QuotaExhausted,
    ScenarioUnavailable,
    _is_daily_quota,
    _is_rate_limit,
    _is_retryable,
)
from convo.projects.day16_adk_agent.model import redact, resolve
from convo.projects.day18_eval_gate import adk_eval
from convo.projects.day18_eval_gate.compare import Summary, compare
from convo.projects.day18_eval_gate.trajectory import disagrees_with_suite, expected_all

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
DAY16_ARTIFACT = Path(__file__).resolve().parents[1] / "day16_adk_agent" / "artifacts" / "adk.json"

log = get_logger("day18.build")


def _drive(scenario: Scenario, model: str) -> tuple[Outcome, tuple[tuple[str, ...], ...]]:
    """One conversation, returned as day 6's Outcome plus its trajectory."""
    for attempt in range(RETRIES):
        try:
            turns, booking = asyncio.run(run_conversation(scenario.utterances, REF, model))
            break
        except Exception as exc:  # noqa: BLE001 - the message may carry the key
            if _is_rate_limit(exc) and _is_daily_quota(exc):
                raise QuotaExhausted(redact(exc)[:300]) from exc
            if not _is_retryable(exc):
                # Not the server being busy: a bad request, a blocked
                # prompt, or a bug in this repository. Excluding it would
                # hide the bug and scoring it would invent a result, so it
                # stops the run where someone has to look at it.
                raise
            if attempt == RETRIES - 1:
                if _is_rate_limit(exc):
                    raise QuotaExhausted(redact(exc)[:300]) from exc
                raise ScenarioUnavailable(redact(exc)[:300]) from exc
            wait = BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)]
            log.info(
                "retrying after a transient error",
                extra={"seconds": wait, "attempt": attempt + 1, "of": RETRIES},
            )
            time.sleep(wait)

    transcript: list[dict[str, str]] = []
    for utterance, turn in zip(scenario.utterances, turns, strict=True):
        transcript.append({"role": "user", "content": utterance})
        transcript.append({"role": "assistant", "content": turn.reply})

    time.sleep(PAUSE_S)
    outcome = Outcome(
        scenario_id=scenario.id,
        booked=booking.fields is not None,
        captured=normalise_booking(booking.fields),
        transcript=transcript,
        escalated=False,
        user_turns=len(scenario.utterances),
        model_calls=booking.calls,
        failed_steps=booking.failures,
        latency_ms=round(sum(t.latency_ms for t in turns), 3),
    )
    return outcome, tuple(turn.tools for turn in turns)


def _day16_pass_rate() -> float | None:
    """Day 16's stored pass rate, for the run-to-run variance line."""
    if not DAY16_ARTIFACT.exists():
        return None
    payload = json.loads(DAY16_ARTIFACT.read_text(encoding="utf-8"))
    return payload.get("adk", {}).get("pass_rate")


def _rescore() -> int:
    """Re-run day 6's scorers over the stored outcomes. No model, no key.

    This is the path that did not exist when day 18 found a bug in
    `score_safety`, and the reason it now does. Everything expensive about a
    run is the conversation; scoring it is microseconds. Keeping only the
    verdicts meant a scorer fix could not be applied to the figures it had
    just invalidated.

    The trajectories and the ADK verdicts are untouched: they do not depend on
    day 6's scorers, and re-deriving them here would risk changing numbers
    this call has no business changing.
    """
    out = ARTIFACT_DIR / "evaluators.json"
    if not out.exists():
        raise SystemExit(f"nothing to re-score: {out} does not exist")
    payload = json.loads(out.read_text(encoding="utf-8"))
    stored = payload.get("outcomes")
    if not stored:
        raise SystemExit(
            "this artifact predates outcome storage, so it cannot be re-scored. "
            "Re-run the build to produce one that can."
        )

    outcomes = {raw["scenario_id"]: Outcome.from_dict(raw) for raw in stored}
    attempted = tuple(s for s in SUITE if s.id in outcomes)
    was = payload["harness"]["pass_rate"]

    run = run_suite(
        "day18-adk",
        lambda scenario: outcomes[scenario.id],
        scenarios=attempted,
        meta=dict(payload["harness"].get("meta", {})) | {"rescored": True},
    )

    cases = {case.case_id: case for case in run.cases}
    rebuilt = {
        scenario_id: adk_eval.TrajectoryVerdict(
            scenario_id=scenario_id,
            score=raw["score"],
            passed=raw["passed"],
            per_turn=tuple(raw["per_turn"]),
            actual=tuple(tuple(t) for t in raw["actual"]),
            expected=tuple(tuple(t) for t in raw["expected"]),
            flat_ok=raw.get("flat_ok", False),
        )
        for scenario_id, raw in payload["trajectories"].items()
    }
    summary = Summary(tuple(compare(attempted, cases, rebuilt)))

    payload["harness"] = run.as_dict()
    payload["comparison"] = [
        {
            "scenario_id": row.scenario_id,
            "spec_ok": row.spec_ok,
            "harness_ok": row.harness_ok,
            "adk_ok": row.adk_ok,
            "adk_score": row.adk_score,
            "adk_detail": row.adk_detail,
            "harness_detail": row.harness_detail,
            "agreement": row.agreement.value,
            "adjudication": row.adjudication.value,
        }
        for row in summary.rows
    ]
    payload["agreement_rate"] = summary.agreement_rate
    payload["spec_agreement"] = {
        "harness": summary.spec_agreement("harness"),
        "adk": summary.spec_agreement("adk"),
    }
    payload["gate_choice"] = summary.gate_choice
    # Day 16's figure is quoted here for the run-to-run comparison, so a
    # re-run of day 16 would silently leave this one stale. Refreshing it is
    # free: it is a number read off a committed file.
    payload["day16_pass_rate"] = _day16_pass_rate()
    payload["rescored"] = True

    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log.info(
        "re-scored from stored outcomes",
        extra={
            "scenarios": len(attempted),
            "pass_rate_was": was,
            "pass_rate_now": run.pass_rate,
            "agreement_rate": summary.agreement_rate,
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="run only the first N scenarios")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="re-score the stored outcomes with the current scorers, calling no model",
    )
    args = parser.parse_args(argv)

    if args.rescore:
        return _rescore()

    scenarios = SUITE[: args.limit] if args.limit else SUITE

    # Check the oracle before spending anything on the model. If the derived
    # trajectory contradicts a scenario's own label, nothing computed from it
    # is worth a quota call.
    expected = expected_all(scenarios)
    for scenario in scenarios:
        complaint = disagrees_with_suite(scenario, expected[scenario.id])
        if complaint:
            raise SystemExit(f"oracle disagrees with the suite on {scenario.id}: {complaint}")
    log.info("oracle agrees with every scenario label", extra={"scenarios": len(scenarios)})

    resolved = resolve()
    log.info("running the suite", extra={"model": resolved.model, "scenarios": len(scenarios)})

    outcomes: dict[str, Outcome] = {}
    actual: dict[str, tuple[tuple[str, ...], ...]] = {}
    attempted: list[Scenario] = []
    quota_note = ""
    unavailable: list[str] = []
    for scenario in scenarios:
        try:
            outcome, trajectory = _drive(scenario, resolved.model)
        except ScenarioUnavailable as exc:
            unavailable.append(scenario.id)
            log.info(
                "scenario excluded, not scored",
                extra={"scenario": scenario.id, "error": str(exc)[:160]},
            )
            continue
        except QuotaExhausted as exc:
            quota_note = str(exc)
            log.info(
                "free tier exhausted",
                extra={"completed": len(attempted), "of": len(scenarios)},
            )
            break
        outcomes[scenario.id] = outcome
        actual[scenario.id] = trajectory
        attempted.append(scenario)
        log.info(
            "scenario done",
            extra={
                "scenario": scenario.id,
                "booked": outcome.booked,
                "tools": sum(len(t) for t in trajectory),
                "progress": f"{len(attempted)}/{len(scenarios)}",
            },
        )

    if not attempted:
        raise SystemExit(
            "the free tier was exhausted before any scenario completed. "
            "Daily quota is per model and resets after 24h."
        )

    run: EvalRun = run_suite(
        "day18-adk",
        lambda scenario: outcomes[scenario.id],
        scenarios=tuple(attempted),
        meta={"model": resolved.model, "framework": "google-adk", "pass": "second"},
    )

    verdicts = {
        scenario.id: adk_eval.evaluate(
            scenario.id, scenario.utterances, actual[scenario.id], expected[scenario.id]
        )
        for scenario in attempted
    }
    cases = {case.case_id: case for case in run.cases}
    summary = Summary(tuple(compare(tuple(attempted), cases, verdicts)))

    payload = {
        "model": resolved.model,
        "scenarios": len(attempted),
        "of_suite": len(SUITE),
        "partial": len(attempted) < len(SUITE),
        "quota_note": quota_note,
        "unavailable": unavailable,
        "threshold": adk_eval.THRESHOLD,
        "harness": run.as_dict(),
        # The raw outcomes, so a scorer change can be re-applied without
        # another run of a rate-limited hosted model. Day 18's own safety bug
        # is why this is here: fixing `score_safety` could not refresh the
        # figures it invalidated, because only the verdicts had been kept.
        "outcomes": [outcomes[scenario.id].as_dict() for scenario in attempted],
        "day16_pass_rate": _day16_pass_rate(),
        "trajectories": {
            scenario.id: {
                "actual": [list(t) for t in actual[scenario.id]],
                "expected": [list(t) for t in expected[scenario.id].per_turn],
                "score": verdicts[scenario.id].score,
                "passed": verdicts[scenario.id].passed,
                "per_turn": list(verdicts[scenario.id].per_turn),
                "flat_ok": verdicts[scenario.id].flat_ok,
                "only_timing": verdicts[scenario.id].only_timing,
                "detail": verdicts[scenario.id].explain(),
            }
            for scenario in attempted
        },
        "comparison": [
            {
                "scenario_id": row.scenario_id,
                "spec_ok": row.spec_ok,
                "harness_ok": row.harness_ok,
                "adk_ok": row.adk_ok,
                "adk_score": row.adk_score,
                "adk_detail": row.adk_detail,
                "harness_detail": row.harness_detail,
                "agreement": row.agreement.value,
                "adjudication": row.adjudication.value,
            }
            for row in summary.rows
        ],
        "agreement_rate": summary.agreement_rate,
        "trajectory_pass_rate": round(
            sum(1 for v in verdicts.values() if v.passed) / len(verdicts), 4
        ),
        "flat_pass_rate": round(sum(1 for v in verdicts.values() if v.flat_ok) / len(verdicts), 4),
        "only_timing": sum(1 for v in verdicts.values() if v.only_timing),
        "spec_agreement": {
            "harness": summary.spec_agreement("harness"),
            "adk": summary.spec_agreement("adk"),
        },
        "gate_choice": summary.gate_choice,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT_DIR / "evaluators.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log.info(
        "artifact written",
        extra={
            "path": str(out),
            "harness_pass_rate": run.pass_rate,
            "agreement_rate": summary.agreement_rate,
            "gate_choice": summary.gate_choice,
            "kilobytes": max(out.stat().st_size // 1024, 1),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
