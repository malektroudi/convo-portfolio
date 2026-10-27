"""Day 19's run, and the preflight that has to pass before it costs anything.

    uv run python -m convo.projects.day19_multi_agent.build --preflight   # free
    uv run python -m convo.projects.day19_multi_agent.build               # needs the key
    uv run python -m convo.projects.day19_multi_agent.build --rescore      # free

The free tier allows 500 requests per model per day, and one pass of day 6's
32 scenarios costs 162 — measured, from day 18's stored outcomes, not
estimated. A routed conversation spends more, because every transfer is a
model call. So there is room for two of these a day at most, and day 19 was
written in the gap between discovering that and the quota resetting.

`--preflight` is everything that can be checked for nothing:

    the topology           `book_table` is held by one desk, and no path
                           leads back to it from the handoff desk
    the oracle vs the suite the derived route agrees with every scenario's
                           own `expect_booked` label
    the precedence         no turn both answers a digression and books, which
                           is the assumption the route derivation rests on
    the two oracles        day 19's derived tool trajectory matches day 18's,
                           derived independently from the same engine
    the baseline           day 16's verdicts on the five scenarios this day
                           exists to re-test, printed before anything is spent

That last one is the part worth insisting on. The experiment is only
meaningful if the single-agent result is known in advance, and writing the
comparison after seeing the new number is how a negative result quietly
becomes a positive one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from convo.core import get_logger, setup
from convo.core.evals import CaseResult, EvalRun
from convo.projects.day05_tool_agent.scenarios import normalise_booking
from convo.projects.day06_eval_harness.harness import engine_runner, run_suite
from convo.projects.day06_eval_harness.outcome import Outcome
from convo.projects.day06_eval_harness.suite import REF, SUITE, Scenario
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
from convo.projects.day18_eval_gate.trajectory import expected_for
from convo.projects.day19_multi_agent import scope
from convo.projects.day19_multi_agent.oracle import (
    arbitration_needed,
    disagrees_with_suite,
    expected_tools,
    route_for,
    routes_for,
)
from convo.projects.day19_multi_agent.runner import run_conversation
from convo.projects.day19_multi_agent.score import Summary, Turn, score_route

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_FILE = ARTIFACT_DIR / "multi_agent.json"

log = get_logger("day19.build")

# The retry taxonomy is day 16's, imported rather than copied. It encodes two
# findings that cost a day's quota each — that a per-minute 429 is not the
# daily wall, and that a 503 must be excluded rather than scored — and a
# second copy of it would eventually drift from the first.


def preflight(verbose: bool = True) -> list[str]:
    """Everything checkable with no model, no key and no quota."""
    problems: list[str] = []

    # Audit what the code builds now, then require the committed artifact to
    # match it. Either alone is weaker: auditing only the artifact would pass
    # a stale file, and auditing only the live tree would leave the published
    # proof unchecked.
    problems.extend(f"topology: {c}" for c in scope.audit(scope.derive()))
    problems.extend(f"committed topology: {c}" for c in scope.reproduces())

    for scenario in SUITE:
        route = route_for(scenario)
        if mismatch := disagrees_with_suite(scenario, route):
            problems.append(f"{scenario.id}: {mismatch}")
        if clash := arbitration_needed(scenario):
            problems.append(f"{scenario.id}: {clash}")
        if expected_tools(route) != expected_for(scenario).per_turn:
            problems.append(
                f"{scenario.id}: day 19's derived tools disagree with day 18's, "
                "so one of the two oracles is wrong"
            )

    if verbose:
        graph = scope.derive()
        log.info(
            "topology holds",
            extra={
                "reachable_from_handoff": sorted(graph.reachable_from(scope.HANDOFF)),
                "tools_from_handoff": sorted(graph.tools_reachable_from(scope.HANDOFF)),
                "tools_from_router": sorted(graph.tools_reachable_from(scope.ROUTER)),
            },
        )
        stored = day16_baseline()
        for scenario_id, verdict in sorted(day16_corrected().items()):
            log.info(
                "day 16's verdict on a handoff scenario",
                extra={
                    "scenario": scenario_id,
                    "single_agent": verdict,
                    "corrected_from": stored[scenario_id]
                    if stored[scenario_id] != verdict
                    else None,
                },
            )

    return problems


DAY16_CORRECTIONS = {"never_books_without_details": "pass"}
"""Verdicts in day 16's stored artifact that the fixed scorer overturns.

Day 16's artifact predates the false-positive fix day 18 forced: its
`score_safety` convicted the bot for saying "I'll get that table reserved for
you", a future-tense offer with no tool call behind it. The stored verdict for
that scenario is `fail` and the correct one is `pass`.

This matters more here than anywhere else. Day 19 is measured against day 16
on exactly these scenarios, and taking the stored verdict at face value would
credit the router with fixing a conversation the single agent never got wrong
— the most flattering possible error, which is reason enough to hard-code the
correction rather than wait for the re-run.

It disappears when day 16 is re-run against the fixed scorer; `day16_stale`
is what says whether that has happened.
"""


def day16_stale() -> bool:
    """Whether day 16's artifact still predates the scorer fix."""
    artifact = Path(__file__).parents[1] / "day16_adk_agent" / "artifacts" / "adk.json"
    if not artifact.exists():
        return False
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    return not payload.get("outcomes")


def day16_corrected() -> dict[str, str]:
    """Day 16's handoff verdicts with the known false positive put right."""
    stored = day16_baseline()
    if not day16_stale():
        return stored
    return {s: DAY16_CORRECTIONS.get(s, v) for s, v in stored.items()}


def day16_baseline() -> dict[str, str]:
    """How day 16's single agent did on the scenarios that reach a handoff.

    Read from day 16's committed artifact, so the comparison day 19 reports is
    against a number that was published before this day was written. These are
    the *stored* verdicts; `day16_corrected` applies the known false positive.
    """
    artifact = Path(__file__).parents[1] / "day16_adk_agent" / "artifacts" / "adk.json"
    if not artifact.exists():
        return {}
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    verdicts = {c["case_id"]: c["verdict"] for c in payload.get("adk", {}).get("cases", [])}
    handoff = {s.id for s in SUITE if route_for(s).handoff_turn is not None}
    return {s: v for s, v in verdicts.items() if s in handoff}


def multi_agent_runner(model: str):
    """Day 6's `Runner` contract, with day 19's per-turn records kept aside.

    The records cannot travel inside an `Outcome` — that shape is shared with
    four other bots and widening it for one day's experiment would make every
    other day's artifact carry a field it has no answer for. So they go into a
    dict the caller reads after the fact.
    """
    records: dict[str, tuple[Turn, ...]] = {}
    transfers: dict[str, int] = {}

    def run(scenario: Scenario) -> Outcome:
        last: Exception | None = None
        for attempt in range(RETRIES):
            try:
                turns, booking = asyncio.run(run_conversation(scenario.utterances, REF, model))
                break
            except Exception as exc:  # noqa: BLE001 - message may carry the key
                last = exc
                if _is_rate_limit(exc) and _is_daily_quota(exc):
                    raise QuotaExhausted(redact(exc)[:300]) from exc
                if not _is_retryable(exc):
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
        else:  # pragma: no cover - the loop always breaks or raises
            raise last  # type: ignore[misc]

        records[scenario.id] = tuple(
            Turn(authors=r.authors, tools=r.tools, by_agent=r.by_agent) for r in turns
        )
        transfers[scenario.id] = sum(r.transfers for r in turns)

        transcript: list[dict[str, str]] = []
        for utterance, record in zip(scenario.utterances, turns, strict=True):
            transcript.append({"role": "user", "content": utterance})
            transcript.append({"role": "assistant", "content": record.reply})

        time.sleep(PAUSE_S)
        return Outcome(
            scenario_id=scenario.id,
            booked=booking.fields is not None,
            captured=normalise_booking(booking.fields),
            transcript=transcript,
            escalated=any(scope.HANDOFF in r.authors for r in turns),
            user_turns=len(scenario.utterances),
            model_calls=booking.calls,
            failed_steps=booking.failures,
            latency_ms=round(sum(r.latency_ms for r in turns), 3),
        )

    return run, records, transfers


def _summary(records: dict[str, tuple[Turn, ...]], attempted: tuple[Scenario, ...]) -> Summary:
    routes = routes_for(attempted)
    return Summary(
        verdicts=tuple(
            score_route(routes[s.id], records[s.id]) for s in attempted if s.id in records
        )
    )


def _payload(
    model: str,
    attempted: tuple[Scenario, ...],
    run: EvalRun,
    cache: dict[str, Outcome],
    summary: Summary,
    transfers: dict[str, int],
    quota_note: str,
    unavailable: list[str],
) -> dict:
    return {
        "model": model,
        "scenarios": len(attempted),
        "of_suite": len(SUITE),
        "partial": len(attempted) < len(SUITE),
        "quota_note": quota_note,
        "unavailable": unavailable,
        "topology": scope.committed().as_dict(),
        "routing": summary.as_dict(),
        "transfers": transfers,
        "run": run.as_dict(),
        "outcomes": [cache[s.id].as_dict() for s in attempted],
        "engine": run_suite("day19-engine", engine_runner, scenarios=attempted).as_dict(),
        "day16_handoff_verdicts": day16_baseline(),
        "day16_handoff_corrected": day16_corrected(),
        "day16_stale": day16_stale(),
    }


def _rescore_stored() -> int:
    """Rebuild the free half of the artifact from the stored run.

    Day 18's lesson, applied before it is needed rather than after: a scorer
    change must be re-appliable without paying for the model again, so the raw
    outcomes are stored and this path re-derives every verdict from them.
    """
    if not ARTIFACT_FILE.exists():
        raise SystemExit(f"nothing to rescore: {ARTIFACT_FILE} does not exist")
    payload = json.loads(ARTIFACT_FILE.read_text(encoding="utf-8"))

    stored = [Outcome.from_dict(raw) for raw in payload.get("outcomes", [])]
    if not stored:
        raise SystemExit("the artifact has no raw outcomes; a real run is needed")

    by_id = {o.scenario_id: o for o in stored}
    attempted = tuple(s for s in SUITE if s.id in by_id)
    run = run_suite(
        "day19-multi-agent",
        lambda scenario: by_id[scenario.id],
        scenarios=attempted,
        meta=payload.get("run", {}).get("meta", {}),
    )
    payload["run"] = run.as_dict()
    payload["topology"] = scope.committed().as_dict()
    payload["engine"] = run_suite("day19-engine", engine_runner, scenarios=attempted).as_dict()
    payload["day16_handoff_verdicts"] = day16_baseline()
    payload["day16_handoff_corrected"] = day16_corrected()
    payload["day16_stale"] = day16_stale()
    payload["rescored"] = True

    ARTIFACT_FILE.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log.info("rescored from stored outcomes", extra={"pass_rate": run.pass_rate})
    return 0


def main(argv: list[str] | None = None) -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="run only the first N scenarios")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="check the topology and both oracles, calling no model",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="rebuild every verdict from the stored outcomes, calling no model",
    )
    parser.add_argument(
        "--write-topology",
        action="store_true",
        help="re-derive and commit the agent graph; needs google-adk, no key",
    )
    args = parser.parse_args(argv)

    if args.write_topology:
        graph = scope.save()
        log.info(
            "topology committed",
            extra={
                "path": str(scope.ARTIFACT_FILE),
                "tools_from_handoff": sorted(graph.tools_reachable_from(scope.HANDOFF)),
            },
        )
        return 0

    if args.preflight:
        problems = preflight()
        for line in problems:
            log.error("preflight", extra={"detail": line})
        if problems:
            return 1
        log.info("preflight passed; the run is worth paying for")
        return 0

    if args.rescore:
        return _rescore_stored()

    # A run that would be scored against a broken oracle is a run that wasted
    # a day's quota, so the free checks gate the expensive ones.
    problems = preflight(verbose=False)
    if problems:
        for line in problems:
            log.error("preflight", extra={"detail": line})
        raise SystemExit("preflight failed; not spending the quota")

    resolved = resolve()
    scenarios = SUITE[: args.limit] if args.limit else SUITE
    live, records, transfers = multi_agent_runner(resolved.model)

    cache: dict[str, Outcome] = {}
    attempted: list[Scenario] = []
    quota_note = ""
    unavailable: list[str] = []
    for scenario in scenarios:
        try:
            cache[scenario.id] = live(scenario)
        except ScenarioUnavailable as exc:
            unavailable.append(scenario.id)
            log.info("scenario excluded, not scored", extra={"scenario": scenario.id})
            log.debug("why it was excluded", extra={"error": str(exc)[:160]})
            continue
        except QuotaExhausted as exc:
            quota_note = str(exc)
            log.info(
                "free tier exhausted",
                extra={"completed": len(attempted), "of": len(scenarios)},
            )
            break
        attempted.append(scenario)
        verdict = score_route(route_for(scenario), records[scenario.id])
        log.info(
            "scenario done",
            extra={
                "scenario": scenario.id,
                "booked": cache[scenario.id].booked,
                "misroutes": len(verdict.misroutes),
                "late_booking": verdict.late_booking,
                "progress": f"{len(attempted)}/{len(scenarios)}",
            },
        )

    if not attempted:
        raise SystemExit(
            "the free tier was exhausted before any scenario completed. "
            "Daily quota is per model and resets after 24h."
        )

    run = run_suite(
        "day19-multi-agent",
        lambda scenario: cache[scenario.id],
        scenarios=tuple(attempted),
        meta={"model": resolved.model, "framework": "google-adk", "topology": "router+4"},
    )
    summary = _summary(records, tuple(attempted))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_FILE.write_text(
        json.dumps(
            _payload(
                resolved.model,
                tuple(attempted),
                run,
                cache,
                summary,
                transfers,
                quota_note,
                unavailable,
            ),
            indent=1,
        ),
        encoding="utf-8",
    )
    log.info(
        "artifact written",
        extra={
            "pass_rate": run.pass_rate,
            "routing_accuracy": summary.routing_accuracy,
            "late_bookings": len(summary.late_bookings),
            "handoff_held": f"{summary.handoff_held}/{len(summary.handoff_scenarios)}",
        },
    )
    return 0


def failed_cases(run: EvalRun) -> list[CaseResult]:
    """Convenience for the tab: the cases that did not pass."""
    return [c for c in run.cases if not c.passed]


if __name__ == "__main__":
    raise SystemExit(main())
