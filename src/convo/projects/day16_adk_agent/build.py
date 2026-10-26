"""Score the ADK agent on day 6's suite, beside day 4's engine and day 5's agent.

    uv sync --extra train --extra agents
    uv run python -m convo.projects.day16_adk_agent.build

The runner below is the only new code in the comparison. Everything that
judges the result — the assertions, the three dimensions, the gate — is day
6's and is shared with the other two bots, which is what makes the three
numbers comparable at all.

A free-tier key has a request-per-minute limit, and a suite of 32 scenarios
at three or four turns each will find it. Rate-limit responses are retried
with backoff rather than counted as failures: a 429 says nothing about
whether the agent books the right table.
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
from convo.projects.day06_eval_harness.harness import engine_runner, run_suite
from convo.projects.day06_eval_harness.outcome import Outcome
from convo.projects.day06_eval_harness.suite import REF, SUITE, Scenario
from convo.projects.day16_adk_agent.agent import run_conversation
from convo.projects.day16_adk_agent.model import redact, resolve

ARTIFACT_DIR = Path(__file__).parent / "artifacts"


class QuotaExhausted(RuntimeError):
    """The free tier ran out, which says nothing about the agent.

    Scoring a quota failure as a failed scenario would understate the bot by
    however many scenarios were left, and the number would look like a
    result. Scenarios that could not be attempted are excluded and counted
    separately instead.
    """


PER_DAY_MARKERS = ("perday", "per day", "requests per day")
"""Substrings that identify a 429 as the *daily* quota rather than per-minute.

Google returns `429 RESOURCE_EXHAUSTED` for both, and the difference decides
whether waiting helps. Day 18 found this the expensive way: a per-minute limit
was read as daily exhaustion, the run stopped after two scenarios, and the
model answered again thirty seconds later. The quota id in the error body is
the only thing that distinguishes them, so it is what gets matched.
"""

BACKOFF_S = (10, 20, 30, 45, 60, 90)
"""Waits between attempts on a 429 that is not identifiably the daily quota.

Four minutes in total, which is long enough that a per-minute window has
certainly rolled over. The previous policy gave up after 8 and 16 seconds and
called it a day's quota — cheap to retry, expensive to get wrong, so this errs
the other way.
"""

RETRIES = len(BACKOFF_S) + 1
PAUSE_S = 1.5
"""Breathing room between scenarios. Cheaper than discovering the rate limit
one scenario at a time."""

log = get_logger("day16.build")


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "resource_exhausted" in text or "rate" in text and "limit" in text


def _is_daily_quota(exc: Exception) -> bool:
    """Whether this 429 is the daily allowance, which waiting will not fix."""
    text = str(exc).lower().replace("_", "")
    return any(marker.replace(" ", "") in text.replace(" ", "") for marker in PER_DAY_MARKERS)


def adk_runner(model: str) -> callable:
    """Day 6's `Runner` contract: a Scenario in, an Outcome out."""

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
                if _is_rate_limit(exc) and attempt == RETRIES - 1:
                    raise QuotaExhausted(redact(exc)[:300]) from exc
                if not _is_rate_limit(exc):
                    log.info(
                        "scenario failed",
                        extra={"scenario": scenario.id, "error": redact(exc)[:160]},
                    )
                    return Outcome(
                        scenario_id=scenario.id,
                        booked=False,
                        captured={},
                        transcript=[],
                        escalated=False,
                        user_turns=len(scenario.utterances),
                        failed_steps=1,
                    )
                wait = BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)]
                log.info(
                    "rate limited, backing off",
                    extra={"seconds": wait, "attempt": attempt + 1, "of": RETRIES},
                )
                time.sleep(wait)
        else:  # pragma: no cover - the loop always breaks or returns
            raise last  # type: ignore[misc]

        transcript: list[dict[str, str]] = []
        for utterance, turn in zip(scenario.utterances, turns, strict=True):
            transcript.append({"role": "user", "content": utterance})
            transcript.append({"role": "assistant", "content": turn.reply})

        time.sleep(PAUSE_S)
        return Outcome(
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

    return run


def _engine(scenarios: tuple[Scenario, ...]) -> dict:
    """Day 4's state machine over the same scenarios, scored the same way.

    Re-run rather than read from day 5's stored comparison, for two reasons.
    Day 5 scored only 12 of these scenarios and its assertions predate day 6's
    dimensions, so that run cannot fill a safety column. And the engine has no
    model in it: re-running it is deterministic and free, which makes it the
    one comparator that never needs a caveat.
    """
    return run_suite("day16-engine", engine_runner, scenarios=scenarios).as_dict()


def _day5(which: str) -> dict | None:
    """Day 5's two agent passes, kept for the only comparison they support.

    These ran on day 5's own 12-scenario suite with untagged assertions, so
    they belong in a pass-rate column and nowhere near the dimension table.
    `results.py` is what enforces that.
    """
    from convo.projects.day05_tool_agent.results import load

    found = load()
    if found is None:
        return None
    return found.grounded.run.as_dict() if which == "grounded" else found.schema_only.run.as_dict()


def _rescore_stored() -> int:
    """Recompute the free half of the artifact, keeping the stored ADK run.

    The ADK run is the only part that costs anything — a day's quota and
    twenty minutes of wall clock. Everything beside it is deterministic and
    free: day 4's engine has no model in it, and day 5's passes are read off
    disk. So when a comparator changes, as it did when the engine column moved
    from day 5's 12-scenario run to day 6's full one, there is no reason to
    pay for the model a second time to find out.

    What this cannot do is re-score ADK itself. The artifact keeps scored
    `CaseResult`s, not the raw `Outcome`s, so a change to day 6's *scorers*
    needs a real run. That is a deliberate limit and not a hidden one.
    """
    out = ARTIFACT_DIR / "adk.json"
    if not out.exists():
        raise SystemExit(f"nothing to rescore: {out} does not exist")
    payload = json.loads(out.read_text(encoding="utf-8"))

    kept = {c["case_id"] for c in payload["adk"]["cases"]}
    attempted = tuple(s for s in SUITE if s.id in kept)
    payload["engine"] = _engine(attempted)
    payload["schema_only"] = _day5("schema")
    payload["grounded"] = _day5("grounded")
    payload["rescored"] = True

    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log.info(
        "comparators rebuilt, ADK run kept",
        extra={"scenarios": len(attempted), "adk_pass_rate": payload["adk"]["pass_rate"]},
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="run only the first N scenarios")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="rebuild the free comparators from the stored ADK run, calling no model",
    )
    args = parser.parse_args(argv)

    if args.rescore:
        return _rescore_stored()

    resolved = resolve()
    scenarios = SUITE[: args.limit] if args.limit else SUITE
    log.info(
        "running the suite",
        extra={"model": resolved.model, "scenarios": len(scenarios)},
    )

    # Each scenario is run exactly once and its Outcome cached, then the
    # cache is handed to day 6's scorer. Letting `run_suite` drive the agent
    # directly would be simpler and would spend the daily quota twice over —
    # once to find out whether the quota holds, once to score it.
    live = adk_runner(resolved.model)
    cache: dict[str, Outcome] = {}
    attempted: list[Scenario] = []
    quota_note = ""
    for scenario in scenarios:
        try:
            cache[scenario.id] = live(scenario)
        except QuotaExhausted as exc:
            quota_note = str(exc)
            log.info(
                "free tier exhausted",
                extra={"completed": len(attempted), "of": len(scenarios)},
            )
            break
        attempted.append(scenario)
        log.info(
            "scenario done",
            extra={
                "scenario": scenario.id,
                "booked": cache[scenario.id].booked,
                "calls": cache[scenario.id].model_calls,
                "progress": f"{len(attempted)}/{len(scenarios)}",
            },
        )

    if not attempted:
        raise SystemExit(
            "the free tier was exhausted before any scenario completed. "
            "Daily quota is per model and resets after 24h."
        )

    run: EvalRun = run_suite(
        "day16-adk",
        lambda scenario: cache[scenario.id],
        scenarios=tuple(attempted),
        meta={"model": resolved.model, "framework": "google-adk"},
    )

    payload = {
        "model": resolved.model,
        "models_listed": resolved.listed,
        "rejected_models": list(resolved.rejected),
        "scenarios": len(attempted),
        "of_suite": len(SUITE),
        "partial": len(attempted) < len(SUITE),
        "quota_note": quota_note,
        "adk": run.as_dict(),
        # The raw outcomes, so a scorer change can be re-applied without
        # paying for another run. Added on day 18, which found a bug in day
        # 6's `score_safety` and then could not correct the figures here
        # because only the verdicts had been kept.
        "outcomes": [cache[scenario.id].as_dict() for scenario in attempted],
        "engine": _engine(tuple(attempted)),
        "schema_only": _day5("schema"),
        "grounded": _day5("grounded"),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT_DIR / "adk.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log.info(
        "artifact written",
        extra={
            "path": str(out),
            "pass_rate": run.pass_rate,
            "kilobytes": max(out.stat().st_size // 1024, 1),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
