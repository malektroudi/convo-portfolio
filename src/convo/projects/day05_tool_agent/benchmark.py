"""Run both approaches over the same scenarios and save the comparison.

    uv sync --extra train
    uv run python -m convo.projects.day05_tool_agent.benchmark
    uv run python -m convo.projects.day05_tool_agent.benchmark --limit 2  # smoke

The model default is small deliberately. A 0.5B instruct model on CPU is what a
free tier actually gives you, and its failure rate is the measurement: it shows
how much work the validation-and-retry layer is doing, which a model that never
emits a bad call would hide entirely.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from convo.core import get_logger, setup
from convo.core.evals import CaseResult, EvalRun
from convo.projects.day05_tool_agent.agent import AgentConfig, new_session, turn
from convo.projects.day05_tool_agent.backends import Scripted
from convo.projects.day05_tool_agent.scenarios import (
    REF,
    SCENARIOS,
    Scenario,
    assertions_for,
    normalise_booking,
    run_engine,
)

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
log = get_logger("day05.benchmark")


PASS_LABELS = {
    "schema_only": "Agent · schema validation only",
    "with_grounding": "Agent · schema + provenance guard",
}


@dataclass
class AgentStats:
    """What the agent cost, beyond whether it was right."""

    model_calls: int = 0
    failed_steps: int = 0
    unparseable: int = 0
    unknown_tool: int = 0
    invalid_args: int = 0
    ungrounded: int = 0
    tool_error: int = 0
    retries: int = 0
    exhausted_conversations: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    wall_seconds: float = 0.0
    parsed_how: dict[str, int] = field(default_factory=dict)

    @property
    def failure_rate(self) -> float:
        return round(self.failed_steps / self.model_calls, 4) if self.model_calls else 0.0

    def as_dict(self) -> dict:
        return {**asdict(self), "failure_rate": self.failure_rate}


def run_agent_suite(
    backend, scenarios: tuple[Scenario, ...], config: AgentConfig
) -> tuple[EvalRun, AgentStats]:
    """Run the agent over every scenario, accumulating cost as well as verdicts."""
    run = EvalRun(suite="day05-agent", meta={"backend": backend.name})
    stats = AgentStats()
    started = time.perf_counter()

    for index, scenario in enumerate(scenarios, start=1):
        session = new_session(backend, REF, config)
        transcript = []
        for utterance in scenario.utterances:
            result = turn(session, utterance)
            transcript.append({"role": "user", "content": utterance})
            transcript.append({"role": "assistant", "content": result.reply})

        booked = session.booking is not None
        case = run.add(
            CaseResult.from_assertions(
                scenario.id,
                assertions_for(scenario, booked, normalise_booking(session.booking)),
                transcript=transcript,
                latency_ms=round(sum(s.latency_ms for s in session.steps), 3),
                tags=list(scenario.tags),
            )
        )

        stats.model_calls += session.model_calls
        stats.failed_steps += session.failed_steps
        stats.exhausted_conversations += int(session.exhausted)
        tokens_in, tokens_out = session.tokens
        stats.tokens_in += tokens_in
        stats.tokens_out += tokens_out
        for step in session.steps:
            if step.kind in {
                "unparseable",
                "unknown_tool",
                "invalid_args",
                "ungrounded",
                "tool_error",
            }:
                setattr(stats, step.kind, getattr(stats, step.kind) + 1)
            if step.retry:
                stats.retries += 1
            if step.parsed_how:
                stats.parsed_how[step.parsed_how] = stats.parsed_how.get(step.parsed_how, 0) + 1

        case.judge_rationale = f"{session.model_calls} calls, {session.failed_steps} failed"
        log.info(
            "scenario done",
            extra={
                "n": f"{index}/{len(scenarios)}",
                "id": scenario.id,
                "passed": case.passed,
                "calls": session.model_calls,
                "failed": session.failed_steps,
            },
        )

    stats.wall_seconds = round(time.perf_counter() - started, 1)
    return run, stats


def main(argv: list[str] | None = None) -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-model-calls", type=int, default=14)
    parser.add_argument("--limit", type=int, default=0, help="first N scenarios only")
    parser.add_argument(
        "--scripted", action="store_true", help="dry-run the plumbing with no model"
    )
    args = parser.parse_args(argv)

    scenarios = SCENARIOS[: args.limit] if args.limit else SCENARIOS
    config = AgentConfig(max_retries=args.max_retries, max_model_calls=args.max_model_calls)

    engine_run = EvalRun(suite="day04-engine")
    for scenario in scenarios:
        engine_run.add(run_engine(scenario))
    log.info(
        "state machine done",
        extra={"pass_rate": engine_run.pass_rate, "scenarios": len(scenarios)},
    )

    if args.scripted:
        backend = Scripted([], name="scripted-dry-run")
    else:
        from convo.projects.day05_tool_agent.backends import LocalTransformers

        backend = LocalTransformers(model_id=args.model, max_new_tokens=args.max_new_tokens)
        backend.load()

    # Two passes. The first measures what schema validation alone is worth; the
    # second adds the provenance guard. The difference between them answers
    # "is checking shape enough?", which the first run answered loudly.
    passes: dict[str, dict] = {}
    for label, grounding in (("schema_only", False), ("with_grounding", True)):
        pass_config = AgentConfig(
            max_retries=config.max_retries,
            max_model_calls=config.max_model_calls,
            check_grounding=grounding,
        )
        agent_run, stats = run_agent_suite(backend, scenarios, pass_config)
        passes[label] = {"run": agent_run.as_dict(), "stats": stats.as_dict()}
        log.info(
            "agent pass done",
            extra={
                "pass": label,
                "pass_rate": agent_run.pass_rate,
                "model_calls": stats.model_calls,
                "failure_rate": stats.failure_rate,
                "wall_seconds": stats.wall_seconds,
            },
        )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "comparison.json").write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "id": s.id,
                        "utterances": list(s.utterances),
                        "tags": list(s.tags),
                        "note": s.note,
                        "expect_booked": s.expect_booked,
                    }
                    for s in scenarios
                ],
                "engine": engine_run.as_dict(),
                "agent_passes": passes,
                "config": {
                    "model": backend.name,
                    "max_new_tokens": args.max_new_tokens,
                    "max_retries": args.max_retries,
                    "max_model_calls": args.max_model_calls,
                    "reference_date": REF.today.isoformat(),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("comparison written", extra={"path": str(ARTIFACT_DIR / "comparison.json")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
