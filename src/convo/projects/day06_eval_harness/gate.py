"""The CI gate: run the suite, compare to a committed baseline, fail loudly.

    uv run python -m convo.projects.day06_eval_harness.gate
    uv run python -m convo.projects.day06_eval_harness.gate --update-baseline

Two things can fail the build, and keeping them separate matters:

  a THRESHOLD breach   the run is below the bar for a dimension, in absolute
                       terms, regardless of history
  a REGRESSION         something that used to pass now does not, even if the
                       absolute numbers still clear the bar

Day 5 is the reason the second exists. A change there eliminated ten false
confirmations while the pass rate did not move; the mirror image — a change
that quietly trades safety for task success — would sail through any gate
watching one number. The baseline makes that visible.

Safety's bar is 1.0 on purpose. A bot that fails to help is recoverable; a bot
that tells a guest their table is booked when it is not sends someone to a
restaurant that is not expecting them. Those do not belong on one scale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from convo.core import get_logger, setup
from convo.core.evals import EFFICIENCY, SAFETY, TASK, EvalRun
from convo.projects.day06_eval_harness.harness import engine_runner, run_suite
from convo.projects.day06_eval_harness.suite import SUITE

BASELINE_DIR = Path(__file__).parent / "baselines"
THRESHOLDS: dict[str, float] = {SAFETY: 1.0, TASK: 0.9, EFFICIENCY: 0.9}

log = get_logger("day06.gate")


def baseline_path(runner_name: str) -> Path:
    return BASELINE_DIR / f"{runner_name}.json"


def report(run: EvalRun, previous: EvalRun | None) -> tuple[list[str], list[str]]:
    """Return (threshold breaches, regression messages)."""
    breaches: list[str] = []
    for dimension, bar in THRESHOLDS.items():
        rate = run.pass_rate_in(dimension)
        if rate < bar:
            offenders = ", ".join(c.case_id for c in run.failures_in(dimension)[:5])
            breaches.append(f"{dimension}: {rate:.1%} is below the {bar:.0%} bar — {offenders}")

    regressions: list[str] = []
    if previous is not None:
        diff = run.compare(previous)
        if diff.regressions:
            regressions.append(
                f"cases that used to pass and now fail: {', '.join(diff.regressions)}"
            )
        for dimension in diff.regressed_dimensions():
            regressions.append(
                f"{dimension} regressed on: {', '.join(diff.by_dimension[dimension])}"
            )
    return breaches, regressions


def main(argv: list[str] | None = None) -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", default="engine", choices=["engine"])
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="record this run as the new baseline instead of judging against it",
    )
    args = parser.parse_args(argv)

    run = run_suite(f"day06-{args.runner}", engine_runner, SUITE)
    rates = run.pass_rate_by_dimension()
    log.info(
        "suite run",
        extra={
            "scenarios": len(SUITE),
            "overall": run.pass_rate,
            **{f"rate_{k}": v for k, v in rates.items()},
        },
    )

    path = baseline_path(args.runner)
    if args.update_baseline:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        run.save(path)
        log.info("baseline written", extra={"path": str(path)})
        return 0

    previous = EvalRun.load(path) if path.exists() else None
    if previous is None:
        log.warning(
            "no baseline to compare against; thresholds only",
            extra={"expected_at": str(path)},
        )

    breaches, regressions = report(run, previous)

    for line in breaches:
        log.error("threshold breach", extra={"detail": line})
    for line in regressions:
        log.error("regression", extra={"detail": line})

    if breaches or regressions:
        print("\nFAILING CASES", file=sys.stderr)
        for case in run.failures:
            for assertion in case.assertions:
                if not assertion.passed:
                    print(
                        f"  {case.case_id} [{assertion.dimension}] "
                        f"{assertion.name}: {assertion.detail}",
                        file=sys.stderr,
                    )
        return 1

    log.info("gate passed", extra={"overall": run.pass_rate, **rates})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
