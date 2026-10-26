"""Both evaluators as CI gates, neither of which may call a model.

    uv run python -m convo.projects.day18_eval_gate.gate
    uv run python -m convo.projects.day18_eval_gate.gate --update-baseline

Day 6's gate already runs in CI against day 4's engine: deterministic, free,
no key. Adding ADK's evaluator beside it looked impossible at first, because
the obvious way to run it is `AgentEvaluator`, which re-runs the agent — and
CI has no `GOOGLE_API_KEY`, no quota and no business having either.

The way through is that a trajectory metric does not need a live agent. It
needs a recorded trajectory and an expected one. Both are in the committed
artifact, so ADK's `TrajectoryEvaluator` runs in CI over stored data and
produces exactly the numbers it produced locally — and if it does not, that is
itself the failure this gate exists to catch.

What can fail the build, and what deliberately cannot:

  a REGRESSION            the recorded comparison got worse — the harness
                          pass rate dropped, the evaluators agree less, or a
                          scenario's agreement flipped
  an IRREPRODUCIBLE score ADK's evaluator, re-run over the same stored
                          trajectories, no longer agrees with what was
                          committed — a scorer changed under us

**Day 6's absolute bars are not applied here, on purpose.** The first version
did apply them, and CI went red and stayed red: this run's safety rate is
90.3% against day 6's 1.0 safety bar, because the ADK agent issues false
confirmations. That is a finding, it is documented, and re-asserting it on
every build is a stuck alarm rather than a gate.

Absolute bars belong to the bot that would ship. Day 6's gate applies them to
day 4's engine, which clears them, and that check is unchanged. Day 18's
subject is an experiment being measured, so the useful question is not "is it
good enough" — it is already known not to be — but "have its recorded numbers
moved without anyone saying so".

The third is the one worth having. Every published figure in this repository
is supposed to be re-derivable from a committed artifact; this is the only day
where a *dependency* does the deriving, and a silent change in ADK's matching
rules would quietly rewrite day 18's conclusions. Pinning the version is not
enough, because the version is installed with `--no-deps` outside the lockfile.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from convo.core import get_logger, setup
from convo.projects.day18_eval_gate.results import load

BASELINE = Path(__file__).parent / "baselines" / "evaluators.json"

MAX_AGREEMENT_DROP = 0.0
"""How far the agreement rate may fall before the build fails.

Zero. Not because agreement is good in itself — the two evaluators measure
different things and some disagreement is the point — but because a *change*
in how often they agree means one of them changed, and that is always worth a
human looking at it. Raising this to tolerate noise would be wrong: nothing
here is stochastic, both evaluators read the same committed file.
"""

log = get_logger("day18.gate")


@dataclass(frozen=True)
class Baseline:
    agreement_rate: float
    verdicts: dict[str, bool]
    """Per scenario: did the two evaluators agree? Keyed by scenario id."""
    harness_pass_rate: float

    @classmethod
    def from_payload(cls, payload: dict) -> Baseline:
        return cls(
            agreement_rate=payload["agreement_rate"],
            verdicts=dict(payload["verdicts"]),
            harness_pass_rate=payload["harness_pass_rate"],
        )

    def as_dict(self) -> dict:
        return {
            "agreement_rate": self.agreement_rate,
            "verdicts": self.verdicts,
            "harness_pass_rate": self.harness_pass_rate,
        }


def current_baseline() -> Baseline:
    found = load()
    if found is None:
        raise SystemExit(
            "no day 18 artifact; run `python -m convo.projects.day18_eval_gate.build` first"
        )
    return Baseline(
        agreement_rate=found.agreement_rate,
        verdicts={row.scenario_id: row.agreement.agree for row in found.rows},
        harness_pass_rate=found.harness.pass_rate,
    )


def recheck_adk() -> list[str]:
    """Re-score the stored trajectories with ADK and compare to what was committed.

    Returns a list of complaints, empty when every score reproduces. Needs
    `google-adk` installed; the caller decides whether its absence is fatal.
    """
    from convo.projects.day18_eval_gate.adk_eval import evaluate
    from convo.projects.day18_eval_gate.trajectory import Expected

    found = load()
    assert found is not None
    complaints: list[str] = []
    for stored in found.trajectories:
        verdict = evaluate(
            stored.scenario_id,
            stored.utterances,
            stored.actual,
            Expected(scenario_id=stored.scenario_id, per_turn=stored.expected),
        )
        if verdict.passed != stored.passed or abs(verdict.score - stored.score) > 1e-9:
            complaints.append(
                f"{stored.scenario_id}: committed {stored.score} "
                f"({'pass' if stored.passed else 'fail'}), "
                f"recomputed {verdict.score} ({'pass' if verdict.passed else 'fail'})"
            )
    return complaints


def report(now: Baseline, previous: Baseline | None) -> list[str]:
    """Regressions against the committed baseline. Empty means nothing moved."""
    if previous is None:
        return []

    regressions: list[str] = []
    drop = previous.agreement_rate - now.agreement_rate
    if drop > MAX_AGREEMENT_DROP:
        regressions.append(
            f"the evaluators agree less than they did: "
            f"{previous.agreement_rate:.1%} -> {now.agreement_rate:.1%}"
        )
    if now.harness_pass_rate < previous.harness_pass_rate:
        regressions.append(
            f"day 6's pass rate on the stored run dropped: "
            f"{previous.harness_pass_rate:.1%} -> {now.harness_pass_rate:.1%}"
        )
    flipped = [
        scenario
        for scenario, agreed in previous.verdicts.items()
        if scenario in now.verdicts and now.verdicts[scenario] != agreed
    ]
    if flipped:
        regressions.append("agreement flipped on: " + ", ".join(sorted(flipped)))
    return regressions


def main(argv: list[str] | None = None) -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="record the current comparison as the new baseline",
    )
    parser.add_argument(
        "--require-adk",
        action="store_true",
        help="fail if google-adk is missing instead of skipping the recheck",
    )
    args = parser.parse_args(argv)

    now = current_baseline()

    if args.update_baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(now.as_dict(), indent=1), encoding="utf-8")
        log.info("baseline written", extra={"path": str(BASELINE)})
        return 0

    previous = (
        Baseline.from_payload(json.loads(BASELINE.read_text(encoding="utf-8")))
        if BASELINE.exists()
        else None
    )
    if previous is None:
        log.warning("no baseline to compare against; thresholds only")

    irreproducible: list[str] = []
    try:
        irreproducible = recheck_adk()
        log.info("ADK's evaluator reproduced every committed score")
    except ImportError as exc:
        if args.require_adk:
            log.error("google-adk is not installed", extra={"detail": str(exc)[:120]})
            return 1
        log.warning(
            "google-adk not installed; skipping the second evaluator",
            extra={"install": "uv pip install --no-deps google-adk"},
        )

    regressions = report(now, previous)

    for line in regressions:
        log.error("regression", extra={"detail": line})
    for line in irreproducible:
        log.error("ADK's score did not reproduce", extra={"detail": line})

    if regressions or irreproducible:
        print("\nWHERE THE EVALUATORS DISAGREE", file=sys.stderr)
        found = load()
        assert found is not None
        for row in found.disagreements:
            print(
                f"  {row.scenario_id}: {row.agreement.value} "
                f"[{row.adjudication.value}] {row.adk_detail or row.harness_detail}",
                file=sys.stderr,
            )
        return 1

    log.info(
        "gate passed",
        extra={
            "harness_pass_rate": now.harness_pass_rate,
            "agreement_rate": now.agreement_rate,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
