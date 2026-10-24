"""The gate has to fail, or it is decoration.

The important case is `test_a_safety_trade_fails_even_though_the_pass_rate_is_flat`.
That is the day 5 shape inverted: a change that trades safety away while the
headline number holds steady. A gate that only watches the overall pass rate
waves it through, which is precisely the hole dimensions were added to close.
"""

from __future__ import annotations

from convo.core.evals import EFFICIENCY, SAFETY, TASK, Assertion, CaseResult, EvalRun
from convo.projects.day06_eval_harness import gate
from convo.projects.day06_eval_harness.harness import engine_runner, run_suite
from convo.projects.day06_eval_harness.suite import SUITE


def case(case_id: str, **dims: bool) -> CaseResult:
    return CaseResult.from_assertions(
        case_id,
        [Assertion(f"{d} check", ok, f"{d}", dimension=d) for d, ok in dims.items()],
    )


def run(*cases: CaseResult) -> EvalRun:
    out = EvalRun(suite="s")
    for c in cases:
        out.add(c)
    return out


class TestThresholds:
    def test_a_clean_run_reports_nothing(self):
        current = run(case("a", safety=True, task=True, efficiency=True))
        breaches, regressions = gate.report(current, None)
        assert breaches == [] and regressions == []

    def test_a_single_safety_failure_breaches_the_bar(self):
        # Safety's bar is 1.0, so one failure is enough.
        current = run(
            case("a", safety=False, task=True, efficiency=True),
            *[case(f"ok{i}", safety=True, task=True, efficiency=True) for i in range(20)],
        )
        breaches, _ = gate.report(current, None)
        assert any(b.startswith(SAFETY) for b in breaches)
        assert any("a" in b for b in breaches), "the offending case is named"

    def test_task_tolerates_a_little_failure(self):
        # Task's bar is 0.9, so one in twenty is fine.
        current = run(
            case("a", task=False, safety=True, efficiency=True),
            *[case(f"ok{i}", task=True, safety=True, efficiency=True) for i in range(20)],
        )
        breaches, _ = gate.report(current, None)
        assert not any(b.startswith(TASK) for b in breaches)

    def test_the_breach_message_states_the_rate_and_the_bar(self):
        current = run(case("a", efficiency=False))
        breaches, _ = gate.report(current, None)
        assert any(EFFICIENCY in b and "0.0%" in b and "90%" in b for b in breaches)


class TestRegressions:
    def test_a_case_that_stops_passing_is_reported(self):
        before = run(case("a", task=True, safety=True, efficiency=True))
        after = run(case("a", task=False, safety=True, efficiency=True))
        _, regressions = gate.report(after, before)
        assert any("used to pass" in r and "a" in r for r in regressions)

    def test_a_safety_trade_fails_even_though_the_pass_rate_is_flat(self):
        # Both runs score 0.0 overall. What changed is WHICH dimension broke.
        before = run(case("a", task=False, safety=True, efficiency=True))
        after = run(case("a", task=True, safety=False, efficiency=True))
        assert before.pass_rate == after.pass_rate

        _, regressions = gate.report(after, before)
        assert any(r.startswith(f"{SAFETY} regressed") for r in regressions), (
            "a flat pass rate must not hide a safety trade"
        )

    def test_an_improvement_is_not_a_regression(self):
        before = run(case("a", safety=False, task=True, efficiency=True))
        after = run(case("a", safety=True, task=True, efficiency=True))
        _, regressions = gate.report(after, before)
        assert regressions == []

    def test_with_no_baseline_only_thresholds_apply(self):
        current = run(case("a", safety=True, task=True, efficiency=True))
        _, regressions = gate.report(current, None)
        assert regressions == []


class TestEndToEnd:
    def test_the_real_suite_passes_its_committed_baseline(self):
        assert gate.main([]) == 0

    def test_a_broken_bot_makes_the_gate_exit_nonzero(self, monkeypatch):
        def liar(scenario):
            outcome = engine_runner(scenario)
            if outcome.captured:
                outcome.captured["name"] = "john doe"
            return outcome

        monkeypatch.setattr(gate, "engine_runner", liar)
        assert gate.main([]) == 1, "an inventing bot must not pass the gate"

    def test_the_committed_baseline_matches_the_current_suite(self):
        path = gate.baseline_path("engine")
        assert path.exists(), "the baseline must be committed, not generated in CI"

        baseline = EvalRun.load(path)
        current = run_suite("check", engine_runner, SUITE)
        assert {c.case_id for c in baseline.cases} == {c.case_id for c in current.cases}, (
            "baseline is stale — re-record it with --update-baseline"
        )
