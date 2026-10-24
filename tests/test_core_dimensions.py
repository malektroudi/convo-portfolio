"""Dimension scoring and the regression gate.

The case that matters most is `test_a_safety_regression_is_caught_when_the_pass_rate_is_flat`.
Day 5 produced exactly that shape — a change that eliminated ten false
confirmations while the pass rate did not move — and a gate that only watches
the headline number would have called the fix a no-op.
"""

from __future__ import annotations

from convo.core.evals import (
    EFFICIENCY,
    SAFETY,
    TASK,
    Assertion,
    CaseResult,
    EvalRun,
)


def case(case_id: str, **dims: bool) -> CaseResult:
    """A case asserting one claim per named dimension."""
    assertions = [
        Assertion(f"{d} check", passed, f"{d}={passed}", dimension=d) for d, passed in dims.items()
    ]
    return CaseResult.from_assertions(case_id, assertions)


def run(*cases: CaseResult, suite: str = "s") -> EvalRun:
    out = EvalRun(suite=suite)
    for c in cases:
        out.add(c)
    return out


class TestAssertionsCarryDimensions:
    def test_a_dimension_round_trips_through_json(self):
        original = run(case("a", task=True, safety=False))
        restored = EvalRun(
            suite="s",
            cases=[CaseResult.from_dict(c) for c in original.as_dict()["cases"]],
        )
        assert restored.cases[0].assertions[0].dimension == TASK
        assert restored.cases[0].dimensions() == {TASK, SAFETY}

    def test_an_unclassified_assertion_still_counts_overall(self):
        c = CaseResult.from_assertions("a", [Assertion("plain", False)])
        assert c.passed is False
        assert c.dimensions() == set()

    def test_an_unclassified_assertion_belongs_to_no_dimension(self):
        c = CaseResult.from_assertions("a", [Assertion("plain", False)])
        assert c.scored_in(TASK) is False
        assert c.passed_in(TASK) is True  # vacuous


class TestCaseLevelDimensions:
    def test_passed_in_is_per_dimension_not_overall(self):
        c = case("a", task=True, safety=False)
        assert c.passed is False, "the case fails overall"
        assert c.passed_in(TASK) is True, "but task is fine"
        assert c.passed_in(SAFETY) is False

    def test_failures_in_names_only_that_dimension(self):
        c = case("a", task=False, safety=False)
        assert [a.dimension for a in c.failures_in(SAFETY)] == [SAFETY]

    def test_a_dimension_nobody_measured_passes_vacuously(self):
        c = case("a", task=True)
        assert c.passed_in(EFFICIENCY) is True
        assert c.scored_in(EFFICIENCY) is False


class TestRunLevelRates:
    def test_rate_is_scoped_to_cases_that_measure_the_dimension(self):
        # Three cases, only two say anything about safety.
        r = run(
            case("a", task=True, safety=True),
            case("b", task=True, safety=False),
            case("c", task=True),
        )
        assert r.pass_rate_in(SAFETY) == 0.5, "2 measured, 1 passed"
        assert r.pass_rate_in(TASK) == 1.0

    def test_adding_cases_that_ignore_a_dimension_cannot_inflate_it(self):
        before = run(case("a", safety=False))
        after = run(case("a", safety=False), case("b", task=True), case("c", task=True))
        assert before.pass_rate_in(SAFETY) == after.pass_rate_in(SAFETY) == 0.0

    def test_an_unmeasured_dimension_reports_zero_not_one(self):
        assert run(case("a", task=True)).pass_rate_in(SAFETY) == 0.0

    def test_by_dimension_covers_every_dimension_present(self):
        r = run(case("a", task=True, safety=False), case("b", efficiency=True))
        assert set(r.pass_rate_by_dimension()) == {TASK, SAFETY, EFFICIENCY}

    def test_failures_in_lists_the_offending_cases(self):
        r = run(case("a", safety=True), case("b", safety=False))
        assert [c.case_id for c in r.failures_in(SAFETY)] == ["b"]


class TestGates:
    def test_each_dimension_is_held_to_its_own_bar(self):
        r = run(
            case("a", task=True, safety=True),
            case("b", task=False, safety=True),
        )
        gates = r.gate_by_dimension({TASK: 0.9, SAFETY: 1.0})
        assert gates == {TASK: False, SAFETY: True}

    def test_safety_can_fail_while_task_passes(self):
        r = run(case("a", task=True, safety=False))
        gates = r.gate_by_dimension({TASK: 1.0, SAFETY: 1.0})
        assert gates[TASK] is True
        assert gates[SAFETY] is False

    def test_a_dimension_with_no_bar_is_not_gated(self):
        r = run(case("a", task=True, safety=False))
        assert set(r.gate_by_dimension({TASK: 1.0})) == {TASK}


class TestRegressionComparison:
    def test_a_safety_regression_is_caught_when_the_pass_rate_is_flat(self):
        # The day 5 shape: the case fails both before and after, so the overall
        # verdict never changes, but WHY it fails moved from task to safety.
        before = run(case("a", task=False, safety=True))
        after = run(case("a", task=True, safety=False))

        assert before.pass_rate == after.pass_rate == 0.0
        diff = after.compare(before)
        assert diff.regressions == [], "no case flipped overall"
        assert diff.by_dimension[SAFETY] == ["a"]
        assert diff.regressed_dimensions() == [SAFETY]
        assert diff.clean is False, "a flat pass rate must not read as clean"

    def test_a_clean_run_is_clean_on_every_axis(self):
        before = run(case("a", task=True, safety=True))
        after = run(case("a", task=True, safety=True))
        diff = after.compare(before)
        assert diff.clean is True
        assert diff.regressed_dimensions() == []

    def test_an_improvement_in_one_dimension_is_not_a_regression(self):
        before = run(case("a", safety=False))
        after = run(case("a", safety=True))
        assert after.compare(before).clean is True

    def test_newly_measuring_a_dimension_reads_as_a_case_drop_not_a_dimension_drop(self):
        # Adding a check that immediately fails does make the case worse
        # overall — you have found a real problem. But safety did not regress:
        # it was never measured before, so the two lists say different things
        # and together they distinguish "newly broken" from "newly looked at".
        before = run(case("a", task=True))
        after = run(case("a", task=True, safety=False))
        diff = after.compare(before)

        assert diff.regressions == ["a"], "the case did get worse"
        assert diff.by_dimension[SAFETY] == [], "but safety did not regress"
        assert diff.clean is False

    def test_a_dimension_dropped_from_a_case_is_not_a_regression_either(self):
        before = run(case("a", task=True, safety=True))
        after = run(case("a", task=True))
        assert after.compare(before).clean is True

    def test_case_level_and_dimension_level_regressions_both_report(self):
        before = run(case("a", task=True, safety=True), case("b", task=True))
        after = run(case("a", task=True, safety=False), case("b", task=False))
        diff = after.compare(before)

        assert diff.regressions == ["a", "b"], "both cases flipped overall"
        assert diff.by_dimension[SAFETY] == ["a"], "but for different reasons"
        assert diff.by_dimension[TASK] == ["b"]
        assert diff.clean is False

    def test_new_and_removed_cases_are_reported_separately(self):
        before = run(case("a", task=True), case("gone", task=True))
        after = run(case("a", task=True), case("new", task=True))
        diff = after.compare(before)
        assert diff.added == ["new"]
        assert diff.removed == ["gone"]
        assert diff.clean is True


class TestSerialisation:
    def test_dimension_rates_ride_along_in_the_artifact(self):
        payload = run(case("a", task=True, safety=False)).as_dict()
        assert payload["pass_rate_by_dimension"] == {TASK: 1.0, SAFETY: 0.0}

    def test_a_run_survives_save_and_load_with_dimensions(self, tmp_path):
        original = run(case("a", task=True, safety=False), case("b", safety=True))
        path = original.save(tmp_path / "run.json")
        loaded = EvalRun.load(path)

        assert loaded.pass_rate_by_dimension() == original.pass_rate_by_dimension()
        assert loaded.compare(original).clean is True
