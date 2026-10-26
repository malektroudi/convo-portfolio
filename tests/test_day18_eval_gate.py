"""Day 18: that the oracle is derived, the two alignments mean what they say,
and the gate can actually run where it has to run.

Nothing here calls a model, and nothing here needs `google-adk` except the
handful of tests that exercise ADK's evaluator directly — those skip. That is
deliberate and it is the same property the gate depends on: CI must be able to
check thresholds and regressions on a machine where the framework cannot be
installed at all.
"""

from __future__ import annotations

import sys
from importlib.abc import MetaPathFinder

import pytest

from convo.core.evals import Assertion, CaseResult
from convo.projects.day06_eval_harness.suite import SUITE
from convo.projects.day18_eval_gate import compare as cmp
from convo.projects.day18_eval_gate import trajectory as traj


class TestTheOracleIsDerived:
    """The expected trajectory must come from day 4, not from me."""

    def test_it_agrees_with_every_scenario_label(self):
        """If the derived trajectory contradicts the suite's own
        `expect_booked`, one of them is wrong and no result built on either is
        worth reading. This is the check `build` runs before spending a single
        model call."""
        for scenario in SUITE:
            expected = traj.expected_for(scenario)
            assert traj.disagrees_with_suite(scenario, expected) == "", scenario.id

    def test_a_booking_scenario_books_exactly_once(self):
        for scenario in SUITE:
            if not scenario.expect_booked:
                continue
            expected = traj.expected_for(scenario)
            calls = sum(t.count(traj.BOOK_TOOL) for t in expected.per_turn)
            assert calls == 1, (scenario.id, calls)

    def test_a_refusing_scenario_never_books(self):
        for scenario in SUITE:
            if scenario.expect_booked:
                continue
            expected = traj.expected_for(scenario)
            assert not expected.books, scenario.id

    def test_one_entry_per_user_turn(self):
        """ADK zips actual against expected with strict=True."""
        for scenario in SUITE:
            expected = traj.expected_for(scenario)
            assert len(expected.per_turn) == len(scenario.utterances), scenario.id

    def test_digressions_call_the_policy_tool(self):
        digressions = [s for s in SUITE if any("digression" in t for t in s.tags)]
        assert digressions, "the suite should still contain digression scenarios"
        for scenario in digressions:
            expected = traj.expected_for(scenario)
            assert any(traj.POLICY_TOOL in t for t in expected.per_turn), scenario.id

    def test_no_trajectory_is_written_by_hand(self):
        """A literal table of tool names per scenario would make the metric
        unfalsifiable — the answer key would be my memory of the run."""
        import inspect

        source = inspect.getsource(traj)
        for scenario in SUITE[:8]:
            assert scenario.id not in source, scenario.id

    def test_it_scores_the_executed_tool_not_the_advertised_one(self):
        from convo.projects.day05_tool_agent.tools import POLICY_TOPICS  # noqa: F401

        assert traj.BOOK_TOOL == "book_table"
        assert traj.POLICY_TOOL == "get_policy"

    def test_every_digression_maps_to_a_real_policy_topic(self):
        from convo.projects.day04_dialog_engine import flow as fl
        from convo.projects.day05_tool_agent.tools import POLICY_TOPICS

        for digression in fl.DIGRESSIONS:
            topic = traj.TOPIC_FOR_DIGRESSION.get(digression.name)
            assert topic, digression.name
            assert topic in POLICY_TOPICS, (digression.name, topic)


class TestTheAdjudicationCannotFavourEither:
    """`compare` must not quietly treat one evaluator as the ground truth."""

    def _case(self, name: str, task_ok: bool, other_ok: bool = True) -> CaseResult:
        return CaseResult.from_assertions(
            name,
            [
                Assertion("booked", task_ok, "", dimension="task"),
                Assertion("no invented details", other_ok, "", dimension="safety"),
            ],
        )

    def test_agreement_is_never_adjudicated(self):
        row = cmp.Row(
            "x",
            spec_ok=True,
            harness_ok=True,
            adk_ok=True,
            adk_score=1.0,
            adk_detail="",
            harness_detail="",
        )
        assert row.adjudication is cmp.Adjudication.AGREED

    def test_passing_a_wrong_outcome_convicts_the_evaluator(self):
        """ADK passed a conversation whose outcome the spec says was wrong."""
        row = cmp.Row(
            "x",
            spec_ok=False,
            harness_ok=False,
            adk_ok=True,
            adk_score=1.0,
            adk_detail="",
            harness_detail="",
        )
        assert row.adjudication is cmp.Adjudication.HARNESS_RIGHT

    def test_the_mirror_image_convicts_the_other_way(self):
        row = cmp.Row(
            "x",
            spec_ok=False,
            harness_ok=True,
            adk_ok=False,
            adk_score=0.5,
            adk_detail="",
            harness_detail="",
        )
        assert row.adjudication is cmp.Adjudication.ADK_RIGHT

    def test_a_route_disagreement_is_left_undecided(self):
        """Outcome correct, evaluators differ: the spec says nothing about the
        route, so scoring this against either would be unfair to the one that
        is measuring the route on purpose."""
        row = cmp.Row(
            "x",
            spec_ok=True,
            harness_ok=True,
            adk_ok=False,
            adk_score=0.5,
            adk_detail="",
            harness_detail="",
        )
        assert row.adjudication is cmp.Adjudication.UNDECIDED

    def test_the_spec_verdict_reads_only_the_task_dimension(self):
        """Safety and efficiency are judgements on top of the spec, not the
        spec. Folding them in would make the spec a copy of day 6's verdict."""
        case = self._case("x", task_ok=True, other_ok=False)
        assert cmp.spec_verdict(SUITE[0], case) is True
        assert case.passed is False

    def test_the_spec_verdict_is_false_without_any_task_assertion(self):
        case = CaseResult.from_assertions("x", [Assertion("odd", True, "", dimension="")])
        assert cmp.spec_verdict(SUITE[0], case) is False

    def test_a_tie_on_spec_agreement_goes_to_the_harness(self):
        """Not favouritism: day 6's is the one that can score a live bot in CI
        with no key, and a gate that cannot run is not a gate."""
        rows = tuple(
            cmp.Row(
                f"s{i}",
                spec_ok=True,
                harness_ok=True,
                adk_ok=True,
                adk_score=1.0,
                adk_detail="",
                harness_detail="",
            )
            for i in range(4)
        )
        summary = cmp.Summary(rows)
        assert summary.spec_agreement("harness") == summary.spec_agreement("adk")
        assert summary.gate_choice == "harness"

    def test_the_gate_switches_when_adk_is_strictly_better(self):
        rows = (
            cmp.Row(
                "a",
                spec_ok=False,
                harness_ok=True,
                adk_ok=False,
                adk_score=0.0,
                adk_detail="",
                harness_detail="",
            ),
            cmp.Row(
                "b",
                spec_ok=True,
                harness_ok=True,
                adk_ok=True,
                adk_score=1.0,
                adk_detail="",
                harness_detail="",
            ),
        )
        summary = cmp.Summary(rows)
        assert summary.spec_agreement("adk") > summary.spec_agreement("harness")
        assert summary.gate_choice == "adk"

    def test_agreement_rate_counts_both_kinds_of_agreeing(self):
        rows = (
            cmp.Row(
                "a",
                spec_ok=True,
                harness_ok=True,
                adk_ok=True,
                adk_score=1.0,
                adk_detail="",
                harness_detail="",
            ),
            cmp.Row(
                "b",
                spec_ok=False,
                harness_ok=False,
                adk_ok=False,
                adk_score=0.0,
                adk_detail="",
                harness_detail="",
            ),
            cmp.Row(
                "c",
                spec_ok=True,
                harness_ok=True,
                adk_ok=False,
                adk_score=0.5,
                adk_detail="",
                harness_detail="",
            ),
        )
        summary = cmp.Summary(rows)
        assert summary.agreed == 2
        assert summary.agreement_rate == round(2 / 3, 4)


class TestNothingHereNeedsTheFramework:
    """The half of the gate that checks thresholds must run without ADK."""

    @pytest.fixture
    def blocked(self):
        class Block(MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "google.adk" or name.startswith("google.adk."):
                    raise ImportError(f"blocked for this test: {name}")
                return None

        blocker = Block()
        sys.meta_path.insert(0, blocker)
        for name in [n for n in sys.modules if n.startswith("convo.projects.day18")]:
            del sys.modules[name]
        try:
            yield
        finally:
            sys.meta_path.remove(blocker)

    def test_every_day18_module_imports(self, blocked):
        import importlib

        for module in ("trajectory", "compare", "adk_eval", "results", "gate", "ui", ""):
            name = f"convo.projects.day18_eval_gate{'.' + module if module else ''}"
            importlib.import_module(name)

    def test_the_oracle_still_derives(self, blocked):
        import importlib

        module = importlib.import_module("convo.projects.day18_eval_gate.trajectory")
        assert module.expected_for(SUITE[0]).books

    def test_the_gate_skips_the_recheck_rather_than_crashing(self, blocked):
        """CI's main job has no ADK. The gate must still check thresholds
        there, and say out loud that it skipped the second evaluator."""
        import importlib

        gate = importlib.import_module("convo.projects.day18_eval_gate.gate")
        results = importlib.import_module("convo.projects.day18_eval_gate.results")
        if results.load() is None:
            pytest.skip("day 18 build not run")
        assert gate.main([]) in (0, 1)

    def test_require_adk_makes_the_absence_fatal(self, blocked):
        import importlib

        gate = importlib.import_module("convo.projects.day18_eval_gate.gate")
        results = importlib.import_module("convo.projects.day18_eval_gate.results")
        if results.load() is None:
            pytest.skip("day 18 build not run")
        assert gate.main(["--require-adk"]) == 1


class TestTheTwoAlignments:
    """Right tools at the wrong time is one behaviour, not thirty failures."""

    @pytest.fixture(autouse=True)
    def _needs_adk(self):
        pytest.importorskip("google.adk", reason="needs google-adk installed --no-deps")

    def test_an_identical_trajectory_passes_both(self):
        from convo.projects.day18_eval_gate.adk_eval import evaluate

        scenario = SUITE[0]
        expected = traj.expected_for(scenario)
        verdict = evaluate(scenario.id, scenario.utterances, expected.per_turn, expected)
        assert verdict.passed
        assert verdict.flat_ok
        assert not verdict.only_timing

    def test_booking_a_turn_early_is_flagged_as_timing_only(self):
        from convo.projects.day18_eval_gate.adk_eval import evaluate

        scenario = SUITE[0]
        expected = traj.expected_for(scenario)
        shifted = [list(t) for t in expected.per_turn]
        at = next(i for i, t in enumerate(shifted) if traj.BOOK_TOOL in t)
        assert at > 0, "needs a scenario that books after the first turn"
        shifted[at].remove(traj.BOOK_TOOL)
        shifted[at - 1].append(traj.BOOK_TOOL)

        verdict = evaluate(
            scenario.id, scenario.utterances, tuple(tuple(t) for t in shifted), expected
        )
        assert not verdict.passed
        assert verdict.flat_ok
        assert verdict.only_timing

    def test_calling_no_tools_at_all_fails_both(self):
        from convo.projects.day18_eval_gate.adk_eval import evaluate

        scenario = SUITE[0]
        expected = traj.expected_for(scenario)
        verdict = evaluate(
            scenario.id, scenario.utterances, tuple(() for _ in expected.per_turn), expected
        )
        assert not verdict.passed
        assert not verdict.flat_ok
        assert not verdict.only_timing

    def test_arguments_are_ignored(self):
        """Day 6 already checks every booking field. Re-checking args inside
        the trajectory metric would make the evaluators agree by construction
        on exactly the cases where a second opinion is the point."""
        import inspect

        from convo.projects.day18_eval_gate import adk_eval

        assert traj.IGNORE_ARGS is True
        assert "ignore_args=True" in inspect.getsource(adk_eval.evaluate)

    def test_a_short_trajectory_is_padded_rather_than_raising(self):
        """ADK zips strict=True; a conversation that ended early called no
        tools on the turns it never reached, and that should score, not crash."""
        from convo.projects.day18_eval_gate.adk_eval import evaluate

        scenario = SUITE[1]
        expected = traj.expected_for(scenario)
        verdict = evaluate(scenario.id, scenario.utterances, ((),), expected)
        assert len(verdict.per_turn) == len(expected.per_turn)
        assert not verdict.passed


class TestTheStoredComparison:
    @pytest.fixture(scope="class")
    def found(self):
        from convo.projects.day18_eval_gate.results import load

        got = load()
        if got is None:
            pytest.skip("day 18 build not run")
        return got

    def test_it_says_how_much_of_the_suite_ran(self, found):
        assert found.scenarios <= found.of_suite
        if found.scenarios < found.of_suite:
            assert found.partial
            assert found.quota_note

    def test_every_scenario_has_both_verdicts(self, found):
        assert {r.scenario_id for r in found.rows} == {t.scenario_id for t in found.trajectories}

    def test_the_trajectory_lengths_line_up(self, found):
        for stored in found.trajectories:
            assert len(stored.actual) == len(stored.expected), stored.scenario_id

    def test_utterances_come_from_the_suite_not_the_artifact(self, found):
        """Storing them would duplicate day 6's suite into a file that then has
        to be kept in step with it."""
        for stored in found.trajectories:
            assert stored.utterances, stored.scenario_id

    def test_rates_are_rates(self, found):
        for value in (
            found.agreement_rate,
            found.trajectory_pass_rate,
            found.flat_pass_rate,
            found.harness.pass_rate,
        ):
            assert 0.0 <= value <= 1.0, value

    def test_the_flat_rate_is_never_below_the_aligned_one(self, found):
        """Collapsing turns can only forgive, never convict."""
        assert found.flat_pass_rate >= found.trajectory_pass_rate

    def test_timing_only_cases_passed_flat_and_failed_aligned(self, found):
        for stored in found.trajectories:
            if stored.only_timing:
                assert stored.flat_ok and not stored.passed, stored.scenario_id

    def test_the_gate_choice_is_one_of_the_two(self, found):
        assert found.gate_choice in ("harness", "adk")
