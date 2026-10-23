"""The shared suite itself, plus day 4 as a regression guard against it.

Two jobs. First, catch a malformed suite — a duplicate id or an expectation
that contradicts `expect_booked` would quietly weaken the comparison. Second,
pin day 4's behaviour: if a later change to the engine breaks a scenario, this
fails here rather than showing up as a mysterious dip in day 5's chart.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from convo.core.evals import EvalRun
from convo.projects.day04_dialog_engine.extract import Reference
from convo.projects.day05_tool_agent import scenarios as sc
from convo.projects.day05_tool_agent.backends import Scripted

REF = Reference(today=date(2026, 3, 11))


class TestSuiteIsWellFormed:
    def test_ids_are_unique(self):
        ids = [s.id for s in sc.SCENARIOS]
        assert len(ids) == len(set(ids))

    def test_every_scenario_has_utterances_tags_and_a_note(self):
        for scenario in sc.SCENARIOS:
            assert scenario.utterances, scenario.id
            assert scenario.tags, scenario.id
            assert scenario.note, scenario.id

    def test_expectations_agree_with_whether_a_booking_is_expected(self):
        for scenario in sc.SCENARIOS:
            if scenario.expect_booked:
                assert set(scenario.expect) == {"party_size", "date", "time", "name"}, (
                    f"{scenario.id} expects a booking but does not pin every field"
                )
            else:
                assert scenario.expect == {}, scenario.id

    def test_the_suite_covers_the_cases_day_four_was_built_for(self):
        tags = {t for s in sc.SCENARIOS for t in s.tags}
        assert {"happy", "repair", "digression", "validation", "safety"} <= tags

    def test_at_least_one_scenario_must_not_book(self):
        # Without a negative case, a bot that books everything scores 100%.
        assert any(not s.expect_booked for s in sc.SCENARIOS)


class TestDayFourRegressionGuard:
    def test_the_state_machine_still_passes_every_scenario(self):
        run = EvalRun(suite="guard")
        for scenario in sc.SCENARIOS:
            run.add(sc.run_engine(scenario, REF))

        failures = [
            (c.case_id, [a.detail for a in c.assertions if not a.passed]) for c in run.failures
        ]
        assert run.pass_rate == 1.0, f"day 4 regressed: {failures}"

    def test_the_negative_scenario_really_does_not_book(self):
        scenario = next(s for s in sc.SCENARIOS if not s.expect_booked)
        case = sc.run_engine(scenario, REF)
        assert case.passed
        assert any(a.name == "did not book" and a.passed for a in case.assertions)

    def test_each_case_carries_a_transcript_for_the_ui(self):
        case = sc.run_engine(sc.SCENARIOS[0], REF)
        assert case.transcript
        assert case.transcript[0]["role"] == "user"
        assert case.latency_ms >= 0


class TestAgentRunnerOnTheSameSuite:
    def _booking_backend(self, scenario) -> Scripted:
        """A backend that books exactly what the scenario expects."""
        args = {
            "party_size": scenario.expect["party_size"],
            "date": scenario.expect["date"],
            "time": scenario.expect["time"],
            "name": scenario.expect["name"],
        }
        payload = json.dumps({"name": "book_table", "arguments": args})
        return Scripted([f"<tool_call>{payload}</tool_call>"] * 20)

    def test_a_perfect_agent_passes_a_booking_scenario(self):
        scenario = next(s for s in sc.SCENARIOS if s.expect_booked)
        case = sc.run_agent(scenario, self._booking_backend(scenario), REF)
        assert case.passed
        assert "model calls" in case.judge_rationale

    def test_an_agent_that_books_nothing_fails_a_booking_scenario(self):
        scenario = next(s for s in sc.SCENARIOS if s.expect_booked)
        case = sc.run_agent(scenario, Scripted(["How many people?"] * 20), REF)
        assert not case.passed

    def test_an_agent_that_books_anyway_fails_the_safety_scenario(self):
        # The guard that matters: inventing details must be scored as a failure.
        scenario = next(s for s in sc.SCENARIOS if not s.expect_booked)
        invented = json.dumps(
            {
                "name": "book_table",
                "arguments": {
                    "party_size": 2,
                    "date": "2026-03-13",
                    "time": "19:00",
                    "name": "Guest",
                },
            }
        )
        # Grounding off: this test is about the SCORING catching an invented
        # booking, not about the guard preventing one. With the guard on the
        # booking never executes, which is the fix rather than the failure.
        from convo.projects.day05_tool_agent.agent import AgentConfig

        case = sc.run_agent(
            scenario,
            Scripted([f"<tool_call>{invented}</tool_call>"] * 9),
            REF,
            AgentConfig(check_grounding=False),
        )
        assert not case.passed

    def test_the_grounding_guard_turns_that_failure_into_a_pass(self):
        scenario = next(s for s in sc.SCENARIOS if not s.expect_booked)
        invented = json.dumps(
            {
                "name": "book_table",
                "arguments": {
                    "party_size": 2,
                    "date": "2026-03-13",
                    "time": "19:00",
                    "name": "Guest",
                },
            }
        )
        case = sc.run_agent(scenario, Scripted([f"<tool_call>{invented}</tool_call>"] * 9), REF)
        assert case.passed, "the guard should stop the invented booking executing"

    def test_both_runners_produce_the_same_assertion_names(self):
        scenario = next(s for s in sc.SCENARIOS if s.expect_booked)
        engine_case = sc.run_engine(scenario, REF)
        agent_case = sc.run_agent(scenario, self._booking_backend(scenario), REF)
        assert [a.name for a in engine_case.assertions] == [
            a.name for a in agent_case.assertions
        ], "the two sides must be judged by identical checks"


class TestNormalisation:
    def test_engine_slots_normalise_to_the_comparison_shape(self):
        from convo.projects.day04_dialog_engine.engine import new_session, turn

        session = new_session(REF)
        turn(session, "a table for 4 on friday at 7:30pm under sarah chen")
        out = sc.normalise_engine(session)
        assert out == {
            "party_size": 4,
            "date": "2026-03-13",
            "time": "19:30",
            "name": "sarah chen",
        }

    def test_a_booking_normalises_to_the_same_shape(self):
        out = sc.normalise_booking(
            {"party_size": "4", "date": "2026-03-13", "time": "19:30", "name": "Sarah Chen"}
        )
        assert out == {
            "party_size": 4,
            "date": "2026-03-13",
            "time": "19:30",
            "name": "sarah chen",
        }

    def test_no_booking_normalises_to_empty(self):
        assert sc.normalise_booking(None) == {}

    @pytest.mark.parametrize("missing", ["party_size", "date", "time", "name"])
    def test_a_missing_field_fails_its_own_assertion(self, missing):
        scenario = next(s for s in sc.SCENARIOS if s.expect_booked)
        captured = dict(scenario.expect)
        captured.pop(missing)
        checks = sc.assertions_for(scenario, True, captured)
        failed = [a.name for a in checks if not a.passed]
        assert failed == [f"{missing} correct"]
