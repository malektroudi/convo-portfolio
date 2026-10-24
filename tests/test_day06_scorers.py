"""Scorer tests, including deliberately broken bots.

Day 4 scores 100% on this suite, which is a legitimate result for a
deterministic state machine — it genuinely cannot invent a slot or echo a card
number back. But a suite that has only ever been run against something that
passes is not evidence of anything. The `TestTheSuiteCanFail` class drives
bots that misbehave in each specific way and asserts the right dimension
catches each one.
"""

from __future__ import annotations

from convo.core.evals import EFFICIENCY, SAFETY, TASK
from convo.projects.day06_eval_harness.harness import engine_runner, run_suite
from convo.projects.day06_eval_harness.outcome import Outcome
from convo.projects.day06_eval_harness.scorers import (
    score_efficiency,
    score_safety,
    score_task,
)
from convo.projects.day06_eval_harness.suite import SARAH, SUITE, Scenario

BOOKING = Scenario(
    "x",
    ("a table for 4", "friday", "7:30pm", "my name is sarah chen", "yes"),
    True,
    SARAH,
    ("task",),
    "fixture",
)
NO_BOOKING = Scenario("y", ("just book something",), False, {}, ("safety",), "fixture")


def outcome(scenario: Scenario, **kwargs) -> Outcome:
    base = {
        "scenario_id": scenario.id,
        "booked": scenario.expect_booked,
        "captured": dict(scenario.expect),
        "transcript": [{"role": "user", "content": u} for u in scenario.utterances],
        "user_turns": len(scenario.utterances),
    }
    base.update(kwargs)
    return Outcome(**base)


def failed(assertions, dimension):
    return [a.name for a in assertions if a.dimension == dimension and not a.passed]


class TestTask:
    def test_a_correct_booking_passes(self):
        assert failed(score_task(BOOKING, outcome(BOOKING)), TASK) == []

    def test_a_wrong_field_fails_that_field(self):
        o = outcome(BOOKING, captured={**SARAH, "time": "21:00"})
        assert failed(score_task(BOOKING, o), TASK) == ["time correct"]

    def test_not_booking_when_expected_fails(self):
        o = outcome(BOOKING, booked=False, captured={})
        assert "booked" in failed(score_task(BOOKING, o), TASK)

    def test_booking_when_not_expected_fails(self):
        o = outcome(NO_BOOKING, booked=True, captured={})
        assert "did not book" in failed(score_task(NO_BOOKING, o), TASK)


class TestSafety:
    def test_a_clean_conversation_passes(self):
        o = outcome(BOOKING)
        o.transcript.append({"role": "assistant", "content": "Booked. See you then."})
        assert failed(score_safety(BOOKING, o), SAFETY) == []

    def test_claiming_a_booking_that_should_not_exist_is_a_false_confirmation(self):
        o = outcome(NO_BOOKING, booked=True)
        o.transcript.append({"role": "assistant", "content": "Booked — see you then."})
        assert "no false confirmation" in failed(score_safety(NO_BOOKING, o), SAFETY)

    def test_claiming_a_booking_with_wrong_details_is_a_false_confirmation(self):
        o = outcome(BOOKING, captured={**SARAH, "name": "john doe"})
        o.transcript.append({"role": "assistant", "content": "All set!"})
        assert "no false confirmation" in failed(score_safety(BOOKING, o), SAFETY)

    def test_an_invented_name_is_caught(self):
        o = outcome(BOOKING, captured={**SARAH, "name": "john doe"})
        assert "no invented details" in failed(score_safety(BOOKING, o), SAFETY)

    def test_a_correctly_parsed_value_is_not_mistaken_for_invented(self):
        # The guest said "friday", the bot stored 2026-03-13. Grounded.
        o = outcome(BOOKING)
        assert "no invented details" not in failed(score_safety(BOOKING, o), SAFETY)

    def test_echoing_a_card_number_is_caught(self):
        scenario = next(s for s in SUITE if s.id == "pii_card_number_volunteered")
        o = outcome(scenario)
        o.transcript.append({"role": "assistant", "content": "Booked on card 4111 1111 1111 1111."})
        assert "no card number echoed back" in failed(score_safety(scenario, o), SAFETY)

    def test_not_echoing_a_card_number_passes(self):
        scenario = next(s for s in SUITE if s.id == "pii_card_number_volunteered")
        o = outcome(scenario)
        o.transcript.append({"role": "assistant", "content": "Booked. See you then."})
        assert failed(score_safety(scenario, o), SAFETY) == []

    def test_echoing_an_email_is_caught(self):
        scenario = next(s for s in SUITE if s.id == "pii_email_volunteered")
        o = outcome(scenario)
        o.transcript.append(
            {"role": "assistant", "content": "Confirmation sent to sarah.chen@example.com"}
        )
        assert "no email address echoed back" in failed(score_safety(scenario, o), SAFETY)

    def test_pii_checks_only_apply_when_pii_is_in_play(self):
        # A conversation with no card number should not be scored on card echo.
        names = [a.name for a in score_safety(BOOKING, outcome(BOOKING))]
        assert "no card number echoed back" not in names


class TestEfficiency:
    def test_within_budget_passes(self):
        assert failed(score_efficiency(BOOKING, outcome(BOOKING)), EFFICIENCY) == []

    def test_too_many_turns_fails(self):
        o = outcome(BOOKING, user_turns=99)
        assert "within the turn budget" in failed(score_efficiency(BOOKING, o), EFFICIENCY)

    def test_an_explicit_budget_is_honoured(self):
        tight = Scenario("t", ("one go",), True, SARAH, ("x",), "n", turn_budget=1)
        o = outcome(tight, user_turns=2)
        assert "within the turn budget" in failed(score_efficiency(tight, o), EFFICIENCY)

    def test_model_calls_are_only_scored_for_bots_that_make_them(self):
        names = [a.name for a in score_efficiency(BOOKING, outcome(BOOKING))]
        assert "within the model-call budget" not in names

        with_calls = score_efficiency(BOOKING, outcome(BOOKING, model_calls=3))
        assert "within the model-call budget" in [a.name for a in with_calls]

    def test_too_many_model_calls_fails(self):
        o = outcome(BOOKING, model_calls=500)
        assert "within the model-call budget" in failed(score_efficiency(BOOKING, o), EFFICIENCY)


class TestTheSuiteCanFail:
    """Mutation checks: break a bot one way and confirm the right axis notices."""

    def test_the_state_machine_passes_everything(self):
        run = run_suite("guard", engine_runner, SUITE)
        assert run.pass_rate == 1.0, [c.case_id for c in run.failures]

    def test_a_bot_that_invents_a_name_fails_safety_but_not_efficiency(self):
        def liar(scenario):
            o = engine_runner(scenario)
            if o.captured:
                o.captured["name"] = "john doe"
            return o

        run = run_suite("liar", liar, SUITE)
        assert run.pass_rate_in(SAFETY) < 1.0
        assert run.pass_rate_in(EFFICIENCY) == 1.0, "efficiency is unaffected"

    def test_a_bot_that_books_everything_fails_both_task_and_safety(self):
        def overeager(scenario):
            o = engine_runner(scenario)
            o.booked = True
            o.transcript.append({"role": "assistant", "content": "Booked — see you!"})
            return o

        run = run_suite("overeager", overeager, SUITE)
        assert run.pass_rate_in(TASK) < 1.0
        assert run.pass_rate_in(SAFETY) < 1.0

    def test_a_bot_that_leaks_pii_fails_only_safety(self):
        def leaky(scenario):
            o = engine_runner(scenario)
            for message in list(o.transcript):
                if message["role"] == "user":
                    o.transcript.append(
                        {"role": "assistant", "content": f"You said: {message['content']}"}
                    )
            return o

        run = run_suite("leaky", leaky, SUITE)
        assert run.pass_rate_in(SAFETY) < 1.0
        assert len(run.failures_in(SAFETY)) >= 3, "three PII scenarios at least"

    def test_a_slow_bot_fails_only_efficiency(self):
        def dawdling(scenario):
            o = engine_runner(scenario)
            o.user_turns = 99
            return o

        run = run_suite("dawdling", dawdling, SUITE)
        assert run.pass_rate_in(EFFICIENCY) == 0.0
        assert run.pass_rate_in(TASK) == 1.0, "task is unaffected"
        assert run.pass_rate_in(SAFETY) == 1.0, "and so is safety"
