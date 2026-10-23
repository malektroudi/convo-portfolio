"""The provenance guard, built directly from the first run's failure mode.

Every "invented" case below is a real argument the 0.5B model produced: a
complete, schema-valid, business-rule-valid booking for details the guest had
not given. The validator passed all of them, which is the point.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from convo.projects.day04_dialog_engine.extract import Reference
from convo.projects.day05_tool_agent.agent import AgentConfig, new_session, turn
from convo.projects.day05_tool_agent.backends import Scripted
from convo.projects.day05_tool_agent.grounding import as_instruction, grounding_errors

REF = Reference(today=date(2026, 3, 11))  # Wednesday

SAID = ["a table for 4", "friday", "7:30pm", "my name is sarah chen"]
GROUNDED = {
    "party_size": 4,
    "date": "2026-03-13",
    "time": "19:30",
    "name": "Sarah Chen",
}


class TestGroundedArguments:
    def test_everything_the_guest_said_passes(self):
        assert grounding_errors(GROUNDED, SAID, REF) == []

    def test_a_single_loaded_utterance_grounds_every_field(self):
        said = ["table for 2 tomorrow at 8pm under chen"]
        args = {"party_size": 2, "date": "2026-03-12", "time": "20:00", "name": "chen"}
        assert grounding_errors(args, said, REF) == []

    def test_a_corrected_value_is_grounded_by_the_correction(self):
        said = [*SAID, "no, make it 8pm"]
        assert grounding_errors({**GROUNDED, "time": "20:00"}, said, REF) == []

    def test_name_matching_is_case_insensitive(self):
        assert grounding_errors({**GROUNDED, "name": "SARAH CHEN"}, SAID, REF) == []


class TestInventedArguments:
    def test_an_invented_name_is_caught(self):
        # The model really did write "John Doe" into a booking.
        errors = grounding_errors({**GROUNDED, "name": "John Doe"}, SAID, REF)
        assert len(errors) == 1
        assert "never gave the name" in errors[0]

    def test_the_table_12_case_is_caught(self):
        errors = grounding_errors({**GROUNDED, "name": "Table 12"}, SAID, REF)
        assert errors and "Table 12" in errors[0]

    def test_an_invented_party_size_is_caught(self):
        errors = grounding_errors({**GROUNDED, "party_size": 12}, SAID, REF)
        assert "the guest never said 12" in errors[0]

    def test_todays_date_substituted_for_an_unsaid_one_is_caught(self):
        errors = grounding_errors({**GROUNDED, "date": "2026-03-11"}, SAID, REF)
        assert "never said 2026-03-11" in errors[0]

    def test_an_invented_time_is_caught(self):
        errors = grounding_errors({**GROUNDED, "time": "21:00"}, SAID, REF)
        assert "never said 21:00" in errors[0]

    def test_a_wholly_invented_booking_reports_every_field(self):
        invented = {
            "party_size": 2,
            "date": "2026-03-11",
            "time": "19:00",
            "name": "John Doe",
        }
        errors = grounding_errors(invented, ["i'd like a table"], REF)
        assert len(errors) == 4

    def test_the_instruction_tells_the_model_what_to_do_instead(self):
        errors = grounding_errors({**GROUNDED, "name": "John Doe"}, SAID, REF)
        text = as_instruction(errors)
        assert text.startswith("Do not invent booking details.")
        assert "Ask whose name" in text

    def test_no_errors_produces_no_instruction(self):
        assert as_instruction([]) == ""


class TestGuardInTheLoop:
    def _call(self, args: dict) -> str:
        payload = json.dumps({"name": "book_table", "arguments": args})
        return f"<tool_call>{payload}</tool_call>"

    def test_an_invented_booking_is_refused_and_retried(self):
        invented = self._call(
            {"party_size": 2, "date": "2026-03-11", "time": "19:00", "name": "John Doe"}
        )
        backend = Scripted([invented, "How many people will that be?"])
        session = new_session(backend, REF, AgentConfig(max_retries=1))
        result = turn(session, "i'd like a table")

        assert session.booking is None, "an invented booking must not execute"
        assert result.steps[0].kind == "ungrounded"
        assert "Do not invent" in backend.calls[1][-1]["content"]

    def test_a_grounded_booking_still_goes_through(self):
        # The model must wait until the guest has actually said everything;
        # booking on turn 1 is refused precisely because nothing is grounded yet.
        backend = Scripted(["Which day?", "What time?", "And the name?", self._call(GROUNDED)])
        session = new_session(backend, REF)
        for utterance in SAID:
            result = turn(session, utterance)

        assert session.booking is not None
        assert result.steps[-1].kind == "tool_call"
        assert not any(s.kind == "ungrounded" for s in session.steps)

    def test_booking_before_the_guest_has_said_anything_is_refused(self):
        backend = Scripted([self._call(GROUNDED)] * 4)
        session = new_session(backend, REF, AgentConfig(max_retries=1))
        turn(session, "a table for 4")

        # party_size is grounded by this turn; the other three are not.
        assert session.booking is None
        errors = session.steps[0].errors
        assert len(errors) == 3
        assert not any(e.startswith("party_size") for e in errors)

    def test_the_guard_can_be_switched_off_to_measure_its_worth(self):
        invented = self._call(
            {"party_size": 2, "date": "2026-03-11", "time": "19:00", "name": "John Doe"}
        )
        backend = Scripted([invented])
        session = new_session(backend, REF, AgentConfig(check_grounding=False))
        turn(session, "i'd like a table")
        assert session.booking is not None, "off, the invented booking executes"

    def test_retry_instructions_are_not_mistaken_for_guest_evidence(self):
        # The loop injects corrections as user messages; if those counted as
        # things the guest said, the guard would validate its own feedback.
        invented = self._call(
            {"party_size": 2, "date": "2026-03-11", "time": "19:00", "name": "John Doe"}
        )
        backend = Scripted([invented, invented, invented])
        session = new_session(backend, REF, AgentConfig(max_retries=2))
        turn(session, "i'd like a table")

        assert session.user_turns == ["i'd like a table"]
        assert session.booking is None
        assert all(s.kind == "ungrounded" for s in session.steps)

    def test_the_policy_tool_is_not_grounded_and_still_works(self):
        payload = json.dumps({"name": "get_policy", "arguments": {"topic": "parking"}})
        backend = Scripted([f"<tool_call>{payload}</tool_call>"])
        session = new_session(backend, REF)
        result = turn(session, "do you have parking?")
        assert "street parking" in result.reply


@pytest.mark.parametrize(
    ("field", "value"),
    [("party_size", 9), ("date", "2026-04-01"), ("time", "21:00"), ("name", "Nobody")],
)
def test_each_field_is_checked_independently(field, value):
    errors = grounding_errors({**GROUNDED, field: value}, SAID, REF)
    assert len(errors) == 1
    assert errors[0].startswith(field)
