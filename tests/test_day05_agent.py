"""The agent loop driven through every failure branch with canned generations.

A scripted backend is what makes this testable at all: the retry logic, the
parser and the limits each get exercised deterministically, in milliseconds,
with no model and no network. The real model's behaviour is measured
separately by the benchmark — these tests are about the machinery that has to
survive it.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from convo.projects.day04_dialog_engine.extract import Reference
from convo.projects.day05_tool_agent import tools as tool_mod
from convo.projects.day05_tool_agent.agent import (
    AgentConfig,
    new_session,
    parse_tool_call,
    turn,
)
from convo.projects.day05_tool_agent.backends import Scripted

REF = Reference(today=date(2026, 3, 11))

GOOD_ARGS = {
    "party_size": 4,
    "date": "2026-03-13",
    "time": "19:30",
    "name": "Sarah Chen",
}


def call_text(name: str, arguments: dict, style: str = "tagged") -> str:
    payload = json.dumps({"name": name, "arguments": arguments})
    if style == "tagged":
        return f"<tool_call>\n{payload}\n</tool_call>"
    if style == "fenced":
        return f"Sure.\n```json\n{payload}\n```"
    return f"I'll book that. {payload}"


def session(responses: list[str], **config):
    # Grounding off by default here: these tests isolate the parse / schema /
    # retry / limit machinery, and they book on turn 1 before the guest has
    # said anything, which the provenance guard would (correctly) refuse.
    # test_day05_grounding.py covers that guard on its own.
    config.setdefault("check_grounding", False)
    backend = Scripted(responses)
    return backend, new_session(backend, REF, AgentConfig(**config))


class TestParsing:
    @pytest.mark.parametrize("style", ["tagged", "fenced", "bare"])
    def test_every_emission_style_is_recognised(self, style):
        parsed = parse_tool_call(call_text("book_table", GOOD_ARGS, style))
        assert parsed is not None
        assert parsed.name == "book_table"
        assert parsed.arguments["party_size"] == 4
        assert parsed.how == style

    def test_prose_with_no_call_parses_to_nothing(self):
        assert parse_tool_call("How many people will that be?") is None

    def test_malformed_json_parses_to_nothing_rather_than_raising(self):
        assert parse_tool_call("<tool_call>{not json at all}</tool_call>") is None

    def test_nested_argument_objects_survive_the_bare_scanner(self):
        nested = {"name": "book_table", "arguments": {"a": {"b": [1, 2]}, "c": 3}}
        parsed = parse_tool_call(f"ok {json.dumps(nested)} done")
        assert parsed is not None
        assert parsed.arguments["a"]["b"] == [1, 2]

    def test_the_tagged_form_wins_over_a_stray_object(self):
        text = '{"name": "wrong", "arguments": {}} ' + call_text("book_table", GOOD_ARGS)
        assert parse_tool_call(text).name == "book_table"

    def test_alternative_key_names_are_accepted(self):
        payload = json.dumps({"tool": "get_policy", "parameters": {"topic": "parking"}})
        parsed = parse_tool_call(payload)
        assert parsed.name == "get_policy"
        assert parsed.arguments["topic"] == "parking"


class TestHappyPath:
    def test_a_valid_call_books_and_stops(self):
        _, s = session([call_text("book_table", GOOD_ARGS)])
        result = turn(s, "table for 4 on friday at 7:30pm under Sarah Chen")

        assert result.booking is not None
        assert result.booking["confirmation"].startswith("BK-")
        assert s.booking is not None
        assert s.done is True
        assert result.steps[-1].kind == "tool_call"
        assert "Booked" in result.reply

    def test_a_plain_reply_is_passed_through_as_speech(self):
        _, s = session(["How many people will that be?"])
        result = turn(s, "i'd like a table")
        assert result.reply == "How many people will that be?"
        assert result.steps[-1].kind == "reply"
        assert s.booking is None

    def test_a_policy_tool_answers_without_ending_the_conversation(self):
        _, s = session([call_text("get_policy", {"topic": "parking"})])
        result = turn(s, "do you have parking?")
        assert "street parking" in result.reply
        assert s.booking is None
        assert s.done is False

    def test_tool_output_is_fed_back_into_the_transcript(self):
        _, s = session([call_text("get_policy", {"topic": "hours"})])
        turn(s, "when do you open?")
        tool_messages = [m for m in s.messages if m["role"] == "tool"]
        assert len(tool_messages) == 1
        assert "5pm to 10pm" in tool_messages[0]["content"]


class TestValidationAndRetry:
    def test_an_invalid_call_is_retried_and_can_succeed(self):
        bad = call_text("book_table", {**GOOD_ARGS, "party_size": "four"})
        _, s = session([bad, call_text("book_table", GOOD_ARGS)])
        result = turn(s, "table for four")

        assert [st.kind for st in result.steps] == ["invalid_args", "tool_call"]
        assert result.booking is not None
        assert result.retries == 1

    def test_the_retry_instruction_names_the_offending_field(self):
        bad = call_text("book_table", {**GOOD_ARGS, "party_size": 40})
        backend, s = session([bad, call_text("book_table", GOOD_ARGS)])
        turn(s, "table for 40")

        correction = backend.calls[1][-1]["content"]
        assert "party_size" in correction
        assert "at most 12" in correction

    def test_retries_are_bounded_and_the_turn_gives_up_cleanly(self):
        bad = call_text("book_table", {"party_size": 4})
        _, s = session([bad, bad, bad, bad], max_retries=2)
        result = turn(s, "book it")

        assert len(result.steps) == 3, "one attempt plus two retries"
        assert all(st.kind == "invalid_args" for st in result.steps)
        assert result.booking is None
        assert "gave up" in result.stopped

    def test_an_unknown_tool_is_reported_with_the_legal_names(self):
        _, s = session([call_text("cancel_everything", {}), call_text("book_table", GOOD_ARGS)])
        backend = s.backend
        turn(s, "book it")
        assert s.steps[0].kind == "unknown_tool"
        assert "book_table" in backend.calls[1][-1]["content"]

    def test_a_business_rule_refusal_is_a_tool_error_not_a_schema_error(self):
        # Schema-valid but 3am is outside service: the rules live in the tool.
        outside = call_text("book_table", {**GOOD_ARGS, "time": "03:00"})
        _, s = session([outside, call_text("book_table", GOOD_ARGS)])
        result = turn(s, "book me in at 3am")

        assert result.steps[0].kind == "tool_error"
        assert "serve between" in result.steps[0].errors[0]
        assert result.booking is not None, "the model gets a chance to correct"

    def test_a_past_date_is_refused_by_the_tool(self):
        past = call_text("book_table", {**GOOD_ARGS, "date": "2026-03-01"})
        _, s = session([past], max_retries=0)
        result = turn(s, "book it for the 1st")
        assert result.steps[0].kind == "tool_error"
        assert "passed" in result.steps[0].errors[0]
        assert s.booking is None


class TestLimits:
    def test_the_conversation_ceiling_stops_an_endless_retry_loop(self):
        bad = call_text("book_table", {"party_size": 4})
        _, s = session([bad] * 40, max_retries=50, max_model_calls=6)
        result = turn(s, "book it")

        assert s.model_calls == 6
        assert s.exhausted is True
        assert "ceiling" in result.stopped

    def test_a_finished_conversation_absorbs_further_turns(self):
        _, s = session([call_text("book_table", GOOD_ARGS)])
        turn(s, "book it")
        result = turn(s, "and another one")
        assert result.stopped == "already finished"
        assert result.steps == []

    def test_an_empty_generation_does_not_count_as_a_reply(self):
        _, s = session(["", call_text("book_table", GOOD_ARGS)])
        result = turn(s, "book it")
        assert result.steps[0].kind == "unparseable"
        assert result.booking is not None


class TestAccounting:
    def test_steps_and_failures_are_counted_across_the_conversation(self):
        bad = call_text("book_table", {"party_size": 4})
        _, s = session([bad, call_text("book_table", GOOD_ARGS)])
        turn(s, "book it")

        assert s.model_calls == 2
        assert s.failed_steps == 1

    def test_tokens_accumulate(self):
        _, s = session([call_text("book_table", GOOD_ARGS)])
        turn(s, "book it")
        tokens_in, tokens_out = s.tokens
        assert tokens_in > 0 and tokens_out > 0

    def test_every_step_records_how_it_was_parsed(self):
        _, s = session([call_text("book_table", GOOD_ARGS, "fenced")])
        turn(s, "book it")
        assert s.steps[-1].parsed_how == "fenced"


class TestToolsMatchDayFourRules:
    def test_the_party_limit_is_the_same_number(self):
        from convo.projects.day04_dialog_engine.flow import MAX_PARTY

        assert tool_mod.BOOK_PARAMETERS["properties"]["party_size"]["maximum"] == MAX_PARTY

    def test_service_hours_come_from_the_same_reference(self):
        out = tool_mod.book_table({**GOOD_ARGS, "time": "23:00"}, REF)
        assert out["ok"] is False
        assert "serve between" in out["error"]

    def test_a_good_booking_summarises_like_day_four_does(self):
        out = tool_mod.book_table(GOOD_ARGS, REF)
        assert out["ok"] is True
        assert out["summary"] == "4 people on Friday 13 March at 7:30pm under Sarah Chen"
