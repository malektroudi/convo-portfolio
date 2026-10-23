"""Validator tests, written around the error TEXT as much as the verdict.

The message is the product here: it goes back into the prompt verbatim as the
retry instruction, so a test that only checked `ok is False` would miss the
thing that decides whether the retry works.
"""

from __future__ import annotations

import pytest

from convo.projects.day05_tool_agent import schema as sc

BOOK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "party_size": {"type": "integer", "minimum": 1, "maximum": 12},
        "date": {
            "type": "string",
            "pattern": r"\d{4}-\d{2}-\d{2}",
            "patternHint": "YYYY-MM-DD",
            "description": "the booking date",
        },
        "time": {"type": "string", "pattern": r"\d{2}:\d{2}", "patternHint": "HH:MM"},
        "name": {"type": "string", "minLength": 2},
        "seating": {"type": "string", "enum": ["indoor", "patio"]},
    },
    "required": ["party_size", "date", "time", "name"],
}

VALID = {"party_size": 4, "date": "2026-03-13", "time": "19:30", "name": "Sarah"}


class TestValidCalls:
    def test_a_good_call_passes_with_no_errors(self):
        result = sc.validate(VALID, BOOK_SCHEMA)
        assert result.ok
        assert result.errors == []
        assert result.as_instruction() == ""

    def test_an_optional_enum_value_is_allowed(self):
        result = sc.validate({**VALID, "seating": "patio"}, BOOK_SCHEMA)
        assert result.ok


class TestMissingAndUnknown:
    def test_a_missing_required_field_is_named_with_its_description(self):
        result = sc.validate({"party_size": 4, "time": "19:30", "name": "S"}, BOOK_SCHEMA)
        assert not result.ok
        assert any(e == "date: required but missing (the booking date)." for e in result.errors)

    def test_a_null_counts_as_missing(self):
        result = sc.validate({**VALID, "date": None}, BOOK_SCHEMA)
        assert any(e.startswith("date: required but missing") for e in result.errors)

    def test_an_unknown_parameter_lists_the_legal_ones(self):
        result = sc.validate({**VALID, "table_number": 7}, BOOK_SCHEMA)
        error = next(e for e in result.errors if e.startswith("table_number"))
        assert "not a parameter of this tool" in error
        assert "party_size" in error and "seating" in error

    def test_unknown_parameters_are_tolerated_when_the_schema_allows_them(self):
        lenient = {**BOOK_SCHEMA, "additionalProperties": True}
        assert sc.validate({**VALID, "extra": 1}, lenient).ok


class TestTypes:
    def test_a_wrong_type_says_what_was_expected_and_what_arrived(self):
        result = sc.validate({**VALID, "party_size": "four people"}, BOOK_SCHEMA)
        assert result.errors == ['party_size: expected an integer, got the string "four people".']

    def test_a_numeric_string_is_coerced_rather_than_rejected(self):
        # Models send "4" constantly; rejecting it wastes a retry on nothing.
        result = sc.validate({**VALID, "party_size": "4"}, BOOK_SCHEMA)
        assert result.ok
        assert result.coerced["party_size"] == 4
        assert isinstance(result.coerced["party_size"], int)

    def test_a_boolean_is_never_accepted_as_a_number(self):
        result = sc.validate({**VALID, "party_size": True}, BOOK_SCHEMA)
        assert not result.ok
        assert "expected an integer, got the boolean true" in result.errors[0]

    def test_arguments_that_are_not_an_object_fail_clearly(self):
        result = sc.validate(["party_size", 4], BOOK_SCHEMA)
        assert not result.ok
        assert "expected an object of named parameters" in result.errors[0]


class TestBoundsPatternsAndEnums:
    def test_a_value_over_the_maximum_states_the_bound(self):
        result = sc.validate({**VALID, "party_size": 40}, BOOK_SCHEMA)
        assert result.errors == ["party_size: must be at most 12, got 40."]

    def test_a_value_under_the_minimum_states_the_bound(self):
        result = sc.validate({**VALID, "party_size": 0}, BOOK_SCHEMA)
        assert "must be at least 1" in result.errors[0]

    def test_a_bad_pattern_shows_the_readable_hint_not_the_regex(self):
        result = sc.validate({**VALID, "date": "13/03/2026"}, BOOK_SCHEMA)
        assert result.errors == ['date: must look like YYYY-MM-DD, got "13/03/2026".']

    def test_a_bad_enum_lists_the_legal_values(self):
        result = sc.validate({**VALID, "seating": "rooftop"}, BOOK_SCHEMA)
        assert 'must be one of "indoor", "patio"' in result.errors[0]

    def test_a_too_short_string_states_the_minimum(self):
        result = sc.validate({**VALID, "name": "S"}, BOOK_SCHEMA)
        assert "at least 2 characters" in result.errors[0]


class TestRetryInstruction:
    def test_all_problems_are_reported_at_once_not_one_per_round_trip(self):
        result = sc.validate({"party_size": 40, "date": "13/03/2026"}, BOOK_SCHEMA)
        # over maximum, bad pattern, and two missing required fields
        assert len(result.errors) == 4

    def test_a_single_problem_reads_as_one_instruction(self):
        result = sc.validate({**VALID, "party_size": 40}, BOOK_SCHEMA)
        text = result.as_instruction()
        assert text.startswith("That call was invalid:")
        assert text.endswith("Call the tool again, corrected.")

    def test_several_problems_are_counted_and_joined(self):
        result = sc.validate({"party_size": 40, "date": "13/03/2026"}, BOOK_SCHEMA)
        text = result.as_instruction()
        assert "had 4 problems" in text
        assert ";" in text

    @pytest.mark.parametrize(
        "bad",
        [
            {**VALID, "party_size": "four"},
            {**VALID, "date": "friday"},
            {**VALID, "seating": "rooftop"},
        ],
    )
    def test_every_instruction_names_the_offending_field(self, bad):
        result = sc.validate(bad, BOOK_SCHEMA)
        assert not result.ok
        field = next(k for k, v in bad.items() if VALID.get(k) != v)
        assert field in result.as_instruction()
