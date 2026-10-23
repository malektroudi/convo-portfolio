"""Extraction tests. Reference date is Wednesday 2026-03-11, fixed on purpose.

The interesting cases here are the collisions: a clock time and a head count
are both small integers, and an ISO date is three of them. Most of these tests
exist because a naive parser gets exactly these wrong.
"""

from __future__ import annotations

from datetime import date, time

import pytest

from convo.projects.day04_dialog_engine import extract as ex

REF = ex.Reference(today=date(2026, 3, 11))  # a Wednesday


class TestYesNo:
    @pytest.mark.parametrize("text", ["yes", "yeah", "sure", "that's right", "Confirm"])
    def test_affirmations(self, text):
        assert ex.parse_yes_no(text) is True

    @pytest.mark.parametrize("text", ["no", "nope", "that's wrong", "incorrect"])
    def test_denials(self, text):
        assert ex.parse_yes_no(text) is False

    @pytest.mark.parametrize("text", ["cancel", "actually make it 6", "change the time"])
    def test_correction_and_abandon_markers_are_not_denials(self, text):
        # "cancel" is routed as abandon and "actually" marks a repair; reading
        # either as "no" would send a correcting user into the amend menu.
        assert ex.parse_yes_no(text) is None

    @pytest.mark.parametrize("text", ["do you have parking?", "what do you serve?"])
    def test_a_question_containing_do_is_not_an_affirmation(self, text):
        # Regression guard: AFFIRM was once built by splitting multi-word
        # phrases, which made "do", "it" and "good" standalone yes-words and
        # silently suppressed every digression phrased as a question.
        assert ex.parse_yes_no(text) is None

    @pytest.mark.parametrize("text", ["sounds good", "go ahead", "do it"])
    def test_multi_word_affirmations_still_work(self, text):
        assert ex.parse_yes_no(text) is True

    def test_a_denial_wins_over_an_affirmation_in_the_same_breath(self):
        # "no, that's fine on friday" is a correction, not agreement.
        assert ex.parse_yes_no("no, ok make it friday") is False

    def test_absent_is_none(self):
        assert ex.parse_yes_no("a table for four") is None


class TestPartySize:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("a table for 4", 4),
            ("four people", 4),
            ("table for two", 2),
            ("just me", 1),
            ("myself", 1),
            ("a couple", 2),
            ("there will be 12 of us", 12),
        ],
    )
    def test_counts(self, text, expected):
        assert ex.parse_party_size(text) == expected

    @pytest.mark.parametrize("text", ["7pm", "7 pm", "19:00", "8 o'clock", "2026-03-14"])
    def test_times_and_iso_dates_are_not_head_counts(self, text):
        assert ex.parse_party_size(text) is None

    def test_a_count_and_a_time_in_one_breath_are_separated(self):
        text = "a table for 4 at 7:30pm"
        assert ex.parse_party_size(text) == 4
        assert ex.parse_time(text) == time(19, 30)

    def test_an_ordinal_date_is_not_a_head_count(self):
        assert ex.parse_party_size("the 14th") is None

    def test_absent_is_none(self):
        assert ex.parse_party_size("hello there") is None


class TestTime:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("7pm", time(19, 0)),
            ("7 pm", time(19, 0)),
            ("7:30pm", time(19, 30)),
            ("19:00", time(19, 0)),
            ("noon", time(12, 0)),
            ("8 o'clock", time(20, 0)),
            ("at 9", None),
        ],
    )
    def test_clock_forms(self, text, expected):
        assert ex.parse_time(text) == expected

    def test_a_bare_hour_with_a_minute_reads_as_evening(self):
        # Dinner service: 7:30 means 19:30, not breakfast.
        assert ex.parse_time("7:30") == time(19, 30)

    def test_midnight_and_midday_meridiems(self):
        assert ex.parse_time("12:15am") == time(0, 15)
        assert ex.parse_time("12:15pm") == time(12, 15)

    def test_a_nonsense_minute_is_refused(self):
        assert ex.parse_time("7:99") is None


class TestDate:
    def test_today_and_tomorrow(self):
        assert ex.parse_date("today", REF) == date(2026, 3, 11)
        assert ex.parse_date("tonight", REF) == date(2026, 3, 11)
        assert ex.parse_date("tomorrow", REF) == date(2026, 3, 12)

    def test_a_weekday_means_the_next_one_to_come(self):
        # Wednesday 11th -> Friday is the 13th.
        assert ex.parse_date("friday", REF) == date(2026, 3, 13)

    def test_todays_weekday_means_a_week_out_not_today(self):
        assert ex.parse_date("wednesday", REF) == date(2026, 3, 18)

    def test_next_weekday_is_the_following_week(self):
        assert ex.parse_date("next friday", REF) == date(2026, 3, 20)

    def test_month_and_day_either_way_round(self):
        assert ex.parse_date("march 14", REF) == date(2026, 3, 14)
        assert ex.parse_date("14 march", REF) == date(2026, 3, 14)
        assert ex.parse_date("march 14th", REF) == date(2026, 3, 14)

    def test_a_month_already_past_rolls_to_next_year(self):
        assert ex.parse_date("january 5", REF) == date(2027, 1, 5)

    def test_a_bare_ordinal_takes_the_next_such_day(self):
        assert ex.parse_date("the 14th", REF) == date(2026, 3, 14)
        # The 5th has gone this month, so it means April.
        assert ex.parse_date("the 5th", REF) == date(2026, 4, 5)

    def test_iso_is_accepted_because_it_is_unambiguous(self):
        assert ex.parse_date("2026-03-14", REF) == date(2026, 3, 14)

    def test_ambiguous_numeric_dates_are_refused_rather_than_guessed(self):
        # 3/4 is April 3rd to most of the world and March 4th in the US.
        # Refusing costs one clarifying turn; guessing books the wrong night.
        assert ex.parse_date("3/4", REF) is None
        assert ex.parse_date("03/04/2026", REF) is None

    def test_an_impossible_date_is_refused(self):
        assert ex.parse_date("february 30", REF) is None

    def test_absent_is_none(self):
        assert ex.parse_date("a table for four", REF) is None


class TestName:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("my name is sarah", "Sarah"),
            ("it's Malek", "Malek"),
            ("under okonkwo", "Okonkwo"),
            ("book it under sarah chen", "Sarah Chen"),
            ("this is priya", "Priya"),
        ],
    )
    def test_carrier_phrases(self, text, expected):
        assert ex.parse_name(text) == expected

    def test_a_bare_word_is_not_treated_as_a_name(self):
        # Free-form name guessing would write "Friday" on the reservation.
        assert ex.parse_name("friday") is None
        assert ex.parse_name("four people") is None

    def test_stopwords_after_a_carrier_are_rejected(self):
        assert ex.parse_name("it's tomorrow") is None
        assert ex.parse_name("i'm here") is None

    def test_absent_is_none(self):
        assert ex.parse_name("7pm please") is None


class TestExtractAll:
    def test_a_fully_loaded_utterance_fills_everything_at_once(self):
        out = ex.extract_all("table for 4 tomorrow at 7:30pm under sarah chen", REF)
        assert out.party_size == 4
        assert out.booking_date == date(2026, 3, 12)
        assert out.booking_time == time(19, 30)
        assert out.name == "Sarah Chen"
        assert set(out.found) >= {"party_size", "booking_date", "booking_time", "name"}
        assert bool(out) is True

    def test_an_empty_utterance_finds_nothing_and_is_falsy(self):
        out = ex.extract_all("hmm", REF)
        assert out.found == []
        assert bool(out) is False

    def test_raw_text_is_preserved_for_the_transcript(self):
        out = ex.extract_all("tomorrow", REF)
        assert out.raw == "tomorrow"
