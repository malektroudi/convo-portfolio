"""The transition table, exercised as conversations rather than unit calls.

Reference date is Wednesday 2026-03-11 throughout, so every expected date is
fixed. The repair and digression classes at the bottom are the ones worth
reading: they are the cases a demo bot gets wrong and a real one has to get
right.
"""

from __future__ import annotations

from datetime import date, time

import pytest

from convo.projects.day04_dialog_engine import flow as fl
from convo.projects.day04_dialog_engine.engine import Session, new_session, turn
from convo.projects.day04_dialog_engine.extract import Reference

REF = Reference(today=date(2026, 3, 11))  # Wednesday


@pytest.fixture
def session() -> Session:
    return new_session(REF)


def say(session: Session, *utterances: str):
    """Run several turns, returning the last result."""
    result = None
    for text in utterances:
        result = turn(session, text)
    return result


def complete(session: Session):
    """Drive the form to the confirmation page."""
    return say(session, "a table for 4", "friday", "7:30pm", "my name is sarah chen")


class TestFormFilling:
    def test_asks_for_the_first_missing_slot(self, session):
        r = turn(session, "i'd like a table")
        assert session.page == fl.COLLECT
        assert r.pending_slot == fl.PARTY_SIZE

    def test_one_slot_per_turn_in_flow_order(self, session):
        r = turn(session, "for 4 people")
        assert r.filled == [fl.PARTY_SIZE]
        assert r.pending_slot == fl.BOOKING_DATE

        r = turn(session, "friday")
        assert session.slots[fl.BOOKING_DATE] == date(2026, 3, 13)
        assert r.pending_slot == fl.BOOKING_TIME

    def test_a_loaded_utterance_fills_several_slots_at_once(self, session):
        r = turn(session, "table for 2 tomorrow at 8pm under okonkwo")
        assert set(r.filled) == {fl.PARTY_SIZE, fl.BOOKING_DATE, fl.BOOKING_TIME, fl.NAME}
        assert session.page == fl.CONFIRM
        assert r.moved is True

    def test_a_complete_form_moves_to_confirmation_and_reads_it_back(self, session):
        r = complete(session)
        assert session.page == fl.CONFIRM
        assert "4 people" in r.reply
        assert "Friday 13 March" in r.reply
        assert "7:30pm" in r.reply
        assert "Sarah Chen" in r.reply
        assert "Shall I book it?" in r.reply

    def test_confirming_books_it(self, session):
        complete(session)
        r = turn(session, "yes please")
        assert session.page == fl.BOOKED
        assert session.done is True
        assert "Booked" in r.reply

    def test_a_terminal_page_absorbs_further_turns(self, session):
        complete(session)
        turn(session, "yes")
        r = turn(session, "another table for 6")
        assert session.page == fl.BOOKED
        assert r.moved is False
        assert "new conversation" in r.reply


class TestValidation:
    def test_an_oversized_party_is_rejected_and_re_prompted(self, session):
        r = turn(session, "a table for 40")
        assert r.rejected and r.rejected[0][0] == fl.PARTY_SIZE
        assert fl.PARTY_SIZE not in session.slots
        assert "up to 12" in r.reply
        assert r.pending_slot == fl.PARTY_SIZE

    def test_a_time_outside_service_is_rejected(self, session):
        say(session, "for 2", "friday")
        r = turn(session, "3am")
        assert r.rejected and r.rejected[0][0] == fl.BOOKING_TIME
        assert fl.BOOKING_TIME not in session.slots

    def test_a_past_date_is_rejected(self, session):
        r = turn(session, "for 2 on 2026-03-01")
        assert any(slot == fl.BOOKING_DATE for slot, _ in r.rejected)
        assert fl.BOOKING_DATE not in session.slots
        # The valid part of the same utterance still lands.
        assert session.slots[fl.PARTY_SIZE] == 2

    def test_a_date_beyond_the_horizon_is_rejected(self, session):
        r = turn(session, "for 2 on 2027-01-01")
        assert any(slot == fl.BOOKING_DATE for slot, _ in r.rejected)
        assert "90 days" in r.reply or "too far" in r.reply

    def test_a_good_value_after_a_rejection_is_accepted(self, session):
        turn(session, "a table for 40")
        r = turn(session, "sorry, 6")
        assert session.slots[fl.PARTY_SIZE] == 6
        assert r.filled == [fl.PARTY_SIZE]


class TestRepair:
    def test_a_correction_mid_form_overwrites_rather_than_appends(self, session):
        say(session, "for 4", "friday")
        r = turn(session, "actually make it 6 people")
        assert r.repaired == [fl.PARTY_SIZE]
        assert r.filled == []
        assert session.slots[fl.PARTY_SIZE] == 6
        assert "Updated the party size" in r.reply

    def test_no_plus_a_correction_at_confirm_repairs_and_re_confirms(self, session):
        complete(session)
        r = turn(session, "no, make it 8pm")

        assert r.repaired == [fl.BOOKING_TIME]
        assert session.slots[fl.BOOKING_TIME] == time(20, 0)
        assert session.page == fl.CONFIRM, "a correction must not open the amend menu"
        assert "Changed the time" in r.reply
        assert "Shall I book it?" in r.reply

    def test_the_classic_no_the_14th_case(self, session):
        complete(session)
        r = turn(session, "no, the 14th")
        assert r.repaired == [fl.BOOKING_DATE]
        assert session.slots[fl.BOOKING_DATE] == date(2026, 3, 14)
        assert session.page == fl.CONFIRM

    def test_a_repair_keeps_every_other_slot(self, session):
        complete(session)
        before = dict(session.slots)
        turn(session, "no, 8pm")
        for slot in (fl.PARTY_SIZE, fl.BOOKING_DATE, fl.NAME):
            assert session.slots[slot] == before[slot]

    def test_an_invalid_correction_at_confirm_is_refused_and_keeps_the_old_value(self, session):
        complete(session)
        r = turn(session, "no, make it 3am")
        assert session.slots[fl.BOOKING_TIME] == time(19, 30)
        assert r.rejected
        assert session.page == fl.CONFIRM

    def test_repairing_several_slots_in_one_breath(self, session):
        complete(session)
        r = turn(session, "actually 6 people at 9pm")
        assert set(r.repaired) == {fl.PARTY_SIZE, fl.BOOKING_TIME}
        assert session.page == fl.CONFIRM


class TestAmend:
    def test_a_bare_no_at_confirm_asks_what_to_change(self, session):
        complete(session)
        r = turn(session, "no")
        assert session.page == fl.AMEND
        assert "what should I change" in r.reply

    def test_naming_a_field_clears_it_and_re_asks(self, session):
        complete(session)
        say(session, "no")
        r = turn(session, "the time")
        assert fl.BOOKING_TIME not in session.slots
        assert session.page == fl.COLLECT
        assert r.pending_slot == fl.BOOKING_TIME

    def test_answering_amend_with_the_value_directly_skips_back_to_confirm(self, session):
        complete(session)
        say(session, "no")
        r = turn(session, "make it 9pm")
        assert session.slots[fl.BOOKING_TIME] == time(21, 0)
        assert session.page == fl.CONFIRM
        assert "Shall I book it?" in r.reply

    def test_an_unclear_amend_re_asks(self, session):
        complete(session)
        say(session, "no")
        r = turn(session, "hmm")
        assert session.page == fl.AMEND
        assert "Which one" in r.reply


class TestDigression:
    def test_an_off_task_question_is_answered_without_losing_the_form(self, session):
        say(session, "for 4")
        r = turn(session, "do you have parking?")

        assert r.digression == "parking"
        assert "street parking" in r.reply
        assert session.page == fl.COLLECT
        assert session.slots[fl.PARTY_SIZE] == 4, "the form must survive the digression"
        assert r.pending_slot == fl.BOOKING_DATE, "and resume at the same slot"

    def test_the_pending_prompt_is_repeated_after_the_answer(self, session):
        say(session, "for 4")
        r = turn(session, "are you wheelchair accessible?")
        assert r.digression == "accessibility"
        assert "step-free" in r.reply
        assert "What day would you like?" in r.reply

    def test_a_digression_at_confirmation_returns_to_confirmation(self, session):
        complete(session)
        r = turn(session, "what's your cancellation policy?")

        assert r.digression == "cancellation_policy"
        assert session.page == fl.CONFIRM
        assert "4 hours" in r.reply
        assert "Shall I book it?" in r.reply

    def test_a_digression_does_not_count_as_a_failed_prompt(self, session):
        say(session, "for 4")
        before = session.reprompts
        turn(session, "do you have vegan options?")
        assert session.reprompts == before

    def test_an_utterance_carrying_a_slot_value_is_not_treated_as_a_digression(self, session):
        # "open" would match the opening-hours digression, but there is a real
        # party size here, so form filling must win.
        r = turn(session, "4 people")
        assert r.digression is None
        assert r.filled == [fl.PARTY_SIZE]

    def test_the_bare_word_cancel_is_not_hijacked_by_the_faq(self, session):
        # Mid-form "cancel" means abandon, not "tell me the refund policy".
        say(session, "for 4")
        r = turn(session, "actually, forget it")
        assert r.digression is None
        assert session.page == fl.HANDOFF


class TestGlobalIntents:
    def test_asking_for_a_human_escalates_from_any_page(self, session):
        say(session, "for 4", "friday")
        r = turn(session, "can i speak to a human")
        assert session.page == fl.HANDOFF
        assert session.done is True
        assert "putting you through" in r.reply

    def test_one_person_is_a_party_size_not_an_escalation(self, session):
        # "person" used to be a handoff trigger; this is the regression guard.
        turn(session, "a table for one person")
        assert session.page == fl.COLLECT
        assert session.slots[fl.PARTY_SIZE] == 1

    def test_start_over_clears_the_form_but_keeps_the_session(self, session):
        complete(session)
        r = turn(session, "start over")
        assert session.slots == {}
        assert session.page == fl.COLLECT
        assert r.pending_slot == fl.PARTY_SIZE
        assert "starting fresh" in r.reply

    def test_abandoning_hands_off(self, session):
        say(session, "for 4")
        r = turn(session, "never mind")
        assert session.page == fl.HANDOFF
        assert "any time" in r.reply


class TestEscalationOnRepeatedFailure:
    def test_three_unparseable_turns_escalate(self, session):
        say(session, "hmm", "uhh")
        assert session.page == fl.COLLECT
        r = turn(session, "???")
        assert session.page == fl.HANDOFF
        assert "person" in r.reply or "someone" in r.reply

    def test_progress_resets_the_counter(self, session):
        say(session, "hmm", "uhh")
        assert session.reprompts == 2
        turn(session, "for 4")
        assert session.reprompts == 0
        # Two more failures no longer tip it over.
        say(session, "hmm", "uhh")
        assert session.page == fl.COLLECT

    def test_repeated_invalid_values_also_escalate(self, session):
        say(session, "for 40", "for 50")
        r = turn(session, "for 60")
        assert session.page == fl.HANDOFF
        assert r.transition.endswith("handoff")


class TestObservability:
    def test_every_turn_records_its_transition_and_page_move(self, session):
        turn(session, "for 4")
        r = session.history[-1]
        assert r.page_before == fl.COLLECT
        assert r.transition
        assert r.latency_ms >= 0

    def test_history_accumulates_in_order(self, session):
        say(session, "for 4", "friday", "7pm")
        assert len(session.history) == 3
        assert [t.user_text for t in session.history] == ["for 4", "friday", "7pm"]

    def test_as_dict_is_serialisable_and_names_the_slots(self, session):
        turn(session, "table for 2 tomorrow at 8pm under chen")
        payload = session.history[-1].as_dict()
        assert payload["page"] == "collect -> confirm"
        assert set(payload["slots"]) == {
            fl.PARTY_SIZE,
            fl.BOOKING_DATE,
            fl.BOOKING_TIME,
            fl.NAME,
        }

    def test_the_same_transcript_replays_identically(self, session):
        script = ["for 4", "friday", "no, the 14th", "7pm", "under chen", "yes"]
        replay = new_session(REF)
        first = [turn(session, t).reply for t in script]
        second = [turn(replay, t).reply for t in script]
        assert first == second, "a fixed clock must make the engine deterministic"
