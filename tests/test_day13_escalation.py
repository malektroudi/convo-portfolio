"""Day 13: the signals, the policies, and what the handoff carries.

These test the contract, not the numbers. The measured recall belongs in the
build, scored against a labelled set that real machinery produced; pinning it
here would turn every change to day 10's threshold into a day 13 test failure
for no reason.

What belongs here is what must hold whatever the numbers say: that a request
for a human is honoured immediately, that a packet never invents a fact, and
that a conversation nobody rescued is counted as a cost rather than dropped.
"""

from __future__ import annotations

import pytest

from convo.projects.day13_escalation import policy as policies
from convo.projects.day13_escalation.conversations import (
    CONVERSATIONS,
    NEEDING,
    RESOLVING,
    Conversation,
    Exchange,
    Need,
)
from convo.projects.day13_escalation.packet import (
    build_packet,
    minimal_packet,
    score_packet,
    transcript_only,
    ungrounded_claims,
)
from convo.projects.day13_escalation.signals import BINDING, read


def convo(name, texts, escalate_at=None, needs=(), **state):
    """A conversation with hand-set machine state, for exercising one rule."""
    exchanges = []
    for index, text in enumerate(texts):
        kwargs = {k: v[index] for k, v in state.items() if index < len(v)}
        exchanges.append(Exchange(user=text, **kwargs))
    return Conversation(name, "faq", exchanges, escalate_at, "", needs)


class TestTheLabelledSet:
    def test_it_contains_conversations_that_must_not_escalate(self):
        """Without these, 'always escalate' scores perfectly and means nothing."""
        assert len(RESOLVING) >= 5
        assert len(NEEDING) >= 5

    def test_every_label_is_a_turn_index_inside_the_conversation(self):
        for conversation in NEEDING:
            assert 0 <= conversation.escalate_at < len(conversation.exchanges), conversation.name

    def test_escalating_conversations_say_what_the_agent_needs(self):
        for conversation in NEEDING:
            assert conversation.needs, conversation.name

    def test_every_conversation_explains_itself(self):
        for conversation in CONVERSATIONS:
            assert len(conversation.note) > 30, conversation.name


class TestSignals:
    def test_asking_for_a_person_is_binding(self):
        history = convo("x", ["hi", "can I speak to a person"]).exchanges
        signal = read(history, 1).by_name("asked_for_human")
        assert signal.fired and signal.binding

    def test_a_table_for_one_person_is_not_asking_for_a_person(self):
        """Day 4 documented this collision; day 13 must not reintroduce it."""
        history = convo("x", ["a table for one person"]).exchanges
        assert not read(history, 0).by_name("asked_for_human").fired

    def test_high_stakes_fires_on_topic_alone(self):
        """Confidence is anti-correlated here, so it is not consulted.

        The two high-stakes conversations needing a person score *higher* than
        the one that does not, because day 10's gates check groundedness, not
        answerability.
        """
        answered = convo("x", ["I was charged twice"], answered=[True]).exchanges
        signal = read(answered, 0).by_name("high_stakes")
        assert signal.fired and signal.binding

    def test_an_ordinary_question_is_not_high_stakes(self):
        history = convo("x", ["what time do you open"]).exchanges
        assert not read(history, 0).by_name("high_stakes").fired

    def test_repeated_refusal_needs_more_than_one(self):
        one = convo("x", ["a", "do you have EV charging"], answered=[True, False]).exchanges
        assert not read(one, 1).by_name("repeated_refusal").fired
        two = convo("x", ["a", "b"], answered=[False, False]).exchanges
        assert read(two, 1).by_name("repeated_refusal").fired

    def test_a_repeated_question_is_recognised_through_rewording(self):
        history = convo("x", ["how much is the deposit", "so how much is the deposit"]).exchanges
        assert read(history, 1).by_name("repeated_question").fired

    def test_two_different_questions_are_not_a_repeat(self):
        history = convo("x", ["where can I park", "do you have vegan options"]).exchanges
        assert not read(history, 1).by_name("repeated_question").fired

    def test_signals_are_sticky_across_turns(self):
        """A signal that fired earlier is still evidence now.

        Without this a customer who repeats themselves and then complains
        presents one signal at each turn and never two at once.
        """
        history = convo("x", ["where can I park", "where do I park", "this is useless"]).exchanges
        fired = {s.name for s in read(history, 2).fired}
        assert "repeated_question" in fired
        assert "frustration" in fired

    def test_only_the_listed_signals_bind(self):
        assert {"asked_for_human", "high_stakes"} == BINDING


class TestPolicies:
    def test_never_escalates_nothing(self):
        assert policies.never(convo("x", ["help"]).exchanges) is None

    def test_always_escalates_on_the_first_exchange(self):
        handoff = policies.always(convo("x", ["hi", "there"]).exchanges)
        assert handoff is not None and handoff.turn == 0

    def test_a_request_for_a_person_is_honoured_on_the_turn_it_is_made(self):
        """Not a turn later. The one rule with no threshold and no override."""
        history = convo("x", ["what time do you open", "put me on to a human"]).exchanges
        handoff = policies.signals_policy()(history)
        assert handoff is not None and handoff.turn == 1

    def test_a_binding_signal_does_not_wait_for_agreement(self):
        history = convo("x", ["I have an allergy question"]).exchanges
        assert policies.signals_policy(soft_limit=3)(history) is not None

    def test_soft_signals_must_reach_the_limit(self):
        history = convo("x", ["where can I park", "where do I park"]).exchanges
        assert policies.signals_policy(soft_limit=2)(history) is None
        assert policies.signals_policy(soft_limit=1)(history) is not None

    def test_dropping_a_signal_removes_its_effect(self):
        history = convo("x", ["I was charged twice"]).exchanges
        assert policies.signals_policy()(history) is not None
        assert policies.without("high_stakes")(history) is None

    def test_turns_late_is_none_when_there_is_nothing_to_compare(self):
        scored = policies.Scored("x", should_escalate=False, expected_turn=None, handoff=None)
        assert scored.turns_late is None

    def test_an_early_handoff_is_negative_not_zero(self):
        handoff = policies.Handoff(0, "r", ("s",), ("e",))
        scored = policies.Scored("x", True, expected_turn=2, handoff=handoff)
        assert scored.turns_late == -2
        assert not scored.on_time

    def test_abandoning_a_conversation_costs_its_remaining_turns(self):
        """A policy that never escalates must not vanish from the average.

        Counting only the conversations a policy *did* escalate would make
        `never` look like it had no timing errors at all.
        """
        conversation = convo("x", ["a", "b", "c", "d"], escalate_at=1)
        report = policies.score("never", policies.never, [conversation])
        assert report.total_turns_late == 3

    def test_recall_and_precision_are_zero_rather_than_undefined(self):
        conversation = convo("x", ["a"], escalate_at=None)
        report = policies.score("never", policies.never, [conversation])
        assert report.recall == 0.0
        assert report.precision == 0.0


class TestThePacket:
    @pytest.fixture
    def escalated(self):
        conversation = convo(
            "billing",
            ["I was charged twice for my deposit"],
            escalate_at=0,
            needs=(Need("topic", "charged"), Need("reason", "charged")),
            answered=[False],
        )
        handoff = policies.signals_policy()(conversation.exchanges)
        assert handoff is not None
        return conversation, handoff

    def test_it_carries_the_customers_own_words(self, escalated):
        conversation, handoff = escalated
        packet = build_packet(conversation, handoff)
        assert "charged twice" in packet.topic

    def test_it_says_why_a_person_is_needed(self, escalated):
        conversation, handoff = escalated
        assert build_packet(conversation, handoff).reason

    def test_it_records_what_was_already_tried(self, escalated):
        conversation, handoff = escalated
        assert any("could not answer" in t for t in build_packet(conversation, handoff).tried)

    def test_a_real_packet_invents_nothing(self, escalated):
        conversation, handoff = escalated
        packet = build_packet(conversation, handoff)
        assert ungrounded_claims(packet, conversation.exchanges) == []

    def test_an_invented_slot_is_caught(self, escalated):
        """Day 5's provenance guard, pointed at the summary.

        A fabricated party size is worse than no party size, because the agent
        will act on it.
        """
        conversation, handoff = escalated
        packet = build_packet(conversation, handoff)
        tampered = type(packet)(**{**packet.__dict__, "slots": {"party_size": "9"}})
        problems = ungrounded_claims(tampered, conversation.exchanges)
        assert problems and "party_size" in problems[0]

    def test_an_invented_citation_is_caught(self, escalated):
        conversation, handoff = escalated
        packet = build_packet(conversation, handoff)
        tampered = type(packet)(
            **{**packet.__dict__, "evidence": ("turn 1: answered from menu-vegan",)}
        )
        assert ungrounded_claims(tampered, conversation.exchanges)

    def test_a_bare_ticket_makes_the_agent_ask_again(self, escalated):
        conversation, handoff = escalated
        score = score_packet(conversation, minimal_packet(conversation, handoff))
        assert score.re_asks == len(conversation.needs)
        assert score.complete == 0.0

    def test_the_full_packet_beats_the_ticket(self, escalated):
        conversation, handoff = escalated
        full = score_packet(conversation, build_packet(conversation, handoff))
        ticket = score_packet(conversation, minimal_packet(conversation, handoff))
        assert full.re_asks < ticket.re_asks

    def test_a_transcript_dump_is_grounded_and_still_not_enough(self, escalated):
        """Completeness alone is not a sufficient metric for a handoff.

        A transcript is grounded by construction and scores well on content,
        while leaving the agent to do all of the reading and all of the
        inference.
        """
        conversation, handoff = escalated
        packet = transcript_only(conversation, handoff)
        score = score_packet(conversation, packet)
        assert score.grounded
        assert score.words < score_packet(conversation, build_packet(conversation, handoff)).words

    def test_the_packet_stops_at_the_handoff_turn(self):
        """Nothing after the handoff happened, so nothing after it is carried."""
        conversation = convo("x", ["hello", "speak to a person", "and another thing"])
        handoff = policies.signals_policy()(conversation.exchanges)
        packet = build_packet(conversation, handoff)
        assert "and another thing" not in packet.transcript


class TestResults:
    def test_it_loads_or_says_it_has_not_been_built(self):
        from convo.projects.day13_escalation.results import load

        results = load()
        if results is None:
            pytest.skip("escalation artifact not built")
        assert results.conversations
        assert results.policies

    def test_the_replayed_state_came_from_the_real_stack(self):
        """At least one refusal and one reprompt, or the build did nothing.

        If every exchange is answered with no reprompts, the machine-state
        fields are still at their defaults and every signal that reads them is
        dead — which is exactly the failure this assertion exists to catch.
        """
        from convo.projects.day13_escalation.results import load

        results = load()
        if results is None:
            pytest.skip("escalation artifact not built")
        exchanges = [e for c in results.conversations for e in c.exchanges]
        assert any(not e.answered for e in exchanges)
        assert any(e.reprompts for e in exchanges)
        assert any(e.evidence_score for e in exchanges)

    def test_the_full_packet_has_the_fewest_re_asks(self):
        from convo.projects.day13_escalation.results import load

        results = load()
        if results is None:
            pytest.skip("escalation artifact not built")
        summary = {row["style"]: row["re_asks"] for row in results.packet_summary()}
        assert summary["full"] < summary["transcript"] < summary["ticket"]

    def test_no_packet_anywhere_invents_a_fact(self):
        from convo.projects.day13_escalation.results import load

        results = load()
        if results is None:
            pytest.skip("escalation artifact not built")
        for row in results.packets:
            assert row["score"]["grounded"], row["packet"]["conversation"]
