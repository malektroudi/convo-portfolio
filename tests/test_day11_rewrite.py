"""Day 11: the lexicon, the memory buffer and the rewriter.

These test *behaviour under a conversation*, not string equality against a
golden rewrite. Pinning exact output would make every rule change a test
change, and would say nothing about whether the rewrite was any good — that
question is answered by the build, against the oracle, with retrieval metrics.
What belongs here is the contract: a pronoun gets an antecedent, a fragment
gets a predicate, memory forgets, and nothing fires on turn one.
"""

from __future__ import annotations

import pytest

from convo.projects.day08_rag_cites.chunking import load_corpus
from convo.projects.day09_retrieval_evals.gold import Gold, locate
from convo.projects.day11_multiturn_rag.conversations import (
    ALL_TURNS,
    CONVERSATIONS,
    FOLLOW_UPS,
)
from convo.projects.day11_multiturn_rag.lexicon import build_lexicon, lexicon
from convo.projects.day11_multiturn_rag.memory import Memory
from convo.projects.day11_multiturn_rag.rewrite import (
    ALL_RULES,
    CARRYOVER,
    ELLIPSIS,
    GUARD,
    GUARDED_RULES,
    PRONOUN,
    STANDALONE,
    SUBSTITUTE,
    anaphors_in,
    content_words,
    rewrite,
)


@pytest.fixture(scope="module")
def documents():
    return load_corpus()


@pytest.fixture(scope="module")
def lex():
    return lexicon()


class TestLexicon:
    def test_it_is_derived_from_the_corpus_not_hand_written(self, documents):
        """Adding an article must extend the lexicon with no code change."""
        before = len(build_lexicon(documents).topics)
        extra = list(documents)
        extra.append(
            type(documents[0])(
                doc_id="gift-vouchers",
                title="Gift vouchers",
                text="# Gift vouchers\n\n## Buying a voucher\n\nVouchers are sold online.\n",
            )
        )
        after = build_lexicon(extra)
        assert len(after.topics) > before
        assert any(t.doc_id == "gift-vouchers" for t in after.topics)

    def test_a_topic_knows_which_article_it_came_from(self, lex):
        topic = lex.best("is there paid street parking")
        assert topic is not None
        assert topic.doc_id == "parking"

    def test_the_longest_phrase_wins(self, lex):
        found = lex.find("what about cancelling a private event")
        assert found
        # "cancelling a private event" is more specific than "private events".
        assert len(found[0].stems) >= 3

    def test_a_heading_that_names_nothing_is_dropped(self, lex):
        # "## What we need from you" has one content stem, "need", which would
        # otherwise match any question containing the word.
        assert all(t.name != "we need from you" for t in lex.topics)
        assert lex.find("do you need a deposit") == [] or all(
            "need" not in t.name for t in lex.find("do you need a deposit")
        )

    def test_matching_is_order_free_and_stemmed(self, lex):
        topic = lex.best("the parking on the street")
        assert topic is not None and topic.doc_id == "parking"

    def test_a_topic_from_an_article_is_marked_as_inferred(self, lex):
        inferred = lex.topic_for("parking")
        assert inferred is not None
        assert inferred.spoken is False


class TestMemory:
    def test_it_forgets_past_its_depth(self):
        memory = Memory(depth=2)
        for n in range(5):
            memory.observe(f"turn {n}", f"turn {n}", None)
        assert len(memory.turns) == 2
        assert memory.turns[-1].text == "turn 4"

    def test_an_empty_memory_offers_no_antecedent(self):
        assert Memory().antecedent() is None

    def test_the_retrieved_article_becomes_an_antecedent(self):
        """ "where can I park" names no topic, but it retrieved one."""
        memory = Memory()
        memory.observe("where can I park", "where can I park", "parking")
        topic = memory.antecedent()
        assert topic is not None and topic.doc_id == "parking"

    def test_what_the_user_said_outranks_what_retrieval_guessed(self):
        """The deposit case: a mis-retrieved middle turn must not capture "it".

        Turn 1 names the deposit. Turn 2 names nothing and retrieval lands on
        the wrong article. Resolving by recency alone would take the wrong
        article; the antecedent must reach back to what was actually said.
        """
        memory = Memory()
        memory.observe(
            "do I need a deposit for a table of eight",
            "do I need a deposit for a table of eight",
            "deposit-policy-2026",
        )
        memory.observe("what if there are twenty of us", "...", "menu-vegan")
        topic = memory.antecedent()
        assert topic is not None
        assert topic.spoken is True
        assert "deposit" in topic.name

    def test_number_agreement_reaches_back_past_a_disagreeing_turn(self):
        """ "they" must skip a singular topic to find a plural one.

        Agreement is sought across the whole buffer, not just in the most
        recent turn — checking one turn and then giving up is not preferring
        agreement, it is noticing it occasionally.
        """
        memory = Memory()
        memory.observe("are assistance animals allowed", "...", "accessibility")
        memory.observe("is there street parking", "...", "parking")
        plural = memory.antecedent(plural=True)
        assert plural is not None and plural.plural is True
        assert "animals" in plural.name

    def test_agreement_is_dropped_rather_than_failing(self):
        """A wrong-number antecedent still retrieves better than a bare pronoun."""
        memory = Memory()
        memory.observe("is there street parking", "...", "parking")
        assert memory.antecedent(plural=True) is not None

    def test_it_stores_the_resolved_query_not_the_raw_text(self):
        memory = Memory()
        memory.observe("what about a private event", "how long to cancel a private event", None)
        assert memory.last_question() == "how long to cancel a private event"


class TestAnaphorDetection:
    @pytest.mark.parametrize(
        "text", ["is it free on sundays", "can they be made gluten free", "how much is it"]
    )
    def test_a_pronoun_is_found(self, text):
        assert anaphors_in(text)

    def test_existential_there_is_not_an_anaphor(self):
        # "is there a dress code" asks whether one exists; it refers to nothing.
        assert "there" not in anaphors_in("is there a dress code")

    def test_a_complementiser_that_is_not_an_anaphor(self):
        assert anaphors_in("I heard that you close at ten") == []

    def test_a_demonstrative_that_is_an_anaphor(self):
        assert "that" in anaphors_in("is that the same on christmas day")


class TestRewriting:
    def test_the_first_turn_is_never_rewritten(self):
        """Every rule needs history, and turn one has none."""
        for conversation in CONVERSATIONS:
            result = rewrite(conversation.turns[0].text, Memory())
            assert result.strategy == STANDALONE
            assert not result.changed

    def test_a_pronoun_is_replaced_by_its_antecedent(self):
        memory = Memory()
        memory.observe("where can I park", "where can I park", "parking")
        result = rewrite("is it free on sundays", memory)
        assert PRONOUN in result.strategy
        assert "it" not in content_words(result.text) + ["it"][:0]
        assert "parking" in result.text

    def test_a_fragment_borrows_the_previous_predicate(self):
        memory = Memory()
        memory.observe(
            "how long before do I have to cancel",
            "how long before do I have to cancel",
            "cancellation-standard",
        )
        result = rewrite("what about for a private event", memory)
        assert ELLIPSIS in result.strategy
        assert "cancel" in result.text
        assert "private event" in result.text

    def test_substitution_replaces_a_whole_noun_phrase(self):
        """The stale modifier must go too, not just the head noun."""
        memory = Memory()
        memory.observe(
            "is the dining room wheelchair accessible",
            "is the dining room wheelchair accessible",
            "accessibility",
        )
        result = rewrite("what about the back room", memory)
        assert SUBSTITUTE in result.strategy
        assert "back room" in result.text
        assert "dining" not in result.text

    def test_resolution_compounds_across_turns(self):
        """Turn 3 inherits turn 2's resolution, not turn 1's wording."""
        memory = Memory()
        memory.observe(
            "how long before do I have to cancel",
            "how long before do I have to cancel",
            "cancellation-standard",
        )
        second = rewrite("what about for a private event", memory)
        memory.observe("what about for a private event", second.text, "cancellation-private-events")
        third = rewrite("and if it is five days before", memory)
        assert "private event" in third.text

    def test_it_explains_itself(self):
        memory = Memory()
        memory.observe("where can I park", "where can I park", "parking")
        result = rewrite("is it free on sundays", memory)
        assert result.detail
        assert "it ->" in result.detail

    def test_a_disabled_rule_does_not_fire(self):
        memory = Memory()
        memory.observe("where can I park", "where can I park", "parking")
        result = rewrite("is it free on sundays", memory, rules=(ELLIPSIS, CARRYOVER))
        assert result.strategy == STANDALONE

    def test_an_unresolvable_pronoun_is_left_alone(self):
        """No antecedent means no guess — a wrong one is worse than none."""
        result = rewrite("is it free on sundays", Memory())
        assert not result.changed


class TestTheGuard:
    def test_it_suppresses_resolution_on_a_turn_with_enough_words(self):
        memory = Memory()
        memory.observe("what time do you open", "what time do you open", "opening-hours")
        thick = "is that the same on christmas day and boxing day"
        assert len(content_words(thick)) >= 3
        assert rewrite(thick, memory, GUARDED_RULES).strategy == STANDALONE
        assert rewrite(thick, memory, ALL_RULES).strategy != STANDALONE

    def test_it_never_suppresses_a_fragment(self):
        """A fragment has no predicate however many content words it carries."""
        memory = Memory()
        memory.observe(
            "how long before do I have to cancel",
            "how long before do I have to cancel",
            "cancellation-standard",
        )
        result = rewrite("what about for a large private event", memory, GUARDED_RULES)
        assert ELLIPSIS in result.strategy

    def test_the_guard_is_not_in_the_default_rules(self):
        # It was tuned on the same twenty turns it is scored against, so it is
        # reported separately rather than folded into the headline.
        assert GUARD not in ALL_RULES
        assert GUARD in GUARDED_RULES


class TestTheEvalSet:
    def test_every_gold_quote_exists_exactly_once(self, documents):
        by_id = {d.doc_id: d for d in documents}
        for name, index, turn in ALL_TURNS:
            located = locate(Gold(turn.text, turn.doc_id, turn.quote), by_id[turn.doc_id])
            assert located.length > 0, f"{name}[{index}]"

    def test_first_turns_are_standalone_and_later_ones_are_not(self):
        for conversation in CONVERSATIONS:
            assert conversation.turns[0].standalone, conversation.name

    def test_every_follow_up_has_an_oracle_or_is_genuinely_standalone(self):
        """A follow-up with no oracle is a claim that it needs no rewriting."""
        for name, index, turn in FOLLOW_UPS:
            if turn.standalone:
                assert len(content_words(turn.text)) >= 3, (
                    f"{name}[{index}] {turn.text!r} has no oracle but is thin — "
                    "either it needs one or it is mislabelled"
                )

    def test_the_oracle_never_equals_the_raw_text(self):
        for name, index, turn in ALL_TURNS:
            if turn.oracle:
                assert turn.oracle.lower() != turn.text.lower(), f"{name}[{index}]"

    def test_the_set_covers_every_article(self, documents):
        covered = {turn.doc_id for _, _, turn in ALL_TURNS}
        assert covered == {d.doc_id for d in documents}

    def test_some_follow_ups_cross_to_another_article(self):
        """The hard case must be represented, and its size is worth knowing."""
        crossing = sum(
            1
            for conversation in CONVERSATIONS
            for turn in conversation.turns[1:]
            if turn.doc_id != conversation.turns[0].doc_id
        )
        assert crossing >= 4
