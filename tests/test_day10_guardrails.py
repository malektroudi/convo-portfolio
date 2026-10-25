"""The two gates, the pipeline, and the red-team scoring.

The over-refusal cases matter as much as the attacks. A guardrail that refuses
everything scores perfectly on injection and is useless, so the red-team set
deliberately contains ordinary questions phrased alarmingly, and the tests
assert that refusing those counts as a failure.
"""

from __future__ import annotations

import pytest

from convo.projects.day08_rag_cites.answer import Citation
from convo.projects.day10_guardrails.groundedness import (
    ANSWER,
    REFUSAL_TEXT,
    REFUSE_NO_CITATION,
    REFUSE_NO_EVIDENCE,
    Evidence,
    decide,
    score_threshold,
)
from convo.projects.day10_guardrails.pipeline import guarded_reply, score_red_team
from convo.projects.day10_guardrails.probes import (
    ATTACK_CLASSES,
    EXPECTATIONS,
    RED_TEAM,
    TEST_CARD,
    UNANSWERABLE,
)
from convo.projects.day10_guardrails.results import load

CITATION = Citation(
    doc_id="parking",
    text="Sundays are free.",
    start=10,
    end=27,
    score=0.9,
    chunk_id="parking#structural:3",
)


def evidence(score: float, citation: Citation | None = CITATION) -> Evidence:
    return Evidence("q", "parking", "parking#structural:3", score, citation)


class TestTheTwoGates:
    def test_strong_evidence_with_a_citation_answers(self):
        decision = decide(evidence(5.0), threshold=0.0)
        assert decision.verdict == ANSWER and decision.answered
        assert decision.text == "Sundays are free."

    def test_weak_evidence_refuses(self):
        decision = decide(evidence(-5.0), threshold=0.0)
        assert decision.verdict == REFUSE_NO_EVIDENCE
        assert not decision.answered and decision.text == REFUSAL_TEXT

    def test_strong_evidence_with_no_citation_still_refuses(self):
        """The citation gate is structural, not advisory."""
        decision = decide(evidence(99.0, citation=None), threshold=0.0)
        assert decision.verdict == REFUSE_NO_CITATION
        assert not decision.answered

    def test_the_threshold_is_the_only_knob_on_the_first_gate(self):
        assert decide(evidence(1.0), threshold=0.0).answered
        assert not decide(evidence(1.0), threshold=2.0).answered


class TestScoreThreshold:
    def test_answering_from_the_wrong_article_is_not_a_success(self):
        wrong = Evidence("q", "parking", "c", 5.0, CITATION)
        metrics = score_threshold([(wrong, "payment")], [], threshold=0.0)
        assert metrics.answer_accuracy == 0.0
        assert metrics.false_refusal == 0.0, "it answered, just wrongly"

    def test_refusing_an_answerable_question_is_a_false_refusal(self):
        metrics = score_threshold([(evidence(-9.0), "parking")], [], threshold=0.0)
        assert metrics.false_refusal == 1.0
        assert metrics.answer_accuracy == 0.0

    def test_answering_an_unanswerable_question_is_a_false_answer(self):
        metrics = score_threshold([], [evidence(5.0)], threshold=0.0)
        assert metrics.false_answer == 1.0
        assert metrics.refusal_recall == 0.0

    def test_refusal_recall_rises_with_the_threshold(self):
        unanswerable = [evidence(-1.0), evidence(1.0)]
        assert score_threshold([], unanswerable, -5.0).refusal_recall == 0.0
        assert score_threshold([], unanswerable, 0.0).refusal_recall == 0.5
        assert score_threshold([], unanswerable, 5.0).refusal_recall == 1.0

    def test_empty_sides_do_not_divide_by_zero(self):
        metrics = score_threshold([], [], 0.0)
        assert metrics.answer_accuracy == 0.0 and metrics.refusal_recall == 0.0


class TestPipeline:
    def test_an_inbound_card_is_redacted_before_anything_else(self):
        reply = guarded_reply(f"my card is {TEST_CARD}", evidence(5.0), 0.0)
        assert TEST_CARD not in reply.question_redacted
        assert reply.redacted_in == 1
        assert any("redacted" in g for g in reply.gates)

    def test_a_clean_question_is_untouched(self):
        reply = guarded_reply("where can I park", evidence(5.0), 0.0)
        assert reply.question_redacted == "where can I park"
        assert reply.redacted_in == 0

    def test_a_refusal_says_so_rather_than_guessing(self):
        reply = guarded_reply("do you have wifi", evidence(-9.0), 0.0)
        assert not reply.answered
        assert reply.reply == REFUSAL_TEXT

    def test_no_evidence_at_all_refuses(self):
        reply = guarded_reply("anything", None, 0.0)
        assert not reply.answered
        assert reply.reason == "no candidates"
        assert reply.reply == REFUSAL_TEXT

    def test_the_outbound_scan_catches_a_leak_and_keeps_the_reply_usable(self):
        leaky = Citation(
            doc_id="payment",
            text=f"Charged to {TEST_CARD}.",
            start=0,
            end=10,
            score=1.0,
            chunk_id="c",
        )
        reply = guarded_reply("ok", Evidence("q", "payment", "c", 5.0, leaky), 0.0)
        assert TEST_CARD not in reply.reply
        assert "the card ending 1111" in reply.reply
        assert reply.scrubbed_out == 1
        assert not reply.leaked

    def test_gates_are_recorded_in_order(self):
        reply = guarded_reply(f"card {TEST_CARD}", evidence(5.0), 0.0)
        assert reply.gates[0].startswith("redacted")


class TestRedTeamScoring:
    def _reply(self, score: float, citation=CITATION):
        return guarded_reply("q", evidence(score, citation), 0.0)

    def test_a_refuse_case_passes_when_it_refuses(self):
        case = next(c for c in RED_TEAM if c.expect == "refuse")
        assert score_red_team(case, self._reply(-9.0)).passed

    def test_a_refuse_case_fails_when_it_answers(self):
        case = next(c for c in RED_TEAM if c.expect == "refuse")
        outcome = score_red_team(case, self._reply(9.0))
        assert not outcome.passed and "answered" in outcome.detail

    def test_a_cite_case_fails_when_it_refuses(self):
        """Over-refusal is a failure, not a safe default."""
        case = next(c for c in RED_TEAM if c.expect == "cite")
        outcome = score_red_team(case, self._reply(-9.0))
        assert not outcome.passed

    def test_a_pii_case_fails_only_on_an_actual_leak(self):
        case = next(c for c in RED_TEAM if c.expect == "no_pii_leak")
        assert score_red_team(case, self._reply(9.0)).passed

    def test_an_unknown_expectation_is_refused(self):
        class Bogus:
            prompt, expect, attack = "x", "vibes", "x"

        with pytest.raises(ValueError, match="unknown expectation"):
            score_red_team(Bogus(), self._reply(9.0))


class TestTheProbeSets:
    def test_every_red_team_case_uses_a_known_expectation(self):
        assert all(c.expect in EXPECTATIONS for c in RED_TEAM)

    def test_the_set_covers_attacks_and_over_refusal(self):
        assert {"injection", "pii", "pressure", "control", "over-refusal"} <= set(ATTACK_CLASSES)

    def test_there_are_cases_that_must_be_answered(self):
        """Without these, refusing everything would score 100%."""
        assert sum(1 for c in RED_TEAM if c.expect == "cite") >= 5

    def test_unanswerable_questions_are_really_absent_from_the_corpus(self):
        from convo.projects.day08_rag_cites.chunking import load_corpus
        from convo.projects.day08_rag_cites.index import tokenise

        corpus = " ".join(d.text.lower() for d in load_corpus())
        corpus_terms = set(tokenise(corpus))
        for probe in UNANSWERABLE:
            terms = set(tokenise(probe.question))
            # At least one content word of each probe must be missing entirely,
            # so a later article cannot quietly make it answerable.
            assert terms - corpus_terms, (
                f"{probe.question!r} shares every word with the corpus; it may "
                f"no longer be unanswerable"
            )

    def test_prompts_are_unique(self):
        prompts = [c.prompt for c in RED_TEAM]
        assert len(prompts) == len(set(prompts))


class TestAgainstTheRealArtifact:
    @pytest.fixture(scope="class")
    def results(self):
        loaded = load()
        if loaded is None:
            pytest.skip("evidence artifact not built")
        return loaded

    def test_the_best_threshold_refuses_everything_unanswerable(self, results):
        best = results.best()
        assert best.refusal_recall == 1.0
        assert best.false_answer == 0.0

    def test_no_red_team_case_leaks_pii_at_the_best_threshold(self, results):
        threshold = results.best().threshold
        for reply in results.replies(threshold):
            assert not reply.leaked, f"PII in reply to {reply.question_redacted!r}"

    def test_every_injection_and_pressure_case_is_refused(self, results):
        threshold = results.best().threshold
        for outcome in results.red_team_outcomes(threshold):
            if outcome.attack in {"injection", "pressure"}:
                assert outcome.passed, f"{outcome.prompt!r}: {outcome.detail}"

    def test_the_red_team_rate_is_not_maximised_by_refusing_everything(self, results):
        """Refusing everything must score worse than the tuned threshold."""
        tuned = results.red_team_rate(results.best().threshold)
        refuse_all = results.red_team_rate(999.0)
        assert tuned > refuse_all
