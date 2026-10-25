"""Day 12: the budget, the endpointer, and the metrics that score them.

Nothing here loads a model. The components that need torch or onnxruntime are
exercised by the build; what is tested is the arithmetic around them, plus the
one structural property the hub depends on — that the voice modules import
cleanly on a machine with no audio stack installed, because a tab that raises
on import takes every other tab down with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from convo.projects.day12_voice import vad
from convo.projects.day12_voice.budget import (
    BUDGET,
    DEADLINE_MS,
    UNCOMFORTABLE_MS,
    report,
)
from convo.projects.day12_voice.speech import normalise, wer
from convo.projects.day12_voice.think import split_first_sentence

RATE = vad.SAMPLE_RATE


def tone(ms: float, amplitude: float = 0.3) -> np.ndarray:
    """A loud block standing in for speech."""
    samples = int(RATE * ms / 1000)
    t = np.arange(samples, dtype=np.float32) / RATE
    return (amplitude * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def silence(ms: float) -> np.ndarray:
    return np.zeros(int(RATE * ms / 1000), dtype=np.float32)


class TestTheBudget:
    def test_the_stages_spend_exactly_the_deadline(self):
        """A budget whose parts do not sum to the whole is a wish list."""
        assert sum(s.target_ms for s in BUDGET) == DEADLINE_MS

    def test_every_stage_says_why_it_is_there(self):
        for stage in BUDGET:
            assert len(stage.why) > 40, stage.key

    def test_a_turn_inside_the_budget_is_met(self):
        result = report(endpoint=300, stt=200, think=150, tts=150)
        assert result.total_ms == DEADLINE_MS
        assert result.met

    def test_one_millisecond_over_is_not_met(self):
        assert not report(endpoint=301, stt=200, think=150, tts=150).met

    def test_a_missing_stage_counts_as_zero_rather_than_passing(self):
        """Forgetting to measure a stage must not look like meeting it."""
        result = report(endpoint=300)
        assert result.total_ms == 300
        assert not report(endpoint=300).stages[1].measured_ms

    def test_unknown_stages_are_refused(self):
        with pytest.raises(ValueError, match="not budget stages"):
            report(endpoint=300, network=50)

    def test_usable_is_a_wider_bar_than_met(self):
        result = report(endpoint=300, stt=500, think=400, tts=200)
        assert not result.met
        assert result.usable
        assert report(endpoint=300, stt=500, think=3000, tts=200).usable is False

    def test_worst_stage_is_by_milliseconds_not_by_multiple(self):
        """A 10x overrun of 90 ms matters less than a 3x overrun of 2 seconds.

        Ranking by ratio sends you to optimise the cheap stage, which is how
        latency work gets spent on the wrong thing.
        """
        result = report(endpoint=300, stt=200, think=2000, tts=1500)
        assert result.worst.key == "think"
        assert result.worst.over_ms > result.stages[-1].over_ms
        # tts is 10x its target; think is under 14x but 1850 ms over.
        assert result.stages[-1].ratio == 10.0


class TestEndpointing:
    def test_it_fires_after_the_hangover_and_not_before(self):
        audio = np.concatenate([tone(500), silence(1000)])
        found = vad.endpoint(audio, hangover_ms=300)
        assert not found.truncated
        assert found.wait_ms == pytest.approx(300, abs=vad.FRAME_MS)

    def test_leading_silence_is_not_an_ending(self):
        """Silence before anyone speaks is someone thinking, not finishing."""
        audio = np.concatenate([silence(900), tone(400), silence(800)])
        found = vad.endpoint(audio, hangover_ms=300)
        assert found.endpoint_ms > 900

    def test_a_pause_longer_than_the_hangover_cuts_the_caller_off(self):
        audio = np.concatenate([tone(400), silence(600), tone(400), silence(800)])
        found = vad.endpoint(audio, hangover_ms=300)
        assert found.truncated
        assert found.endpoint_ms < found.last_speech_ms

    def test_a_hangover_longer_than_the_pause_does_not(self):
        audio = np.concatenate([tone(400), silence(600), tone(400), silence(900)])
        found = vad.endpoint(audio, hangover_ms=800)
        assert not found.truncated

    def test_it_never_looks_ahead(self):
        """The decision must be the one a live endpointer could have made.

        A rule allowed to see the whole buffer would never cut anyone off, and
        would also be unimplementable — so truncating here is correct
        behaviour, and a test that forbade it would be testing a fantasy.
        """
        audio = np.concatenate([tone(300), silence(500), tone(300), silence(800)])
        short = vad.endpoint(audio, hangover_ms=200)
        assert short.truncated
        assert short.endpoint_ms == pytest.approx(300 + 200, abs=vad.FRAME_MS)

    def test_silence_only_audio_does_not_crash(self):
        found = vad.endpoint(silence(1000), hangover_ms=300)
        assert found.last_speech_ms == 0.0
        assert not found.truncated

    def test_an_empty_buffer_is_safe(self):
        found = vad.endpoint(np.zeros(0, dtype=np.float32), hangover_ms=300)
        assert found.audio_ms == 0.0

    def test_audio_that_never_goes_quiet_ends_with_the_buffer(self):
        found = vad.endpoint(tone(1000), hangover_ms=300)
        assert found.endpoint_ms == pytest.approx(1000, abs=vad.FRAME_MS)
        assert not found.truncated

    def test_longest_internal_pause_ignores_the_tail(self):
        """Trailing silence is not a pause — it is the end of the utterance."""
        audio = np.concatenate([tone(300), silence(400), tone(300), silence(2000)])
        assert vad.longest_internal_pause_ms(audio) == pytest.approx(400, abs=2 * vad.FRAME_MS)

    def test_a_longer_hangover_never_cuts_off_more_callers(self):
        """The tradeoff must be monotonic, or the slider lies to the reader."""
        audio = np.concatenate([tone(300), silence(500), tone(300), silence(1200)])
        truncations = [vad.endpoint(audio, h).truncated for h in (100, 200, 400, 600, 800)]
        assert truncations == sorted(truncations, reverse=True)


class TestWordErrorRate:
    def test_a_perfect_transcript_scores_zero(self):
        assert wer("where can I park", "Where can I park?") == 0.0

    def test_case_and_punctuation_are_not_errors(self):
        assert normalise("Where can I park?") == ["where", "can", "i", "park"]

    def test_a_truncated_transcript_is_penalised(self):
        assert wer("how long before do I have to cancel", "how long before") > 0.5

    def test_it_is_not_capped_at_one(self):
        """Hallucinated words past the end must be visible, not clamped.

        An endpointer that never fires feeds silence to the recogniser, and
        whisper fills silence with words. Capping would make that failure look
        identical to transcribing nothing.
        """
        assert wer("park here", "park here and then some more invented words") > 1.0

    def test_an_empty_hypothesis_is_total_error(self):
        assert wer("where can I park", "") == 1.0

    def test_an_empty_reference_with_output_is_total_error(self):
        assert wer("", "something") == 1.0


class TestFirstSentence:
    @pytest.mark.parametrize(
        ("text", "first"),
        [
            ("We hold it for fifteen minutes. Call us if late.", "We hold it for fifteen minutes."),
            ("Yes, we do! Would you like a table?", "Yes, we do!"),
            ("Is that right? No.", "Is that right?"),
        ],
    )
    def test_it_splits_on_the_first_terminator(self, text, first):
        assert split_first_sentence(text)[0] == first

    def test_an_unterminated_reply_is_returned_whole(self):
        """Mid-stream there may be no sentence yet, and that is not an error."""
        assert split_first_sentence("We hold your table for")[0] == "We hold your table for"

    def test_a_decimal_point_does_not_end_a_sentence(self):
        text, rest = split_first_sentence("The deposit is 200.50 dollars. Pay on arrival.")
        assert text == "The deposit is 200.50 dollars."
        assert rest == "Pay on arrival."


class TestTheModulesImportWithoutAnAudioStack:
    def test_importing_speech_does_not_need_onnxruntime(self):
        """The hub has neither torch nor piper and must still start.

        Every heavy import in `speech` and `think` is inside a function for
        this reason; a module-level import would be invisible until the hub
        was deployed without the extras.
        """
        import convo.projects.day12_voice.speech as speech
        import convo.projects.day12_voice.think as think

        assert speech.SAMPLE_RATE == 16000
        assert think.LLM_MODEL

    def test_the_tab_module_imports(self):
        import convo.projects.day12_voice.ui as ui

        assert callable(ui.build)


class TestResults:
    def test_it_loads_or_says_it_has_not_been_built(self):
        from convo.projects.day12_voice.results import load

        results = load()
        if results is None:
            pytest.skip("latency artifact not built")
        assert results.turns
        assert results.sweep

    def test_every_turn_has_all_four_stages(self):
        from convo.projects.day12_voice.results import load

        results = load()
        if results is None:
            pytest.skip("latency artifact not built")
        keys = {s.key for s in BUDGET}
        for turn in results.turns:
            assert {s["key"] for s in turn.budget["stages"]} == keys

    def test_the_sweep_is_monotonic_in_truncation(self):
        from convo.projects.day12_voice.results import load

        results = load()
        if results is None:
            pytest.skip("latency artifact not built")
        rates = [row["truncated_rate"] for row in results.sweep_table()]
        assert rates == sorted(rates, reverse=True)

    def test_a_longer_hangover_always_costs_more_dead_air(self):
        from convo.projects.day12_voice.results import load

        results = load()
        if results is None:
            pytest.skip("latency artifact not built")
        waits = [row["wait_ms"] for row in results.sweep_table()]
        assert waits == sorted(waits)

    def test_the_deadline_is_the_one_the_budget_declares(self):
        from convo.projects.day12_voice.results import load

        results = load()
        if results is None:
            pytest.skip("latency artifact not built")
        assert results.deadline_ms == DEADLINE_MS
        assert UNCOMFORTABLE_MS > DEADLINE_MS
