"""Run real audio through the real pipeline and record where the time went.

    uv sync --extra train --extra voice
    uv run python -m convo.projects.day12_voice.build

Every number in the artifact is measured on this machine: Piper synthesises
the caller's question, whisper-tiny.en transcribes it back, one of two reply
paths answers, and Piper speaks the answer. Nothing is simulated and nothing
is a published benchmark figure.

**The input audio is synthetic, and that flatters the STT.** A TTS voice has
no room tone, no clipping, no accent the model has not seen and no second
person talking over it. The word error rate here should be read as a floor,
not as a result — what this day measures honestly is *latency*, where
synthetic input is close to fair, because whisper spends the same time on
clean audio as on dirty audio of the same length.

Two axes are crossed, because each on its own gives a misleading answer:

  reply path    cited (days 8-11, no model) against generated (0.5B, streamed)
  TTS strategy  speak the first sentence when it exists, or wait for all of it
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from convo.core import get_logger, setup
from convo.projects.day12_voice import vad
from convo.projects.day12_voice.budget import report
from convo.projects.day12_voice.speech import (
    STT_MODEL,
    TTS_VOICE,
    Speaker,
    Transcriber,
    pad_with_silence,
    pcm_to_float,
    wer,
    write_wav,
)
from convo.projects.day12_voice.think import LLM_MODEL, Generator, cited_reply

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

DEFAULT_HANGOVER_MS = 300.0
"""The budget's endpointing target, used for every timing run."""

HANGOVERS = (100.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0, 1200.0)
"""Swept to show what the endpointing number actually costs either way."""

# Questions day 10 already has verified evidence for, so the cited path can
# answer them and the generated path gets the same passage as context. Chosen
# to vary in length, because STT time scales with audio and a set of
# same-length questions would hide that.
QUESTIONS: tuple[str, ...] = (
    "how long before do I have to cancel",
    "where can I park",
    "do you have vegan options",
    "how much is the deposit for a private event",
    "is the dining room wheelchair accessible",
    "what happens if I cancel a private event five days before",
    "do you need my card details to hold a booking",
    "are assistance dogs allowed",
)


@dataclass(frozen=True)
class Utterance:
    """A question to speak, optionally with a hesitation in the middle."""

    question: str
    before: str = ""
    """Spoken, then a gap, then the rest. Empty means say it straight through."""
    after: str = ""
    gap_ms: float = 0.0

    @property
    def hesitant(self) -> bool:
        return bool(self.before and self.after)

    @property
    def label(self) -> str:
        """`gap_ms` is what was spliced in; the pause the endpointer actually
        sees is longer, because Piper pads each half with its own silence. The
        measured value is recorded per row and is the one worth quoting."""
        return f"{self.question} [hesitation]" if self.hesitant else self.question


# Piper reads a sentence straight through: the longest silence inside a
# synthesised question here is 40 ms, so no hangover in the sweep would ever
# fire early and the endpointing tradeoff would look one-sided and free.
#
# Real callers stop mid-sentence — to think, to read a date off a screen, to
# let someone else finish talking — and that pause is what cuts them off. So
# these utterances are **constructed**: each half is synthesised separately and
# a silence of exactly `gap_ms` is spliced between them. The pause lengths are
# the variable under test, which is why they are stated rather than sampled.
HESITATIONS: tuple[Utterance, ...] = (
    Utterance(
        "how long before do I have to cancel",
        before="how long before",
        after="do I have to cancel",
        gap_ms=250.0,
    ),
    Utterance(
        "how much is the deposit for a private event",
        before="how much is the deposit",
        after="for a private event",
        gap_ms=450.0,
    ),
    Utterance(
        "do you need my card details to hold a booking",
        before="do you need my card details",
        after="to hold a booking",
        gap_ms=700.0,
    ),
)

ALL_UTTERANCES: tuple[Utterance, ...] = tuple(Utterance(q) for q in QUESTIONS) + HESITATIONS

log = get_logger("day12.build")


def _context_for(question: str) -> str:
    """The passage day 10 would have cited, used as the generator's context."""
    from convo.projects.day10_guardrails.results import load as load_guard

    results = load_guard()
    if results is None:
        return ""
    for evidence, _ in results.answerable:
        if evidence.question == question and evidence.citation:
            return evidence.citation.text
    return ""


def speak(speaker: Speaker, utterance: Utterance) -> np.ndarray:
    """Synthesise an utterance, splicing in its hesitation if it has one."""

    def say(text: str) -> np.ndarray:
        return pcm_to_float(b"".join(p.pcm for p in speaker.synthesize(text)))

    if not utterance.hesitant:
        return pad_with_silence(say(utterance.question))

    gap = np.zeros(int(vad.SAMPLE_RATE * utterance.gap_ms / 1000), dtype=np.float32)
    return pad_with_silence(np.concatenate([say(utterance.before), gap, say(utterance.after)]))


def main() -> int:
    setup()
    speaker = Speaker()
    transcriber = Transcriber()
    generator = Generator()

    turns: list[dict] = []
    sweep_rows: list[dict] = []
    sample_written = False

    for utterance in ALL_UTTERANCES:
        question = utterance.question
        # --- the caller speaks -------------------------------------------------
        audio = speak(speaker, utterance)
        pause_ms = vad.longest_internal_pause_ms(audio)

        # --- endpointing sweep: what the hangover costs both ways --------------
        for hangover in HANGOVERS:
            found = vad.endpoint(audio, hangover)
            heard, stt_ms = transcriber.transcribe(audio[: found.kept_samples])
            sweep_rows.append(
                {
                    "question": question,
                    "label": utterance.label,
                    "hesitant": utterance.hesitant,
                    "gap_ms": utterance.gap_ms,
                    "pause_ms": pause_ms,
                    "hangover_ms": hangover,
                    "wait_ms": found.wait_ms,
                    "truncated": found.truncated,
                    "endpoint_ms": found.endpoint_ms,
                    "last_speech_ms": found.last_speech_ms,
                    "stt_ms": round(stt_ms, 2),
                    "heard": heard,
                    "wer": wer(question, heard),
                }
            )

        # --- the timed turn, at the budget's endpointing target ----------------
        found = vad.endpoint(audio, DEFAULT_HANGOVER_MS)
        heard, stt_ms = transcriber.transcribe(audio[: found.kept_samples])
        context = _context_for(question)

        replies = {
            "cited": cited_reply(question),
            "generated": generator.reply(heard or question, context),
        }

        for path, reply in replies.items():
            first_sentence_audio_ms, first_pieces = speaker.first_audio_ms(reply.first_sentence)
            whole_audio_ms, whole_pieces = speaker.first_audio_ms(reply.text)

            for strategy, tts_ms, think_ms, played in (
                ("stream", first_sentence_audio_ms, reply.first_sentence_ms, first_pieces),
                ("wait", whole_audio_ms, reply.complete_ms, whole_pieces),
            ):
                budget = report(
                    endpoint=found.wait_ms,
                    stt=stt_ms,
                    think=think_ms,
                    tts=tts_ms,
                )
                turns.append(
                    {
                        "question": question,
                        "label": utterance.label,
                        "hesitant": utterance.hesitant,
                        "heard": heard,
                        "wer": wer(question, heard),
                        "path": path,
                        "strategy": strategy,
                        "reply": reply.as_dict(),
                        "audio_ms": found.audio_ms,
                        "pause_ms": pause_ms,
                        "first_audio_duration_ms": round(
                            played[0].duration_ms if played else 0.0, 2
                        ),
                        "reply_audio_ms": round(sum(p.duration_ms for p in whole_pieces), 2),
                        "budget": budget.as_dict(),
                    }
                )

            if path == "cited" and not sample_written and not utterance.hesitant:
                ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
                write_wav(
                    ARTIFACT_DIR / "sample_reply.wav",
                    pcm_to_float(b"".join(p.pcm for p in whole_pieces)),
                )
                sample_written = True

        log.info(
            "turn measured",
            extra={
                "question": utterance.label,
                "stt_ms": round(stt_ms, 1),
                "cited_ms": round(replies["cited"].first_sentence_ms, 1),
                "generated_first_sentence_ms": round(replies["generated"].first_sentence_ms, 1),
            },
        )

    payload = {
        "stt_model": STT_MODEL,
        "tts_voice": TTS_VOICE,
        "llm_model": LLM_MODEL,
        "sample_rate": vad.SAMPLE_RATE,
        "default_hangover_ms": DEFAULT_HANGOVER_MS,
        "silence_floor": vad.SILENCE_FLOOR,
        "turns": turns,
        "sweep": sweep_rows,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT_DIR / "latency.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log.info(
        "artifact written",
        extra={
            "path": str(out),
            "kilobytes": out.stat().st_size // 1024,
            "turns": len(turns),
            "sweep_rows": len(sweep_rows),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
