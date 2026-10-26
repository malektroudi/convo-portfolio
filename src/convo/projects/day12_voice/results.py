"""Load the measured timings and aggregate them, with no audio stack present.

Medians rather than means throughout. A latency distribution is not symmetric
— one cold cache or one scheduler hiccup drags a mean somewhere no request
actually went — and the number a person experiences is the typical turn, not
the average of the typical turn and an outlier. Where the tail matters it is
reported as a tail (p90) rather than smuggled into the centre.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from convo.projects.day12_voice.budget import BUDGET, BY_KEY, DEADLINE_MS

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
LATENCY_FILE = ARTIFACT_DIR / "latency.json"
SAMPLE_WAV = ARTIFACT_DIR / "sample_reply.wav"

PATHS = ("cited", "generated")
STRATEGIES = ("stream", "wait")


@dataclass(frozen=True)
class Turn:
    question: str
    label: str
    hesitant: bool
    heard: str
    wer: float
    path: str
    strategy: str
    reply: dict
    audio_ms: float
    pause_ms: float
    first_audio_duration_ms: float
    reply_audio_ms: float
    budget: dict

    @property
    def total_ms(self) -> float:
        return self.budget["total_ms"]

    @property
    def met(self) -> bool:
        return self.budget["met"]

    def stage_ms(self, key: str) -> float:
        for stage in self.budget["stages"]:
            if stage["key"] == key:
                return stage["measured_ms"]
        return 0.0

    @property
    def truncated(self) -> bool:
        """Whether the endpointer cut this caller off mid-sentence.

        Derived from the sign of the endpointing stage rather than stored:
        `wait_ms` is the gap between the endpoint and the last speech frame,
        so it goes negative exactly when the rule fired early.

        Such a turn has **no valid end-to-end latency**. The deadline runs
        from the moment the caller stops speaking, and here the system
        answered before that moment arrived — the total is not a fast
        response, it is an interruption with a minus sign in front of it.

        Day 15's warehouse is what found these, by reporting a mean latency
        below the median, which is impossible without negative values.
        """
        return self.stage_ms("endpoint") < 0

    @property
    def stays_ahead(self) -> bool:
        """Whether the first clip plays for longer than the rest takes to make.

        The question that decides whether a reply is heard as one sentence or
        as a sentence, a gap, and then more words. First-audio latency gets all
        the attention and this is what the listener actually notices.
        """
        remaining = self.reply["complete_ms"] - self.reply["first_sentence_ms"]
        return self.first_audio_duration_ms >= remaining


@dataclass(frozen=True)
class SweepRow:
    question: str
    label: str
    hesitant: bool
    gap_ms: float
    pause_ms: float
    hangover_ms: float
    wait_ms: float
    truncated: bool
    endpoint_ms: float
    last_speech_ms: float
    stt_ms: float
    heard: str
    wer: float


@dataclass(frozen=True)
class Day12Results:
    turns: tuple[Turn, ...]
    sweep: tuple[SweepRow, ...]
    meta: dict[str, Any]

    def select(self, path: str, strategy: str) -> tuple[Turn, ...]:
        return tuple(t for t in self.turns if t.path == path and t.strategy == strategy)

    def timed(self, path: str, strategy: str) -> tuple[Turn, ...]:
        """Turns with a meaningful end-to-end latency: the ones not cut off.

        Every latency aggregate uses this. A truncated turn is not a fast
        turn, and leaving them in let three interruptions carry the only
        budget-meeting totals in the set."""
        return tuple(t for t in self.select(path, strategy) if not t.truncated)

    def truncated_count(self, path: str, strategy: str) -> int:
        return sum(1 for t in self.select(path, strategy) if t.truncated)

    def median_total(self, path: str, strategy: str) -> float:
        rows = self.timed(path, strategy)
        return round(statistics.median([t.total_ms for t in rows]), 1) if rows else 0.0

    def median_stage(self, key: str, path: str = "cited", strategy: str = "stream") -> float:
        rows = self.select(path, strategy)
        return round(statistics.median([t.stage_ms(key) for t in rows]), 1) if rows else 0.0

    def p90_total(self, path: str, strategy: str) -> float:
        values = sorted(t.total_ms for t in self.timed(path, strategy))
        if not values:
            return 0.0
        index = min(len(values) - 1, int(round(0.9 * (len(values) - 1))))
        return round(values[index], 1)

    def met_rate(self, path: str, strategy: str) -> float:
        rows = self.timed(path, strategy)
        return round(sum(t.met for t in rows) / len(rows), 4) if rows else 0.0

    def median_turn(self, path: str, strategy: str) -> Turn | None:
        """The turn whose total is the median — a real turn, not a composite.

        Summing per-stage medians gives a profile no turn actually had, and
        the sum does not equal the median total (974 ms against 958 ms here).
        A waterfall is a trace, so it should be a trace of something that
        happened.
        """
        rows = sorted(self.timed(path, strategy), key=lambda t: t.total_ms)
        return rows[len(rows) // 2] if rows else None

    def stage_table(self, path: str = "cited", strategy: str = "stream") -> list[dict]:
        """Each budget stage of the median turn, measured against its target."""
        turn = self.median_turn(path, strategy)
        out = []
        for stage in BUDGET:
            measured = round(turn.stage_ms(stage.key), 1) if turn else 0.0
            out.append(
                {
                    "stage": stage.label,
                    "key": stage.key,
                    "target_ms": stage.target_ms,
                    "measured_ms": measured,
                    "over_ms": round(measured - stage.target_ms, 1),
                    "ratio": round(measured / stage.target_ms, 2) if stage.target_ms else 0.0,
                }
            )
        return out

    def worst_stage(self, path: str, strategy: str) -> dict:
        return max(self.stage_table(path, strategy), key=lambda row: row["over_ms"])

    def token_vs_sentence(self) -> dict:
        """How far apart first-token and first-sentence are, per path."""
        out = {}
        for path in PATHS:
            rows = self.select(path, "stream")
            if not rows:
                continue
            out[path] = {
                "first_token_ms": round(
                    statistics.median([t.reply["first_token_ms"] for t in rows]), 1
                ),
                "first_sentence_ms": round(
                    statistics.median([t.reply["first_sentence_ms"] for t in rows]), 1
                ),
            }
            out[path]["gap_ms"] = round(
                out[path]["first_sentence_ms"] - out[path]["first_token_ms"], 1
            )
        return out

    def sweep_table(self) -> list[dict]:
        """Per hangover: what it costs in dead air and in cut-off callers."""
        by_hangover: dict[float, list[SweepRow]] = {}
        for row in self.sweep:
            by_hangover.setdefault(row.hangover_ms, []).append(row)

        out = []
        for hangover in sorted(by_hangover):
            rows = by_hangover[hangover]
            cut = [r for r in rows if r.truncated]
            kept = [r for r in rows if not r.truncated]
            out.append(
                {
                    "hangover_ms": hangover,
                    "wait_ms": round(statistics.median([r.wait_ms for r in kept]), 1)
                    if kept
                    else 0.0,
                    "truncated": len(cut),
                    "of": len(rows),
                    "truncated_rate": round(len(cut) / len(rows), 4),
                    "wer": round(statistics.mean([r.wer for r in rows]), 4),
                    "wer_fluent": round(
                        statistics.mean([r.wer for r in rows if not r.hesitant]), 4
                    ),
                    "wer_hesitant": round(statistics.mean([r.wer for r in rows if r.hesitant]), 4)
                    if any(r.hesitant for r in rows)
                    else 0.0,
                }
            )
        return out

    def best_hangover(self) -> dict | None:
        """The shortest hangover that cuts nobody off.

        Shortest rather than safest: every extra millisecond past the point
        where truncation stops is dead air bought for nothing.
        """
        clean = [row for row in self.sweep_table() if row["truncated"] == 0]
        return min(clean, key=lambda row: row["hangover_ms"]) if clean else None

    @property
    def deadline_ms(self) -> float:
        return DEADLINE_MS


def _turn(row: dict) -> Turn:
    return Turn(
        question=row["question"],
        label=row.get("label", row["question"]),
        hesitant=row.get("hesitant", False),
        heard=row["heard"],
        wer=row["wer"],
        path=row["path"],
        strategy=row["strategy"],
        reply=row["reply"],
        audio_ms=row["audio_ms"],
        pause_ms=row["pause_ms"],
        first_audio_duration_ms=row["first_audio_duration_ms"],
        reply_audio_ms=row["reply_audio_ms"],
        budget=row["budget"],
    )


def _sweep(row: dict) -> SweepRow:
    return SweepRow(
        question=row["question"],
        label=row.get("label", row["question"]),
        hesitant=row.get("hesitant", False),
        gap_ms=row.get("gap_ms", 0.0),
        pause_ms=row.get("pause_ms", 0.0),
        hangover_ms=row["hangover_ms"],
        wait_ms=row["wait_ms"],
        truncated=row["truncated"],
        endpoint_ms=row["endpoint_ms"],
        last_speech_ms=row["last_speech_ms"],
        stt_ms=row["stt_ms"],
        heard=row["heard"],
        wer=row["wer"],
    )


@lru_cache(maxsize=1)
def load() -> Day12Results | None:
    """Return the measured timings, or None if the build has not been run."""
    if not LATENCY_FILE.exists():
        return None
    payload = json.loads(LATENCY_FILE.read_text(encoding="utf-8"))
    return Day12Results(
        turns=tuple(_turn(r) for r in payload["turns"]),
        sweep=tuple(_sweep(r) for r in payload["sweep"]),
        meta={
            "stt_model": payload["stt_model"],
            "tts_voice": payload["tts_voice"],
            "llm_model": payload["llm_model"],
            "default_hangover_ms": payload["default_hangover_ms"],
            "silence_floor": payload["silence_floor"],
        },
    )


__all__ = ["BY_KEY", "Day12Results", "SweepRow", "Turn", "load", "PATHS", "STRATEGIES"]
