"""The latency budget: what each stage is allowed, and where it came from.

A budget written after measuring is not a budget, it is a description. So this
is set first, from the one number that is not negotiable — **a person waiting
for a voice assistant to answer starts to feel the gap at around 800 ms**, and
is fairly sure the thing is broken by two seconds. Everything below divides
that 800 ms between the stages that have to happen inside it.

The stages are a budget, not a prediction. Three of the four turn out to be
wrong, and that is the useful part: a budget's job is to make the overrun
visible and attributable rather than to be met.

One definition matters more than the rest. The deadline is measured from the
moment the **user stops speaking** to the moment the **first audio sample is
ready to play** — not to the end of the reply. A voice turn is not a request
and a response; it is a gap of silence the person is sitting through, and
nothing after the first sample is in that gap.
"""

from __future__ import annotations

from dataclasses import dataclass

DEADLINE_MS = 800.0
"""End of the user's speech to first audio out. The whole budget."""

UNCOMFORTABLE_MS = 2000.0
"""Past here a listener assumes the line has dropped and starts talking again."""


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    target_ms: float
    why: str


BUDGET: tuple[Stage, ...] = (
    Stage(
        "endpoint",
        "Endpointing",
        300.0,
        "Silence held after speech stops before the turn is called finished. "
        "Pure latency — nothing is computed during it — and the only stage "
        "whose target is a product decision rather than an engineering one.",
    ),
    Stage(
        "stt",
        "STT finalise",
        200.0,
        "Endpoint to final transcript. Partial transcripts do not count: "
        "nothing downstream can start on a partial that may still change.",
    ),
    Stage(
        "think",
        "First speakable sentence",
        150.0,
        "Transcript to the first *complete sentence* of the reply. Not "
        "time-to-first-token — a token cannot be spoken, and the gap between "
        "the two is where voice differs from chat.",
    ),
    Stage(
        "tts",
        "TTS first audio",
        150.0,
        "First sentence to first audio sample. Synthesis of the rest overlaps "
        "with playback, so only the first sentence is on the critical path.",
    ),
)

BY_KEY: dict[str, Stage] = {s.key: s for s in BUDGET}

assert sum(s.target_ms for s in BUDGET) == DEADLINE_MS, "the stages must spend the whole budget"


@dataclass(frozen=True)
class StageResult:
    """One stage, measured against its target."""

    key: str
    measured_ms: float

    @property
    def stage(self) -> Stage:
        return BY_KEY[self.key]

    @property
    def over_ms(self) -> float:
        return round(self.measured_ms - self.stage.target_ms, 2)

    @property
    def ratio(self) -> float:
        return round(self.measured_ms / self.stage.target_ms, 2)

    @property
    def met(self) -> bool:
        return self.measured_ms <= self.stage.target_ms


@dataclass(frozen=True)
class BudgetReport:
    """A whole turn, stage by stage, against the budget."""

    stages: tuple[StageResult, ...]

    @property
    def total_ms(self) -> float:
        return round(sum(s.measured_ms for s in self.stages), 2)

    @property
    def met(self) -> bool:
        return self.total_ms <= DEADLINE_MS

    @property
    def usable(self) -> bool:
        """Past this a listener assumes the call has dropped."""
        return self.total_ms <= UNCOMFORTABLE_MS

    @property
    def worst(self) -> StageResult:
        """The stage that overruns by the most, which is where to spend effort.

        By absolute milliseconds rather than by ratio: a stage that is 10x its
        target but only 90 ms over is a rounding error next to one that is 3x
        and 2 seconds over, and optimising the first because its multiple looks
        alarming is how latency work gets wasted.
        """
        return max(self.stages, key=lambda s: s.over_ms)

    def as_dict(self) -> dict:
        return {
            "total_ms": self.total_ms,
            "met": self.met,
            "stages": [
                {
                    "key": s.key,
                    "measured_ms": round(s.measured_ms, 2),
                    "target_ms": s.stage.target_ms,
                    "over_ms": s.over_ms,
                    "ratio": s.ratio,
                }
                for s in self.stages
            ],
        }


def report(**measured_ms: float) -> BudgetReport:
    """Build a report from keyword milliseconds, one per stage key."""
    unknown = set(measured_ms) - set(BY_KEY)
    if unknown:
        raise ValueError(f"not budget stages: {sorted(unknown)}")
    return BudgetReport(
        tuple(StageResult(s.key, float(measured_ms.get(s.key, 0.0))) for s in BUDGET)
    )
