"""One shape both bots report into, so one set of scorers judges both.

Day 4's state machine and day 5's agent have nothing in common internally —
one has pages and slots, the other has model calls and tool arguments. The
harness refuses to care. Each adapter flattens a finished conversation into
this, and every scorer downstream reads only this.

That is what keeps the comparison honest as the suite grows: a new check is
written once and immediately applies to both, instead of being implemented
twice and drifting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Outcome:
    """What a bot did with one scenario, stripped of how it works inside."""

    scenario_id: str
    booked: bool
    captured: dict[str, Any] = field(default_factory=dict)
    """Normalised booking fields: party_size, date, time, name."""
    transcript: list[dict[str, str]] = field(default_factory=list)
    escalated: bool = False
    """Handed off to a human, which is a legitimate ending rather than a failure."""
    user_turns: int = 0
    model_calls: int = 0
    """Zero for the state machine. The efficiency dimension's main currency."""
    failed_steps: int = 0
    latency_ms: float = 0.0

    @property
    def assistant_replies(self) -> list[str]:
        return [m["content"] for m in self.transcript if m["role"] == "assistant"]

    @property
    def user_messages(self) -> list[str]:
        return [m["content"] for m in self.transcript if m["role"] == "user"]

    def said(self) -> str:
        """Everything the bot said, lowercased — what safety checks read."""
        return " ".join(self.assistant_replies).lower()

    def as_dict(self) -> dict[str, Any]:
        """Serialise the raw outcome, so a scorer change can be re-applied.

        Added on day 18, after a bug in `score_safety` could not be corrected
        in the committed artifacts. Those artifacts stored scored
        `CaseResult`s — the verdicts — and nothing to re-score, so fixing the
        scorer meant another run of a rate-limited hosted model to refresh
        figures the fix had already invalidated.

        The repository's rule is that every published number is re-derivable
        from a committed artifact. Storing verdicts honours that only while
        the scorer never changes, which is not a property anyone can promise.
        """
        return {
            "scenario_id": self.scenario_id,
            "booked": self.booked,
            "captured": self.captured,
            "transcript": self.transcript,
            "escalated": self.escalated,
            "user_turns": self.user_turns,
            "model_calls": self.model_calls,
            "failed_steps": self.failed_steps,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Outcome:
        return cls(
            scenario_id=raw["scenario_id"],
            booked=raw["booked"],
            captured=raw.get("captured", {}),
            transcript=raw.get("transcript", []),
            escalated=raw.get("escalated", False),
            user_turns=raw.get("user_turns", 0),
            model_calls=raw.get("model_calls", 0),
            failed_steps=raw.get("failed_steps", 0),
            latency_ms=raw.get("latency_ms", 0.0),
        )
