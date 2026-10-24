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
