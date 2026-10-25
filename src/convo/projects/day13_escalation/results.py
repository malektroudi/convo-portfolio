"""Load the measured escalation run. No model, no retrieval stack.

Everything here is recomputed from the stored record, so the tab can show the
case behind any number — which signal fired, on which exchange, with what
evidence — rather than asking a reader to trust a recall figure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from convo.projects.day13_escalation.conversations import Conversation, Exchange, Need

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
RUN_FILE = ARTIFACT_DIR / "escalation.json"

PACKET_STYLES = ("ticket", "transcript", "full")
STYLE_NOTE = {
    "ticket": "what most deployed bots send: a flag and nothing else",
    "transcript": "everything the customer typed, and no reading of it",
    "full": "topic, reason, what was tried, slots already given, transcript",
}


@dataclass(frozen=True)
class Day13Results:
    conversations: tuple[Conversation, ...]
    policies: dict[str, dict]
    ablation: dict[str, dict]
    counterfactual: dict
    packets: tuple[dict, ...]
    readings: tuple[dict, ...]
    meta: dict[str, Any]

    def by_name(self, name: str) -> Conversation | None:
        return next((c for c in self.conversations if c.name == name), None)

    @property
    def needing(self) -> list[Conversation]:
        return [c for c in self.conversations if c.escalate_at is not None]

    @property
    def resolving(self) -> list[Conversation]:
        return [c for c in self.conversations if c.escalate_at is None]

    def policy_rows(self) -> list[dict]:
        return list(self.policies.values())

    def best_policy(self) -> dict:
        """Fewest turns late, then fewest wasted handoffs.

        Ordered that way on purpose: a customer stuck in a loop is the cost
        this day exists to measure, and a policy that escalates everything has
        no turns late at all — so wasted handoffs break the tie rather than
        being traded against it.
        """
        return min(
            self.policy_rows(),
            key=lambda r: (r["turns_late"], r["wasted_handoffs"], -r["recall"]),
        )

    def packets_for(self, style: str) -> list[dict]:
        return [p for p in self.packets if p["style"] == style]

    def packet_summary(self) -> list[dict]:
        out = []
        for style in PACKET_STYLES:
            rows = self.packets_for(style)
            if not rows:
                continue
            scores = [r["score"] for r in rows]
            out.append(
                {
                    "style": style,
                    "note": STYLE_NOTE[style],
                    "complete": round(sum(s["complete"] for s in scores) / len(scores), 4),
                    "re_asks": sum(s["re_asks"] for s in scores),
                    "grounded": sum(1 for s in scores if s["grounded"]),
                    "of": len(scores),
                    "words": round(sum(s["words"] for s in scores) / len(scores)),
                }
            )
        return out

    def signals_at(self, conversation: str, turn: int) -> list[dict]:
        for row in self.readings:
            if row["conversation"] == conversation and row["turn"] == turn:
                return row["signals"]
        return []

    def packet_for(self, conversation: str, style: str) -> dict | None:
        return next(
            (
                p
                for p in self.packets
                if p["style"] == style and p["packet"]["conversation"] == conversation
            ),
            None,
        )


def _conversation(row: dict) -> Conversation:
    return Conversation(
        name=row["name"],
        kind=row["kind"],
        exchanges=[Exchange(**e) for e in row["exchanges"]],
        escalate_at=row["escalate_at"],
        note=row["note"],
        needs=tuple(Need(n["key"], n["must_contain"], n["why"]) for n in row["needs"]),
    )


@lru_cache(maxsize=1)
def load() -> Day13Results | None:
    """Return the measured run, or None if the build has not happened."""
    if not RUN_FILE.exists():
        return None
    payload = json.loads(RUN_FILE.read_text(encoding="utf-8"))
    return Day13Results(
        conversations=tuple(_conversation(c) for c in payload["conversations"]),
        policies=payload["policies"],
        ablation=payload["ablation"],
        counterfactual=payload.get("counterfactual", {}),
        packets=tuple(payload["packets"]),
        readings=tuple(payload["readings"]),
        meta={"threshold": payload["threshold"], "strategy": payload["strategy"]},
    )
