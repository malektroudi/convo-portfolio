"""Load day 20's artifact. Imports nothing from the A2A SDK.

The same split days 18 and 19 settled on, for the same reason: the tab and the
static site have to render without the dependency that produced the figures.
Here it costs nothing — `a2a-sdk` resolves as an ordinary dependency, unlike
`google-adk` — but keeping the boundary means the site build cannot acquire a
new dependency by accident, which is how day 19 took Pages down.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_FILE = ARTIFACT_DIR / "a2a.json"


@dataclass(frozen=True)
class Crossed:
    """One packet, one carrier, as stored."""

    conversation: str
    carrier: str
    wire_bytes: int
    integrity: bool
    payload_hash_matches: bool
    reconstructed: bool
    equal_by_value: bool
    drift: tuple[dict[str, str], ...]
    complete_sender: float
    complete_receiver: float | None
    complete_agrees: bool | None
    grounded_sender: bool
    provenance_recheckable: bool
    sent_words: int
    history_words: int
    history_multiplier: float

    @property
    def clean(self) -> bool:
        return (
            self.integrity
            and self.payload_hash_matches
            and self.reconstructed
            and not self.drift
            and self.provenance_recheckable
        )


@dataclass(frozen=True)
class Results:
    packets: int
    of_conversations: int
    carriers: tuple[str, ...]
    protocol: str
    sdk: str
    card: dict[str, Any]
    crossings: tuple[Crossed, ...]
    transport: tuple[dict[str, Any], ...]
    provenance_with_history: dict[str, dict[str, Any]]

    def by_carrier(self, carrier: str) -> tuple[Crossed, ...]:
        return tuple(c for c in self.crossings if c.carrier == carrier)

    def tally(self, carrier: str, field: str) -> str:
        rows = self.by_carrier(carrier)
        return f"{sum(bool(getattr(r, field)) for r in rows)}/{len(rows)}"

    @property
    def any_clean(self) -> int:
        """Crossings where every check a receiver could run passed.

        Zero, and that is the headline rather than a disappointment: the two
        honest carriers fail different checks and neither can re-establish
        provenance.
        """
        return sum(c.clean for c in self.crossings)

    @property
    def drift_fields(self) -> tuple[str, ...]:
        return tuple(sorted({d["field"] for c in self.crossings for d in c.drift}))

    @property
    def mean_multiplier(self) -> float:
        """Average cost of carrying the conversation as well, in words."""
        rows = [c for c in self.crossings if c.history_multiplier]
        if not rows:
            return 0.0
        return round(sum(c.history_multiplier for c in rows) / len(rows), 2)

    @property
    def grounded_with_history(self) -> str:
        values = self.provenance_with_history.values()
        return f"{sum(1 for v in values if v['grounded'])}/{len(self.provenance_with_history)}"

    @property
    def replies(self) -> dict[str, str]:
        """One example reply per carrier, from the real exchange."""
        found: dict[str, str] = {}
        for row in self.transport:
            found.setdefault(row["carrier"], row["reply"])
        return found


@lru_cache(maxsize=1)
def load() -> Results | None:
    if not ARTIFACT_FILE.exists():
        return None
    payload = json.loads(ARTIFACT_FILE.read_text(encoding="utf-8"))
    return Results(
        packets=payload["packets"],
        of_conversations=payload["of_conversations"],
        carriers=tuple(payload["carriers"]),
        protocol=payload["protocol"],
        sdk=payload["sdk"],
        card=dict(payload["card"]),
        crossings=tuple(
            Crossed(
                conversation=raw["conversation"],
                carrier=raw["carrier"],
                wire_bytes=raw["wire_bytes"],
                integrity=raw["integrity"],
                payload_hash_matches=raw["payload_hash_matches"],
                reconstructed=raw["reconstructed"],
                equal_by_value=raw["equal_by_value"],
                drift=tuple(raw["drift"]),
                complete_sender=raw["complete_sender"],
                complete_receiver=raw["complete_receiver"],
                complete_agrees=raw["complete_agrees"],
                grounded_sender=raw["grounded_sender"],
                provenance_recheckable=raw["provenance_recheckable"],
                sent_words=raw["sent_words"],
                history_words=raw["history_words"],
                history_multiplier=raw["history_multiplier"],
            )
            for raw in payload["crossings"]
        ),
        transport=tuple(payload["transport"]),
        provenance_with_history=dict(payload["provenance_with_history"]),
    )
