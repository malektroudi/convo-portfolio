"""Two faithful ways to put day 13's packet on an A2A wire, and what each loses.

A2A's `Part` is a oneof: a part carries `text`, or `raw` bytes, or a `url`, or
`data` — a `google.protobuf.Value`. Day 13's packet can honestly travel as
either of two of those, and the choice is not cosmetic:

    TEXT    `packet.text`, the rendering an agent reads. What most systems
            send, because it is what a person wants.
    DATA    `packet.as_dict()`, the structure. What a receiving *program*
            needs if it is going to check anything.

Both round-trip byte-identically through protobuf. They are not equally
useful on arrival, and the difference is the point of the day.

**The text form cannot be parsed back.** `Packet.text` joins `slots` on `", "`,
`tried` on `"; "`, and prefixes each transcript line with `"  customer: "`.
None of that is recoverable: a slot value containing a comma, or a `tried`
entry containing a semicolon, is indistinguishable from two entries. So a
receiver holding a text part cannot rebuild the fields day 13's scorer reads,
and cannot re-check anything.

**The data form round-trips, and quietly changes a type.** `Value` has one
numeric kind and it is a double, so `words: 237` arrives as `237.0`. The dict
still compares **equal**, because `237 == 237.0` in Python — which means the
obvious test passes and the loss is real. Anything that re-serialises the
recovered packet writes `"words": 237.0`, and a schema expecting an integer
rejects a packet nothing corrupted.

That is why `type_drift` exists and why it is reported beside the equality
check rather than instead of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from google.protobuf import json_format

from convo.projects.day13_escalation.packet import Packet

TEXT = "text"
DATA = "data"
CARRIERS = (TEXT, DATA)


def to_text_part(packet: Packet):
    """The packet as the rendering a human reads."""
    from a2a.types import Part

    return Part(text=packet.text)


def to_data_part(packet: Packet):
    """The packet as structure, which is the only form a receiver can check."""
    from a2a.types import Part

    part = Part()
    json_format.ParseDict(packet.as_dict(), part.data)
    return part


def message_with(part, carrier: str):
    """One A2A message carrying one part, from the bot to the desk."""
    from a2a.types import Message, Role

    message = Message(message_id=f"handoff-{carrier}", role=Role.ROLE_AGENT)
    message.parts.append(part)
    return message


def to_message(packet: Packet, carrier: str):
    part = to_text_part(packet) if carrier == TEXT else to_data_part(packet)
    return message_with(part, carrier)


def round_trip(message) -> tuple[bytes, Any]:
    """Serialise and parse with the real wire format, not a copy of the object.

    `SerializeToString` is what A2A puts on a gRPC wire and what its JSON
    transport encodes from, so this exercises the same lossiness a real call
    would — without a server, a port, or a second process. The transport is
    demonstrated separately in `desk.py`; this is what *measures*.
    """
    from a2a.types import Message

    wire = message.SerializeToString()
    back = Message()
    back.ParseFromString(wire)
    return wire, back


def from_data_part(part) -> dict[str, Any]:
    return json_format.MessageToDict(part.data)


def recover(message, carrier: str) -> dict[str, Any] | None:
    """The packet fields a receiver can get back, or None if it cannot.

    None for a text part, and that is the finding rather than a limitation of
    this function — see the module docstring.
    """
    if carrier == TEXT:
        return None
    return from_data_part(message.parts[0])


def as_packet(fields: dict[str, Any]) -> Packet:
    """Rebuild day 13's dataclass, so the receiver can use day 13's own scorer.

    Tuples go out as JSON lists and have to come back as tuples, because
    `Packet` is frozen and compares by value — a `Packet` holding lists is not
    equal to the one that was sent even when every character matches.
    """
    return Packet(
        conversation=fields["conversation"],
        topic=fields["topic"],
        reason=fields["reason"],
        tried=tuple(fields.get("tried", ())),
        slots=dict(fields.get("slots", {})),
        transcript=tuple(fields.get("transcript", ())),
        evidence=tuple(fields.get("evidence", ())),
    )


@dataclass(frozen=True)
class Drift:
    """One field whose type changed while its value compared equal."""

    field: str
    sent: str
    received: str
    value: str

    def as_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "sent": self.sent,
            "received": self.received,
            "value": self.value,
        }


def type_drift(sent: dict[str, Any], received: dict[str, Any]) -> tuple[Drift, ...]:
    """Fields that survived by value and not by type.

    The check the obvious one hides. `sent == received` is `True` for a packet
    whose integer word count came back a float, so a test written the natural
    way reports a clean round trip. This reports the types.
    """
    found: list[Drift] = []
    for key, before in sorted(sent.items()):
        if key not in received:
            continue
        after = received[key]
        if type(before) is not type(after):
            found.append(
                Drift(
                    field=key,
                    sent=type(before).__name__,
                    received=type(after).__name__,
                    value=f"{before!r} -> {after!r}",
                )
            )
    return tuple(found)
