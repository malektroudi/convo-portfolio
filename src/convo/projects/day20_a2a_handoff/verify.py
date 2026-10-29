"""What the receiving agent can actually check, in three descending levels.

Day 13 scored its packet on three properties: **grounded** (every fact traces
to something said or retrieved), **complete** (the facts the customer would
hate to repeat are present), and **short** (reported, not thresholded). All
three were measured inside one process, against the conversation.

Across an agent boundary only some of that survives, and the brief for this
day — "with the provenance guard intact" — turns out to be the thing that
cannot be delivered. Not for want of a better serialisation: the guard needs
an input the protocol has no reason to carry.

    INTEGRITY       the bytes that arrived are the bytes that were sent.
                    A hash settles it. Cheap, and worth having.

    RECONSTRUCTION  a `Packet` equal to the one sent can be rebuilt, so day
                    13's `complete` check can be re-run by the receiver.
                    True for a data part, impossible for a text part.

    PROVENANCE      `ungrounded_claims` can be re-run. **False for both.**
                    Its signature is `(packet, history)`, and the history is
                    the conversation — which the receiver does not have and
                    which A2A was never going to send.

The third line is the day's result. A provenance guard that lives in the
sender's process is not a property the receiver holds; it is a claim the
receiver is taking on trust. Checking the packet against its own transcript
would be circular — a sender that invents a slot can invent the line that
supports it — so the only honest ways out are to ship the conversation too,
which is measured here, or to not claim the guard crossed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from convo.projects.day13_escalation.conversations import Conversation
from convo.projects.day13_escalation.packet import Packet, score_packet, ungrounded_claims
from convo.projects.day20_a2a_handoff.wire import (
    Drift,
    as_packet,
    recover,
    round_trip,
    to_message,
    type_drift,
)


def digest(payload: dict | str) -> str:
    """A stable hash of what was sent, for the integrity check."""
    text = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Crossing:
    """One packet, one carrier, and what the far side could do with it."""

    conversation: str
    carrier: str
    wire_bytes: int
    integrity: bool
    """Transport integrity: the bytes that arrived are the bytes that were sent."""

    payload_hash_matches: bool
    """Whether re-hashing the *decoded* payload reproduces the sender's hash.

    The standard integrity pattern — sender hashes the payload and puts the
    digest in metadata, receiver re-hashes what it decoded — and for a data
    part it **fails on a packet nothing tampered with**. `words: 237` decodes
    as `237.0`, the two serialise to different JSON, and the digests differ.

    Reported separately from `integrity` because they are different claims and
    this day's finding is that they come apart: the bytes are intact, the
    decoded payload compares equal, and the hash says corrupted. A system that
    hashes decoded payloads will reject its own traffic; the hash belongs over
    the wire bytes, or over a canonical form that pins number types.
    """

    reconstructed: bool
    equal_by_value: bool
    drift: tuple[Drift, ...]
    complete_sender: float
    complete_receiver: float | None
    """`None` when the receiver cannot compute it at all, never 0.0."""

    grounded_sender: bool
    provenance_recheckable: bool
    sent_words: int
    history_words: int

    @property
    def clean(self) -> bool:
        """Every check a receiver could run, passing. Nothing is clean here."""
        return (
            self.integrity
            and self.payload_hash_matches
            and self.reconstructed
            and not self.drift
            and self.provenance_recheckable
        )

    @property
    def complete_agrees(self) -> bool | None:
        if self.complete_receiver is None:
            return None
        return abs(self.complete_receiver - self.complete_sender) < 1e-9

    @property
    def history_multiplier(self) -> float:
        """What carrying the conversation as well would cost, in words.

        The price of making provenance re-checkable on the far side, and the
        reason day 13's third property — short — is the one that pays for it.
        """
        if not self.sent_words:
            return 0.0
        return round((self.sent_words + self.history_words) / self.sent_words, 2)

    def as_dict(self) -> dict:
        return {
            "conversation": self.conversation,
            "carrier": self.carrier,
            "wire_bytes": self.wire_bytes,
            "integrity": self.integrity,
            "payload_hash_matches": self.payload_hash_matches,
            "reconstructed": self.reconstructed,
            "equal_by_value": self.equal_by_value,
            "drift": [d.as_dict() for d in self.drift],
            "complete_sender": self.complete_sender,
            "complete_receiver": self.complete_receiver,
            "complete_agrees": self.complete_agrees,
            "grounded_sender": self.grounded_sender,
            "provenance_recheckable": self.provenance_recheckable,
            "sent_words": self.sent_words,
            "history_words": self.history_words,
            "history_multiplier": self.history_multiplier,
        }


def cross(conversation: Conversation, packet: Packet, carrier: str) -> Crossing:
    """Send one packet one way and report what the receiver can establish."""
    sent_fields = packet.as_dict()
    message = to_message(packet, carrier)
    wire, arrived = round_trip(message)

    sender_score = score_packet(conversation, packet)
    history_words = sum(len(e.user.split()) for e in conversation.exchanges)

    received_fields = recover(arrived, carrier)
    if received_fields is None:
        # A text part. It arrived intact — the integrity check is against the
        # rendering, which is what was sent — and nothing downstream of it can
        # be re-derived, because the rendering is not reversible.
        return Crossing(
            conversation=conversation.name,
            carrier=carrier,
            wire_bytes=len(wire),
            integrity=arrived.parts[0].text == packet.text,
            # A string has no number in it to re-type, so the hash of the
            # decoded payload does reproduce. The text carrier is the one that
            # passes both integrity checks and fails everything after them.
            payload_hash_matches=digest(arrived.parts[0].text) == digest(packet.text),
            reconstructed=False,
            equal_by_value=False,
            drift=(),
            complete_sender=sender_score.complete,
            complete_receiver=None,
            grounded_sender=sender_score.grounded,
            provenance_recheckable=False,
            sent_words=packet.words,
            history_words=history_words,
        )

    rebuilt = as_packet(received_fields)
    receiver_score = score_packet(conversation, rebuilt)

    return Crossing(
        conversation=conversation.name,
        carrier=carrier,
        wire_bytes=len(wire),
        # Protobuf parsed what protobuf wrote; the bytes are intact by
        # construction and the check is here so its passing is on the record
        # beside the hash that does not.
        integrity=round_trip(message)[0] == wire,
        payload_hash_matches=digest(received_fields) == digest(sent_fields),
        reconstructed=rebuilt == packet,
        equal_by_value=received_fields == sent_fields,
        drift=type_drift(sent_fields, received_fields),
        complete_sender=sender_score.complete,
        # Re-derived from the rebuilt packet, which is the honest version of
        # "the receiver can check completeness": it reads the same needs off
        # the same sections. It needs the conversation's `needs` list, which a
        # real receiver would hold as its own definition of a usable handoff.
        complete_receiver=receiver_score.complete,
        grounded_sender=sender_score.grounded,
        # The line this day exists to report. `ungrounded_claims(packet,
        # history)` cannot run without the conversation, and the conversation
        # is not in the packet, not in the message, and not something A2A has
        # any reason to carry.
        provenance_recheckable=False,
        sent_words=packet.words,
        history_words=history_words,
    )


def provenance_with_history(conversation: Conversation, packet: Packet) -> tuple[bool, int]:
    """What the receiver *would* conclude if the conversation crossed too.

    Run so that the trade-off is a measured number rather than an assertion:
    carrying the history makes the guard re-runnable and agrees with the
    sender, and it costs `history_multiplier` times the words.
    """
    problems = ungrounded_claims(packet, list(conversation.exchanges))
    return not problems, len(problems)
