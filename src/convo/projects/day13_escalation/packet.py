"""What the human receives, and whether it is any good.

The half of escalation nobody builds. A handoff that arrives as "customer
needs help" is not a handoff — the agent opens with "can you tell me what
happened?", the customer repeats everything they already typed, and the
escalation has cost them more time than the bot did.

Three properties, in descending order of how much they matter:

  **grounded**   every fact in the packet traces to something that was said
                 or something the system recorded. A summary that invents a
                 party size is worse than no summary, because the agent will
                 act on it. This is day 5's provenance guard pointed at a
                 different artifact.

  **complete**   the facts the customer would be furious to repeat are all
                 present. Measured against a per-conversation list written
                 from the customer's side rather than from what happens to
                 be easy to serialise.

  **short**      an agent reads this while the customer waits. A packet that
                 contains the entire transcript and nothing else is complete,
                 grounded, and useless.

The first two are measured. The third is reported as a length and left to the
reader, because "too long" has no defensible threshold and inventing one would
be the kind of precision this portfolio keeps arguing against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from convo.projects.day13_escalation.conversations import Conversation, Exchange, Need
from convo.projects.day13_escalation.policy import Handoff


@dataclass(frozen=True)
class Packet:
    """The handoff, as the agent sees it."""

    conversation: str
    topic: str
    reason: str
    tried: tuple[str, ...]
    slots: dict[str, str]
    transcript: tuple[str, ...]
    evidence: tuple[str, ...]

    def section(self, key: str) -> str:
        """One section as flat text, for checking facts against."""
        value = {
            "topic": self.topic,
            "reason": self.reason,
            "tried": " ".join(self.tried),
            "slots": " ".join(f"{k}={v}" for k, v in self.slots.items()),
            "transcript": " ".join(self.transcript),
            "evidence": " ".join(self.evidence),
        }.get(key, "")
        return value

    @property
    def text(self) -> str:
        lines = [
            f"Topic: {self.topic}",
            f"Why a person: {self.reason}",
        ]
        if self.slots:
            lines.append("Already given: " + ", ".join(f"{k} = {v}" for k, v in self.slots.items()))
        if self.tried:
            lines.append("Already tried: " + "; ".join(self.tried))
        if self.evidence:
            lines.append("What the system found: " + "; ".join(self.evidence))
        lines.append("Transcript:")
        lines.extend(f"  customer: {t}" for t in self.transcript)
        return "\n".join(lines)

    @property
    def words(self) -> int:
        return len(self.text.split())

    def as_dict(self) -> dict:
        return {
            "conversation": self.conversation,
            "topic": self.topic,
            "reason": self.reason,
            "tried": list(self.tried),
            "slots": dict(self.slots),
            "transcript": list(self.transcript),
            "evidence": list(self.evidence),
            "words": self.words,
        }


def _topic(history: list[Exchange], upto: int) -> str:
    """What the conversation is about, in the customer's own words.

    The first substantive thing they said, not a paraphrase and not a label
    from a classifier. A topic line that reads "billing enquiry" when the
    customer wrote "I was charged twice" has thrown away the only detail the
    agent needs, and gained nothing.
    """
    for exchange in history[: upto + 1]:
        if len(exchange.user.split()) >= 3:
            return exchange.user
    return history[0].user if history else ""


def build_packet(
    conversation: Conversation, handoff: Handoff, history: list[Exchange] | None = None
) -> Packet:
    """Assemble the packet from what actually happened, and nothing else."""
    history = history if history is not None else conversation.exchanges
    upto = min(handoff.turn, len(history) - 1)
    seen = history[: upto + 1]

    tried: list[str] = []
    evidence: list[str] = []
    for index, exchange in enumerate(seen):
        if not exchange.answered:
            tried.append(f"turn {index + 1}: could not answer {exchange.user!r}")
        elif exchange.doc_id:
            evidence.append(f"turn {index + 1}: answered from {exchange.doc_id}")
        if exchange.reprompts:
            tried.append(f"turn {index + 1}: {exchange.reprompts} reprompts on the same slot")

    slots: dict[str, str] = {}
    for exchange in seen:
        slots.update(exchange.slots)

    reason = handoff.reason
    if handoff.evidence:
        reason = f"{reason} ({'; '.join(e for e in handoff.evidence if e)})"

    return Packet(
        conversation=conversation.name,
        topic=_topic(seen, upto),
        reason=reason,
        tried=tuple(tried),
        slots=slots,
        transcript=tuple(e.user for e in seen),
        evidence=tuple(evidence),
    )


# --------------------------------------------------------------------- scoring


@dataclass(frozen=True)
class NeedCheck:
    need: Need
    met: bool


@dataclass
class PacketScore:
    """How good one packet is, by the three properties that matter."""

    conversation: str
    checks: list[NeedCheck] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)
    words: int = 0

    @property
    def complete(self) -> float:
        if not self.checks:
            return 1.0
        return round(sum(c.met for c in self.checks) / len(self.checks), 4)

    @property
    def re_asks(self) -> int:
        """Facts the agent would have to ask the customer for again."""
        return sum(1 for c in self.checks if not c.met)

    @property
    def grounded(self) -> bool:
        return not self.ungrounded

    def as_dict(self) -> dict:
        return {
            "conversation": self.conversation,
            "complete": self.complete,
            "re_asks": self.re_asks,
            "grounded": self.grounded,
            "ungrounded": list(self.ungrounded),
            "words": self.words,
            "needs": [
                {"key": c.need.key, "want": c.need.must_contain, "met": c.met} for c in self.checks
            ],
        }


def ungrounded_claims(packet: Packet, history: list[Exchange]) -> list[str]:
    """Facts in the packet that nothing in the conversation supports.

    The check is deliberately mechanical: every slot value and every quoted
    transcript line must appear in what the customer actually typed, and every
    cited document must be one the system actually retrieved. It cannot catch
    a *misleading* summary, only an invented one — which is the failure mode
    that gets an agent to act on something that never happened.
    """
    said = " ".join(e.user for e in history).lower()
    retrieved = {e.doc_id for e in history if e.doc_id}
    problems: list[str] = []

    for key, value in packet.slots.items():
        if value and value.lower() not in said:
            problems.append(f"slot {key}={value!r} was never said")

    for line in packet.transcript:
        if line.lower() not in said:
            problems.append(f"transcript line {line!r} was never said")

    for item in packet.evidence:
        doc = item.rsplit(" ", 1)[-1]
        if doc and doc not in retrieved:
            problems.append(f"cited {doc!r}, which was never retrieved")

    return problems


def score_packet(
    conversation: Conversation, packet: Packet, history: list[Exchange] | None = None
) -> PacketScore:
    history = history if history is not None else conversation.exchanges
    checks = [
        NeedCheck(need, need.must_contain.lower() in packet.section(need.key).lower())
        for need in conversation.needs
    ]
    return PacketScore(
        conversation=conversation.name,
        checks=checks,
        ungrounded=ungrounded_claims(packet, history),
        words=packet.words,
    )


def minimal_packet(conversation: Conversation, handoff: Handoff) -> Packet:
    """The handoff most systems actually send: a ticket with no context.

    Here as a baseline rather than as a straw man — "customer needs
    assistance" plus a transcript dump is what an escalation looks like in a
    great many deployed bots, and the re-ask rate it produces is the number
    the full packet has to beat.
    """
    return Packet(
        conversation=conversation.name,
        topic="customer needs assistance",
        reason="escalated",
        tried=(),
        slots={},
        transcript=(),
        evidence=(),
    )


def transcript_only(
    conversation: Conversation, handoff: Handoff, history: list[Exchange] | None = None
) -> Packet:
    """Everything the customer said and nothing else — the other easy option.

    Complete by construction and grounded by construction, which is exactly
    why completeness alone is not a sufficient metric: this packet makes the
    agent do all the reading and all the inference, and it scores perfectly
    on two of the three properties.
    """
    history = history if history is not None else conversation.exchanges
    upto = min(handoff.turn, len(history) - 1)
    seen = history[: upto + 1]
    return Packet(
        conversation=conversation.name,
        topic=_topic(seen, upto),
        reason="escalated",
        tried=(),
        slots={},
        transcript=tuple(e.user for e in seen),
        evidence=(),
    )
