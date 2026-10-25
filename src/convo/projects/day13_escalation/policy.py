"""When to hand off, and the metric that says whether you did it in time.

Escalation gets scored as a classification problem — did it escalate, should
it have — and that is the wrong shape. Precision and recall treat a handoff
at turn six as a success when the conversation was already lost at turn two,
and a customer who spent four more turns being misunderstood does not care
that the system eventually agreed with them.

So the headline metric here is **turns late**: the distance between the
exchange the conversation should have been handed over on and the one it
actually was. Nought is correct, positive is a customer kept waiting, and
negative is a handoff that fired before anything had gone wrong — an
expensive person pulled in to answer something the bot had in hand.

Both errors are real and they are not symmetric, which is why they are
reported apart rather than summed into one score.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from convo.projects.day13_escalation.signals import Reading, read

if TYPE_CHECKING:
    from convo.projects.day13_escalation.conversations import Conversation, Exchange


@dataclass(frozen=True)
class Handoff:
    """A decision to escalate, and the case for it."""

    turn: int
    reason: str
    signals: tuple[str, ...]
    evidence: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "turn": self.turn,
            "reason": self.reason,
            "signals": list(self.signals),
            "evidence": list(self.evidence),
        }


Policy = Callable[[list["Exchange"]], Handoff | None]


def never(history: list[Exchange]) -> Handoff | None:
    """Pure self-service. The floor, and worth measuring rather than assuming.

    A bot that never escalates is not obviously wrong — most conversations do
    not need a person, and a system with no handoff at all still resolves the
    majority of them. What it costs is exactly the conversations that needed
    one, which is the number this baseline exists to show.
    """
    return None


def always(history: list[Exchange]) -> Handoff | None:
    """Escalate immediately. The other floor: perfect recall, no value added."""
    if not history:
        return None
    return Handoff(0, "escalate everything", ("always",), ("no conditions",))


def explicit_only(history: list[Exchange]) -> Handoff | None:
    """What day 4 shipped: the customer asks, or the repair loop gives up.

    This is where most production bots stop, and it is a reasonable place to
    stop. Measuring it as a named policy rather than as "no policy" is the
    only way to say whether the extra signals buy anything.
    """
    for turn in range(len(history)):
        reading = read(history, turn)
        asked = reading.by_name("asked_for_human")
        if asked and asked.fired:
            return Handoff(
                turn, "customer asked for a person", ("asked_for_human",), (asked.evidence,)
            )
        if history[turn].reprompts >= 3:
            return Handoff(
                turn,
                "repair loop exhausted",
                ("repair_loop",),
                (f"{history[turn].reprompts} reprompts",),
            )
    return None


def _from_reading(reading: Reading, soft_limit: int) -> Handoff | None:
    binding = reading.binding
    if binding:
        return Handoff(
            reading.turn,
            f"binding signal: {binding[0].name}",
            tuple(s.name for s in binding),
            tuple(s.evidence for s in binding),
        )
    soft = [s for s in reading.fired if not s.binding]
    if len(soft) >= soft_limit:
        return Handoff(
            reading.turn,
            f"{len(soft)} signals agreed",
            tuple(s.name for s in soft),
            tuple(s.evidence for s in soft),
        )
    return None


def signals_policy(soft_limit: int = 2) -> Policy:
    """Binding signals fire alone; weak ones have to agree with each other.

    `soft_limit` is the only tunable here, and it is deliberately the only
    one: a per-signal weight vector would have four more numbers to fit on
    sixteen conversations, which is how you get a policy that works on the set
    it was fitted to and nowhere else.
    """

    def policy(history: list[Exchange]) -> Handoff | None:
        for turn in range(len(history)):
            found = _from_reading(read(history, turn), soft_limit)
            if found is not None:
                return found
        return None

    return policy


def without(signal_name: str, soft_limit: int = 2) -> Policy:
    """The full policy with one signal disabled, for the ablation."""

    def policy(history: list[Exchange]) -> Handoff | None:
        for turn in range(len(history)):
            reading = read(history, turn)
            reading.signals = [s for s in reading.signals if s.name != signal_name]
            found = _from_reading(reading, soft_limit)
            if found is not None:
                return found
        return None

    return policy


@dataclass(frozen=True)
class Scored:
    """One conversation under one policy."""

    name: str
    should_escalate: bool
    expected_turn: int | None
    handoff: Handoff | None

    @property
    def escalated(self) -> bool:
        return self.handoff is not None

    @property
    def correct_decision(self) -> bool:
        return self.escalated == self.should_escalate

    @property
    def turns_late(self) -> int | None:
        """Positive is a customer left waiting; negative is a person wasted.

        None when the comparison is meaningless — either the conversation
        never needed a handoff, or the policy never gave one, and in both
        cases there is no distance to measure. Those are counted as decision
        errors instead, which is what they are.
        """
        if self.expected_turn is None or self.handoff is None:
            return None
        return self.handoff.turn - self.expected_turn

    @property
    def on_time(self) -> bool:
        return self.turns_late == 0


@dataclass(frozen=True)
class PolicyReport:
    name: str
    scored: tuple[Scored, ...]

    @property
    def needed(self) -> list[Scored]:
        return [s for s in self.scored if s.should_escalate]

    @property
    def not_needed(self) -> list[Scored]:
        return [s for s in self.scored if not s.should_escalate]

    @property
    def caught(self) -> int:
        return sum(1 for s in self.needed if s.escalated)

    @property
    def missed(self) -> int:
        return len(self.needed) - self.caught

    @property
    def false_handoffs(self) -> int:
        return sum(1 for s in self.not_needed if s.escalated)

    @property
    def recall(self) -> float:
        return round(self.caught / len(self.needed), 4) if self.needed else 0.0

    @property
    def precision(self) -> float:
        escalated = sum(1 for s in self.scored if s.escalated)
        return round(self.caught / escalated, 4) if escalated else 0.0

    @property
    def on_time(self) -> int:
        return sum(1 for s in self.needed if s.on_time)

    @property
    def total_turns_late(self) -> int:
        """Turns customers spent stuck after the point of no return.

        Summed rather than averaged: this is a count of wasted exchanges, and
        a conversation that was never escalated at all contributes its whole
        remaining length rather than dropping out of the average and
        flattering the policy that abandoned it.
        """
        total = 0
        for item in self.needed:
            if item.handoff is None:
                total += max(0, _length(item) - (item.expected_turn or 0))
            else:
                total += max(0, item.handoff.turn - (item.expected_turn or 0))
        return total

    @property
    def wasted_handoffs(self) -> int:
        """Handoffs on conversations that did not need one, plus early ones."""
        early = sum(1 for s in self.needed if (s.turns_late or 0) < 0)
        return self.false_handoffs + early

    def as_dict(self) -> dict:
        return {
            "policy": self.name,
            "recall": self.recall,
            "precision": self.precision,
            "caught": self.caught,
            "missed": self.missed,
            "false_handoffs": self.false_handoffs,
            "on_time": self.on_time,
            "of_needed": len(self.needed),
            "turns_late": self.total_turns_late,
            "wasted_handoffs": self.wasted_handoffs,
        }


_LENGTHS: dict[str, int] = {}


def _length(item: Scored) -> int:
    return _LENGTHS.get(item.name, 0)


def score(name: str, policy: Policy, conversations: list[Conversation]) -> PolicyReport:
    """Run a policy over every conversation and score the decisions."""
    scored = []
    for conversation in conversations:
        _LENGTHS[conversation.name] = len(conversation.exchanges)
        scored.append(
            Scored(
                name=conversation.name,
                should_escalate=conversation.escalate_at is not None,
                expected_turn=conversation.escalate_at,
                handoff=policy(conversation.exchanges),
            )
        )
    return PolicyReport(name, tuple(scored))
