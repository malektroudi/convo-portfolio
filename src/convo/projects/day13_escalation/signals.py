"""Signals that a conversation is going wrong, and what each one is evidence of.

Escalation is usually built as one threshold on one score, which is what days
2, 10 and 12 each did for their own problem. It does not work here, and the
reason is worth stating: the signals are not commensurable. "The customer
asked for a human" and "the retriever scored 0.4" are not two readings of the
same quantity that can be averaged — one is a request that must be honoured
immediately and the other is weak evidence that becomes meaningful only if it
keeps happening.

So each signal reports three things: whether it fired, the evidence it fired
on, and whether it is **binding**. A binding signal escalates on its own, now.
A non-binding one only counts towards a total, and only when it persists.

Every signal reads machine state the system already produces — a refusal from
day 10's citation gate, a reprompt count from day 4's engine, the customer's
own words. None of them need a model, and none of them need the conversation
to be labelled, because at inference time nothing is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from convo.projects.day13_escalation.conversations import Exchange

# Day 4 built this list and documented why "person" is not in it: "a table for
# one person" is a party size. Reused rather than rewritten, because a second
# copy would drift and only one of them would have the comment.
from convo.projects.day04_dialog_engine.flow import HUMAN_PATTERNS

FRUSTRATION_PATTERNS = (
    "this is useless",
    "you are useless",
    "you're useless",
    "not what i asked",
    "that's not what i asked",
    "i already said",
    "i already told you",
    "for the third time",
    "again",
    "still waiting",
    "forget it",
    "ridiculous",
)

# Topics where being confidently wrong is expensive rather than annoying. The
# test is not whether the bot *can* answer — day 10 answers allergen questions
# from the corpus perfectly well — but whether a wrong answer hurts someone.
HIGH_STAKES_PATTERNS = (
    "allerg",
    "anaphyla",
    "epipen",
    "coeliac",
    "celiac",
    "charged twice",
    "double charged",
    "wrong amount",
    "refund",
    "complaint",
    "complain",
    "food poisoning",
    "ill after",
    "sick after",
    "lawyer",
    "legal",
)

BINDING = frozenset({"asked_for_human", "high_stakes_unanswered"})
"""Signals that escalate on their own, without waiting for a second opinion."""


@dataclass(frozen=True)
class Signal:
    name: str
    fired: bool
    evidence: str = ""
    turn: int = -1
    """The exchange index the signal fired on."""

    @property
    def binding(self) -> bool:
        return self.fired and self.name in BINDING

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "fired": self.fired,
            "evidence": self.evidence,
            "turn": self.turn,
            "binding": self.binding,
        }


@dataclass
class Reading:
    """Every signal's state as of one exchange."""

    turn: int
    signals: list[Signal] = field(default_factory=list)

    @property
    def fired(self) -> list[Signal]:
        return [s for s in self.signals if s.fired]

    @property
    def binding(self) -> list[Signal]:
        return [s for s in self.signals if s.binding]

    @property
    def soft_count(self) -> int:
        return len([s for s in self.fired if not s.binding])

    def by_name(self, name: str) -> Signal | None:
        return next((s for s in self.signals if s.name == name), None)


def _matches(text: str, patterns: tuple[str, ...]) -> str:
    lowered = text.lower()
    for pattern in patterns:
        if pattern in lowered:
            return pattern
    return ""


def asked_for_human(history: list[Exchange], turn: int) -> Signal:
    """The customer said so. Nothing outranks this and nothing should.

    A system that keeps trying after being asked for a person is not being
    helpful, it is refusing. This is the one signal with no threshold, no
    accumulation and no override.
    """
    hit = _matches(history[turn].user, HUMAN_PATTERNS)
    return Signal("asked_for_human", bool(hit), f"said {hit!r}" if hit else "", turn)


def high_stakes_unanswered(history: list[Exchange], turn: int) -> Signal:
    """A high-stakes topic the system could not answer from its evidence.

    Both halves are needed. An allergen question day 10 answers with a
    verified citation is a *good* outcome and must not escalate — routing it
    to a human would make the guardrail work pointless. The same question with
    a refusal attached is somebody with a severe allergy being told "I don't
    know", and that is a person, immediately.
    """
    exchange = history[turn]
    hit = _matches(exchange.user, HIGH_STAKES_PATTERNS)
    fired = bool(hit) and not exchange.answered
    return Signal(
        "high_stakes_unanswered",
        fired,
        f"{hit!r} and the system could not answer" if fired else "",
        turn,
    )


def repeated_refusal(history: list[Exchange], turn: int, limit: int = 2) -> Signal:
    """Day 10 refused more than once in this conversation.

    One refusal is the guardrail working. A second means the customer is
    somewhere the corpus does not cover, and no amount of rephrasing will
    change that — continuing is asking them to guess the magic words.
    """
    refusals = [i for i, e in enumerate(history[: turn + 1]) if not e.answered]
    fired = len(refusals) >= limit
    return Signal(
        "repeated_refusal",
        fired,
        f"{len(refusals)} refusals by turn {turn + 1}" if fired else "",
        turn,
    )


def repair_loop(history: list[Exchange], turn: int, limit: int = 2) -> Signal:
    """Day 4's engine reprompted for the same slot too many times.

    The reprompt counter already exists and already triggers day 4's own
    handoff at three. Two is used here because by the third the customer has
    said the same thing three ways and heard the same question back.
    """
    reprompts = history[turn].reprompts
    fired = reprompts >= limit
    return Signal(
        "repair_loop",
        fired,
        f"{reprompts} reprompts on the same slot" if fired else "",
        turn,
    )


def repeated_question(history: list[Exchange], turn: int) -> Signal:
    """The customer asked something they have already asked.

    The strongest *behavioural* evidence that an answer did not land, and it
    needs no sentiment model: people do not repeat themselves when they got
    what they wanted. Compared on content words so "where can I park" and
    "so where do I park" count as the same question.
    """
    from convo.projects.day08_rag_cites.index import tokenise

    current = set(tokenise(history[turn].user))
    if len(current) < 2:
        return Signal("repeated_question", False, "", turn)
    for earlier in history[:turn]:
        previous = set(tokenise(earlier.user))
        if not previous:
            continue
        overlap = len(current & previous) / max(len(current | previous), 1)
        if overlap >= 0.6:
            return Signal(
                "repeated_question",
                True,
                f"already asked {earlier.user!r}",
                turn,
            )
    return Signal("repeated_question", False, "", turn)


def frustration(history: list[Exchange], turn: int) -> Signal:
    """Lexical frustration markers, and deliberately a weak signal.

    This is the signal most likely to be wrong — plenty of people are blunt
    without being stuck, and plenty of stuck people are polite throughout. It
    is non-binding for that reason, and the ablation measures whether it earns
    its place rather than assuming it does.
    """
    hit = _matches(history[turn].user, FRUSTRATION_PATTERNS)
    return Signal("frustration", bool(hit), f"said {hit!r}" if hit else "", turn)


DETECTORS = (
    asked_for_human,
    high_stakes_unanswered,
    repeated_refusal,
    repair_loop,
    repeated_question,
    frustration,
)

ALL_SIGNALS: tuple[str, ...] = tuple(d.__name__ for d in DETECTORS)


def read(history: list[Exchange], turn: int) -> Reading:
    """Every signal's state as of `turn`, using only what is known by then."""
    return Reading(turn=turn, signals=[detector(history, turn) for detector in DETECTORS])
