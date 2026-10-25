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

BINDING = frozenset({"asked_for_human", "high_stakes"})
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


def _matches(text: str, patterns) -> str:
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


def high_stakes(history: list[Exchange], turn: int) -> Signal:
    """A topic where being confidently wrong hurts somebody. Topic alone.

    This signal was built the obvious way first — fire only when the system
    *also* failed to answer, so that a question day 10 handles with a verified
    citation would not waste a person. The measurement killed it. On this set
    the two high-stakes conversations that need a human score **higher** than
    the one that does not:

        -3.04  "I was charged twice for my deposit"      needs a person
        -3.70  "does the set menu contain sesame..."     needs a person
        -5.97  "I have a nut allergy, what should I do"  bot can handle it

    Confidence is not merely a weak signal here, it is pointing the wrong way.
    The reason is structural rather than bad luck: day 10's gates verify that
    an answer is *grounded in the corpus*, not that it is *responsive to the
    question*. "Charged twice, want a refund" retrieves the cancellation
    policy and cites it perfectly — a well-grounded answer to a question
    nobody asked. Groundedness and answerability come apart exactly where the
    cost of being wrong is highest.

    So this consults no confidence at all. On a listed topic, a person. The
    price is visible in the precision column rather than argued away.
    """
    hit = _matches(history[turn].user, HIGH_STAKES_PATTERNS)
    return Signal("high_stakes", bool(hit), f"said {hit!r}" if hit else "", turn)


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

    text = history[turn].user
    current = set(tokenise(text))
    # Guard on the utterance, not the content set. An earlier version required
    # two content words and so could never fire on "where can I park", which
    # stems down to {park} — a perfectly real question that people do repeat.
    # What the guard is actually for is "yes" and "ok", and those are short
    # *utterances*, which is the thing to measure.
    if len(text.split()) < 3 or not current:
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
    high_stakes,
    repeated_refusal,
    repair_loop,
    repeated_question,
    frustration,
)

ALL_SIGNALS: tuple[str, ...] = tuple(d.__name__ for d in DETECTORS)


def read(history: list[Exchange], turn: int) -> Reading:
    """Every signal's state as of `turn`, using only what is known by then.

    Signals are **sticky**: one that fired on an earlier exchange is still
    evidence now. Evaluating each turn in isolation would mean a customer who
    repeated themselves at turn two and swore at turn three presents one
    signal at each, never two at once, and a policy requiring agreement would
    watch the conversation fail one signal at a time.

    Stickiness only ever adds evidence, so it can make a policy escalate
    *earlier*, never later. Nothing here forgets, which is defensible over
    conversations this short and would not be over a long one — a signal from
    forty turns ago is not evidence about now.
    """
    readings = []
    for detector in DETECTORS:
        latest = detector(history, turn)
        if not latest.fired:
            for earlier in range(turn):
                previous = detector(history, earlier)
                if previous.fired:
                    latest = previous
                    break
        readings.append(latest)
    return Reading(turn=turn, signals=readings)
