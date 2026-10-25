"""Sixteen conversations, labelled with when a person should have taken over.

The label is a **turn index, not a flag**. Labelling escalation as yes/no
would let a policy that hands over on the last exchange of a four-turn
disaster score as correct, and that policy is worthless — the customer already
spent the whole conversation failing. The index is the first exchange at which
a reasonable supervisor, reading over the bot's shoulder, would have stepped
in.

`escalate_at = None` means the bot should finish the conversation itself.
Seven of the sixteen are those, and they are the point: a set where everything
escalates measures nothing, because "always escalate" would score perfectly
and the expensive mistake — pulling a person in to answer a question the bot
had in hand — would be invisible.

Each conversation also carries `needs`: the facts a human must be handed to
pick it up without starting again. That list is written from the customer's
side — what they would be furious to repeat — rather than from what the system
happens to find easy to serialise.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Exchange:
    """One user turn and what the system did with it.

    The machine-state fields are filled by the build from the real pipeline —
    `answered` from day 10's citation gate, `reprompts` from day 4's engine —
    rather than being asserted here. A labelled set that also hand-writes the
    system's behaviour is testing the labels against themselves.
    """

    user: str
    reply: str = ""
    answered: bool = True
    reprompts: int = 0
    doc_id: str = ""
    evidence_score: float = 0.0
    slots: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "user": self.user,
            "reply": self.reply,
            "answered": self.answered,
            "reprompts": self.reprompts,
            "doc_id": self.doc_id,
            "evidence_score": round(self.evidence_score, 3),
            "slots": dict(self.slots),
        }


@dataclass(frozen=True)
class Need:
    """One fact the receiving agent must be given, and how to check for it."""

    key: str
    must_contain: str
    why: str = ""


@dataclass
class Conversation:
    name: str
    kind: str
    """`faq` runs through day 10's guarded pipeline; `booking` through day 4."""
    exchanges: list[Exchange]
    escalate_at: int | None
    note: str = ""
    needs: tuple[Need, ...] = ()

    @property
    def user_turns(self) -> list[str]:
        return [e.user for e in self.exchanges]


def _faq(name, turns, escalate_at, note, needs=()):
    return Conversation(
        name=name,
        kind="faq",
        exchanges=[Exchange(user=t) for t in turns],
        escalate_at=escalate_at,
        note=note,
        needs=needs,
    )


def _booking(name, turns, escalate_at, note, needs=()):
    return Conversation(
        name=name,
        kind="booking",
        exchanges=[Exchange(user=t) for t in turns],
        escalate_at=escalate_at,
        note=note,
        needs=needs,
    )


CONVERSATIONS: tuple[Conversation, ...] = (
    # ---------------------------------------------------------------- resolve
    _faq(
        "answers-cleanly",
        ["where can I park", "is it free on sundays"],
        None,
        "Two questions the corpus answers. Escalating this wastes a person.",
    ),
    _faq(
        "allergen-answered",
        ["I have a nut allergy, what should I do"],
        None,
        "High stakes, but day 10 answers it with a verified citation. The "
        "guardrail working is not a reason to escalate.",
    ),
    _faq(
        "one-refusal-then-moves-on",
        ["do you have EV charging", "what time do you open"],
        None,
        "One refusal is the guardrail doing its job; the customer moves on.",
    ),
    _booking(
        "books-without-trouble",
        ["I'd like a table for four", "friday at 8pm", "yes"],
        None,
        "The happy path. Day 4 fills every slot first time.",
    ),
    _faq(
        "digression-then-answer",
        ["is the dining room wheelchair accessible", "and is there parking"],
        None,
        "A topic change the bot handles. Not a failure.",
    ),
    _booking(
        "corrects-itself",
        ["table for six", "actually make it four", "saturday 7pm", "yes"],
        None,
        "A repair that succeeds. Repair is normal, not a crisis.",
    ),
    _faq(
        "blunt-but-fine",
        ["dress code", "ok and what time is last orders"],
        None,
        "Terse, not frustrated. The frustration signal must not fire here.",
    ),
    # ---------------------------------------------------------------- escalate
    _faq(
        "asks-for-a-person",
        ["what time do you open", "can I speak to a person"],
        1,
        "The one signal with no threshold. Honoured on the exchange it is said.",
        needs=(
            Need("transcript", "speak to a person", "what they asked for"),
            Need("reason", "asked", "why they were handed over"),
        ),
    ),
    _faq(
        "allergen-refused",
        ["does the set menu contain sesame for a severe allergy"],
        0,
        "High stakes the corpus cannot answer. Telling someone with a severe "
        "allergy 'I don't know' is the worst available outcome.",
        needs=(
            Need("topic", "allerg", "what it is about"),
            Need("reason", "allerg", "why a person is needed"),
            Need("tried", "could not", "that the bot already failed"),
        ),
    ),
    _faq(
        "keeps-refusing",
        [
            "do you have EV charging",
            "is there a charging point in the garage",
            "anywhere nearby I could charge",
        ],
        1,
        "Three ways of asking something the corpus does not contain. The "
        "second refusal is the point of no return.",
        needs=(
            Need("topic", "charg", "what they actually want"),
            Need("tried", "could not", "that rephrasing will not help"),
        ),
    ),
    _booking(
        "stuck-on-a-slot",
        [
            "I want to book",
            "for the day after the bank holiday",
            "the one in may",
            "the monday one",
        ],
        2,
        "A date the extractor cannot parse, asked three ways. Day 4's own "
        "handoff fires at three reprompts; by then it is already too late.",
        needs=(
            Need("tried", "reprompt", "which slot failed"),
            Need("transcript", "bank holiday", "what they actually said"),
        ),
    ),
    _faq(
        "billing-complaint",
        ["I was charged twice for my deposit and want a refund"],
        0,
        "Money, a complaint, and nothing in the corpus. Immediate.",
        needs=(
            Need("topic", "charg", "what went wrong"),
            Need("reason", "charg", "why a person is needed"),
        ),
    ),
    _faq(
        "repeats-the-question",
        [
            "how much is the deposit for a private event",
            "no I mean how much is the deposit for a private event",
            "this is useless",
        ],
        1,
        "The customer repeats themselves, which is the clearest behavioural "
        "evidence an answer did not land. Frustration confirms it a turn later.",
        needs=(
            Need("topic", "deposit", "what they asked"),
            Need("transcript", "useless", "their own words"),
        ),
    ),
    _booking(
        "asks-for-human-mid-booking",
        ["table for two", "actually just put me through to someone"],
        1,
        "Partly-filled form, then a request for a person. The slots collected "
        "so far are exactly what must not be lost.",
        needs=(
            Need("slots", "2", "the party size already given"),
            Need("reason", "asked", "why they were handed over"),
        ),
    ),
    _faq(
        "food-safety",
        ["my friend was ill after eating here last night"],
        0,
        "Nothing in a help centre should answer this.",
        needs=(
            Need("topic", "ill", "what happened"),
            Need("reason", "ill", "why it is urgent"),
        ),
    ),
    _booking(
        "abandons-then-escalates",
        ["book me a table", "never mind", "actually I need to speak to a manager"],
        2,
        "An abandoned form then an explicit request. The handoff must not "
        "carry the dead booking as though it were live.",
        needs=(Need("reason", "asked", "why they were handed over"),),
    ),
)

NEEDING: tuple[Conversation, ...] = tuple(c for c in CONVERSATIONS if c.escalate_at is not None)
RESOLVING: tuple[Conversation, ...] = tuple(c for c in CONVERSATIONS if c.escalate_at is None)
