"""Twelve conversations, and the standalone question each follow-up means.

Every turn carries three things: what the user typed, a gold quote from the
corpus that answers it, and an **oracle** — the question a person would write
if they had to make that turn stand on its own.

The oracle is the point. A rewriter can always be shown to beat doing nothing,
because doing nothing on "is it free on Sundays" retrieves noise. The harder
and more useful question is how much of the *available* improvement it
captures, and that needs a ceiling. The oracle is that ceiling: it is what a
perfect rewriter would produce, written by hand before any rule existed, so
the rules were not tuned to hit it.

Two things the oracle is not. It is not a prediction of what an LLM rewriter
would produce — a model would make its own mistakes, and claiming otherwise
would turn a measured ceiling into a guessed one. And it is not necessarily
the best possible *query*: a hand-written standalone question is sometimes
worse for BM25 than a clumsy machine one that happens to repeat a keyword.
Where that happens the build reports it rather than hiding it.

Turn 1 of each conversation is standalone by construction. Those turns are
scored separately, because averaging them in would dilute the whole effect
with questions that never needed rewriting.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    """One user turn in a conversation, with its answer and its ceiling."""

    text: str
    """What the user typed."""
    doc_id: str
    quote: str
    """The passage that answers it. Matched whitespace-insensitively."""
    oracle: str = ""
    """The standalone form. Empty means the turn already stands alone."""

    @property
    def standalone(self) -> bool:
        return not self.oracle

    @property
    def resolved(self) -> str:
        return self.oracle or self.text


@dataclass(frozen=True)
class Conversation:
    name: str
    turns: tuple[Turn, ...]
    note: str = ""
    """What this conversation is testing, shown in the tab."""


CONVERSATIONS: tuple[Conversation, ...] = (
    Conversation(
        "cancelling",
        (
            Turn(
                "how long before do I have to cancel",
                "cancellation-standard",
                "A standard table booking can be cancelled free of charge up to four hours before the reservation time.",
            ),
            Turn(
                "what about for a private event",
                "cancellation-private-events",
                "A private event booking must be cancelled at least fourteen days before the event date for a full refund of the deposit.",
                oracle="how long before do I have to cancel a private event",
            ),
            Turn(
                "and if it is five days before",
                "cancellation-private-events",
                "Cancelling with less than seven days' notice means the deposit is not refunded.",
                oracle="what happens if I cancel a private event five days before",
            ),
        ),
        note="The follow-up moves to a different article with a near-identical wording.",
    ),
    Conversation(
        "parking",
        (
            Turn(
                "where can I park",
                "parking",
                "There is paid street parking on Dundas, metered until 9pm on weekdays and until 6pm on Saturdays.",
            ),
            Turn(
                "is it free on sundays",
                "parking",
                "Sundays are free.",
                oracle="is street parking free on sundays",
            ),
            Turn(
                "how far is the subway",
                "parking",
                "the nearest subway station is about twelve minutes on foot.",
                oracle="how far is the subway station from the restaurant",
            ),
        ),
        note="A bare pronoun, then a turn that is already almost standalone.",
    ),
    Conversation(
        "accessibility",
        (
            Turn(
                "is the dining room wheelchair accessible",
                "accessibility",
                "The dining room is step-free from the street.",
            ),
            Turn(
                "what about the back room",
                "accessibility",
                "The back room is reached by three steps and is not step-free,",
                oracle="is the back room step free",
            ),
            Turn(
                "is there a washroom I can use",
                "accessibility",
                "There is an accessible washroom on the same floor as the dining room, with grab rails and a 150cm turning circle.",
                oracle="is there an accessible washroom",
            ),
        ),
        note="A contrast within one article: the answer is the opposite of turn 1's.",
    ),
    Conversation(
        "vegan",
        (
            Turn(
                "do you have vegan options",
                "menu-vegan",
                "A separate vegan menu runs alongside the main menu and changes with the season.",
            ),
            Turn(
                "can they be made gluten free",
                "menu-vegan",
                "Most vegan dishes can be made gluten-free.",
                oracle="can vegan dishes be made gluten free",
            ),
            Turn(
                "do I have to tell you in advance",
                "menu-vegan",
                "You do not need to tell us in advance to eat vegan, but it helps for groups of eight or more so the kitchen can prepare enough.",
                oracle="do I have to tell you in advance that I eat vegan",
            ),
        ),
        note="A plural pronoun whose antecedent is never named in the user's words.",
    ),
    Conversation(
        "hours",
        (
            Turn(
                "what time do you open",
                "opening-hours",
                "We serve dinner from 5pm to 10pm, seven days a week.",
            ),
            Turn(
                "is that the same on christmas day",
                "holiday-hours",
                "We are closed on Christmas Day, Boxing Day and New Year's Day.",
                oracle="are you open on christmas day",
            ),
            Turn(
                "when do you go back to normal",
                "holiday-hours",
                "Regular dinner service resumes on 2 January.",
                oracle="when do you return to normal hours after the holidays",
            ),
        ),
        note="The topic crosses articles mid-conversation and never comes back.",
    ),
    Conversation(
        "deposit",
        (
            Turn(
                "do I need a deposit for a table of eight",
                "deposit-policy-2026",
                "Standard tables of twelve or fewer do not require a deposit.",
            ),
            Turn(
                "what if there are twenty of us",
                "deposit-policy-2026",
                "A deposit applies to private events and to any booking for more than twelve guests.",
                oracle="do I need a deposit for a booking of twenty guests",
            ),
            Turn(
                "how much is it",
                "deposit-policy-2026",
                "From 1 March 2026 the private event deposit is a flat 200 CAD regardless of party size.",
                oracle="how much is the private event deposit",
            ),
        ),
        note="'it' refers two turns back, not to the turn before it.",
    ),
    Conversation(
        "late",
        (
            Turn(
                "how long do you hold my table if I am late",
                "late-arrival",
                "We hold your table for fifteen minutes past the booking time.",
            ),
            Turn(
                "what if I call ahead",
                "late-arrival",
                "If we know you are coming we will usually hold the table longer,",
                oracle="what if I call ahead to say I am running late",
            ),
            Turn(
                "and if I am an hour late",
                "late-arrival",
                "If you arrive more than forty-five minutes after your booking time, the kitchen may not be able to serve the full menu,",
                oracle="what happens if I arrive an hour late",
            ),
        ),
        note="Three sections of one article, each needing the turn before it.",
    ),
    Conversation(
        "large-groups",
        (
            Turn(
                "we are a group of twenty, can you seat us",
                "large-groups",
                "Parties larger than twelve are handled by the events team as a private event.",
            ),
            Turn(
                "do we have to choose a set menu",
                "large-groups",
                "Groups over sixteen choose from a set menu rather than ordering individually,",
                oracle="do large groups have to choose a set menu",
            ),
        ),
        note="A follow-up that is grammatical on its own but underspecified.",
    ),
    Conversation(
        "booking",
        (
            Turn(
                "how do I book a table",
                "booking-how-to",
                "You can book a table through the website, through the chat assistant, or by calling the restaurant directly during opening hours.",
            ),
            Turn(
                "do you need my card for that",
                "booking-how-to",
                "We do not ask for card details to hold a regular table.",
                oracle="do you need my card details to hold a booking",
            ),
            Turn(
                "how far ahead can I do it",
                "booking-how-to",
                "Bookings open sixty days ahead for regular service and ninety days ahead for private events.",
                oracle="how far ahead can I book a table",
            ),
        ),
        note="Two different pronouns pointing at the same first turn.",
    ),
    Conversation(
        "allergens",
        (
            Turn(
                "I have a severe nut allergy, what should I do",
                "allergens",
                "For a severe allergy, please call ahead rather than relying on the chat assistant,",
            ),
            Turn(
                "can you guarantee there is none in the kitchen",
                "allergens",
                "We cannot guarantee a completely allergen-free environment,",
                oracle="can you guarantee an allergen free kitchen",
            ),
        ),
        note="A negative pronoun — 'none' — which no rule here handles.",
    ),
    Conversation(
        "payment",
        (
            Turn(
                "can we split the bill",
                "payment",
                "We can split a bill up to six ways.",
            ),
            Turn(
                "is there a charge for a table of eight",
                "payment",
                "A service charge of 18% is added automatically to tables of eight or more.",
                oracle="is there a service charge for a table of eight",
            ),
        ),
        note="A follow-up that is standalone but ambiguous with the deposit article.",
    ),
    Conversation(
        "dress-code",
        (
            Turn(
                "is there a dress code",
                "dress-code",
                "There is no dress code.",
            ),
            Turn(
                "what about at a private event",
                "dress-code",
                "Hosts of private events sometimes set their own dress code for their guests.",
                oracle="is there a dress code at a private event",
            ),
        ),
        note="The same fragment as the cancelling conversation, answered by a different article.",
    ),
)


ALL_TURNS: tuple[tuple[str, int, Turn], ...] = tuple(
    (conversation.name, index, turn)
    for conversation in CONVERSATIONS
    for index, turn in enumerate(conversation.turns)
)

FOLLOW_UPS: tuple[tuple[str, int, Turn], ...] = tuple(
    (name, index, turn) for name, index, turn in ALL_TURNS if index > 0
)
