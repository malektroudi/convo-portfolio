"""The question set, with the article each one should land on.

These are embedded into the artifact so the hub can run dense retrieval with no
encoder, and they become day 9's labelled evaluation set. Several exist purely
to probe the traps the corpus was built with:

  near-duplicate   a standard cancellation and a private-event cancellation,
                   worded similarly with different numbers
  superseded       the 2026 deposit policy versus the old per-head figure
  split            opening hours versus holiday hours
  unanswerable     nothing in the corpus mentions EV charging
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    text: str
    expect_doc: str | None
    """The article that should be retrieved. None means unanswerable."""
    probe: str = ""
    """Which designed failure mode this question is aimed at, if any."""


QUESTIONS: tuple[Question, ...] = (
    Question("how do I book a table", "booking-how-to"),
    Question("do you need my card details to hold a booking", "booking-how-to"),
    Question("how far ahead can I book", "booking-how-to"),
    Question("how long before do I have to cancel", "cancellation-standard", "near-duplicate"),
    Question(
        "I need to cancel a private event, do I lose the deposit",
        "cancellation-private-events",
        "near-duplicate",
    ),
    Question(
        "what happens if I cancel a private event five days before",
        "cancellation-private-events",
        "near-duplicate",
    ),
    Question("how much is the deposit for a private event", "deposit-policy-2026", "superseded"),
    Question("is the deposit per person or a flat fee", "deposit-policy-2026", "superseded"),
    Question("do I need a deposit for a table of eight", "deposit-policy-2026"),
    Question("we are a group of twenty, can you seat us", "large-groups"),
    Question("do large groups have to choose a set menu", "large-groups"),
    Question("what time do you open", "opening-hours", "split"),
    Question("do you serve lunch", "opening-hours"),
    Question("when does the kitchen take its last order", "opening-hours"),
    Question("are you open on christmas day", "holiday-hours", "split"),
    Question("what are your hours on new year's eve", "holiday-hours", "split"),
    Question("when do you go back to normal hours after the holidays", "holiday-hours"),
    Question("I have a severe nut allergy, what should I do", "allergens"),
    Question("can you guarantee an allergen free kitchen", "allergens"),
    Question("do you have vegan options", "menu-vegan"),
    Question("can vegan dishes be made gluten free", "menu-vegan"),
    Question("is the dining room wheelchair accessible", "accessibility"),
    Question("is the back room step free", "accessibility"),
    Question("are assistance dogs allowed", "accessibility"),
    Question("where can I park", "parking"),
    Question("is street parking free on sundays", "parking"),
    Question("how long do you hold my table if I am late", "late-arrival"),
    Question("what if I arrive an hour late", "late-arrival"),
    Question("is there a dress code", "dress-code"),
    Question("can we split the bill", "payment"),
    Question("is service charge added automatically", "payment"),
    Question("do gift cards expire", "payment"),
    Question("do you have EV charging", None, "unanswerable"),
    Question("can I bring my own wine", None, "unanswerable"),
)

ANSWERABLE = tuple(q for q in QUESTIONS if q.expect_doc is not None)
UNANSWERABLE = tuple(q for q in QUESTIONS if q.expect_doc is None)
