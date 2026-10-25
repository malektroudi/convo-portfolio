"""Gold answer spans, labelled once and valid for every chunking.

Day 8 ended with dense@3 at 100% because the labels were document-level and
there are only fourteen documents — a question that lands anywhere in the right
article counts as a hit, which is far too easy to improve on. Day 9 needs
labels that say *where* the answer is.

Labelling chunks directly would be a mistake: there are three chunkings, so
every label would have to be redone for each, and changing a chunker would
invalidate the ground truth it is being judged against. So the label is a quote
from the source document. Chunk relevance is then *derived* — a chunk is
relevant if it covers enough of the gold span — which makes one set of labels
work for every chunking, including ones not written yet.

Quotes are matched whitespace-insensitively because the corpus is hard-wrapped,
so a sentence frequently contains a newline that nobody would type in a label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from convo.projects.day08_rag_cites.chunking import Chunk, Document


@dataclass(frozen=True)
class Gold:
    """One question and the exact words that answer it."""

    question: str
    doc_id: str
    quote: str
    """A quote from the document. Whitespace in it matches any whitespace run."""


@dataclass(frozen=True)
class Span:
    doc_id: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def locate(gold: Gold, document: Document) -> Span:
    """Find the gold quote's exact offsets, ignoring how the source wraps."""
    pattern = r"\s+".join(re.escape(part) for part in gold.quote.split())
    matches = list(re.finditer(pattern, document.text))
    if not matches:
        raise ValueError(f"{gold.doc_id}: quote not found — {gold.quote!r}")
    if len(matches) > 1:
        raise ValueError(
            f"{gold.doc_id}: quote appears {len(matches)} times, so it is ambiguous "
            f"— {gold.quote!r}"
        )
    found = matches[0]
    return Span(gold.doc_id, found.start(), found.end())


def covers(chunk: Chunk, span: Span, threshold: float = 0.5) -> bool:
    """Whether a chunk carries enough of the answer to count as relevant.

    Half the gold span, not all of it: a chunk boundary that clips the last few
    words still leaves a passage a reader can use, and demanding total
    containment would punish every chunker for the same unavoidable edge.
    """
    if chunk.doc_id != span.doc_id:
        return False
    overlap = min(chunk.end, span.end) - max(chunk.start, span.start)
    return overlap > 0 and overlap / span.length >= threshold


def relevant(chunks: list[Chunk], span: Span, threshold: float = 0.5) -> set[str]:
    """The chunk ids that count as correct answers for this span."""
    return {c.chunk_id for c in chunks if covers(c, span, threshold)}


# One label per answerable question in day 8's set, plus the harder paraphrases
# added below. The quote is the smallest passage that actually answers.
GOLD: tuple[Gold, ...] = (
    Gold(
        "how do I book a table",
        "booking-how-to",
        "You can book a table through the website, through the chat assistant, or by calling the restaurant directly during opening hours.",
    ),
    Gold(
        "do you need my card details to hold a booking",
        "booking-how-to",
        "We do not ask for card details to hold a regular table.",
    ),
    Gold(
        "how far ahead can I book",
        "booking-how-to",
        "Bookings open sixty days ahead for regular service and ninety days ahead for private events.",
    ),
    Gold(
        "how long before do I have to cancel",
        "cancellation-standard",
        "A standard table booking can be cancelled free of charge up to four hours before the reservation time.",
    ),
    Gold(
        "I need to cancel a private event, do I lose the deposit",
        "cancellation-private-events",
        "A private event booking must be cancelled at least fourteen days before the event date for a full refund of the deposit.",
    ),
    Gold(
        "what happens if I cancel a private event five days before",
        "cancellation-private-events",
        "Cancelling with less than seven days' notice means the deposit is not refunded.",
    ),
    Gold(
        "how much is the deposit for a private event",
        "deposit-policy-2026",
        "From 1 March 2026 the private event deposit is a flat 200 CAD regardless of party size.",
    ),
    Gold(
        "is the deposit per person or a flat fee",
        "deposit-policy-2026",
        "From 1 March 2026 the private event deposit is a flat 200 CAD regardless of party size.",
    ),
    Gold(
        "do I need a deposit for a table of eight",
        "deposit-policy-2026",
        "Standard tables of twelve or fewer do not require a deposit.",
    ),
    Gold(
        "we are a group of twenty, can you seat us",
        "large-groups",
        "Parties larger than twelve are handled by the events team as a private event.",
    ),
    Gold(
        "do large groups have to choose a set menu",
        "large-groups",
        "Groups over sixteen choose from a set menu rather than ordering individually,",
    ),
    Gold(
        "what time do you open",
        "opening-hours",
        "We serve dinner from 5pm to 10pm, seven days a week.",
    ),
    Gold("do you serve lunch", "opening-hours", "We do not currently serve lunch."),
    Gold(
        "when does the kitchen take its last order",
        "opening-hours",
        "The kitchen takes its last order at 9:30pm,",
    ),
    Gold(
        "are you open on christmas day",
        "holiday-hours",
        "We are closed on Christmas Day, Boxing Day and New Year's Day.",
    ),
    Gold(
        "what are your hours on new year's eve",
        "holiday-hours",
        "On Christmas Eve and New Year's Eve we serve a single seating at 6pm with a set menu.",
    ),
    Gold(
        "when do you go back to normal hours after the holidays",
        "holiday-hours",
        "Regular dinner service resumes on 2 January.",
    ),
    Gold(
        "I have a severe nut allergy, what should I do",
        "allergens",
        "For a severe allergy, please call ahead rather than relying on the chat assistant,",
    ),
    Gold(
        "can you guarantee an allergen free kitchen",
        "allergens",
        "We cannot guarantee a completely allergen-free environment,",
    ),
    Gold(
        "do you have vegan options",
        "menu-vegan",
        "A separate vegan menu runs alongside the main menu and changes with the season.",
    ),
    Gold(
        "can vegan dishes be made gluten free",
        "menu-vegan",
        "Most vegan dishes can be made gluten-free.",
    ),
    Gold(
        "is the dining room wheelchair accessible",
        "accessibility",
        "The dining room is step-free from the street.",
    ),
    Gold(
        "is the back room step free",
        "accessibility",
        "The back room is reached by three steps and is not step-free,",
    ),
    Gold(
        "are assistance dogs allowed",
        "accessibility",
        "Assistance animals are welcome throughout the restaurant, including the bar.",
    ),
    Gold(
        "where can I park",
        "parking",
        "There is paid street parking on Dundas, metered until 9pm on weekdays and until 6pm on Saturdays.",
    ),
    Gold("is street parking free on sundays", "parking", "Sundays are free."),
    Gold(
        "how long do you hold my table if I am late",
        "late-arrival",
        "We hold your table for fifteen minutes past the booking time.",
    ),
    Gold(
        "what if I arrive an hour late",
        "late-arrival",
        "If you arrive more than forty-five minutes after your booking time, the kitchen may not be able to serve the full menu,",
    ),
    Gold("is there a dress code", "dress-code", "There is no dress code."),
    Gold("can we split the bill", "payment", "We can split a bill up to six ways."),
    Gold(
        "is service charge added automatically",
        "payment",
        "A service charge of 18% is added automatically to tables of eight or more.",
    ),
    Gold(
        "do gift cards expire",
        "payment",
        "Gift cards can be used against any bill including the service charge. They do not expire.",
    ),
)

# Harder questions, added because day 8's set was answerable almost entirely by
# keyword overlap. These are phrased the way someone actually asks — no shared
# vocabulary with the passage — so lexical retrieval has nothing to grip.
HARD: tuple[Gold, ...] = (
    Gold(
        "my mum uses a walker, will she manage the back room",
        "accessibility",
        "The back room is reached by three steps and is not step-free,",
    ),
    Gold(
        "we might be twenty minutes behind, will you give the table away",
        "late-arrival",
        "We hold your table for fifteen minutes past the booking time.",
    ),
    Gold("is it smart or casual in the evening", "dress-code", "There is no dress code."),
    Gold(
        "will the kitchen still be going at half nine",
        "opening-hours",
        "The kitchen takes its last order at 9:30pm,",
    ),
    Gold(
        "do I pay anything up front for sixteen people",
        "deposit-policy-2026",
        "A deposit applies to private events and to any booking for more than twelve guests.",
    ),
    Gold(
        "my friend is coeliac, is there anything plant based for her",
        "menu-vegan",
        "Most vegan dishes can be made gluten-free.",
    ),
    Gold(
        "changed my mind the morning of, am I charged",
        "cancellation-standard",
        "A standard table booking can be cancelled free of charge up to four hours before the reservation time.",
    ),
    Gold(
        "can we settle up separately, there are four of us",
        "payment",
        "We can split a bill up to six ways.",
    ),
    Gold(
        "is the 25th a normal evening for you",
        "holiday-hours",
        "We are closed on Christmas Day, Boxing Day and New Year's Day.",
    ),
    Gold("anywhere to leave the car nearby on a sunday", "parking", "Sundays are free."),
)

ALL: tuple[Gold, ...] = GOLD + HARD
