"""The flow as data: pages, slots, validators, and the off-task answers.

This is deliberately the same shape Conversational Agents (formerly Dialogflow
CX) gives you — a page holds a form of required parameters, a parameter has a
prompt and a validator, and routes move you between pages — but written out so
the mechanism is visible instead of drawn in a console. Day 19 rebuilds exactly
this flow in the managed product and compares.

Nothing here imports a model. The whole point of day 4 is that a task-oriented
bot is a state machine, and that the hard parts are repair and digression
rather than language understanding.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, time, timedelta
from typing import Any

from convo.projects.day04_dialog_engine.extract import Reference

# A validator returns an error message to say back, or None if the value is fine.
Validator = Callable[[Any, Reference], "str | None"]

PARTY_SIZE = "party_size"
BOOKING_DATE = "booking_date"
BOOKING_TIME = "booking_time"
NAME = "name"

MAX_PARTY = 12


def validate_party_size(value: int, ref: Reference) -> str | None:
    if value < 1:
        return "I need at least one guest — how many people should I book for?"
    if value > MAX_PARTY:
        return (
            f"We can only seat up to {MAX_PARTY} at one table. For a larger group "
            f"I'd need to pass you to the events team. How many people is it?"
        )
    return None


def validate_date(value: date, ref: Reference) -> str | None:
    if value < ref.today:
        # No %-d here: that directive does not exist on Windows.
        return f"{fmt_date(value)} has already passed. Which day did you mean?"
    if value > ref.today + timedelta(days=ref.horizon_days):
        return (
            f"We only take bookings {ref.horizon_days} days ahead, so that is too "
            "far out. Could you pick a nearer date?"
        )
    return None


def validate_time(value: time, ref: Reference) -> str | None:
    if not (ref.opens <= value <= ref.closes):
        return (
            f"We serve between {fmt_time(ref.opens)} and {fmt_time(ref.closes)}. "
            "What time works inside that?"
        )
    return None


def validate_name(value: str, ref: Reference) -> str | None:
    if len(value.replace(" ", "")) < 2:
        return "I didn't catch that name — could you spell it out?"
    return None


def fmt_time(value: time) -> str:
    """7:30pm, or 7pm when it lands on the hour."""
    hour = value.hour % 12 or 12
    suffix = "am" if value.hour < 12 else "pm"
    return f"{hour}:{value.minute:02d}{suffix}" if value.minute else f"{hour}{suffix}"


def fmt_date(value: date) -> str:
    return value.strftime("%A %d %B").replace(" 0", " ")


@dataclass(frozen=True)
class SlotSpec:
    """One parameter a page needs before it can move on."""

    name: str
    prompt: str
    source: str
    """The `Extracted` field this slot is filled from."""
    validator: Validator
    display: Callable[[Any], str]
    label: str
    aliases: tuple[str, ...] = ()
    """Words a user might use to name this slot when amending."""
    phrase: str = ""
    """Preposition that joins this value into a spoken summary ("at 8pm")."""


@dataclass(frozen=True)
class Page:
    """A state. Its form must be complete before the flow leaves it."""

    name: str
    slots: tuple[str, ...] = ()
    terminal: bool = False
    description: str = ""


BOOKING_SLOTS: dict[str, SlotSpec] = {
    PARTY_SIZE: SlotSpec(
        name=PARTY_SIZE,
        label="Party size",
        prompt="How many people will that be for?",
        source="party_size",
        validator=validate_party_size,
        display=lambda v: f"{v} {'person' if v == 1 else 'people'}",
        aliases=("party", "size", "people", "guests", "cover", "covers", "number"),
        phrase="",
    ),
    BOOKING_DATE: SlotSpec(
        name=BOOKING_DATE,
        label="Date",
        prompt="What day would you like?",
        source="booking_date",
        validator=validate_date,
        display=fmt_date,
        aliases=("date", "day", "night"),
        phrase="on",
    ),
    BOOKING_TIME: SlotSpec(
        name=BOOKING_TIME,
        label="Time",
        prompt="What time suits you?",
        source="booking_time",
        validator=validate_time,
        display=fmt_time,
        aliases=("time", "hour", "slot"),
        phrase="at",
    ),
    NAME: SlotSpec(
        name=NAME,
        label="Name",
        prompt="And what name should I put it under?",
        source="name",
        validator=validate_name,
        display=lambda v: str(v),
        aliases=("name", "booking name", "reservation name"),
        phrase="under",
    ),
}

COLLECT = "collect"
CONFIRM = "confirm"
AMEND = "amend"
BOOKED = "booked"
HANDOFF = "handoff"

BOOKING_PAGES: dict[str, Page] = {
    COLLECT: Page(
        COLLECT,
        slots=(PARTY_SIZE, BOOKING_DATE, BOOKING_TIME, NAME),
        description="Fill the form, one missing parameter at a time.",
    ),
    CONFIRM: Page(
        CONFIRM,
        description="Read the booking back and wait for a yes, a no, or a correction.",
    ),
    AMEND: Page(
        AMEND,
        description="The user said no but did not say what to change. Ask which field.",
    ),
    BOOKED: Page(BOOKED, terminal=True, description="Done."),
    HANDOFF: Page(
        HANDOFF,
        terminal=True,
        description="Too many failed prompts in a row. Escalate to a human.",
    ),
}


@dataclass(frozen=True)
class Digression:
    """An off-task question answerable from any page without losing the form.

    This is the case that separates a real engine from a demo. The user asks
    about parking halfway through giving you a date; you answer, and then you
    pick the form back up at exactly the parameter you were waiting on.
    """

    name: str
    patterns: tuple[str, ...]
    answer: str


DIGRESSIONS: tuple[Digression, ...] = (
    Digression(
        # Deliberately not the bare word "cancel": mid-form that almost always
        # means "abandon this booking", and hijacking it with an FAQ answer is
        # one of the most common ways a real bot infuriates people.
        "cancellation_policy",
        ("cancellation", "can i cancel", "refund", "policy", "change it later"),
        "You can cancel free of charge up to 4 hours before your booking.",
    ),
    Digression(
        "opening_hours",
        ("open", "opening", "hours", "close", "closing", "what time do you"),
        "We serve dinner from 5pm to 10pm, seven days a week.",
    ),
    Digression(
        "parking",
        ("park", "parking", "car", "drive"),
        "There's paid street parking on Dundas, and a garage one block east.",
    ),
    Digression(
        "dietary",
        ("vegan", "vegetarian", "gluten", "allergy", "allergies", "halal", "kosher"),
        "Yes — we have vegan, vegetarian and gluten-free menus, and the kitchen "
        "handles allergies if you mention them on arrival.",
    ),
    Digression(
        "accessibility",
        ("wheelchair", "accessible", "step free", "stepfree", "lift", "elevator"),
        "The dining room is step-free and there's an accessible washroom.",
    ),
    Digression(
        "dress_code",
        ("dress code", "dress", "smart", "formal", "wear"),
        "No dress code — come as you are.",
    ),
)

# Not the bare word "person": "a table for one person" is a party size, not a
# request to escalate, and that collision would hand off real bookings.
HUMAN_PATTERNS = (
    "human",
    "real person",
    "speak to a person",
    "talk to a person",
    "agent",
    "someone else",
    "speak to someone",
    "talk to someone",
    "manager",
    "representative",
)
RESTART_PATTERNS = ("start over", "start again", "restart", "from scratch")
ABANDON_PATTERNS = (
    "never mind",
    "nevermind",
    "forget it",
    "stop",
    "quit",
    "no thanks",
    "cancel",  # mid-form this means "drop the booking", not the refund FAQ
)


@dataclass(frozen=True)
class Flow:
    """A named set of pages plus the slots they draw on."""

    name: str
    pages: dict[str, Page]
    slots: dict[str, SlotSpec]
    start: str
    digressions: tuple[Digression, ...] = field(default=DIGRESSIONS)
    greeting: str = ""
    max_reprompts: int = 3

    def page(self, name: str) -> Page:
        return self.pages[name]

    def spec(self, name: str) -> SlotSpec:
        return self.slots[name]


BOOKING_FLOW = Flow(
    name="restaurant_booking",
    pages=BOOKING_PAGES,
    slots=BOOKING_SLOTS,
    start=COLLECT,
    greeting="Hi — I can book you a table. How many people, and when?",
)
