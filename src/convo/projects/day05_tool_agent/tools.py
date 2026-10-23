"""The booking tools, enforcing the SAME business rules as day 4.

This matters for the comparison. If the agent's tools were more permissive
than the state machine's validators, day 5 would look better by being allowed
to do things day 4 refuses. Both sides are held to one set of rules: at most
twelve covers, inside service hours, within the booking horizon.

The tool layer is also where the two approaches genuinely differ. Day 4
validates before it ever stores a value; day 5 lets the model propose a whole
call and validates at the boundary. The second shape is more flexible and
fails in more places, which is exactly what day 5 is measuring.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

from convo.projects.day04_dialog_engine.extract import Reference
from convo.projects.day04_dialog_engine.flow import MAX_PARTY, fmt_date, fmt_time

POLICY_TOPICS = {
    "cancellation": "You can cancel free of charge up to 4 hours before your booking.",
    "hours": "We serve dinner from 5pm to 10pm, seven days a week.",
    "parking": "There's paid street parking on Dundas, and a garage one block east.",
    "dietary": "We have vegan, vegetarian and gluten-free menus, and the kitchen "
    "handles allergies if you mention them on arrival.",
    "accessibility": "The dining room is step-free and there's an accessible washroom.",
    "dress_code": "No dress code — come as you are.",
}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[dict[str, Any], Reference], dict[str, Any]]
    grounded: bool = False
    """Whether every argument must be traceable to something the guest said."""

    def spec(self) -> dict[str, Any]:
        """The shape every tool-calling chat template expects."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


DATE_PATTERN = r"\d{4}-\d{2}-\d{2}"
TIME_PATTERN = r"\d{1,2}:\d{2}"

BOOK_PARAMETERS = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "party_size": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_PARTY,
            "description": "number of guests",
        },
        "date": {
            "type": "string",
            "pattern": DATE_PATTERN,
            "patternHint": "YYYY-MM-DD",
            "description": "the booking date",
        },
        "time": {
            "type": "string",
            "pattern": TIME_PATTERN,
            "patternHint": "HH:MM on a 24-hour clock",
            "description": "the booking time",
        },
        "name": {"type": "string", "minLength": 2, "description": "name for the booking"},
    },
    "required": ["party_size", "date", "time", "name"],
}

POLICY_PARAMETERS = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "topic": {"type": "string", "enum": sorted(POLICY_TOPICS)},
    },
    "required": ["topic"],
}


def _parse_date(value: str) -> date | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _parse_time(value: str) -> time | None:
    for fmt in ("%H:%M", "%I:%M%p", "%I:%M %p"):
        try:
            return datetime.strptime(str(value).strip().upper(), fmt).time()
        except ValueError:
            continue
    return None


def book_table(args: dict[str, Any], ref: Reference) -> dict[str, Any]:
    """Apply the same rules day 4's validators apply, and say why on refusal."""
    when = _parse_date(str(args.get("date", "")))
    at = _parse_time(args.get("time", ""))

    if when is None:
        return {"ok": False, "error": "date could not be read; use YYYY-MM-DD."}
    if at is None:
        return {"ok": False, "error": "time could not be read; use HH:MM, 24-hour."}
    if when < ref.today:
        return {"ok": False, "error": f"{fmt_date(when)} has already passed."}
    if when > ref.today + timedelta(days=ref.horizon_days):
        return {
            "ok": False,
            "error": f"we only take bookings {ref.horizon_days} days ahead.",
        }
    if not (ref.opens <= at <= ref.closes):
        return {
            "ok": False,
            "error": f"we serve between {fmt_time(ref.opens)} and {fmt_time(ref.closes)}.",
        }

    return {
        "ok": True,
        "confirmation": "BK-" + when.strftime("%m%d") + at.strftime("%H%M"),
        "party_size": int(args["party_size"]),
        "date": when.isoformat(),
        "time": at.strftime("%H:%M"),
        "name": str(args["name"]),
        "summary": (
            f"{args['party_size']} "
            f"{'person' if int(args['party_size']) == 1 else 'people'} on "
            f"{fmt_date(when)} at {fmt_time(at)} under {args['name']}"
        ),
    }


def get_policy(args: dict[str, Any], ref: Reference) -> dict[str, Any]:
    topic = str(args.get("topic", ""))
    if topic not in POLICY_TOPICS:
        return {"ok": False, "error": f"unknown topic {topic!r}."}
    return {"ok": True, "topic": topic, "answer": POLICY_TOPICS[topic]}


BOOK_TABLE = Tool(
    name="book_table",
    description=(
        "Book a restaurant table. Call this only once every detail is known: "
        "number of guests, date, time and the name for the booking."
    ),
    parameters=BOOK_PARAMETERS,
    run=book_table,
    grounded=True,
)

GET_POLICY = Tool(
    name="get_policy",
    description=(
        "Answer a question about the restaurant that is not part of the booking "
        "itself, such as parking, opening hours or the cancellation policy."
    ),
    parameters=POLICY_PARAMETERS,
    run=get_policy,
)

TOOLS: tuple[Tool, ...] = (BOOK_TABLE, GET_POLICY)
BY_NAME: dict[str, Tool] = {t.name: t for t in TOOLS}


def specs(tools: tuple[Tool, ...] = TOOLS) -> list[dict[str, Any]]:
    return [t.spec() for t in tools]


SYSTEM_PROMPT = """You are the booking assistant for a restaurant. Today is {today}.

Collect the number of guests, the date, the time and a name. Ask for whatever is
missing, one or two items at a time, in plain friendly sentences.

Rules:
- Call book_table only when you have all four details. Dates are YYYY-MM-DD and
  times are HH:MM on a 24-hour clock, so 7:30pm is 19:30.
- The restaurant seats at most {max_party} people per table and serves between
  {opens} and {closes}.
- If the guest asks something that is not about their booking, call get_policy.
- If the guest corrects a detail, use the corrected value.
Never invent a detail the guest has not given you."""


def system_prompt(ref: Reference) -> str:
    return SYSTEM_PROMPT.format(
        today=ref.today.isoformat(),
        max_party=MAX_PARTY,
        opens=fmt_time(ref.opens),
        closes=fmt_time(ref.closes),
    )
