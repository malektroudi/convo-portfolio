"""Deterministic entity extraction. No model, no LLM, no network.

Every parser takes text and a `Reference` (the "now" the conversation happens
in) and returns a value or None. Passing the clock in rather than reading it
means a conversation is reproducible: the same transcript always yields the
same slots, which is what makes the day-6 eval suite possible at all.

The parsers deliberately refuse ambiguity rather than guessing. "3/4" is the
third of April to most of the world and the fourth of March in the US, and a
booking bot that silently picks one will book the wrong table for someone. An
unparsed date becomes a clarifying question, which is the correct product
behaviour and costs one extra turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, time, timedelta

WEEKDAYS = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}

MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "a": 1,
    "an": 1,
    "couple": 2,
    "pair": 2,
}

# Single tokens only. Splitting multi-word phrases into a word set is a trap:
# it turns "do", "it", "go" and "good" into affirmations, and then "do you have
# parking?" parses as a yes and suppresses the digression that should fire.
AFFIRM_WORDS = frozenset(
    [
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "ok",
        "okay",
        "correct",
        "confirm",
        "confirmed",
        "definitely",
        "absolutely",
        "affirmative",
        "right",
    ]
)
AFFIRM_PHRASES = ("sounds good", "go ahead", "that's right", "thats right", "do it", "please do")

# "cancel" is NOT a denial: mid-booking it means abandon, which the engine
# routes separately. "actually" and "change" are correction markers, not
# denials — "actually make it 6" is a repair and must not read as "no".
DENY_WORDS = frozenset(["no", "nope", "nah", "wrong", "incorrect", "negative"])
DENY_PHRASES = ("not right", "that's wrong", "thats wrong", "don't", "dont", "no thanks")


@dataclass(frozen=True)
class Reference:
    """The clock and calendar the conversation happens in."""

    today: date
    opens: time = time(17, 0)
    closes: time = time(22, 0)
    horizon_days: int = 90


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def parse_yes_no(text: str) -> bool | None:
    """True for an affirmation, False for a denial, None if neither.

    A denial anywhere in the utterance wins, because "no, make it Friday"
    is a correction and must not be read as agreement.
    """
    lowered = text.lower()
    words = set(_words(text))

    if words & DENY_WORDS or any(p in lowered for p in DENY_PHRASES):
        return False
    if words & AFFIRM_WORDS or any(p in lowered for p in AFFIRM_PHRASES):
        return True
    return None


def parse_party_size(text: str) -> int | None:
    """A head count, from digits or number words. None if absent."""
    lowered = text.lower()
    if re.search(r"\b(just me|only me|myself|by myself|for one)\b", lowered):
        return 1

    # A number touching a digit separator belongs to a time or an ISO date, not
    # to a head count: "19:00" and "2026-03-14" must not yield 19 or 3.
    for match in re.finditer(
        r"(?<![\d:/\-.])\b(\d{1,2})\b(?![\d:/\-.]|\s*(?:am|pm|o'?clock))", lowered
    ):
        value = int(match.group(1))
        if 1 <= value <= 99:
            return value

    for word, value in NUMBER_WORDS.items():
        if word in {"a", "an"}:
            continue
        if re.search(rf"\b{word}\b", lowered):
            return value

    # "a table for a person" — only count "a"/"an" next to an explicit noun.
    if re.search(r"\b(a|an)\s+(person|guest|seat)\b", lowered):
        return 1
    return None


def parse_time(text: str) -> time | None:
    """A wall-clock time. Bare hours are read as evening for a dinner booking."""
    lowered = text.lower()
    if re.search(r"\bnoon|midday\b", lowered):
        return time(12, 0)

    match = re.search(r"\b(\d{1,2})[:.](\d{2})\s*(am|pm)?\b", lowered)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        meridiem = match.group(3)
        if minute > 59:
            return None
        hour = _apply_meridiem(hour, meridiem)
        return time(hour, minute) if 0 <= hour <= 23 else None

    match = re.search(r"\b(\d{1,2})\s*(am|pm)\b", lowered)
    if match:
        hour = _apply_meridiem(int(match.group(1)), match.group(2))
        return time(hour, 0) if 0 <= hour <= 23 else None

    match = re.search(r"\b(\d{1,2})\s*o'?clock\b", lowered)
    if match:
        hour = int(match.group(1))
        # Bare "8 o'clock" in a dinner reservation means the evening.
        hour = hour + 12 if hour < 12 else hour
        return time(hour, 0) if hour <= 23 else None
    return None


def _apply_meridiem(hour: int, meridiem: str | None) -> int:
    if meridiem == "pm" and hour < 12:
        return hour + 12
    if meridiem == "am" and hour == 12:
        return 0
    if meridiem is None and 1 <= hour <= 11:
        # No am/pm given. For an evening-only restaurant, 7 means 19:00.
        return hour + 12
    return hour


def parse_date(text: str, ref: Reference) -> date | None:
    """A calendar date relative to `ref.today`. Ambiguous numerics are refused."""
    lowered = text.lower()

    if re.search(r"\b(today|tonight|this evening)\b", lowered):
        return ref.today
    if re.search(r"\btomorrow\b", lowered):
        return ref.today + timedelta(days=1)

    match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", lowered)
    if match:  # ISO is unambiguous, so it is the one numeric form accepted.
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None

    for name, month in MONTHS.items():
        pattern = (
            rf"\b{name}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b|\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{name}\b"
        )
        match = re.search(pattern, lowered)
        if match:
            day = int(match.group(1) or match.group(2))
            return _next_occurrence_of(ref.today, month, day)

    match = re.search(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b|\b(\d{1,2})(?:st|nd|rd|th)\b", lowered)
    if match:
        day = int(match.group(1) or match.group(2))
        return _next_day_of_month(ref.today, day)

    for name, weekday in WEEKDAYS.items():
        if not re.search(rf"\b{name}\b", lowered):
            continue
        ahead = (weekday - ref.today.weekday()) % 7
        # "friday" is the next Friday to come. "next friday" is the one in the
        # following week, which is the common reading but genuinely contested —
        # a production bot should confirm this one back to the user.
        if re.search(rf"\bnext\s+{name}\b", lowered):
            ahead = ahead + 7 if ahead else 7
        elif ahead == 0:
            ahead = 7
        return ref.today + timedelta(days=ahead)
    return None


def _next_occurrence_of(today: date, month: int, day: int) -> date | None:
    """That month and day, this year or next, whichever is still ahead."""
    for year in (today.year, today.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate >= today:
            return candidate
    return None


def _next_day_of_month(today: date, day: int) -> date | None:
    """The next time that day-of-month comes round."""
    month, year = today.month, today.year
    for _ in range(3):
        try:
            candidate = date(year, month, day)
        except ValueError:
            candidate = None
        if candidate and candidate >= today:
            return candidate
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return None


_PERSON = r"([a-z][a-z'\-]+(?:\s+[a-z][a-z'\-]+)?)"
NAME_PATTERNS = (
    rf"\b(?:my name is|name's|i'm|i am|it's|its|this is|under)\s+{_PERSON}",
    rf"\b(?:book|reserve|put)\s+(?:it\s+)?(?:under|for)\s+{_PERSON}",
)

# Words that look like a name to the patterns above but never are.
NAME_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "me",
        "myself",
        "us",
        "we",
        "you",
        "it",
        "that",
        "this",
        "here",
        "there",
        "now",
        "today",
        "tomorrow",
        "tonight",
        "yes",
        "no",
        "ok",
        "okay",
        "sure",
        "please",
        "thanks",
        "table",
        "people",
        "person",
        "guests",
        "booking",
        "reservation",
        "dinner",
        "lunch",
        "time",
        "date",
        "name",
    ]
)


@dataclass
class Extracted:
    """Everything one utterance yielded. Empty fields simply were not present."""

    party_size: int | None = None
    booking_date: date | None = None
    booking_time: time | None = None
    name: str | None = None
    yes_no: bool | None = None
    raw: str = ""
    found: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.found)


def parse_name(text: str) -> str | None:
    """A person's name, only from an explicit carrier phrase.

    Free-form name guessing is a trap: "friday" and "table" are perfectly good
    names syntactically. Requiring "my name is" / "under" trades recall for not
    writing "Tomorrow" on the reservation.
    """
    lowered = text.lower()
    for pattern in NAME_PATTERNS:
        match = re.search(pattern, lowered)
        if not match:
            continue
        candidate = match.group(1).strip()
        parts = [p for p in candidate.split() if p not in NAME_STOPWORDS]
        if not parts:
            continue
        return " ".join(part.capitalize() for part in parts)
    return None


def extract_all(text: str, ref: Reference) -> Extracted:
    """Run every parser over one utterance."""
    out = Extracted(raw=text)
    out.party_size = parse_party_size(text)
    out.booking_date = parse_date(text, ref)
    out.booking_time = parse_time(text)
    out.name = parse_name(text)
    out.yes_no = parse_yes_no(text)

    for field_name in ("party_size", "booking_date", "booking_time", "name", "yes_no"):
        if getattr(out, field_name) is not None:
            out.found.append(field_name)
    return out
