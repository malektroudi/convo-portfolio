"""The third guard: is this argument something the guest actually said?

Schema validation checks SHAPE. The first run of day 5 showed that shape was
never the problem: across 20 model calls the 0.5B model emitted exactly one
malformed call and zero unparseable ones. What it did instead was call
book_table on the opening utterance with a complete, well-formed, entirely
invented booking — names like "John Doe" and "Table 12", today's date, an
arbitrary time. Every one of those passed the validator and the business rules,
because inventing plausible data is not a schema violation.

That is the structural difference between the two architectures. Day 4 cannot
hallucinate a slot: a slot is only ever filled by an extractor reading the
guest's own words. Day 5 can, because the model writes the arguments itself.

So this module closes the gap by using day 4's parsers as an oracle. Before a
booking executes, every argument must be traceable to something the guest said.
It is a provenance check, not a correctness check — and it is day 10's
groundedness work in miniature.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from convo.projects.day04_dialog_engine.extract import Reference, extract_all


def _evidence(user_texts: list[str], ref: Reference):
    """What the guest demonstrably said, according to day 4's parsers."""
    parsed = [extract_all(text, ref) for text in user_texts]
    return {
        "party_size": {p.party_size for p in parsed if p.party_size is not None},
        "date": {p.booking_date for p in parsed if p.booking_date is not None},
        "time": {p.booking_time for p in parsed if p.booking_time is not None},
    }


def grounding_errors(arguments: dict[str, Any], user_texts: list[str], ref: Reference) -> list[str]:
    """Arguments the guest never supplied, phrased as retry instructions."""
    evidence = _evidence(user_texts, ref)
    spoken = " ".join(user_texts).lower()
    errors: list[str] = []

    if "party_size" in arguments:
        try:
            size = int(arguments["party_size"])
        except (TypeError, ValueError):
            size = None
        if size is not None and size not in evidence["party_size"]:
            errors.append(
                f"party_size: the guest never said {size}. Ask how many people instead of assuming."
            )

    if "date" in arguments:
        try:
            when = datetime.strptime(str(arguments["date"]), "%Y-%m-%d").date()
        except ValueError:
            when = None
        if when is not None and when not in evidence["date"]:
            errors.append(
                f"date: the guest never said {arguments['date']}. Ask which day "
                f"instead of assuming."
            )

    if "time" in arguments:
        at = None
        for fmt in ("%H:%M", "%I:%M%p"):
            try:
                at = datetime.strptime(str(arguments["time"]).strip().upper(), fmt).time()
                break
            except ValueError:
                continue
        if at is not None and at not in evidence["time"]:
            errors.append(
                f"time: the guest never said {arguments['time']}. Ask what time "
                f"instead of assuming."
            )

    if "name" in arguments and arguments["name"]:
        # Lenient on purpose: day 4 only reads a name after a carrier phrase, so
        # requiring an exact extractor match would refuse names it simply cannot
        # see. Word presence is enough to rule out an invented one.
        words = [w for w in str(arguments["name"]).lower().split() if w]
        if words and not all(word in spoken for word in words):
            errors.append(
                f'name: the guest never gave the name "{arguments["name"]}". '
                f"Ask whose name the booking is under."
            )

    return errors


def as_instruction(errors: list[str]) -> str:
    """The correction handed back to the model."""
    if not errors:
        return ""
    if len(errors) == 1:
        return f"Do not invent booking details. {errors[0]}"
    joined = " ".join(errors)
    return f"Do not invent booking details. {joined}"
