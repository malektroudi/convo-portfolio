"""Scripted conversations both approaches are judged on, with assertions.

One suite, two runners. Day 4's state machine and day 5's agent see exactly
the same utterances and are scored by the same checks, which is the only way
the comparison means anything.

Results are `convo.core.evals` types — the ones written on day 1 for precisely
this. Day 6 extends this suite rather than inventing another.

The clock is fixed at Wednesday 2026-03-11 so "friday" is always 2026-03-13.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from convo.core.evals import Assertion, CaseResult
from convo.projects.day04_dialog_engine import flow as fl
from convo.projects.day04_dialog_engine.engine import new_session as new_engine_session
from convo.projects.day04_dialog_engine.engine import turn as engine_turn
from convo.projects.day04_dialog_engine.extract import Reference

REF = Reference(today=date(2026, 3, 11))  # a Wednesday


@dataclass(frozen=True)
class Scenario:
    id: str
    utterances: tuple[str, ...]
    expect_booked: bool
    expect: dict[str, object] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    note: str = ""


FRIDAY = "2026-03-13"
SARAH = {"party_size": 4, "date": FRIDAY, "time": "19:30", "name": "sarah chen"}

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "happy_one_shot",
        ("a table for 2 tomorrow at 8pm under chen", "yes"),
        True,
        {"party_size": 2, "date": "2026-03-12", "time": "20:00", "name": "chen"},
        ("happy",),
        "Everything in one utterance.",
    ),
    Scenario(
        "happy_slot_by_slot",
        ("i'd like a table", "4 people", "friday", "7:30pm", "my name is sarah chen", "yes"),
        True,
        SARAH,
        ("happy", "form"),
        "One parameter per turn, the ordinary case.",
    ),
    Scenario(
        "repair_at_confirmation",
        ("a table for 4", "friday", "7:30pm", "my name is sarah chen", "no, make it 8pm", "yes"),
        True,
        {**SARAH, "time": "20:00"},
        ("repair",),
        "The headline case: a denial carrying a correction.",
    ),
    Scenario(
        "repair_the_date",
        ("a table for 4", "friday", "7:30pm", "my name is sarah chen", "no, the 14th", "yes"),
        True,
        {**SARAH, "date": "2026-03-14"},
        ("repair",),
        '"no, the 14th" must change the date, not restart.',
    ),
    Scenario(
        "repair_mid_form",
        ("a table for 4", "actually make it 6", "friday", "7:30pm", "under sarah chen", "yes"),
        True,
        {**SARAH, "party_size": 6},
        ("repair",),
        "A correction before the form is even complete.",
    ),
    Scenario(
        "repair_two_fields",
        (
            "a table for 4",
            "friday",
            "7:30pm",
            "my name is sarah chen",
            "actually 6 people at 9pm",
            "yes",
        ),
        True,
        {**SARAH, "party_size": 6, "time": "21:00"},
        ("repair",),
        "Two corrections in one breath.",
    ),
    Scenario(
        "digression_parking",
        (
            "a table for 4",
            "do you have parking?",
            "friday",
            "7:30pm",
            "my name is sarah chen",
            "yes",
        ),
        True,
        SARAH,
        ("digression",),
        "Off-task question mid-form; the form must survive.",
    ),
    Scenario(
        "digression_at_confirmation",
        (
            "a table for 4",
            "friday",
            "7:30pm",
            "my name is sarah chen",
            "what's your cancellation policy?",
            "yes",
        ),
        True,
        SARAH,
        ("digression",),
        "Off-task question at the confirmation step.",
    ),
    Scenario(
        "reject_oversized_party",
        ("a table for 40", "sorry, 6", "friday", "7:30pm", "under sarah chen", "yes"),
        True,
        {**SARAH, "party_size": 6},
        ("validation",),
        "Over the seating limit, then corrected.",
    ),
    Scenario(
        "reject_closed_time",
        ("a table for 4", "friday", "3am", "7:30pm", "under sarah chen", "yes"),
        True,
        SARAH,
        ("validation",),
        "Outside service hours, then corrected.",
    ),
    Scenario(
        "reject_past_date",
        ("a table for 4", "2026-03-01", "friday", "7:30pm", "under sarah chen", "yes"),
        True,
        SARAH,
        ("validation",),
        "A date that has already gone.",
    ),
    Scenario(
        "never_books_without_details",
        ("i'd like a table", "hmm", "not sure yet"),
        False,
        {},
        ("safety",),
        "Must not invent details. Booking here is a failure.",
    ),
)


def normalise_engine(session) -> dict[str, object]:
    out: dict[str, object] = {}
    if fl.PARTY_SIZE in session.slots:
        out["party_size"] = int(session.slots[fl.PARTY_SIZE])
    if fl.BOOKING_DATE in session.slots:
        out["date"] = session.slots[fl.BOOKING_DATE].isoformat()
    if fl.BOOKING_TIME in session.slots:
        out["time"] = session.slots[fl.BOOKING_TIME].strftime("%H:%M")
    if fl.NAME in session.slots:
        out["name"] = str(session.slots[fl.NAME]).lower()
    return out


def normalise_booking(booking: dict | None) -> dict[str, object]:
    if not booking:
        return {}
    return {
        "party_size": int(booking["party_size"]),
        "date": str(booking["date"]),
        "time": str(booking["time"]),
        "name": str(booking["name"]).lower(),
    }


def assertions_for(scenario: Scenario, booked: bool, captured: dict) -> list[Assertion]:
    checks = [
        Assertion(
            "booked" if scenario.expect_booked else "did not book",
            booked == scenario.expect_booked,
            f"expected booked={scenario.expect_booked}, got {booked}",
        )
    ]
    for key, expected in scenario.expect.items():
        actual = captured.get(key)
        checks.append(
            Assertion(
                f"{key} correct", actual == expected, f"expected {expected!r}, got {actual!r}"
            )
        )
    return checks


def run_engine(scenario: Scenario, ref: Reference = REF) -> CaseResult:
    """Day 4's state machine against one scenario."""
    session = new_engine_session(ref)
    transcript: list[dict[str, str]] = []
    for utterance in scenario.utterances:
        result = engine_turn(session, utterance)
        transcript.append({"role": "user", "content": utterance})
        transcript.append({"role": "assistant", "content": result.reply})

    booked = session.page == fl.BOOKED
    captured = normalise_engine(session)
    latency = sum(t.latency_ms for t in session.history)
    return CaseResult.from_assertions(
        scenario.id,
        assertions_for(scenario, booked, captured),
        transcript=transcript,
        latency_ms=round(latency, 3),
        tags=list(scenario.tags),
    )


def run_agent(scenario: Scenario, backend, ref: Reference = REF, config=None) -> CaseResult:
    """Day 5's tool-calling agent against the same scenario."""
    from convo.projects.day05_tool_agent.agent import new_session as new_agent_session
    from convo.projects.day05_tool_agent.agent import turn as agent_turn

    session = new_agent_session(backend, ref, config)
    transcript: list[dict[str, str]] = []
    for utterance in scenario.utterances:
        result = agent_turn(session, utterance)
        transcript.append({"role": "user", "content": utterance})
        transcript.append({"role": "assistant", "content": result.reply})

    booked = session.booking is not None
    captured = normalise_booking(session.booking)
    tokens_in, tokens_out = session.tokens
    case = CaseResult.from_assertions(
        scenario.id,
        assertions_for(scenario, booked, captured),
        transcript=transcript,
        latency_ms=round(sum(s.latency_ms for s in session.steps), 3),
        tags=list(scenario.tags),
    )
    case.judge_rationale = (
        f"{session.model_calls} model calls, {session.failed_steps} failed, "
        f"{tokens_in} in / {tokens_out} out tokens"
    )
    return case
