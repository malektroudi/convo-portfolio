"""The booking bot a third time, in ADK, on day 5's tools and day 6's suite.

Day 4 built it as a state machine with no model in it and scores 100% on the
suite. Day 5 rebuilt it as a tool-calling agent around a 0.5B model and
scored 8.3%, which said a great deal about the model and little about the
architecture. This is the version that separates those: a real framework and
a real model, judged by the same scorers.

**Two things are deliberately held fixed** so the comparison measures what it
claims to.

The *tools* are day 5's, imported and called directly — `book_table` applies
exactly the same validators day 4's engine applies, down to the opening hours
and the booking horizon. A reimplementation here would mean any difference in
the results could be a difference in the rules.

The *scorers* are day 6's, untouched. This file produces an `Outcome` and
nothing else; every assertion, dimension and gate downstream is shared with
the other two bots. That is the whole point of the `Outcome` shape existing.

**What is genuinely different** is worth naming, because it is what a
framework buys. Day 5 hand-wrote a JSON-Schema validator, a parser for three
different tool-call formats, bounded retries and a model-call ceiling. ADK
derives the tool schema from the Python signature, handles the call loop, and
keeps session state. Day 5's provenance guard has no equivalent and is not
used here — whether that matters is the question the run answers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from convo.projects.day04_dialog_engine.extract import Reference
from convo.projects.day05_tool_agent.tools import book_table, get_policy, system_prompt

APP = "day16-booking"
MAX_TURNS_PER_UTTERANCE = 6
"""Tool-call rounds ADK may take for one user utterance before being cut off.

Day 5 capped model calls for the same reason: an agent that cannot finish is
better stopped than left looping, and the efficiency dimension counts what it
spent either way.
"""


@dataclass
class Booking:
    """What the agent committed, captured when `book_table` succeeds.

    Read out of the tool rather than out of the model's prose: the booking is
    whatever the tool accepted, not whatever the reply claims. Day 5 learned
    that distinction the hard way — a model will happily describe a
    reservation it never made.
    """

    fields: dict[str, Any] | None = None
    calls: int = 0
    failures: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def build_tools(ref: Reference, booking: Booking):
    """Day 5's tools, wrapped so ADK can derive a schema from the signature."""

    def book_a_table(party_size: int, date: str, time: str, name: str) -> dict:
        """Book a table. Only call this once every detail has been given by the guest.

        Args:
            party_size: number of guests, 1 to 12.
            date: the date as YYYY-MM-DD.
            time: 24-hour time as HH:MM.
            name: the name to hold the booking under.
        """
        args = {"party_size": party_size, "date": date, "time": time, "name": name}
        result = book_table(args, ref)
        booking.tool_calls.append({"tool": "book_table", "args": args, "ok": result["ok"]})
        if result.get("ok"):
            booking.fields = {
                "party_size": result["party_size"],
                "date": result["date"],
                "time": result["time"],
                "name": result["name"],
            }
        else:
            booking.failures += 1
        return result

    def restaurant_policy(topic: str) -> dict:
        """Look up a restaurant policy.

        Args:
            topic: one of cancellation, parking, dietary, accessibility, dress_code.
        """
        result = get_policy({"topic": topic}, ref)
        booking.tool_calls.append({"tool": "get_policy", "args": {"topic": topic}, "ok": True})
        return result

    return [book_a_table, restaurant_policy]


def build_agent(ref: Reference, booking: Booking, model: str):
    """An ADK agent with the same instructions and tools day 5's agent had."""
    from google.adk.agents import Agent

    from convo.projects.day16_adk_agent.model import ensure_env

    # ADK's internal client reads the key from the environment, so it has to
    # be there before the agent is constructed.
    ensure_env()

    return Agent(
        name="booking_assistant",
        model=model,
        instruction=system_prompt(ref),
        tools=build_tools(ref, booking),
    )


@dataclass
class Turn:
    reply: str
    latency_ms: float
    model_calls: int


async def run_conversation(
    utterances: tuple[str, ...], ref: Reference, model: str
) -> tuple[list[Turn], Booking]:
    """Drive one scripted conversation through ADK and report what happened.

    Session state lives in ADK's in-memory service, so each scenario is a
    fresh session — a leaked session between scenarios would let one booking
    satisfy the next one's assertions.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    booking = Booking()
    agent = build_agent(ref, booking, model)
    runner = InMemoryRunner(agent=agent, app_name=APP)
    session = await runner.session_service.create_session(app_name=APP, user_id="guest")

    turns: list[Turn] = []
    for utterance in utterances:
        started = time.perf_counter()
        reply_parts: list[str] = []
        calls = 0
        async for event in runner.run_async(
            user_id="guest",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=utterance)]),
        ):
            content = getattr(event, "content", None)
            if content and getattr(content, "parts", None):
                for part in content.parts:
                    if getattr(part, "function_call", None):
                        calls += 1
                    text = getattr(part, "text", None)
                    if text and getattr(event, "author", "") != "user":
                        reply_parts.append(text)
            if calls > MAX_TURNS_PER_UTTERANCE:
                break
        booking.calls += max(calls, 1)
        turns.append(
            Turn(
                reply=" ".join(p.strip() for p in reply_parts if p.strip()),
                latency_ms=(time.perf_counter() - started) * 1000,
                model_calls=max(calls, 1),
            )
        )

    return turns, booking
