"""Drive one scripted conversation through the router and record who answered.

Deliberately a near-copy of day 16's runner, and the differences are the
measurement, so they are listed here rather than left to a reader's diff.

**It records the author of every turn.** Day 16 recorded tools; day 19 needs
desks, because the topology already decides what each desk can do. ADK reports
an author per event, so the authors of one user turn are collected in order
and kept as a tuple — the router and then the specialist it transferred to is
the normal shape, not an anomaly.

**It counts transfers separately.** `transfer_to_agent` is an ordinary
function call in ADK's event stream, so routing spends model calls that day
16's single agent never spent. Those calls are counted into `model_calls`
exactly as day 16 counted its own, because that is what day 6's efficiency
dimension measures and changing the rule per bot would make the column
meaningless. They are *also* counted on their own, so the write-up can say how
much of any efficiency gap is the price of the guarantee rather than the agent
being slower at its job.

**Its per-utterance cap is higher.** Day 16 allowed 6 tool-call rounds for one
user utterance. A routed turn spends one of those on the transfer before the
specialist does anything, so keeping 6 would cut day 19's conversations off
earlier than day 16's and show up as a behavioural difference. 10 restores the
same headroom for actual work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from convo.projects.day04_dialog_engine.extract import Reference
from convo.projects.day19_multi_agent.agents import APP, ROUTER, Booking, build_agent

MAX_ROUNDS_PER_UTTERANCE = 10
TRANSFER_TOOL = "transfer_to_agent"


@dataclass
class TurnRecord:
    reply: str
    authors: tuple[str, ...]
    tools: tuple[str, ...]
    by_agent: tuple[tuple[str, str], ...]
    transfers: int
    model_calls: int
    latency_ms: float


async def run_conversation(
    utterances: tuple[str, ...], ref: Reference, model: str
) -> tuple[list[TurnRecord], Booking]:
    """One fresh session per scenario, as in day 16.

    A leaked session would let one scenario's booking satisfy the next one's
    assertions, and with a router it would also carry the previous
    conversation's desk — so a handoff in scenario 12 would silently gag
    scenario 13.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    booking = Booking()
    agent = build_agent(ref, booking, model)
    runner = InMemoryRunner(agent=agent, app_name=APP)
    session = await runner.session_service.create_session(app_name=APP, user_id="guest")

    records: list[TurnRecord] = []
    for utterance in utterances:
        started = time.perf_counter()
        reply_parts: list[str] = []
        authors: list[str] = []
        calls = 0
        transfers = 0
        before = len(booking.by_agent)

        async for event in runner.run_async(
            user_id="guest",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=utterance)]),
        ):
            author = getattr(event, "author", "") or ""
            content = getattr(event, "content", None)
            if content and getattr(content, "parts", None):
                for part in content.parts:
                    call = getattr(part, "function_call", None)
                    if call is not None:
                        calls += 1
                        if getattr(call, "name", "") == TRANSFER_TOOL:
                            transfers += 1
                    text = getattr(part, "text", None)
                    if text and text.strip() and author != "user":
                        reply_parts.append(text)
            if author and author != "user" and author not in authors:
                authors.append(author)
            if calls > MAX_ROUNDS_PER_UTTERANCE:
                break

        booking.calls += max(calls, 1)
        executed = booking.by_agent[before:]
        records.append(
            TurnRecord(
                reply=" ".join(p.strip() for p in reply_parts if p.strip()),
                # The router authors nearly every turn by issuing the
                # transfer, so it is kept in the tuple rather than filtered
                # out: `score_route` asks whether the expected desk is among
                # the authors, and hiding the router would make a turn the
                # router answered alone indistinguishable from an empty one.
                authors=tuple(authors) or (ROUTER,),
                tools=tuple(tool for _, tool in executed),
                by_agent=tuple(executed),
                transfers=transfers,
                model_calls=max(calls, 1),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        )

    return records, booking
