"""The booking bot as a router and four specialists, on day 5's tools.

Day 16 built it as one ADK agent holding every tool, and it failed the same
two scenarios on both runs: it booked a table after the guest said "never
mind", and it booked a table after the guest asked for a human. Day 6's
suite calls those safety failures, and they are the only failures in this
repository that have reproduced across every run of every model.

The multi-agent claim is that splitting the bot fixes exactly this. Not
because a specialist is smarter, but because **a tool it does not hold is a
tool it cannot call**. So the topology here is not "sequential and parallel
agents on day 4's flow", which is what day 19's brief said and would have
measured nothing: it is a deliberate attempt to make one known defect
unreachable, and then to find out what the attempt costs.

    booking_router      no tools. Decides which desk answers.
      slot_collector    no tools. Collects the four details.
      booking_desk      book_a_table. The ONLY holder.
      policy_desk       restaurant_policy.
      handoff_desk      no tools, and no way back out.

`handoff_desk` is the whole experiment. Day 4's engine makes HANDOFF a
terminal page: once the guest asks for a person, no transition leads
anywhere, so booking is not a thing the engine can go on to do. ADK has an
exact equivalent in `disallow_transfer_to_parent` and
`disallow_transfer_to_peers`, and with both set the desk cannot hand control
back. The booking tool is then out of scope for the rest of the conversation,
by construction rather than by instruction.

**What this does not guarantee**, and the write-up must not claim: the router
can still route a "never mind" turn to `booking_desk` instead of
`handoff_desk`, and then the booking happens. The defect is not eliminated.
It moves from *the agent ignored its instructions* to *the router made a
visible, logged, wrong decision* — which is a better failure to have, because
it is attributable and testable, but it is a different claim from impossible.
`scope.py` is what keeps the stronger claim honest: it checks the reachability
statically, with no model and no quota.

Day 5's tools and day 6's scorers are unchanged, as in day 16. The one
variable this design cannot control for is the role sentence each specialist
carries — four short prompts exist here that did not exist on day 16, so a
difference in results is a difference in *topology plus those sentences*, not
topology alone. Saying so is cheaper than pretending otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from convo.projects.day04_dialog_engine.extract import Reference
from convo.projects.day05_tool_agent.tools import book_table, get_policy, system_prompt

APP = "day19-multi-agent"

ROUTER = "booking_router"
COLLECTOR = "slot_collector"
BOOKER = "booking_desk"
POLICY = "policy_desk"
HANDOFF = "handoff_desk"

SPECIALISTS = (COLLECTOR, BOOKER, POLICY, HANDOFF)
EVERY_AGENT = (ROUTER, *SPECIALISTS)

BOOK_TOOL = "book_table"
POLICY_TOOL = "get_policy"
"""Day 5's executed tool names, not the ADK wrapper's advertised ones.

Same reasoning as day 18: the wrapper is what the model sees, the tool is
what runs, and a measurement keyed to the advertised name would change
silently if the wrapper were renamed.
"""

WRAPPER_FOR = {BOOK_TOOL: "book_a_table", POLICY_TOOL: "restaurant_policy"}
"""Executed tool -> the wrapper function the model is offered.

`scope.py` audits the topology by reading `Agent.tools`, which holds the
wrappers, while every measurement in days 16, 18 and 19 is keyed to the
executed tool. One mapping, in one place, so the static audit and the
recorded trajectory cannot drift apart.
"""

ROLES = {
    ROUTER: (
        "You are the front desk. Read the guest's latest message and transfer them "
        "to the right colleague, then let that colleague answer.\n"
        "- Asking for a person, a manager, or to speak to someone real: transfer to "
        f"{HANDOFF}.\n"
        "- Abandoning the booking — 'never mind', 'forget it', 'cancel that': transfer "
        f"to {HANDOFF}.\n"
        f"- A question about the restaurant rather than their booking: transfer to {POLICY}.\n"
        "- Confirming a booking you have all four details for: transfer to "
        f"{BOOKER}.\n"
        f"- Anything else about making a booking: transfer to {COLLECTOR}.\n"
        "You hold no tools. Do not attempt to book anything yourself."
    ),
    COLLECTOR: (
        "You collect the four booking details and nothing else: the number of guests, "
        "the date, the time, and a name. Ask for whatever is missing in plain friendly "
        "sentences.\n"
        "You cannot book a table — you do not have the tool. When you hold all four "
        "details, read them back and ask the guest to confirm."
    ),
    BOOKER: (
        "You write confirmed bookings and do nothing else. Call book_a_table once, with "
        "the details the guest has already given.\n"
        "Never invent a detail. If any of the four is missing or was never stated by the "
        f"guest, do not call the tool — say what is missing and transfer back to {COLLECTOR}."
    ),
    POLICY: (
        "You answer questions about the restaurant by calling restaurant_policy. Answer "
        f"the question, then transfer back to {ROUTER} so the booking can continue."
    ),
    HANDOFF: (
        "A colleague is taking over. Tell the guest a member of staff will be with them "
        "shortly, warmly and in one or two sentences.\n"
        "You hold no tools and there is nothing further to do. Do not offer to book, "
        "change, or confirm anything."
    ),
}
"""One role sentence per agent. The uncontrolled variable named in the module
docstring: day 16's single agent had day 5's prompt and nothing else."""


@dataclass
class Booking:
    """What the agents committed, and which of them committed it.

    `by_agent` is the addition day 19 needs. Day 16 recorded that
    `book_table` ran; the question here is *who* ran it, because a booking
    written by an agent the router should never have reached is a different
    defect from a booking with the wrong date in it.
    """

    fields: dict[str, Any] | None = None
    calls: int = 0
    failures: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    by_agent: list[tuple[str, str]] = field(default_factory=list)
    """(agent name, tool name) in execution order."""

    def tools_run_by(self, agent: str) -> tuple[str, ...]:
        return tuple(tool for who, tool in self.by_agent if who == agent)


def build_tools(ref: Reference, booking: Booking, owner: str):
    """Day 5's tools, wrapped so ADK derives a schema and we record the owner.

    `owner` is baked in at construction rather than read from the event
    stream. ADK reports the author of an *event*, and a tool result event's
    author is not always the agent whose function was invoked; closing over
    the owning agent means the record cannot drift from the topology.
    """

    def book_a_table(party_size: int, date: str, time: str, name: str) -> dict:
        """Book a table. Only call this for details the guest has actually given.

        Args:
            party_size: number of guests, 1 to 12.
            date: the date as YYYY-MM-DD.
            time: 24-hour time as HH:MM.
            name: the name to hold the booking under.
        """
        args = {"party_size": party_size, "date": date, "time": time, "name": name}
        result = book_table(args, ref)
        booking.tool_calls.append({"tool": BOOK_TOOL, "args": args, "ok": result["ok"]})
        booking.by_agent.append((owner, BOOK_TOOL))
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
        booking.tool_calls.append({"tool": POLICY_TOOL, "args": {"topic": topic}, "ok": True})
        booking.by_agent.append((owner, POLICY_TOOL))
        return result

    return {BOOK_TOOL: book_a_table, POLICY_TOOL: restaurant_policy}


def build_agent(ref: Reference, booking: Booking, model: str, *, require_key: bool = True):
    """The router and its four desks, with day 5's rules on every one of them.

    Constructing this calls no model and needs no key — verified, not assumed,
    because `require_key=False` is what lets `scope.py` audit the topology in
    CI, and a wrong guess there would have been the fourth time I asserted a
    dependency property without running it.
    """
    from google.adk.agents import Agent

    from convo.projects.day16_adk_agent.model import ensure_env

    # A run needs the key in the environment before the agent is built, since
    # ADK's client reads it from there. An audit does not, and must not: CI has
    # no key and the topology is the same object either way. Day 19 reuses day
    # 16's loader rather than adding a second one, so there is still exactly
    # one place in the repository that touches the key.
    if require_key:
        ensure_env()

    rules = system_prompt(ref)

    def desk(name: str, tools: list) -> Any:
        return Agent(
            name=name,
            model=model,
            instruction=f"{ROLES[name]}\n\n{rules}",
            tools=tools,
            # A desk may hand control back to the router, except the one that
            # must not: see the module docstring.
            disallow_transfer_to_parent=name == HANDOFF,
            disallow_transfer_to_peers=name == HANDOFF,
        )

    booker_tools = build_tools(ref, booking, BOOKER)
    policy_tools = build_tools(ref, booking, POLICY)

    return Agent(
        name=ROUTER,
        model=model,
        instruction=f"{ROLES[ROUTER]}\n\n{rules}",
        tools=[],
        sub_agents=[
            desk(COLLECTOR, []),
            desk(BOOKER, [booker_tools[BOOK_TOOL]]),
            desk(POLICY, [policy_tools[POLICY_TOOL]]),
            desk(HANDOFF, []),
        ],
    )
