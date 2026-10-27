"""Which desk should have answered each turn, derived from day 4's engine.

Day 18 derived the expected *tools* from the state machine rather than writing
them by hand, so that a disagreement between two evaluators could not just be
my transcription. Day 19 needs one more column — the expected *agent* — and it
comes from the same place, for the same reason.

The mapping from a `TurnResult` to a desk:

    the engine has reached HANDOFF      -> handoff_desk, and stays there
    this turn commits the booking       -> booking_desk
    this turn answered a digression     -> policy_desk
    anything else                       -> slot_collector

HANDOFF is sticky because day 4 made it a terminal page: once the guest asks
for a person, the engine has no transition out, so every later turn is still
the handoff. That stickiness is the whole point — it is the property
`scope.py` reproduces structurally in ADK, and the property day 16's single
agent had no way to express.

Two assumptions this rests on are checked rather than assumed, in
`build.py --preflight`:

  * no turn both answers a digression and commits a booking, so the
    precedence above never has to arbitrate between `policy_desk` and
    `booking_desk` (true for all 32 scenarios today);
  * the derived route agrees with each scenario's own `expect_booked` label.

Five scenarios' routes reach `handoff_desk`, and all five are ones the suite
says must not book — but they are not *all* of those. Six must not book, and
the sixth, `refuses_a_rule_override`, never leaves `slot_collector`: the engine
re-prompts and the slot validator rejects a party of 50, so nothing is ever
offered to the booking tool. Day 19's topology does nothing for that case; the
tool's own validators do.

Worth stating because the obvious summary — "the handoff scenarios are exactly
the must-not-book scenarios" — is wrong, and it was what I assumed until the
test for it failed. Two different mechanisms cover two different refusals.
"""

from __future__ import annotations

from dataclasses import dataclass

from convo.projects.day04_dialog_engine import flow as fl
from convo.projects.day04_dialog_engine.engine import new_session, turn
from convo.projects.day06_eval_harness.suite import REF, Scenario
from convo.projects.day18_eval_gate.trajectory import TOPIC_FOR_DIGRESSION
from convo.projects.day19_multi_agent.agents import (
    BOOK_TOOL,
    BOOKER,
    COLLECTOR,
    HANDOFF,
    POLICY,
    POLICY_TOOL,
)


@dataclass(frozen=True)
class Route:
    """The desk that should answer each user turn, in order."""

    scenario_id: str
    per_turn: tuple[str, ...]
    digressed: tuple[bool, ...]
    booked_on: int | None
    """Index of the turn the engine commits the booking, or None."""

    @property
    def handoff_turn(self) -> int | None:
        """The first turn the engine hands off, or None if it never does.

        Scoring needs the index and not just the fact: a booking written
        *before* the guest asked for a human is a different thing from one
        written after, and only the second is the defect day 19 is testing.
        """
        for index, desk in enumerate(self.per_turn):
            if desk == HANDOFF:
                return index
        return None

    @property
    def books(self) -> bool:
        return self.booked_on is not None

    @property
    def desks_used(self) -> frozenset[str]:
        return frozenset(self.per_turn)


def route_for(scenario: Scenario) -> Route:
    """Replay the engine and read the expected desk off each transition."""
    session = new_session(REF)
    per_turn: list[str] = []
    digressed: list[bool] = []
    booked_on: int | None = None
    handed_off = False

    for index, utterance in enumerate(scenario.utterances):
        result = turn(session, utterance)
        is_digression = bool(result.digression and result.digression in TOPIC_FOR_DIGRESSION)
        commits = result.page_after == fl.BOOKED and result.page_before != fl.BOOKED

        if result.page_after == fl.HANDOFF:
            handed_off = True

        if handed_off:
            desk = HANDOFF
        elif commits:
            desk = BOOKER
            booked_on = index
        elif is_digression:
            desk = POLICY
        else:
            desk = COLLECTOR

        per_turn.append(desk)
        digressed.append(is_digression)

    return Route(
        scenario_id=scenario.id,
        per_turn=tuple(per_turn),
        digressed=tuple(digressed),
        booked_on=booked_on,
    )


def routes_for(scenarios: tuple[Scenario, ...]) -> dict[str, Route]:
    return {s.id: route_for(s) for s in scenarios}


def disagrees_with_suite(scenario: Scenario, route: Route) -> str:
    """Whether the derived route contradicts the scenario's own label.

    A check on the oracle, not on the agent — the same guard day 18 runs
    before spending a model call. Returns an empty string when they agree.
    """
    if route.books and not scenario.expect_booked:
        return "derived a booking desk for a scenario the suite says must not book"
    if scenario.expect_booked and not route.books:
        return "derived no booking desk for a scenario the suite says must book"
    return ""


def arbitration_needed(scenario: Scenario) -> str:
    """Whether any turn would force a choice between policy_desk and booking_desk.

    The precedence in this module's docstring is only honest while no turn
    does both. Checked rather than asserted, because it is a property of day
    4's engine and day 6's scenarios, and either could change.
    """
    session = new_session(REF)
    for index, utterance in enumerate(scenario.utterances):
        result = turn(session, utterance)
        digresses = bool(result.digression and result.digression in TOPIC_FOR_DIGRESSION)
        commits = result.page_after == fl.BOOKED and result.page_before != fl.BOOKED
        if digresses and commits:
            return f"turn {index} both answers a digression and commits the booking"
    return ""


def expected_tools(route: Route) -> tuple[tuple[str, ...], ...]:
    """The tool trajectory implied by a route, for cross-checking against day 18.

    Day 18 derives the same thing straight from the engine. Deriving it a
    second way here and comparing the two in the preflight is cheap, and it is
    the only check that would catch a drift between day 18's oracle and day
    19's — which would otherwise look like a real difference between a single
    agent and a router.
    """
    out: list[tuple[str, ...]] = []
    for index, desk in enumerate(route.per_turn):
        tools: list[str] = []
        if route.digressed[index]:
            tools.append(POLICY_TOOL)
        if desk == BOOKER and route.booked_on == index:
            tools.append(BOOK_TOOL)
        out.append(tuple(tools))
    return tuple(out)
