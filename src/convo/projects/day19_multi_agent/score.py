"""Scoring the router, on top of day 6's unchanged verdict on the booking.

Day 6 still decides whether a conversation passed: same assertions, same three
dimensions, same thresholds as days 4, 5, 16 and 18. Nothing here touches
that, because the whole value of the `Outcome` shape is that five bots are
judged by one scorer.

What day 19 adds is the dimension day 16 could not have: **who answered**. A
single agent has one author for every turn, so "did it route correctly" is not
a question that can be asked of it. With a router it is the only question that
matters, because the topology already settles what each desk is able to do.

Three findings are separated deliberately, because they have different
consequences and conflating them is how a multi-agent write-up becomes a
sales pitch:

    a MISROUTE          the wrong desk answered, and nothing bad followed.
                        Noise, mostly: several desks give similar replies.
    a LATE BOOKING      `book_table` ran on or after the turn the engine hands
                        off. This is day 16's reproducing defect, and the one
                        thing the topology is meant to prevent.
    a SCOPE BREACH      `book_table` ran under an agent that does not hold it.
                        Should be impossible — `scope.py` proves the graph
                        allows no path to it. Checked anyway, at runtime,
                        because a proof about a graph is not a proof about a
                        framework's behaviour, and day 18 was a lesson in the
                        difference.
"""

from __future__ import annotations

from dataclasses import dataclass

from convo.projects.day19_multi_agent.agents import BOOK_TOOL, BOOKER, HANDOFF
from convo.projects.day19_multi_agent.oracle import Route


@dataclass(frozen=True)
class Turn:
    """What one user turn actually produced, as recorded by the runner."""

    authors: tuple[str, ...]
    """Desks that authored any part of the reply, in order. More than one is
    normal: the router transfers and the specialist answers within one turn."""

    tools: tuple[str, ...]
    """Executed tool names, in order."""

    by_agent: tuple[tuple[str, str], ...] = ()
    """(agent, tool) for each execution, so a scope breach is attributable."""


@dataclass(frozen=True)
class Verdict:
    scenario_id: str
    expected: tuple[str, ...]
    turns: tuple[Turn, ...]
    misroutes: tuple[int, ...]
    late_booking_turn: int | None
    scope_breaches: tuple[tuple[str, str], ...]

    @property
    def scored_turns(self) -> int:
        return len(self.expected)

    @property
    def routed_right(self) -> int:
        return self.scored_turns - len(self.misroutes)

    @property
    def routing_accuracy(self) -> float | None:
        """None, not 0.0, when there was nothing to score.

        Day 5's lesson, kept: an unmeasured dimension that renders as zero is
        a claim nobody made.
        """
        if not self.scored_turns:
            return None
        return round(self.routed_right / self.scored_turns, 4)

    @property
    def late_booking(self) -> bool:
        return self.late_booking_turn is not None

    @property
    def clean(self) -> bool:
        return not self.misroutes and not self.late_booking and not self.scope_breaches

    def as_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "expected": list(self.expected),
            "actual": [list(t.authors) for t in self.turns],
            "tools": [list(t.tools) for t in self.turns],
            "misroutes": list(self.misroutes),
            "late_booking_turn": self.late_booking_turn,
            "scope_breaches": [list(b) for b in self.scope_breaches],
            "routing_accuracy": self.routing_accuracy,
        }


def score_route(route: Route, turns: tuple[Turn, ...]) -> Verdict:
    """Compare what the router did against what day 4's engine implies.

    A turn counts as routed correctly when the expected desk is *among* the
    authors, not when it is the only one: ADK's root agent emits the transfer
    itself, so the router legitimately appears on nearly every turn. Demanding
    an exact single author would score the framework's own mechanics as
    failures and bury the real signal, which is the same mistake day 18's
    turn-aligned trajectory metric made visible.
    """
    scored = min(len(route.per_turn), len(turns))
    misroutes = tuple(
        index for index in range(scored) if route.per_turn[index] not in turns[index].authors
    )

    handoff_at = route.handoff_turn
    late: int | None = None
    if handoff_at is not None:
        for index in range(handoff_at, len(turns)):
            if BOOK_TOOL in turns[index].tools:
                late = index
                break

    breaches = tuple(
        (agent, tool)
        for turn in turns
        for agent, tool in turn.by_agent
        if tool == BOOK_TOOL and agent != BOOKER
    )

    return Verdict(
        scenario_id=route.scenario_id,
        expected=route.per_turn[:scored],
        turns=turns[:scored],
        misroutes=misroutes,
        late_booking_turn=late,
        scope_breaches=breaches,
    )


@dataclass(frozen=True)
class Summary:
    """Day 19's routing figures. Day 6's pass rate is reported beside it, not here."""

    verdicts: tuple[Verdict, ...]

    @property
    def turns(self) -> int:
        return sum(v.scored_turns for v in self.verdicts)

    @property
    def routing_accuracy(self) -> float | None:
        if not self.turns:
            return None
        right = sum(v.routed_right for v in self.verdicts)
        return round(right / self.turns, 4)

    @property
    def late_bookings(self) -> tuple[Verdict, ...]:
        return tuple(v for v in self.verdicts if v.late_booking)

    @property
    def scope_breaches(self) -> tuple[Verdict, ...]:
        return tuple(v for v in self.verdicts if v.scope_breaches)

    @property
    def handoff_scenarios(self) -> tuple[Verdict, ...]:
        """The five conversations the whole day is about."""
        return tuple(v for v in self.verdicts if HANDOFF in v.expected)

    @property
    def handoff_held(self) -> int:
        """Of those, how many ended without a booking written after the handoff."""
        return sum(1 for v in self.handoff_scenarios if not v.late_booking)

    def as_dict(self) -> dict:
        return {
            "turns": self.turns,
            "routing_accuracy": self.routing_accuracy,
            "late_bookings": [v.scenario_id for v in self.late_bookings],
            "scope_breaches": [v.scenario_id for v in self.scope_breaches],
            "handoff_scenarios": [v.scenario_id for v in self.handoff_scenarios],
            "handoff_held": self.handoff_held,
            "verdicts": [v.as_dict() for v in self.verdicts],
        }
