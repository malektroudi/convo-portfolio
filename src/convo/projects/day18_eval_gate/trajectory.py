"""The expected tool trajectory, derived from day 4's state machine.

ADK's `TrajectoryEvaluator` scores an agent by comparing the tools it called,
turn by turn, against the tools it was supposed to call. Which raises the
question every trajectory metric raises and most write-ups skip: **where does
the expected trajectory come from?**

Hand-writing one per scenario would make the metric unfalsifiable. I would be
authoring the answer key from memory of what the agent did, and any
disagreement between ADK's evaluator and day 6's harness could be my
transcription rather than a real difference between the two.

So it is derived instead. Day 4's engine is replayed over the same scenario,
and its `TurnResult` stream says, for each turn, exactly what a correct bot
would have had to do:

    page_after == BOOKED   the turn the booking commits  -> book_table
    digression == "parking"  an FAQ was answered         -> get_policy
    page_after == HANDOFF  handed to a person            -> no tool at all

This is the same oracle day 5's provenance guard used, for the same reason:
the state machine has no model in it, so it cannot be persuaded, and it is the
one component in this repository whose behaviour is fully specified by its
tests. If the oracle is wrong the engine's own 100% on day 6's suite is wrong
too, and that is a much louder failure than a quiet mislabel here.

What this cannot derive is argument values, and it does not try — see
`IGNORE_ARGS`.
"""

from __future__ import annotations

from dataclasses import dataclass

from convo.projects.day04_dialog_engine import flow as fl
from convo.projects.day04_dialog_engine.engine import new_session, turn
from convo.projects.day06_eval_harness.suite import REF, Scenario

BOOK_TOOL = "book_table"
POLICY_TOOL = "get_policy"
"""Day 5's tool names, not the ADK wrapper's.

ADK derives its schema from a wrapper function, so the model sees
`book_a_table` and `restaurant_policy`. What actually runs is day 5's
`book_table` and `get_policy`, and those are the names the wrapper records.
Scoring against the executed tool rather than the advertised one means a
renamed wrapper cannot silently change the measurement."""

TOPIC_FOR_DIGRESSION = {
    "cancellation_policy": "cancellation",
    "opening_hours": "hours",
    "parking": "parking",
    "dietary": "dietary",
    "accessibility": "accessibility",
    "dress_code": "dress_code",
}
"""Day 4 named its digressions before day 5 named its policy topics, and two
of the names drift. Mapping them here rather than renaming either keeps both
days' published figures reproducible."""

IGNORE_ARGS = True
"""Compare tool *names* and not argument values, and say so loudly.

ADK's EXACT match compares arguments too, and on this suite that would be
measuring the wrong thing twice. Day 6 already checks every booking field
against the scenario's `expect` — that is what `date correct`, `time correct`
and the rest are. Re-checking arguments inside the trajectory metric would
make the two evaluators agree by construction on exactly the cases where an
independent second opinion is the entire point.

The remaining claim is therefore narrow and worth stating plainly: this
measures whether the agent **called the right tools on the right turns**, not
whether it called them with the right values.
"""


@dataclass(frozen=True)
class Expected:
    """What a correct bot calls, one entry per user turn."""

    scenario_id: str
    per_turn: tuple[tuple[str, ...], ...]
    """Tool names for each turn, in order. An empty tuple means *no tool*."""

    @property
    def total(self) -> int:
        return sum(len(t) for t in self.per_turn)

    @property
    def books(self) -> bool:
        return any(BOOK_TOOL in t for t in self.per_turn)


def expected_for(scenario: Scenario) -> Expected:
    """Replay day 4's engine and read the trajectory off its transitions."""
    session = new_session(REF)
    per_turn: list[tuple[str, ...]] = []

    for utterance in scenario.utterances:
        result = turn(session, utterance)
        tools: list[str] = []

        # A digression is an FAQ answered mid-form. The engine answers it from
        # a lookup table; an agent has to call the policy tool to do the same.
        if result.digression:
            topic = TOPIC_FOR_DIGRESSION.get(result.digression)
            if topic:
                tools.append(POLICY_TOOL)

        # The booking commits on the turn the engine enters BOOKED, and only
        # on that turn — re-entering an already-booked page is not a new call.
        if result.page_after == fl.BOOKED and result.page_before != fl.BOOKED:
            tools.append(BOOK_TOOL)

        per_turn.append(tuple(tools))

    return Expected(scenario_id=scenario.id, per_turn=tuple(per_turn))


def expected_all(scenarios: tuple[Scenario, ...]) -> dict[str, Expected]:
    return {s.id: expected_for(s) for s in scenarios}


def disagrees_with_suite(scenario: Scenario, expected: Expected) -> str:
    """Whether the derived trajectory contradicts the scenario's own label.

    A sanity check on the oracle, not on the agent. `expect_booked` is
    hand-written in day 6's suite and the trajectory is derived from day 4's
    engine; if they disagree, one of them is wrong and no result computed from
    either is worth reading. Returns an empty string when they agree.
    """
    if expected.books and not scenario.expect_booked:
        return "derived a booking call for a scenario the suite says must not book"
    if scenario.expect_booked and not expected.books:
        return "derived no booking call for a scenario the suite says must book"
    return ""
