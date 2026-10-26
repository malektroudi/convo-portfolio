"""ADK's `TrajectoryEvaluator`, driven from stored runs rather than live ones.

ADK's own entry point for this is `AgentEvaluator`, which re-runs the agent.
That is the wrong shape for a repository whose whole design is that expensive
work happens once and commits an artifact — and on a free key it is also the
difference between a gate CI can run and one it cannot.

So the evaluator is used directly, on `Invocation` objects rebuilt from the
committed run. The scoring is identical; it is the same class with the same
inputs. What changes is that a gate reading a file needs no API key, no quota
and no network, which is the only way the second evaluator gets into CI at
all.

**What is reachable on a free tier, and what is not.** ADK ships a dozen
evaluators and most of them are out of reach here, which is a finding rather
than an inconvenience:

    TrajectoryEvaluator        works — deterministic, no model
    FinalResponseMatchV1       needs `rouge_score`, installable but unused
                               (see `WHY_NOT_ROUGE`)
    FinalResponseMatchV2       LLM-as-judge: needs a model, so needs quota
    HallucinationsV1           LLM-as-judge, same
    SafetyEvaluator            needs `agentplatform` — not installable
    LocalEvalService           needs `agentplatform` — not installable
    MetricEvaluatorRegistry    needs `agentplatform` — not installable
    multi_turn_* evaluators    need `agentplatform` — not installable

The registry is the part worth dwelling on. It is the component that maps a
metric name to an evaluator, so without it there is no supported way to ask
ADK for "these four metrics" — you import the classes yourself, as below.
`agentplatform` is not on PyPI, so this is not a question of paying for
Vertex; the import simply cannot be satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass

from convo.projects.day18_eval_gate.trajectory import Expected

THRESHOLD = 1.0
"""The bar for a conversation's trajectory score.

Set to 1.0, which means *every* turn matched. ADK averages the per-turn 1.0/0.0
scores, so a lower bar would let a bot that called the wrong tool on one turn
of six pass at 0.83 — and on this suite the single most important turn is the
one that commits the booking. A metric that averages away the only turn that
touches a database is not a gate.

This also makes the comparison with day 6 fair in the one way that matters:
day 6's verdict is per conversation and binary, so the trajectory verdict has
to be too, or the two are not answering the same question.
"""

WHY_NOT_ROUGE = """FinalResponseMatchV1 scores the final reply against a reference
reply with ROUGE-1. This suite has no reference replies — day 6 asserts facts
about outcomes, not wordings — so using it would mean hand-writing 32 golden
responses and then measuring n-gram overlap against them. That rewards a bot
for phrasing its confirmation the way I happened to phrase mine, which is not
a property anyone wants. It is left out deliberately, not because it fails to
install."""


@dataclass(frozen=True)
class TrajectoryVerdict:
    """ADK's trajectory result for one conversation, in both alignments."""

    scenario_id: str
    score: float
    passed: bool
    per_turn: tuple[float, ...]
    actual: tuple[tuple[str, ...], ...]
    expected: tuple[tuple[str, ...], ...]
    flat_ok: bool = False
    """Whether the same tools were called, ignoring *which turn* they landed on.

    This exists because the first run of this comparison failed almost every
    booking scenario for one reason, repeated: the agent calls `book_table`
    the moment it holds four slots, while day 4's engine books on the
    following turn, after the guest says yes. Turn-aligned EXACT scores that
    as two wrong turns in every conversation.

    Separating the two alignments turns "ADK's evaluator fails 90% of the
    suite" into something you can act on. `passed` and `flat_ok` both false
    means the agent called the wrong tools. `passed` false with `flat_ok`
    true means it called the right tools at the wrong time — one behaviour,
    not thirty failures, and the difference between the two rates measures
    exactly that behaviour.
    """

    @property
    def only_timing(self) -> bool:
        """Right tools, wrong turns: the confirmation-step difference."""
        return self.flat_ok and not self.passed

    @property
    def first_bad_turn(self) -> int | None:
        for index, score in enumerate(self.per_turn):
            if score < 1.0:
                return index
        return None

    def explain(self) -> str:
        """Why it failed, in terms of tools rather than scores."""
        turn = self.first_bad_turn
        if turn is None:
            return ""
        got = list(self.actual[turn]) if turn < len(self.actual) else []
        want = list(self.expected[turn]) if turn < len(self.expected) else []
        if got and not want:
            return f"turn {turn}: called {got} where a correct bot calls nothing"
        if want and not got:
            return f"turn {turn}: called nothing where a correct bot calls {want}"
        return f"turn {turn}: called {got}, expected {want}"


def _flatten(per_turn: tuple[tuple[str, ...], ...]) -> list[str]:
    return [name for turn in per_turn for name in turn]


def _invocations(per_turn: tuple[tuple[str, ...], ...], utterances: tuple[str, ...]):
    """Rebuild ADK `Invocation`s from recorded tool names."""
    from google.adk.evaluation.eval_case import IntermediateData, Invocation
    from google.genai import types

    built = []
    for index, tools in enumerate(per_turn):
        text = utterances[index] if index < len(utterances) else ""
        built.append(
            Invocation(
                invocation_id=str(index),
                user_content=types.Content(role="user", parts=[types.Part(text=text)]),
                intermediate_data=IntermediateData(
                    # Args are deliberately empty; see trajectory.IGNORE_ARGS.
                    tool_uses=[types.FunctionCall(name=name, args={}) for name in tools]
                ),
            )
        )
    return built


def evaluate(
    scenario_id: str,
    utterances: tuple[str, ...],
    actual: tuple[tuple[str, ...], ...],
    expected: Expected,
) -> TrajectoryVerdict:
    """Score one conversation's trajectory with ADK's own evaluator."""
    from google.adk.evaluation.eval_metrics import EvalMetric, ToolTrajectoryCriterion
    from google.adk.evaluation.trajectory_evaluator import TrajectoryEvaluator

    want = expected.per_turn
    # ADK zips actual against expected with strict=True, so a length mismatch
    # raises rather than scoring. Pad instead: a conversation that ended early
    # called no tools on the turns it never reached, which is the truthful
    # reading and keeps the failure visible in the score rather than as a
    # crash.
    turns = max(len(actual), len(want))
    actual = tuple(list(actual) + [()] * (turns - len(actual)))
    want = tuple(list(want) + [()] * (turns - len(want)))

    metric = EvalMetric(
        metric_name="tool_trajectory_avg_score",
        criterion=ToolTrajectoryCriterion(
            threshold=THRESHOLD,
            match_type=ToolTrajectoryCriterion.MatchType.EXACT,
            ignore_args=True,
        ),
    )
    result = TrajectoryEvaluator(eval_metric=metric).evaluate_invocations(
        actual_invocations=_invocations(actual, utterances),
        expected_invocations=_invocations(want, utterances),
    )
    per_turn = tuple(r.score or 0.0 for r in result.per_invocation_results)
    score = result.overall_score or 0.0

    # The same comparison with turn boundaries collapsed, scored by the same
    # evaluator so the two numbers are produced by one implementation.
    flat = TrajectoryEvaluator(eval_metric=metric).evaluate_invocations(
        actual_invocations=_invocations((tuple(_flatten(actual)),), (" ".join(utterances),)),
        expected_invocations=_invocations((tuple(_flatten(want)),), (" ".join(utterances),)),
    )

    return TrajectoryVerdict(
        scenario_id=scenario_id,
        score=round(score, 4),
        passed=score >= THRESHOLD,
        per_turn=per_turn,
        actual=actual,
        expected=want,
        flat_ok=(flat.overall_score or 0.0) >= THRESHOLD,
    )
