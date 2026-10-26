"""Project the measured runs of days 10 to 13 into the event schema.

These are not synthetic logs. Every row comes from a pipeline that actually
ran: day 10's gates against its own thresholds, day 11's rewriter against its
conversations, day 12's audio through whisper and Piper, day 13's policies
over replayed exchanges. The warehouse is being fed the project's real
history, which is the only reason any number it produces is worth reading.

The one thing that is **not** real is time. Nothing in this repo recorded a
wall-clock timestamp per turn, so there is no honest way to put one here — and
rather than invent an arrival rate, the schema has no timestamp at all. That
rules out every trend, cohort and seasonality question, which is a large part
of what analytics usually means, and saying so is better than shipping a chart
whose x-axis is fiction.
"""

from __future__ import annotations

from convo.projects.day15_analytics.events import (
    CHAT,
    NO_CITATION,
    NO_EVIDENCE,
    NOT_RETRIEVED,
    VOICE,
    TurnEvent,
)


def _refusal_reason(reason: str) -> str:
    """Map a day's own wording onto the closed set.

    Exactly the translation layer the schema exists to make unnecessary: these
    strings are prose written for a human reading a tab, and the warehouse
    must not depend on them. Doing it once here is the cost of retrofitting a
    schema onto systems that predate it.
    """
    lowered = reason.lower()
    if "below threshold" in lowered or "evidence" in lowered:
        return NO_EVIDENCE
    if "cite" in lowered or "citation" in lowered or "span" in lowered:
        return NO_CITATION
    return NOT_RETRIEVED


def from_day10() -> list[TurnEvent]:
    """Single-turn questions through the guardrail gates."""
    from convo.projects.day10_guardrails.groundedness import decide
    from convo.projects.day10_guardrails.results import load

    found = load()
    if found is None:
        return []
    threshold = found.best().threshold
    events: list[TurnEvent] = []

    def add(evidence, kind: str, index: int, tags: tuple[str, ...] = ()) -> None:
        decision = decide(evidence, threshold)
        events.append(
            TurnEvent(
                conversation_id=f"d10-{kind}-{index:03d}",
                turn_index=0,
                surface=CHAT,
                source_day=10,
                question_chars=len(evidence.question),
                is_follow_up=False,
                answered=decision.answered,
                refusal_reason="" if decision.answered else _refusal_reason(decision.reason),
                cited_doc_id=evidence.doc_id if decision.answered else "",
                evidence_score=round(evidence.score, 4),
                tags=tags,
            )
        )

    for index, (evidence, _) in enumerate(found.answerable):
        add(evidence, "ans", index)
    for index, evidence in enumerate(found.unanswerable):
        add(evidence, "una", index, ("unanswerable",))
    for index, (case, evidence) in enumerate(found.red_team):
        if evidence is None:
            continue
        add(evidence, "red", index, ("red_team", case.attack))
    return events


def from_day11() -> list[TurnEvent]:
    """Multi-turn conversations, with whether the rewriter touched the turn."""
    from convo.projects.day11_multiturn_rag.results import load

    found = load()
    if found is None:
        return []
    events: list[TurnEvent] = []
    for row in found.turns("rewrite", "hybrid+rerank"):
        relevant = found.relevance.get(row.key, set())
        # "Answered" here means retrieval put a relevant chunk first. Day 11
        # ran no guardrail gate, so this is the closest equivalent decision
        # that day actually made, and it is labelled as such in the README
        # rather than quietly pooled with day 10's gated answers.
        answered = bool(row.ranked) and row.ranked[0] in relevant
        events.append(
            TurnEvent(
                conversation_id=f"d11-{row.conversation}",
                turn_index=row.index,
                surface=CHAT,
                source_day=11,
                question_chars=len(row.text),
                is_follow_up=row.index > 0,
                answered=answered,
                refusal_reason="" if answered else NOT_RETRIEVED,
                cited_doc_id=row.ranked[0].split("#")[0] if answered else "",
                rewritten=row.rewritten,
                rewrite_rule=row.strategy if row.rewritten else "",
            )
        )
    return events


def from_day12() -> list[TurnEvent]:
    """Voice turns, which are the only rows carrying a measured latency."""
    from convo.projects.day12_voice.results import load

    found = load()
    if found is None:
        return []
    events: list[TurnEvent] = []
    # `timed` rather than `select`: a turn the endpointer cut off has no valid
    # end-to-end latency, and feeding its negative total into a warehouse is
    # how this was noticed in the first place.
    for index, turn in enumerate(found.timed("cited", "stream")):
        answered = "don't have anything" not in turn.reply["text"].lower()
        events.append(
            TurnEvent(
                conversation_id=f"d12-{index:03d}",
                turn_index=0,
                surface=VOICE,
                source_day=12,
                question_chars=len(turn.question),
                is_follow_up=False,
                answered=answered,
                refusal_reason="" if answered else NO_EVIDENCE,
                latency_ms=round(turn.total_ms, 2),
                tags=("hesitation",) if turn.hesitant else (),
            )
        )
    return events


def from_day13() -> list[TurnEvent]:
    """Conversations where a policy decided whether to hand over, and when."""
    from convo.projects.day13_escalation.policy import signals_policy
    from convo.projects.day13_escalation.results import load

    found = load()
    if found is None:
        return []
    policy = signals_policy(soft_limit=1)
    events: list[TurnEvent] = []

    for conversation in found.conversations:
        handoff = policy(conversation.exchanges)
        for index, exchange in enumerate(conversation.exchanges):
            escalated_here = handoff is not None and handoff.turn == index
            events.append(
                TurnEvent(
                    conversation_id=f"d13-{conversation.name}",
                    turn_index=index,
                    surface=CHAT,
                    source_day=13,
                    question_chars=len(exchange.user),
                    is_follow_up=index > 0,
                    answered=exchange.answered,
                    refusal_reason="" if exchange.answered else NO_EVIDENCE,
                    cited_doc_id=exchange.doc_id,
                    evidence_score=round(exchange.evidence_score, 4)
                    if exchange.evidence_score
                    else None,
                    escalated=escalated_here,
                    escalation_reason=handoff.reason if escalated_here else "",
                    tags=("needs_human",) if conversation.escalate_at is not None else (),
                )
            )
            if handoff is not None and index >= handoff.turn:
                # Nothing after the handoff happened: the person took over.
                break
    return events


SOURCES = (from_day10, from_day11, from_day12, from_day13)


def collect() -> list[TurnEvent]:
    """Every event the project can honestly produce, in source order."""
    events: list[TurnEvent] = []
    for source in SOURCES:
        events.extend(source())
    return events
