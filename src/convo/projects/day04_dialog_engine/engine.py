"""The state machine. One function of (session, utterance) -> (session, result).

Every turn returns a `TurnResult` recording what was extracted, which slots
were filled or overwritten, which were rejected and why, where the flow moved
and the reason it moved. That record is the live state inspector in the UI and
the assertion surface for day 6's eval harness — the engine is observable by
construction rather than by adding logging later.

The order of checks in `turn()` is the design. Digressions are tested before
form filling, so an off-task question never gets parsed as a date. Corrections
are tested before a plain yes/no, so "no, make it Friday" repairs the date
instead of dropping the user into an amend menu.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date as date_cls
from typing import Any

from convo.core import span
from convo.projects.day04_dialog_engine.extract import (
    Extracted,
    Reference,
    extract_all,
)
from convo.projects.day04_dialog_engine.flow import (
    ABANDON_PATTERNS,
    AMEND,
    BOOKED,
    BOOKING_FLOW,
    COLLECT,
    CONFIRM,
    HANDOFF,
    HUMAN_PATTERNS,
    RESTART_PATTERNS,
    Digression,
    Flow,
)


@dataclass
class TurnResult:
    """Everything one turn did. The inspector renders this verbatim."""

    user_text: str
    reply: str
    page_before: str
    page_after: str
    transition: str
    extracted: Extracted
    filled: list[str] = field(default_factory=list)
    repaired: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    pending_slot: str | None = None
    digression: str | None = None
    reprompts: int = 0
    slots_after: dict[str, str] = field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def moved(self) -> bool:
        return self.page_before != self.page_after

    def as_dict(self) -> dict[str, Any]:
        return {
            "user": self.user_text,
            "reply": self.reply,
            "page": f"{self.page_before} -> {self.page_after}",
            "transition": self.transition,
            "filled": self.filled,
            "repaired": self.repaired,
            "rejected": [{"slot": s, "why": w} for s, w in self.rejected],
            "pending_slot": self.pending_slot,
            "digression": self.digression,
            "reprompts": self.reprompts,
            "slots": self.slots_after,
        }


@dataclass
class Session:
    """One conversation. Serialisable, so a transcript can be replayed exactly."""

    flow: Flow = field(default=BOOKING_FLOW)
    ref: Reference = field(default_factory=lambda: Reference(today=date_cls.today()))
    page: str = ""
    slots: dict[str, Any] = field(default_factory=dict)
    reprompts: int = 0
    history: list[TurnResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.page:
            self.page = self.flow.start

    @property
    def done(self) -> bool:
        return self.flow.page(self.page).terminal

    def filled_slots(self) -> dict[str, str]:
        """Slot values rendered for display, in the flow's own order."""
        out: dict[str, str] = {}
        for name in self.flow.page(COLLECT).slots:
            if name in self.slots:
                out[name] = self.flow.spec(name).display(self.slots[name])
        return out

    def missing_slots(self) -> list[str]:
        return [n for n in self.flow.page(COLLECT).slots if n not in self.slots]

    def summary(self) -> str:
        """The booking as a sentence fragment: "4 people on Friday at 8pm"."""
        parts = [f"{self.flow.spec(n).phrase} {v}".strip() for n, v in self.filled_slots().items()]
        return " ".join(parts)


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(p)}\b", lowered) for p in patterns)


def _find_digression(text: str, flow: Flow) -> Digression | None:
    for digression in flow.digressions:
        if _matches(text, digression.patterns):
            return digression
    return None


def _named_slots(text: str, flow: Flow) -> list[str]:
    """Slots the user referred to by name, e.g. "change the time"."""
    lowered = text.lower()
    hits: list[str] = []
    for name in flow.page(COLLECT).slots:
        spec = flow.spec(name)
        if any(re.search(rf"\b{re.escape(a)}\b", lowered) for a in spec.aliases):
            hits.append(name)
    return hits


def _apply_extracted(session: Session, found: Extracted, result: TurnResult) -> None:
    """Fill or repair every slot this utterance supplied, validating each."""
    for name in session.flow.page(COLLECT).slots:
        spec = session.flow.spec(name)
        value = getattr(found, spec.source, None)
        if value is None:
            continue

        error = spec.validator(value, session.ref)
        if error:
            result.rejected.append((name, error))
            continue

        if name in session.slots and session.slots[name] != value:
            result.repaired.append(name)
        elif name not in session.slots:
            result.filled.append(name)
        session.slots[name] = value


def _prompt_for_next(session: Session, result: TurnResult, prefix: str = "") -> None:
    """Ask for the first missing slot, or move to confirmation when complete."""
    missing = session.missing_slots()
    if missing:
        slot = missing[0]
        result.pending_slot = slot
        result.reply = f"{prefix}{session.flow.spec(slot).prompt}".strip()
        session.page = COLLECT
        return
    session.page = CONFIRM
    result.pending_slot = None
    result.reply = f"{prefix}{_confirmation_question(session)}".strip()


def _confirmation_question(session: Session) -> str:
    return f"So that's {session.summary()}. Shall I book it?"


def turn(session: Session, text: str) -> TurnResult:
    """Advance the conversation by one user utterance."""
    with span("dialogue.turn", page=session.page) as turn_span:
        result = _turn_inner(session, text)
    result.latency_ms = round(turn_span.duration_ms, 3)
    result.slots_after = session.filled_slots()
    result.reprompts = session.reprompts
    session.history.append(result)
    return result


def _turn_inner(session: Session, text: str) -> TurnResult:
    page_before = session.page
    found = extract_all(text, session.ref)
    result = TurnResult(
        user_text=text,
        reply="",
        page_before=page_before,
        page_after=page_before,
        transition="",
        extracted=found,
    )

    def finish(transition: str) -> TurnResult:
        result.transition = transition
        result.page_after = session.page
        return result

    # A terminal page absorbs everything; a new booking needs a new session.
    if session.done:
        result.reply = "That's all set. Start a new conversation to book again."
        return finish("terminal page, no transitions")

    # 1. Escalation requested outright.
    if _matches(text, HUMAN_PATTERNS):
        session.page = HANDOFF
        result.reply = "Of course — putting you through to someone now."
        return finish("human requested -> handoff")

    # 2. Start over: keep the session, drop the form.
    if _matches(text, RESTART_PATTERNS):
        session.slots.clear()
        session.reprompts = 0
        session.page = COLLECT
        _prompt_for_next(session, result, "No problem, starting fresh. ")
        return finish("restart -> collect, form cleared")

    # 3. Abandon, but only when it is not carrying a correction.
    if _matches(text, ABANDON_PATTERNS) and not found.found:
        session.page = HANDOFF
        result.reply = "No problem — I'll leave it there. Come back any time."
        return finish("abandoned -> handoff")

    # 4. Digression BEFORE form filling, so "do you have parking" is never
    #    mined for a date. The page does not change; we answer and re-ask.
    digression = _find_digression(text, session.flow)
    if digression and not found.found:
        result.digression = digression.name
        session.page = page_before
        if page_before == CONFIRM:
            result.reply = f"{digression.answer} {_confirmation_question(session)}"
        else:
            _prompt_for_next(session, result, f"{digression.answer} ")
        return finish(f"digression '{digression.name}', returned to {page_before}")

    if page_before == COLLECT:
        return _collect(session, found, result, finish)
    if page_before == CONFIRM:
        return _confirm(session, found, text, result, finish)
    if page_before == AMEND:
        return _amend(session, found, text, result, finish)
    result.reply = "Sorry, I lost my place. Could you say that again?"
    return finish("unknown page")


def _collect(session, found, result, finish):
    _apply_extracted(session, found, result)

    if result.rejected and not (result.filled or result.repaired):
        session.reprompts += 1
        if session.reprompts >= session.flow.max_reprompts:
            session.page = HANDOFF
            result.reply = "I'm not getting this right — let me pass you to someone who can help."
            return finish("too many rejections -> handoff")
        result.reply = result.rejected[0][1]
        result.pending_slot = result.rejected[0][0]
        return finish("value rejected, re-prompting same slot")

    if not (result.filled or result.repaired):
        session.reprompts += 1
        if session.reprompts >= session.flow.max_reprompts:
            session.page = HANDOFF
            result.reply = "I'm having trouble understanding — let me get a person for you."
            return finish("no progress -> handoff")
        _prompt_for_next(session, result, "Sorry, I didn't catch that. ")
        return finish("nothing extracted, re-prompting")

    session.reprompts = 0
    prefix = ""
    if result.repaired:
        labels = ", ".join(session.flow.spec(n).label.lower() for n in result.repaired)
        prefix = f"Updated the {labels}. "
    if result.rejected:
        prefix += result.rejected[0][1] + " "
    _prompt_for_next(session, result, prefix)
    moved = "-> confirm" if session.page == CONFIRM else "slot filled"
    return finish(f"{'repair, ' if result.repaired else ''}{moved}")


def _confirm(session, found, text, result, finish):
    # A correction arriving with the confirmation is a repair, not a rejection:
    # "no, make it 8pm" must change the time and re-confirm, not open a menu.
    correcting = [f for f in found.found if f != "yes_no"]
    if correcting:
        _apply_extracted(session, found, result)
        if result.rejected and not (result.filled or result.repaired):
            result.reply = result.rejected[0][1]
            return finish("correction rejected at confirm")
        session.reprompts = 0
        changed = result.repaired + result.filled
        labels = ", ".join(session.flow.spec(n).label.lower() for n in changed)
        _prompt_for_next(session, result, f"Changed the {labels}. " if labels else "")
        return finish(f"repair at confirm ({labels or 'no change'})")

    if found.yes_no is True:
        session.page = BOOKED
        session.reprompts = 0
        result.reply = f"Booked — {session.summary()}. See you then."
        return finish("confirmed -> booked")

    if found.yes_no is False:
        session.page = AMEND
        session.reprompts = 0
        result.reply = (
            "No problem — what should I change? The party size, the date, the time, or the name?"
        )
        return finish("denied without a correction -> amend")

    session.reprompts += 1
    if session.reprompts >= session.flow.max_reprompts:
        session.page = HANDOFF
        result.reply = "Let me pass you to someone to finish this off."
        return finish("no confirmation -> handoff")
    result.reply = f"Sorry — yes or no? {_confirmation_question(session)}"
    return finish("no yes/no, re-asking confirmation")


def _amend(session, found, text, result, finish):
    # The user may answer the amend question with the new value directly
    # ("make it 8pm") or by naming the field ("the time").
    if found.found and [f for f in found.found if f != "yes_no"]:
        _apply_extracted(session, found, result)
        if result.filled or result.repaired:
            session.reprompts = 0
            _prompt_for_next(session, result)
            return finish("amended by value -> confirm")

    named = _named_slots(text, session.flow)
    if named:
        for name in named:
            session.slots.pop(name, None)
        session.reprompts = 0
        session.page = COLLECT
        _prompt_for_next(session, result)
        return finish(f"cleared {', '.join(named)} -> collect")

    session.reprompts += 1
    if session.reprompts >= session.flow.max_reprompts:
        session.page = HANDOFF
        result.reply = "I'll pass you to someone who can sort this out."
        return finish("amend unresolved -> handoff")
    result.reply = (
        "Which one would you like to change — the party size, the date, the time, or the name?"
    )
    return finish("amend target unclear, re-asking")


def new_session(ref: Reference, flow: Flow = BOOKING_FLOW) -> Session:
    """A fresh conversation at the flow's start page."""
    return Session(flow=flow, ref=ref, page=flow.start)
