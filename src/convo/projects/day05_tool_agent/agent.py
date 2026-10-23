"""The tool-calling loop: parse, validate, retry, stop.

Three guards, each earning its place:

  parsing     a small model emits tool calls in whatever shape it feels like —
              tagged, fenced, bare, or prose with JSON buried in it
  validation  a syntactically fine call can still be semantically wrong, and
              the error text goes back as the retry instruction
  limits      a bounded retry per turn and a hard cap on model calls per
              conversation, because an agent with neither will loop forever on
              a model that cannot satisfy the schema

Day 4 needed none of this: a state machine cannot emit a malformed transition.
That is the whole comparison in one sentence, and the step log below is what
lets the UI show it turn by turn.
"""

from __future__ import annotations

import json
import re
import time as time_mod
from dataclasses import dataclass, field
from typing import Any, Protocol

from convo.projects.day04_dialog_engine.extract import Reference
from convo.projects.day05_tool_agent import tools as tool_mod
from convo.projects.day05_tool_agent.grounding import (
    as_instruction as grounding_instruction,
)
from convo.projects.day05_tool_agent.grounding import grounding_errors
from convo.projects.day05_tool_agent.schema import validate

# Qwen-style tags, a fenced block, or a bare object. Checked in that order
# because the tagged form is unambiguous and the bare form is a guess.
TAGGED = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
FENCED = re.compile(r"```(?:json|tool_code)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class ModelCall:
    """One generation, with what it cost."""

    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0


class Backend(Protocol):
    """Anything that can continue a chat. Keeps the loop model-agnostic."""

    name: str

    def generate(
        self, messages: list[dict[str, Any]], tool_specs: list[dict[str, Any]]
    ) -> ModelCall: ...


@dataclass
class ParsedCall:
    name: str
    arguments: Any
    how: str
    """Which parser matched: tagged, fenced, bare."""


def parse_tool_call(text: str) -> ParsedCall | None:
    """Pull a tool call out of whatever the model actually emitted."""
    for pattern, how in ((TAGGED, "tagged"), (FENCED, "fenced")):
        match = pattern.search(text)
        if match:
            parsed = _load(match.group(1))
            if parsed:
                return ParsedCall(parsed[0], parsed[1], how)

    # Bare object anywhere in the text, balanced-brace scanned rather than
    # regexed, so nested argument objects survive.
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    parsed = _load(text[start : end + 1])
                    if parsed:
                        return ParsedCall(parsed[0], parsed[1], "bare")
                    break
    return None


def _load(blob: str) -> tuple[str, Any] | None:
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("name") or payload.get("tool") or payload.get("function")
    if not isinstance(name, str):
        return None
    arguments = payload.get("arguments", payload.get("parameters", {}))
    return name, arguments


@dataclass
class Step:
    """One model call and what became of it."""

    kind: str
    """reply | tool_call | unparseable | unknown_tool | invalid_args |
    ungrounded | tool_error"""
    raw: str = ""
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    parsed_how: str | None = None
    retry: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0

    @property
    def failed(self) -> bool:
        return self.kind in {
            "unparseable",
            "unknown_tool",
            "invalid_args",
            "ungrounded",
            "tool_error",
        }


@dataclass
class AgentConfig:
    max_retries: int = 2
    """Extra attempts per user turn after an invalid call."""
    max_model_calls: int = 14
    """Hard ceiling for a whole conversation. Without it, a model that cannot
    satisfy the schema will retry until something else kills the process."""
    check_grounding: bool = True
    """Refuse tool arguments the guest never supplied. Off measures what schema
    validation alone is worth."""


@dataclass
class AgentSession:
    backend: Backend
    ref: Reference
    config: AgentConfig = field(default_factory=AgentConfig)
    messages: list[dict[str, Any]] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    booking: dict[str, Any] | None = None
    user_turns: list[str] = field(default_factory=list)
    """Only what the guest actually typed — the grounding oracle's evidence.
    Retry instructions are injected as user messages too, so session.messages
    cannot be used for this."""
    exhausted: bool = False
    """True once the conversation hit max_model_calls."""

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages = [{"role": "system", "content": tool_mod.system_prompt(self.ref)}]

    @property
    def done(self) -> bool:
        return self.booking is not None or self.exhausted

    @property
    def model_calls(self) -> int:
        return len(self.steps)

    @property
    def failed_steps(self) -> int:
        return sum(s.failed for s in self.steps)

    @property
    def tokens(self) -> tuple[int, int]:
        return (
            sum(s.tokens_in for s in self.steps),
            sum(s.tokens_out for s in self.steps),
        )


@dataclass
class TurnResult:
    user_text: str
    reply: str
    steps: list[Step] = field(default_factory=list)
    booking: dict[str, Any] | None = None
    stopped: str = ""

    @property
    def retries(self) -> int:
        return sum(1 for s in self.steps if s.retry > 0)


def turn(session: AgentSession, text: str) -> TurnResult:
    """Advance the conversation by one user utterance."""
    result = TurnResult(user_text=text, reply="")
    if session.done:
        result.reply = "That's all set."
        result.stopped = "already finished"
        return result

    session.messages.append({"role": "user", "content": text})
    session.user_turns.append(text)
    specs = tool_mod.specs()

    for attempt in range(session.config.max_retries + 1):
        if session.model_calls >= session.config.max_model_calls:
            session.exhausted = True
            result.stopped = "hit the conversation's model-call ceiling"
            result.reply = "I'm having trouble with this — let me pass you to someone."
            return result

        started = time_mod.perf_counter()
        call = session.backend.generate(session.messages, specs)
        step = Step(
            kind="reply",
            raw=call.text,
            retry=attempt,
            tokens_in=call.tokens_in,
            tokens_out=call.tokens_out,
            latency_ms=call.latency_ms or (time_mod.perf_counter() - started) * 1000,
        )

        parsed = parse_tool_call(call.text)
        if parsed is None:
            # No tool call: treat it as something to say to the guest.
            reply = _strip(call.text)
            step.kind = "reply" if reply else "unparseable"
            session.steps.append(step)
            result.steps.append(step)
            session.messages.append({"role": "assistant", "content": call.text})
            if reply:
                result.reply = reply
                return result
            session.messages.append({"role": "user", "content": "Please answer, or call a tool."})
            continue

        step.tool, step.arguments, step.parsed_how = (
            parsed.name,
            parsed.arguments if isinstance(parsed.arguments, dict) else None,
            parsed.how,
        )
        session.messages.append({"role": "assistant", "content": call.text})

        tool = tool_mod.BY_NAME.get(parsed.name)
        if tool is None:
            step.kind = "unknown_tool"
            legal = ", ".join(sorted(tool_mod.BY_NAME))
            step.errors = [f"{parsed.name}: no such tool. Available tools: {legal}."]
            _record(session, result, step, step.errors[0])
            continue

        check = validate(parsed.arguments, tool.parameters)
        if not check.ok:
            step.kind = "invalid_args"
            step.errors = check.errors
            _record(session, result, step, check.as_instruction())
            continue

        step.arguments = check.coerced

        # Third guard: shape was fine, but did the guest actually say this?
        if session.config.check_grounding and tool.grounded:
            ungrounded = grounding_errors(check.coerced, session.user_turns, session.ref)
            if ungrounded:
                step.kind = "ungrounded"
                step.errors = ungrounded
                _record(session, result, step, grounding_instruction(ungrounded))
                continue

        outcome = tool.run(check.coerced, session.ref)
        step.result = outcome
        if not outcome.get("ok"):
            step.kind = "tool_error"
            step.errors = [str(outcome.get("error", "tool failed"))]
            _record(session, result, step, f"That didn't work: {step.errors[0]}")
            continue

        step.kind = "tool_call"
        session.steps.append(step)
        result.steps.append(step)
        session.messages.append({"role": "tool", "name": tool.name, "content": json.dumps(outcome)})

        if tool.name == tool_mod.BOOK_TABLE.name:
            session.booking = outcome
            result.booking = outcome
            result.reply = f"Booked — {outcome['summary']}. See you then."
            return result

        # A non-terminal tool (a policy answer): let the model use the result.
        result.reply = str(outcome.get("answer", ""))
        return result

    result.stopped = f"gave up after {session.config.max_retries + 1} attempts"
    result.reply = "I'm not getting that right — let me pass you to someone."
    return result


def _record(session: AgentSession, result: TurnResult, step: Step, instruction: str) -> None:
    """Log a failed step and hand the model the correction to retry against."""
    session.steps.append(step)
    result.steps.append(step)
    session.messages.append({"role": "user", "content": instruction})


def _strip(text: str) -> str:
    """The speakable part of a model response."""
    cleaned = TAGGED.sub("", text)
    cleaned = FENCED.sub("", cleaned)
    cleaned = re.sub(r"<\|?/?[a-z_]+\|?>", "", cleaned)
    return cleaned.strip()


def new_session(backend: Backend, ref: Reference, config: AgentConfig | None = None):
    return AgentSession(backend=backend, ref=ref, config=config or AgentConfig())
