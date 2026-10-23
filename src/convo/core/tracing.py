"""Minimal span tracing, so every project can show where a turn spent its time.

Day 5 renders these as a trace timeline and day 12 as a latency waterfall, so
the shape here is deliberately flat and JSON-serialisable.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    name: str
    start_ms: float
    duration_ms: float
    parent: str | None = None
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    attrs: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent": self.parent,
            "start_ms": round(self.start_ms, 2),
            "duration_ms": round(self.duration_ms, 2),
            **({"attrs": self.attrs} if self.attrs else {}),
        }


@dataclass
class Trace:
    """A single conversational turn's worth of spans."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    spans: list[Span] = field(default_factory=list)
    _origin: float = field(default_factory=time.perf_counter)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._origin) * 1000, 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "total_ms": self.total_ms,
            "spans": [s.as_dict() for s in self.spans],
        }

    def waterfall(self) -> list[tuple[str, float, float]]:
        """(name, offset_ms, duration_ms) rows, ready to plot."""
        return [(s.name, round(s.start_ms, 2), round(s.duration_ms, 2)) for s in self.spans]


_current_trace: ContextVar[Trace | None] = ContextVar("current_trace", default=None)
_current_span: ContextVar[str | None] = ContextVar("current_span", default=None)


@contextmanager
def trace(trace_id: str | None = None) -> Iterator[Trace]:
    """Start a new trace for one turn."""
    active = Trace(trace_id=trace_id) if trace_id else Trace()
    token = _current_trace.set(active)
    try:
        yield active
    finally:
        _current_trace.reset(token)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    """Time a block. Outside a trace() this is a no-op recorder, never an error."""
    active = _current_trace.get()
    started = time.perf_counter()
    offset = (started - active._origin) * 1000 if active else 0.0
    record = Span(
        name=name, start_ms=offset, duration_ms=0.0, parent=_current_span.get(), attrs=attrs
    )
    token = _current_span.set(record.span_id)
    try:
        yield record
    finally:
        _current_span.reset(token)
        record.duration_ms = (time.perf_counter() - started) * 1000
        if active is not None:
            active.spans.append(record)


def current_trace() -> Trace | None:
    return _current_trace.get()
