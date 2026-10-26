"""The 21-day plan as data, plus the contract a day's project implements.

The hub renders whatever is registered here. A day that has not been built yet
still gets a tab showing its brief, so the hub is honest about progress instead
of hiding the unfinished work.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

Status = Literal["planned", "shipped"]

# A builder lays out one day's Gradio components. It runs inside an open
# gr.Tab(), so it adds components rather than returning them.
Builder = Callable[[], None]


@dataclass(frozen=True)
class Project:
    """One day of the plan.

    A `planned` day's brief is an intention and may describe anything. A
    `shipped` day's brief is a **claim about code that exists**, and is
    amended when the day lands differently from the plan — which is most
    days, because the plan was written on day one.

    This was not the rule to begin with, and by day 15 eight of fifteen
    shipped briefs still described the plan: a Cloud Run deployment that
    needs billing, a judge rubric that needs a model, a demo GIF nobody
    recorded, a "real" corpus that is synthetic by design, a summarising
    buffer that only bounds, suggested replies, and a BQML model fitted to
    nine positive examples. The README reports its results honestly and the
    index above it did not, which is the worse of the two places to be wrong
    — it is what a reader skims first.
    """

    day: int
    week: int
    slug: str
    title: str
    brief: str
    status: Status = "planned"
    builder: Builder | None = None

    @property
    def label(self) -> str:
        mark = "" if self.status == "shipped" else " ·"
        return f"{self.day:02d}{mark} {self.title}"

    @property
    def tab_label(self) -> str:
        """Short enough that all seven of a week's tabs fit on one row.

        The full titles run to fifty characters, which pushed everything past
        day three into an overflow menu that does not open reliably — hiding
        day 4, the best thing in week 1. The slug carries the meaning; the full
        title is the first heading inside the tab anyway.
        """
        mark = "" if self.status == "shipped" else "·"
        return f"{self.day:02d}{mark} {self.slug.replace('-', ' ')}"


_REGISTRY: dict[int, Project] = {}


def register(project: Project) -> Project:
    """Add or replace a day. Last writer wins, which is what imports want."""
    _REGISTRY[project.day] = project
    return project


def ship(day: int, builder: Builder) -> Project:
    """Attach a UI to a planned day and mark it shipped."""
    if day not in _REGISTRY:
        raise KeyError(f"day {day} is not in the plan")
    updated = replace(_REGISTRY[day], status="shipped", builder=builder)
    _REGISTRY[day] = updated
    return updated


def get(day: int) -> Project:
    return _REGISTRY[day]


def projects() -> list[Project]:
    return [_REGISTRY[day] for day in sorted(_REGISTRY)]


def by_week() -> dict[int, list[Project]]:
    grouped: dict[int, list[Project]] = {}
    for project in projects():
        grouped.setdefault(project.week, []).append(project)
    return grouped


def shipped() -> list[Project]:
    return [p for p in projects() if p.status == "shipped"]


@dataclass(frozen=True)
class Week:
    number: int
    title: str
    note: str


WEEKS: tuple[Week, ...] = (
    Week(1, "Dialogue mechanics", "Runs on a laptop or a free Kaggle session."),
    Week(
        2, "Grounding, voice, guardrails", "Every day here adds a measurement, not just a feature."
    ),
    Week(
        3,
        "Google's agent stack, free tier only",
        "Replanned on day 15: no billing account, so ADK and a free Gemini key "
        "instead of Vertex. Every day re-runs an existing week-1 or week-2 "
        "measurement, so the framework is compared against something.",
    ),
)

PLAN: tuple[Project, ...] = (
    Project(
        1,
        1,
        "hub",
        "Repo skeleton and a hub in a container",
        "Monorepo, a shared core package, and a hub in a container CI builds and boots.",
    ),
    Project(
        2,
        1,
        "intent-nlu",
        "Intent classifier with an out-of-scope door",
        "Fine-tune a small encoder on CLINC150. Tune the rejection threshold deliberately.",
    ),
    Project(
        3,
        1,
        "embed-router",
        "Embedding router against the fine-tuned model",
        "kNN over embeddings against the fine-tuned model, on ten intents it never saw.",
    ),
    Project(
        4,
        1,
        "dialog-engine",
        "Deterministic dialogue engine",
        "Pages, slots, validators, confirmation, repair and digression. No LLM anywhere.",
    ),
    Project(
        5,
        1,
        "tool-agent",
        "The same bot, tool-calling",
        "Schema validation, bounded retries, a turn limit, and a trace timeline per turn.",
    ),
    Project(
        6,
        1,
        "eval-harness",
        "Conversational eval harness",
        "Golden multi-turn conversations, mechanical assertions, three dimensions, a CI gate.",
    ),
    Project(
        7,
        1,
        "writeup-w1",
        "Write it up",
        "README, architecture diagram, comparison tables, and the defects writing them found.",
    ),
    Project(
        8,
        2,
        "rag-cites",
        "RAG that actually cites",
        "A corpus built around known failure modes, three chunkings, exact-offset citations.",
    ),
    Project(
        9,
        2,
        "retrieval-evals",
        "Make retrieval measurably better",
        "hit@k, MRR, nDCG. Then hybrid BM25 and a reranker, and report the lift.",
    ),
    Project(
        10,
        2,
        "guardrails",
        "Groundedness and guardrails",
        "Refuse when unsupported, enforce citations, redact PII, score a red-team set.",
    ),
    Project(
        11,
        2,
        "multiturn-rag",
        "Multi-turn RAG",
        "Query rewriting, a bounded memory buffer, anaphora, against a hand-written ceiling.",
    ),
    Project(
        12,
        2,
        "voice",
        "Voice, with a latency budget",
        "Streamed STT and TTS, instrumented: STT finalise, time-to-first-token, first audio.",
    ),
    Project(
        13,
        2,
        "escalation",
        "Escalation and agent assist",
        "Escalation signals, policies scored on when they fire, and what the agent receives.",
    ),
    Project(
        14,
        2,
        "deploy-demo",
        "Deploy and demo",
        "A static site generated from the committed artifacts, live on GitHub Pages for nothing.",
    ),
    Project(
        15,
        3,
        "bq-analytics",
        "Conversation analytics in BigQuery",
        "An event schema, then containment both ways, content gaps and latency in portable SQL.",
    ),
    Project(
        16,
        3,
        "adk-agent",
        "The booking bot in ADK",
        "Day 4/5's bot a third time, on day 5's tools and day 6's suite. Framework vs hand-rolled.",
    ),
    Project(
        17,
        3,
        "adk-memory",
        "ADK sessions and memory vs day 11",
        "Day 11's rewriter and buffer against ADK's own session and memory services.",
    ),
    Project(
        18,
        3,
        "adk-evals",
        "ADK's evaluator against day 6's harness",
        "Score the same suite both ways, then run whichever disagrees less as a CI gate.",
    ),
    Project(
        19,
        3,
        "adk-multi-agent",
        "Multi-agent against one state machine",
        "ADK's sequential and parallel agents on day 4's flow. Does splitting it buy anything?",
    ),
    Project(
        20,
        3,
        "a2a-handoff",
        "A2A carrying day 13's handoff packet",
        "Day 13's escalation across an agent boundary over A2A, with the provenance guard intact.",
    ),
    Project(
        21,
        3,
        "capstone",
        "Close it as one product",
        "One narrative, one architecture diagram, one cost table — which this time reads $0.",
    ),
)

for _project in PLAN:
    register(_project)
