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
    Week(3, "On Google Cloud", "Tear every endpoint down the same day you create it."),
)

PLAN: tuple[Project, ...] = (
    Project(
        1,
        1,
        "hub",
        "Repo skeleton and a hub deployed empty",
        "Monorepo, a shared core package, and a hub containerised to Cloud Run on day one.",
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
        "kNN over embeddings. Add a new intent with five examples and watch fine-tuning lose.",
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
        "Golden multi-turn conversations, mechanical assertions, a judge rubric, a CI gate.",
    ),
    Project(
        7,
        1,
        "writeup-w1",
        "Write it up",
        "README, architecture diagram, demo GIF, comparison tables. No new code.",
    ),
    Project(
        8,
        2,
        "rag-cites",
        "RAG that actually cites",
        "One real corpus, three chunking strategies, citations that point at the source span.",
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
        "Query rewriting, a summarisation buffer, anaphora. Measured on multi-turn only.",
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
        "Frustration detection, an escalation policy, a handoff summary and suggested replies.",
    ),
    Project(
        14,
        2,
        "deploy-demo",
        "Deploy and demo",
        "Everything live on Cloud Run, cold start handled, a three-minute walkthrough recorded.",
    ),
    Project(
        15,
        3,
        "bq-analytics",
        "Conversation analytics in BigQuery",
        "Containment, escalation and failure rates in SQL. One BQML escalation model.",
    ),
    Project(
        16,
        3,
        "custom-training",
        "A custom training job on the platform",
        "Day 2's classifier as a container: custom training job, Model Registry, batch predict.",
    ),
    Project(
        17,
        3,
        "serve-monitor",
        "Serve it, monitor it, tear it down",
        "An endpoint, online prediction, skew and drift monitoring, undeployed the same day.",
    ),
    Project(
        18,
        3,
        "pipeline",
        "Pipeline it, with an eval gate",
        "KFP: ingest, train, evaluate, a conditional gate on pass rate, then deploy.",
    ),
    Project(
        19,
        3,
        "conversational-agents",
        "Build day 4 again, managed",
        "The same bot in Conversational Agents, with a Cloud Run webhook. Then compare.",
    ),
    Project(
        20,
        3,
        "managed-rag",
        "Managed RAG, measured against yours",
        "Day 9's eval set, unchanged, against Agent Search. Publish the comparison.",
    ),
    Project(
        21,
        3,
        "capstone",
        "Close it as one product",
        "One narrative, one architecture diagram, one cost table, three headline repos.",
    ),
)

for _project in PLAN:
    register(_project)
