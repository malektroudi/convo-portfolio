"""The event schema, which is the actual deliverable of an analytics day.

The SQL in this project is unremarkable — counts, rates, a percentile. What
decides whether any of it is *possible* is what the system wrote down while
the conversation was happening, and that decision is made months earlier by
somebody who is not thinking about analytics.

Two rules account for most of it.

**Log the decision, not the prose.** A row that stores the assistant's reply
as text can answer nothing: "how often do we refuse" becomes a substring
search for whatever the refusal copy happened to say that quarter, and the
number changes when a copywriter does. A row that stores `answered=false,
refusal_reason='no citation verified'` answers it exactly, and keeps
answering it after the wording changes.

**Log the thing you will want to segment by, even when it is obvious now.**
`surface` is the clearest case here: every metric below splits differently for
voice and chat, and a schema without it cannot be fixed retrospectively —
the rows are already written.

What this schema deliberately does not carry is the customer's words. Day 10
built a PII redactor and it would be strange to pipe raw utterances into a
warehouse in the next breath. Length is kept because it is useful and
harmless; the text is not.

One field exists only for honesty. `source_day` records which measured run
each row was projected from, so every number downstream can be traced back to
the artifact that produced it rather than to a log nobody can find.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

SCHEMA_VERSION = 1

CHAT = "chat"
VOICE = "voice"


@dataclass(frozen=True)
class TurnEvent:
    """One user turn, as the system would have logged it at the time."""

    conversation_id: str
    turn_index: int
    surface: str
    source_day: int
    """Which day's measured run this row came from. Provenance, not analysis."""

    # --- what was asked ------------------------------------------------
    question_chars: int
    """Length only. The text is deliberately absent; day 10 redacts for a
    reason and a warehouse is the last place to undo it."""
    is_follow_up: bool
    """Whether this turn depended on an earlier one to make sense."""

    # --- what the system decided ---------------------------------------
    answered: bool
    refusal_reason: str = ""
    """Empty when answered. A stable enum, never the user-facing copy."""
    cited_doc_id: str = ""
    evidence_score: float | None = None
    """The reranker's relevance for the chosen passage. Null where the path
    had no retrieval step at all, which is different from a score of zero."""

    # --- what happened to the conversation ------------------------------
    escalated: bool = False
    escalation_reason: str = ""
    rewritten: bool = False
    rewrite_rule: str = ""

    # --- cost ------------------------------------------------------------
    latency_ms: float | None = None
    """End of input to first response. Null where the day did not measure it,
    rather than zero, which would drag every average it touches."""

    tags: tuple[str, ...] = field(default_factory=tuple)
    """Free labels: `red_team`, `high_stakes`, `paraphrase`."""

    def as_row(self) -> dict:
        row = asdict(self)
        row["tags"] = list(self.tags)
        return row


FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in fields(TurnEvent))

# BigQuery types for the same schema, used by the loader. Kept beside the
# dataclass so the two cannot drift: a test asserts they cover the same names.
BIGQUERY_SCHEMA: tuple[tuple[str, str, str], ...] = (
    ("conversation_id", "STRING", "REQUIRED"),
    ("turn_index", "INT64", "REQUIRED"),
    ("surface", "STRING", "REQUIRED"),
    ("source_day", "INT64", "REQUIRED"),
    ("question_chars", "INT64", "REQUIRED"),
    ("is_follow_up", "BOOL", "REQUIRED"),
    ("answered", "BOOL", "REQUIRED"),
    ("refusal_reason", "STRING", "NULLABLE"),
    ("cited_doc_id", "STRING", "NULLABLE"),
    ("evidence_score", "FLOAT64", "NULLABLE"),
    ("escalated", "BOOL", "REQUIRED"),
    ("escalation_reason", "STRING", "NULLABLE"),
    ("rewritten", "BOOL", "REQUIRED"),
    ("rewrite_rule", "STRING", "NULLABLE"),
    ("latency_ms", "FLOAT64", "NULLABLE"),
    ("tags", "STRING", "REPEATED"),
)

# Refusal reasons, as a closed set. The point of the schema is that this list
# is short, stable, and owned by the system rather than by its copy.
NO_EVIDENCE = "evidence_below_threshold"
NO_CITATION = "no_span_verified"
NOT_RETRIEVED = "nothing_retrieved"
REFUSAL_REASONS = (NO_EVIDENCE, NO_CITATION, NOT_RETRIEVED)


def write_ndjson(events: list[TurnEvent], path: Path) -> int:
    """Newline-delimited JSON: what `bq load` takes, and what DuckDB reads.

    One format for both engines, so the thing loaded into the warehouse is
    byte-identical to the thing queried locally. Any disagreement between them
    is then a query or an engine difference, never a loading difference.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event.as_row(), separators=(",", ":")) + "\n")
    return len(events)
