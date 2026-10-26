"""Load the stored query results. No warehouse, no DuckDB, no network."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
RESULTS_FILE = ARTIFACT_DIR / "analytics.json"
EVENTS_FILE = ARTIFACT_DIR / "turns.ndjson"


@dataclass(frozen=True)
class Day15Results:
    events: int
    conversations: int
    sources: list[int]
    engine: str
    bigquery_available: bool
    descriptions: dict[str, str]
    results: list[dict]
    bigquery_results: list[dict]
    disagreements: list[dict]
    load_commands: list[str]
    meta: dict[str, Any]

    def query(self, name: str) -> dict | None:
        return next((r for r in self.results if r["name"] == name), None)

    def rows(self, name: str) -> list[dict]:
        found = self.query(name)
        return found["rows"] if found else []

    def one(self, name: str) -> dict:
        found = self.rows(name)
        return found[0] if found else {}

    @property
    def containment(self) -> dict:
        return self.one("01_containment")

    @property
    def gap(self) -> float:
        """The distance between the metric that gets reported and the truth."""
        row = self.containment
        if not row:
            return 0.0
        return round(row["containment_naive"] - row["containment_honest"], 4)

    @property
    def names(self) -> list[str]:
        return [r["name"] for r in self.results]

    def describe(self, name: str) -> str:
        return self.descriptions.get(name, "")


@lru_cache(maxsize=1)
def load() -> Day15Results | None:
    """Return the stored analytics, or None if the build has not been run."""
    if not RESULTS_FILE.exists():
        return None
    payload = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    return Day15Results(
        events=payload["events"],
        conversations=payload["conversations"],
        sources=payload["sources"],
        engine=payload["engine"],
        bigquery_available=payload.get("bigquery_available", False),
        descriptions=payload.get("descriptions", {}),
        results=payload["results"],
        bigquery_results=payload.get("bigquery_results", []),
        disagreements=payload.get("disagreements", []),
        load_commands=payload.get("load_commands", []),
        meta={"schema_version": payload.get("schema_version", 0)},
    )
