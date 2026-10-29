"""Load day 21's artifact. Imports no other day.

The same split every day since 18 has used, and here it does real work: day
21's build imports fourteen days' `results` modules to resolve its claims, and
the Gradio tab must not. A tab that pulled in every artifact in the repository
to draw one table would make the hub's first paint depend on all of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_FILE = ARTIFACT_DIR / "capstone.json"


@dataclass(frozen=True)
class Audited:
    key: str
    where: tuple[str, ...]
    published: str
    derived: str | None
    kind: str
    note: str
    verdict: str

    @property
    def holds(self) -> bool:
        return self.verdict == "holds"


@dataclass(frozen=True)
class Results:
    shipped: int
    of_days: int
    committed_bytes: int
    image_ceiling_mb: int
    blocked_in_image: tuple[str, ...]
    claims: tuple[dict[str, Any], ...]
    audit: tuple[Audited, ...]
    tally: dict[str, int]
    inventory: tuple[dict[str, Any], ...]
    pipeline: tuple[dict[str, Any], ...]
    costs: tuple[dict[str, Any], ...]
    gaps: tuple[dict[str, Any], ...]

    @property
    def megabytes(self) -> float:
        return round(self.committed_bytes / 1024 / 1024, 1)

    @property
    def leads(self) -> tuple[dict[str, Any], ...]:
        return tuple(c for c in self.claims if c["lead"])

    @property
    def days_with_artifacts(self) -> int:
        return sum(1 for row in self.inventory if row["files"])

    @property
    def days_rebuilt_in_ci(self) -> int:
        return sum(1 for row in self.inventory if row["rebuilt_in_ci"])

    @property
    def unverifiable(self) -> tuple[dict[str, Any], ...]:
        """Claims no artifact can settle. One, and it is the best one."""
        return tuple(c for c in self.claims if c["kind"] == "testimony")

    def kinds(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for claim in self.claims:
            counts[claim["kind"]] = counts.get(claim["kind"], 0) + 1
        return counts


@lru_cache(maxsize=1)
def load() -> Results | None:
    if not ARTIFACT_FILE.exists():
        return None
    raw = json.loads(ARTIFACT_FILE.read_text(encoding="utf-8"))
    return Results(
        shipped=raw["shipped"],
        of_days=raw["of_days"],
        committed_bytes=raw["committed_bytes"],
        image_ceiling_mb=raw["image_ceiling_mb"],
        blocked_in_image=tuple(raw["blocked_in_image"]),
        claims=tuple(raw["claims"]),
        audit=tuple(
            Audited(
                key=row["key"],
                where=tuple(row["where"]),
                published=row["published"],
                derived=row["derived"],
                kind=row["kind"],
                note=row["note"],
                verdict=row["verdict"],
            )
            for row in raw["audit"]
        ),
        tally=dict(raw["tally"]),
        inventory=tuple(raw["inventory"]),
        pipeline=tuple(raw["pipeline"]),
        costs=tuple(raw["costs"]),
        gaps=tuple(raw["gaps"]),
    )
