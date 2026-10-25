"""Day 10: groundedness, citation enforcement, PII redaction and a red team.

`pii`, `groundedness`, `pipeline`, `probes` and `results` are dependency-light
and safe to import anywhere. `build` needs the `train` extra and is used by the
CLI only.
"""

from convo.core import ship
from convo.projects.day10_guardrails.ui import build

ship(10, build)

__all__ = ["build"]
