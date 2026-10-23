"""Day 5: the booking bot as a tool-calling agent, measured against day 4.

`schema`, `tools`, `agent`, `scenarios` and `results` are dependency-light and
safe to import anywhere. `backends.LocalTransformers` and `benchmark` need the
`train` extra and are used by the CLI only.
"""

from convo.core import ship
from convo.projects.day05_tool_agent.ui import build

ship(5, build)

__all__ = ["build"]
