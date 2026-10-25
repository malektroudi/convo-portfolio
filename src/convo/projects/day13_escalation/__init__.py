"""Day 13: escalation and agent assist.

`signals`, `policy`, `packet`, `conversations` and `results` are
dependency-light and safe to import anywhere. `build` needs the `train` extra
and is used by the CLI only.
"""

from convo.core import ship
from convo.projects.day13_escalation.ui import build

ship(13, build)

__all__ = ["build"]
