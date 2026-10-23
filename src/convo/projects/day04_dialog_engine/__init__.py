"""Day 4: a task-oriented dialogue engine with no model in it.

`extract`, `flow` and `engine` are pure Python and safe to import anywhere —
there is no torch, no network and no artifact to build, which is why this day
needs no benchmark step.
"""

from convo.core import ship
from convo.projects.day04_dialog_engine.ui import build

ship(4, build)

__all__ = ["build"]
