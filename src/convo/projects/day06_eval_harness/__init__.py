"""Day 6: the conversational eval harness and the gate CI blocks on.

Everything here is deterministic and dependency-light — no model, no network —
because these checks run on every push. The judge-backed scoring that needs a
model is deliberately out of the gate's path.
"""

from convo.core import ship
from convo.projects.day06_eval_harness.ui import build

ship(6, build)

__all__ = ["build"]
