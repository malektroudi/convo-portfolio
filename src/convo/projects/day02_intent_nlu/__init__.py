"""Day 2: intent classification with inference-time out-of-scope rejection.

`analysis` and `results` are numpy-only and safe to import anywhere. `data` and
`train` need the `train` extra and are imported only by the training CLI.
"""

from convo.core import ship
from convo.projects.day02_intent_nlu.ui import build

ship(2, build)

__all__ = ["build"]
