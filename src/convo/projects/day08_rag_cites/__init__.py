"""Day 8: retrieval with citations that point at exact source spans.

`chunking`, `index`, `answer`, `questions` and `results` are numpy-only and
safe to import anywhere. `build` needs the `train` extra and is used by the
CLI only.
"""

from convo.core import ship
from convo.projects.day08_rag_cites.ui import build

ship(8, build)

__all__ = ["build"]
