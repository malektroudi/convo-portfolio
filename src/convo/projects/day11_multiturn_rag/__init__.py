"""Day 11: multi-turn RAG — query rewriting, a memory buffer, anaphora.

`lexicon`, `memory`, `rewrite`, `conversations` and `results` are
dependency-light and safe to import anywhere. `build` needs the `train` extra
and is used by the CLI only.
"""

from convo.core import ship
from convo.projects.day11_multiturn_rag.ui import build

ship(11, build)

__all__ = ["build"]
