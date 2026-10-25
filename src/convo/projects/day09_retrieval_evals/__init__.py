"""Day 9: labelled retrieval evaluation, hybrid fusion and a reranker.

`gold` and `results` are dependency-light and safe to import anywhere. `build`
needs the `train` extra — it runs the encoder and the cross-encoder — and is
used by the CLI only.
"""

from convo.core import ship
from convo.projects.day09_retrieval_evals.ui import build

ship(9, build)

__all__ = ["build"]
