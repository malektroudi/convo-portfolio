"""Day 3: an embedding router, measured against day 2's fine-tuned classifier.

`router`, `analysis` and `results` are numpy-only and safe to import anywhere.
`embed` and `benchmark` need the `train` extra and are used by the CLI only.
"""

from convo.core import ship
from convo.projects.day03_embed_router.ui import build

ship(3, build)

__all__ = ["build"]
