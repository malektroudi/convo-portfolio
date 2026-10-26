"""Day 18: ADK's evaluator against day 6's harness, both as CI gates.

`trajectory`, `compare`, `results`, `ui` and the threshold half of `gate` are
dependency-light and import without `google-adk` — which is the property the
gate relies on, since CI's main job cannot install the framework at all.
`adk_eval` imports ADK lazily, inside the functions that score.

As in day 16, the tab builder is not re-exported as `build`: this package has
a `build` module too.
"""

from convo.core import ship
from convo.projects.day18_eval_gate.ui import build as build_tab

ship(18, build_tab)

__all__ = ["build_tab"]
