"""Each day's project is a module here that calls `ship(day, build)` on import.

Add a day by writing `dayNN_slug.py` with a `build()` that lays out its Gradio
components, then add the import below. Days that are not imported stay
"planned" and the hub shows their brief instead of a UI.
"""

from __future__ import annotations

from convo.projects import (
    day01_hub,  # noqa: F401  (import registers the day)
    day02_intent_nlu,  # noqa: F401
    day03_embed_router,  # noqa: F401
    day04_dialog_engine,  # noqa: F401
    day05_tool_agent,  # noqa: F401
    day06_eval_harness,  # noqa: F401
    day07_writeup,  # noqa: F401
    day08_rag_cites,  # noqa: F401
    day09_retrieval_evals,  # noqa: F401
    day10_guardrails,  # noqa: F401
)

__all__ = [
    "day08_rag_cites",
    "day09_retrieval_evals",
    "day10_guardrails",
    "day01_hub",
    "day02_intent_nlu",
    "day03_embed_router",
    "day04_dialog_engine",
    "day05_tool_agent",
    "day06_eval_harness",
    "day07_writeup",
]
