"""Shared primitives every day's project imports.

Nothing here knows about any specific model, provider or dataset. That is the
point: the projects change, these do not.
"""

from convo.core.costs import Call, CostMeter, Price, UnpricedModel
from convo.core.evals import (
    DIMENSIONS,
    EFFICIENCY,
    SAFETY,
    TASK,
    Assertion,
    CaseResult,
    Comparison,
    EvalRun,
)
from convo.core.logging import get_logger, setup
from convo.core.registry import (
    PLAN,
    WEEKS,
    Project,
    Week,
    by_week,
    get,
    projects,
    register,
    ship,
    shipped,
)
from convo.core.tracing import Span, Trace, current_trace, span, trace

__all__ = [
    "DIMENSIONS",
    "EFFICIENCY",
    "PLAN",
    "SAFETY",
    "TASK",
    "WEEKS",
    "Assertion",
    "Call",
    "CaseResult",
    "Comparison",
    "CostMeter",
    "EvalRun",
    "Price",
    "Project",
    "Span",
    "Trace",
    "UnpricedModel",
    "Week",
    "by_week",
    "current_trace",
    "get",
    "get_logger",
    "projects",
    "register",
    "setup",
    "ship",
    "shipped",
    "span",
    "trace",
]
