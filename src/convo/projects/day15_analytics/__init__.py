"""Day 15: conversation analytics, in a warehouse that needs no billing.

`events`, `emit` and `results` are dependency-light and safe to import
anywhere. `warehouse` and `build` need the `analytics` extra for DuckDB, and
the BigQuery path additionally needs the Cloud SDK and a signed-in project.
"""

from convo.core import ship
from convo.projects.day15_analytics.ui import build

ship(15, build)

__all__ = ["build"]
