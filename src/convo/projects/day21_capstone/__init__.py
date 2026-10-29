"""Day 21: close it as one product, and audit what it published.

`results` and `ui` import no other day. `claims`, `ledger` and `build` import
fourteen of them, because a claim is only worth publishing if the code that
recomputes it is one call away from the artifact that produced it.

As in days 16, 18, 19 and 20, the tab builder is not re-exported as `build`:
this package has a `build` module too.
"""

from convo.core import ship
from convo.projects.day21_capstone.ui import build as build_tab

ship(21, build_tab)

__all__ = ["build_tab"]
