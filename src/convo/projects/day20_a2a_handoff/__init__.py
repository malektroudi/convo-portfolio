"""Day 20: day 13's handoff packet across a real A2A boundary.

`results` and `ui` import nothing from `a2a-sdk`; `wire`, `desk`, `send` and
`build` do, and it is an ordinary declared dependency rather than the
`--no-deps` arrangement `google-adk` needs. That contrast is part of the day.

As in days 16, 18 and 19, the tab builder is not re-exported as `build`: this
package has a `build` module too.
"""

from convo.core import ship
from convo.projects.day20_a2a_handoff.ui import build as build_tab

ship(20, build_tab)

__all__ = ["build_tab"]
