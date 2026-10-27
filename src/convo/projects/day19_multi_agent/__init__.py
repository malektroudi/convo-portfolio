"""Day 19: a router and four desks against day 16's single agent.

`agents`, `oracle`, `score` and `results` import without `google-adk`, which
imports lazily inside `build_agent`. `scope` therefore needs the framework
only when it builds the tree, and needs no key even then — which is what lets
CI check the topology in a job that cannot run the bot.

As in days 16 and 18, the tab builder is not re-exported as `build`: this
package has a `build` module too, and re-exporting the function over it is a
shadowing bug days 8-13 and 15 still carry.
"""

from convo.core import ship
from convo.projects.day19_multi_agent.ui import build as build_tab

ship(19, build_tab)

__all__ = ["build_tab"]
