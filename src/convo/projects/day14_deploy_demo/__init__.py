"""Day 14: the deployable demo, which is a static site rather than a service.

No artifact of its own. The thing this day produced is the site, generated
from every other day's artifacts by `convo.site`, and the tab exists to point
at it and to say what moving to static hosting cost.
"""

from convo.core import ship
from convo.projects.day14_deploy_demo.ui import build

ship(14, build)

__all__ = ["build"]
