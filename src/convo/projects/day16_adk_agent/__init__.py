"""Day 16: the booking bot a third time, in Google's Agent Development Kit.

`results` and `ui` are dependency-light and safe to import anywhere — they
read the committed artifact and nothing else. `agent`, `model` and `build`
need the `agents` extra for `google-adk` and a `GOOGLE_API_KEY` in `.env`,
and import ADK lazily so the hub never pays for it.

The tab builder is deliberately *not* re-exported as `build`, which is what
every earlier day does. Eight of them have a `build.py` module as well, so in
those packages the name means two things and
`from convo.projects.dayNN_x import build` quietly hands back the Gradio
builder instead of the module. Nothing depends on the re-export, and this is
the first day whose tests needed the module, so it stops here.
"""

from convo.core import ship
from convo.projects.day16_adk_agent.ui import build as build_tab

ship(16, build_tab)

__all__ = ["build_tab"]
