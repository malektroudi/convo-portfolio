"""Day 12: voice, with a latency budget.

`budget`, `vad` and `results` are dependency-light and safe to import
anywhere. `speech`, `think` and `build` need the `train` and `voice` extras
and are used by the CLI only — the hub reads committed timings and plays a
committed wav, so neither torch nor onnxruntime reaches the serving image.
"""

from convo.core import ship
from convo.projects.day12_voice.ui import build

ship(12, build)

__all__ = ["build"]
