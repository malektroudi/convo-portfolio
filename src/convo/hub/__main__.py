"""Entry point. Honours $PORT so the same command works locally and on Cloud Run."""

from __future__ import annotations

import os

from convo.core import get_logger, setup
from convo.hub.app import build_hub


def main() -> None:
    setup()
    log = get_logger("convo.hub")
    port = int(os.environ.get("PORT", "7860"))
    host = os.environ.get("HOST", "0.0.0.0")  # noqa: S104 - Cloud Run requires this
    log.info("hub starting", extra={"port": port, "host": host})
    demo = build_hub()
    demo.queue(default_concurrency_limit=4).launch(
        server_name=host,
        server_port=port,
        # Cloud Run's image has no Node runtime, and SSR needs one.
        ssr_mode=False,
        quiet=True,
    )


if __name__ == "__main__":
    main()
