"""Generate the static site.

    uv run python -m convo.site.build --out site

The deliverable this repo has been missing. Day 1 was built around Cloud Run,
which needs a billing account even inside the free tier; free Gradio hosting
no longer exists either. What does exist, for nothing, is static hosting — and
the architecture every day has followed makes that the easy path rather than a
retreat: heavy work on a laptop produces small committed artifacts, and the
Gradio hub was only ever one way of rendering them.

So this renders them a second way. The two front ends share no display code,
which makes the site a genuine check on the hub rather than a copy of it: a
number that disagrees means one of them is reading the artifact wrong.

What the site cannot do is the part worth stating plainly. The hub lets you
type a question and watch BM25 answer it; the site has no Python at request
time, so the one interactive piece here is day 10's threshold, recomputed in
the browser from the stored scores. Everything else is a reading, not a demo.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

from convo.core import get_logger, setup
from convo.site import data, render

log = get_logger("site.build")


def findings() -> list[render.Finding]:
    """The index cards, resolved from the artifacts as the page is generated.

    This was a tuple of hand-typed strings, and the hub had a different tuple
    of hand-typed strings. Day 21 audited both: they disagreed about what the
    portfolio's best finding was, and three of the seven figures between them
    had drifted from the artifacts they described. The per-day pages were all
    correct, because they compute.

    So both front ends now read one list of claims and resolve each figure at
    build time. Which claims lead is a property of the claim, not of this
    module, so the two pages cannot disagree again.
    """
    from convo.projects.day21_capstone.claims import MEANS, leads

    cards = []
    for claim in leads():
        value = claim.value()
        cards.append(
            render.Finding(
                value if value is not None else "no artifact",
                claim.headline,
                claim.body,
                claim.day,
                claim.kind,
                MEANS[claim.kind],
            )
        )
    return cards


STATUS = (
    "<strong>How to read this.</strong> Each page carries a "
    "<em>what this does not show</em> note, because a portfolio that only "
    "reports its wins is a sales page. Every figure above is recomputed from a "
    "committed artifact when this page is generated, except the one card marked "
    "<em>testimony</em>: it describes a training run whose artifact was "
    "overwritten by the run that fixed it, so it is true and unfalsifiable at "
    "once, and day 21 argues about what to do with a number like that. The "
    "repository has the code, the tests and the full write-up, including the "
    "days where the result was that the idea did not work."
)


def _threshold_payload() -> dict | None:
    """Day 10's stored scores, reduced to what the browser slider needs.

    Three fields per question and nothing else: the reranker score, whether a
    span verified, and whether the citation came from the right article. That
    is the whole input to the decision, so the page can recompute it honestly
    rather than shipping a table of precomputed answers.
    """
    from convo.projects.day10_guardrails.results import load

    found = load()
    if found is None:
        return None
    return {
        "answerable": [
            {
                "s": round(evidence.score, 4),
                "c": bool(evidence.citation),
                "ok": bool(evidence.citation and evidence.citation.doc_id == expected),
            }
            for evidence, expected in found.answerable
        ],
        "unanswerable": [
            {"s": round(evidence.score, 4), "c": bool(evidence.citation), "ok": False}
            for evidence in found.unanswerable
        ],
    }


def _threshold_range() -> dict:
    """Where the slider starts and stops, taken from the scores themselves.

    The default is day 10's own balanced-best threshold, so the page opens on
    the setting that day argues for rather than on a number typed into a
    template.
    """
    from convo.projects.day10_guardrails.results import load

    found = load()
    scores = [e.score for e, _ in found.answerable] + [e.score for e in found.unanswerable]
    low, high = round(min(scores) - 1, 1), round(max(scores) + 1, 1)
    default = round(found.best().threshold, 2)
    # Roughly seventy stops across whatever range the scores occupy, so the
    # slider has the same feel regardless of the model behind it.
    step = round((high - low) / 72, 2) or 0.25

    # Shift the floor so the default sits exactly on a stop. A browser snaps
    # `value` to the nearest `min + n*step`, which silently moved the opening
    # position off the measured threshold — the page would have said it opens
    # on day 10's balanced best and opened 0.15 away from it. The metrics were
    # identical either way, which is exactly why it would not have been
    # noticed.
    low = round(default - step * math.ceil((default - low) / step), 2)
    return {"default": default, "low": low, "high": high, "step": step}


def generate(out: Path) -> dict:
    """Write the whole site. Returns a summary for the log and the tests."""
    site = data.collect()

    if out.exists():
        shutil.rmtree(out)
    (out / "days").mkdir(parents=True)

    payload = _threshold_payload()
    written = []

    for page in site.pages:
        if not page.shipped:
            continue
        extra = ""
        if page.day == 10 and payload is not None:
            extra = render.threshold_demo(payload, **_threshold_range())
        target = out / "days" / f"day{page.day:02d}.html"
        target.write_text(render.day_page(page, extra), encoding="utf-8")
        written.append(target)

    index = out / "index.html"
    index.write_text(render.index_page(site, findings(), STATUS), encoding="utf-8")
    written.append(index)

    # Tells GitHub Pages not to run the output through Jekyll, which would
    # otherwise ignore any file or directory beginning with an underscore.
    (out / ".nojekyll").write_text("", encoding="utf-8")

    total = sum(p.stat().st_size for p in written)
    return {
        "pages": len(written),
        "bytes": total,
        "kilobytes": total // 1024,
        "largest": max(written, key=lambda p: p.stat().st_size).name,
        "interactive": payload is not None,
    }


def main(argv: list[str] | None = None) -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("site"))
    args = parser.parse_args(argv)

    summary = generate(args.out)
    log.info("site generated", extra={"out": str(args.out), **summary})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
