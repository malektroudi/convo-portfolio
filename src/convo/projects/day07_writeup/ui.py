"""Day 7's tab: what week 1 actually produced, and what I got wrong.

No new machinery. The day's output is the README's opening, the two
architecture diagrams and this page — the part of a portfolio that decides
whether anyone reads the rest.
"""

from __future__ import annotations

import gradio as gr

from convo.core import projects, shipped

REUSE = (
    (
        "Day 3 → core",
        "Day 3 needed day 2's threshold and confusion maths over cosine "
        "similarities instead of softmax probabilities. Rather than copy it, the "
        "maths moved into <code>core/classification.py</code>, which takes any "
        "score array whose row maximum is a comparable confidence. Day 2 became "
        "a thin adapter and its numbers were verified unchanged — that is what "
        "makes the two approaches honestly comparable rather than "
        "comparable-looking.",
    ),
    (
        "Day 4 → day 5",
        "Day 5 needed to know whether a model had invented a booking detail. "
        "Day 4's deterministic parsers were already the authority on what the "
        "guest had actually said, so they became the oracle: the rule-based "
        "component vouches for the generative one. That is the real answer to "
        "“deterministic or generative?” — both, and here is the seam.",
    ),
    (
        "Day 5 → day 6",
        "The eval harness reuses day 5 twice: its twelve scenarios became the "
        "suite's core, imported rather than copied so there is one definition of "
        "each conversation, and its grounding guard became a safety scorer. When "
        "a naive version of that scorer produced three false positives, the fix "
        "was to delete it and use day 5's, not to write a second worse oracle.",
    ),
)

REGRETS = (
    (
        "Check the deployment constraint first",
        "Day 1 was built around Cloud Run — containerised, sized for the free "
        "tier, deploy scripts written. Cloud Run needs a billing account even "
        "for free usage, and this project has none. The container is real and "
        "verified in CI, but the headline deliverable — a link you can put in an "
        "application — still does not exist. That question belonged on day 1.",
    ),
    (
        "Sanity-check a training run after one epoch, not eight",
        "Day 2's first run used a BERT-base recipe on a 6-layer encoder with a "
        "150-way head. Forty-one minutes produced a model whose maximum "
        "confidence was 0.092. Printing the confidence distribution after epoch "
        "one would have caught it in five minutes. The check exists now; it "
        "should have existed first.",
    ),
    (
        "Write the eval suite from the task, not from a working bot",
        "Day 6's scenarios were written after day 4 worked, and day 4 scores "
        "100% on them. Partly honest — a state machine genuinely cannot invent a "
        "slot — and partly circular. The mutation checks that break a bot "
        "deliberately and assert the right dimension notices are a patch for "
        "this, not a substitute for writing the suite first.",
    ),
    (
        "A 0.5B model cannot settle an architectural question",
        "Day 5's agent scored 8.3%, which says a great deal about Qwen2.5-0.5B "
        "on CPU and very little about tool-calling agents. The shape of the "
        "finding generalises — schema validation does not constrain semantics — "
        "but the numbers do not. Re-running it against a Gemini Flash free-tier "
        "key would make the comparison worth far more.",
    ),
    (
        "Gradio was right for one demo and wrong for twenty-one",
        "Several tabs build HTML as strings inside Python because Gradio has no "
        "component for what the page needs. The data is already JSON artifacts; "
        "a small static front end reading those same files would be cleaner, "
        "cheaper to host, and would sidestep free Gradio hosting no longer "
        "existing. The artifacts-as-interface split is right — what renders them "
        "is not load-bearing and should have been decided later.",
    ),
)


def _cards(items, colour: str) -> str:
    return "".join(
        f'<div style="border-left:3px solid {colour};padding:2px 0 2px 14px;'
        f'margin:14px 0">'
        f'<div style="font-weight:600;margin-bottom:4px">{title}</div>'
        f'<div style="font-size:0.88rem;opacity:0.78;line-height:1.6">{body}</div>'
        f"</div>"
        for title, body in items
    )


def build() -> None:
    done = len(shipped())
    total = len(projects())

    gr.Markdown(
        f"### Day 07 · Write it up\n"
        f"No new machinery today. Week 1 closed with **{done} of {total} days** "
        f"shipped, 362 tests, and CI green on Linux — including a Docker build "
        f"that had silently never worked until it was first exercised.\n\n"
        f"Half a day of writing is worth more than a seventh project nobody can "
        f"read, which is the entire argument for this day existing."
    )

    gr.Markdown(
        "#### The days are not independent demos\n"
        "Each one reuses what the last built, and the reuse is the architecture."
    )
    gr.HTML(_cards(REUSE, "#2a78d6"))

    gr.Markdown(
        "#### What I would do differently\n"
        "Written while it is still inconvenient to admit, rather than after it "
        "stops costing anything."
    )
    gr.HTML(_cards(REGRETS, "#eb6834"))

    gr.Markdown(
        "#### Where it stands\n"
        "The container is verified in CI on every push — it boots, serves, runs "
        "as a non-root user, and a dedicated job fails the build if `torch` ever "
        "leaks into the 588 MB image. What is missing is a billing account, not "
        "code. Run it locally with `uv run python -m convo.hub`."
    )
