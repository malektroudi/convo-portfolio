"""Day 14's tab: where the public site is, and what it cannot do.

This tab exists in the hub, which is the thing the site replaced for public
use, so it is mostly a signpost. The one job worth doing carefully is being
straight about what the static version loses — somebody reading the site has
no way to find that out except by being told.
"""

from __future__ import annotations

import gradio as gr

from convo.site.data import collect

SITE_URL = "https://malektroudi.github.io/convo-portfolio/"
REPO_URL = "https://github.com/malektroudi/convo-portfolio"

LOST = (
    (
        "Free-text retrieval",
        "Day 8 lets you type any question and watch BM25 answer it. The site "
        "has no Python at request time, so it shows the stored question set "
        "instead.",
    ),
    (
        "The dialogue inspector",
        "Day 4 steps through pages, slots and repair as you talk to it. "
        "Nothing of that survives without a server.",
    ),
    (
        "Threshold sliders over real model output",
        "Day 2 recomputes accuracy from 5,500 stored logits as you move the "
        "bar. Shipping those to a browser would be a 3 MB download for one "
        "slider.",
    ),
)

KEPT = (
    "Day 10's threshold, reimplemented in about twenty lines of JavaScript "
    "over the stored reranker scores. It reproduces the Python scorer exactly "
    "at every threshold, which a test checks by recomputing from the shipped "
    "payload and comparing against `score_threshold` directly."
)


def build() -> None:
    site = collect()
    shipped = len(site.shipped)

    gr.Markdown(
        f"### Day 14 · Deploy and demo\n"
        f"**[{SITE_URL}]({SITE_URL})** — {shipped} day pages, generated from the "
        f"committed artifacts and served for nothing.\n\n"
        f"Day 1 was built around Cloud Run, which needs a billing account even "
        f"inside its free tier, and free Gradio hosting no longer exists. The "
        f"way out was already in the architecture: every day writes a small "
        f"committed artifact, and this hub was only ever one way of rendering "
        f"them. The site renders them a second way, through code that shares "
        f"no display logic with these tabs — so when the two agree, that is "
        f"evidence rather than a copy."
    )

    with gr.Row():
        gr.Button("Open the site", link=SITE_URL, variant="primary")
        gr.Button("Read the code", link=REPO_URL)

    gr.Markdown("#### What the static version loses")
    rows = "".join(
        f"<tr><td style='padding:7px 14px 7px 0;white-space:nowrap'>"
        f"<strong>{title}</strong></td>"
        f"<td style='padding:7px 0;font-size:0.9rem;opacity:0.78'>{body}</td></tr>"
        for title, body in LOST
    )
    gr.HTML(
        f"<table style='border-collapse:collapse;font-size:0.92rem'><tbody>{rows}</tbody></table>"
    )

    gr.Markdown(f"#### What made it across\n{KEPT}")

    gr.Markdown(
        "#### Rebuild it\n"
        "```bash\n"
        "uv run python -m convo.site.build --out site\n"
        "```\n"
        "Nothing is committed: the site is generated at deploy time, so the "
        "HTML cannot drift from the artifacts it reads. CI builds it on every "
        "push as well, so a broken generator fails on the pull request rather "
        "than on the push meant to publish it."
    )
