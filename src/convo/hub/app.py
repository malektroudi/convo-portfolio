"""The hub: one tabbed app, 21 slots, honest about which are empty.

The overview's findings used to be a tuple of hand-typed strings here, and the
static site had a different hand-typed tuple of its own. Day 21 audited both:
the two disagreed about what the portfolio's best result was, and three of the
seven figures between them had gone stale against the artifacts they described.

They now come from `day21_capstone.claims`, resolved on every launch. Nothing
in this module may contain a metric, and a test enforces that.
"""

from __future__ import annotations

import gradio as gr

import convo.projects  # noqa: F401  (importing registers every shipped day)
from convo.core import WEEKS, Project, by_week, projects, shipped
from convo.projects.day21_capstone import claims, ledger


def _progress_line() -> str:
    done = len(shipped())
    total = len(projects())
    filled = "#" * done + "." * (total - done)
    return f"**{done} of {total} days shipped**  \n`{filled}`"


# A colour per kind of number rather than per card. The reader can see at a
# glance that one of these figures is a different sort of thing from the rest,
# which is the point day 21 landed on.
TONE = {
    claims.DERIVED: "#2a78d6",
    claims.PINNED: "#1baf7a",
    claims.TESTIMONY: "#eb6834",
    claims.UNSTORED: "#c98a16",
}


def _findings() -> str:
    cards = []
    for claim in claims.leads():
        figure = claim.value()
        cards.append(
            f'<div style="flex:1 1 230px;min-width:0;padding:14px 16px;'
            f'border:1px solid rgba(128,128,128,0.25);border-radius:6px">'
            f'<div style="font-size:0.68rem;letter-spacing:0.05em;'
            f'text-transform:uppercase;opacity:0.55">Day {claim.day:02d} '
            f"&middot; {claim.kind}</div>"
            f'<div style="font-size:1.75rem;font-weight:600;line-height:1.15;'
            f'color:{TONE[claim.kind]};margin:2px 0 4px">'
            f"{figure if figure is not None else 'no artifact'}</div>"
            f'<div style="font-size:0.85rem;font-weight:600;margin-bottom:6px">'
            f"{claim.headline}</div>"
            f'<div style="font-size:0.8rem;opacity:0.72;line-height:1.5">'
            f"{claim.body}</div></div>"
        )
    return f'<div style="display:flex;gap:12px;flex-wrap:wrap">{"".join(cards)}</div>'


def _landing() -> None:
    gr.Markdown(
        "# Conversational AI portfolio\n"
        "A booking assistant built four different ways — a fine-tuned classifier, "
        "an embedding router, a deterministic state machine and a tool-calling "
        "agent — then measured against each other on one eval suite, with a CI "
        "gate that blocks on a safety regression."
    )

    kinds = sorted({c.kind for c in claims.leads()})
    gr.Markdown(f"### What it found ({len(claims.leads())} findings)")
    gr.Markdown(
        "_Every figure below is resolved from a committed artifact as this page "
        "loads, except where the card says otherwise. Day 21 audited the "
        "hand-typed list that used to be here and found three stale numbers and "
        "one with no artifact at all, so the list stopped being hand-typed._  \n"
        + " · ".join(f"**{kind}** — {claims.MEANS[kind]}" for kind in kinds)
    )
    gr.HTML(_findings())

    gr.Markdown(
        "### What runs where\n"
        "Training happens on a laptop and produces small files. This app reads "
        "those files and carries **no ML framework at all** — which is why it "
        "can run free, and why every slider here is instant rather than "
        "re-running a model.\n\n"
        "```\n"
        "laptop (torch)  ──►  committed artifacts  ──►  this hub (no torch)\n"
        "```\n"
        + "\n".join(f"- **{stage.name}** — {stage.size}" for stage in ledger.pipeline())
        + "\n\nThe middle figure is the bargain, and it is measured here rather "
        "than remembered — the version this page used to carry was typed in when "
        "two days had artifacts and was never updated. Day 21 has the audit.\n\n"
        + f"CI refuses to find {', '.join(f'`{m}`' for m in ledger.BLOCKED_IN_IMAGE)} "
        "inside the built image. That list is the architecture stated as a "
        "prohibition."
    )

    gr.Markdown("### Progress")
    gr.Markdown(_progress_line())
    for week in WEEKS:
        days = by_week().get(week.number, [])
        done = sum(p.status == "shipped" for p in days)
        rows = "\n".join(
            f"- **{p.day:02d}** {p.title} — {'shipped' if p.status == 'shipped' else 'planned'}"
            for p in days
        )
        gr.Markdown(
            f"### Week {week.number} · {week.title} ({done}/{len(days)})\n_{week.note}_\n\n{rows}"
        )


def _placeholder(project: Project) -> None:
    gr.Markdown(
        f"### Day {project.day:02d} · {project.title}\n"
        f"{project.brief}\n\n"
        f"---\n"
        f"Not built yet. When it is, add `src/convo/projects/day"
        f"{project.day:02d}_{project.slug.replace('-', '_')}.py` with a `build()` "
        f"and import it in `convo/projects/__init__.py`."
    )


def build_hub() -> gr.Blocks:
    # analytics_enabled=False keeps Gradio from calling home on every launch.
    # The Dockerfile sets the env var too; this covers local runs as well.
    with gr.Blocks(
        title="Conversational AI portfolio",
        fill_width=True,
        analytics_enabled=False,
    ) as demo:
        # Nested by week on purpose: 21 tabs in one strip overflow the bar and
        # every day past the third becomes unreachable by clicking. Four
        # top-level tabs with seven each fits at laptop width.
        grouped = by_week()
        with gr.Tabs():
            with gr.Tab("Overview"):
                _landing()
            for week in WEEKS:
                days = grouped.get(week.number, [])
                done = sum(p.status == "shipped" for p in days)
                with gr.Tab(f"Week {week.number} ({done}/{len(days)})"):
                    with gr.Tabs():
                        for project in days:
                            with gr.Tab(project.tab_label):
                                if project.builder is not None:
                                    project.builder()
                                else:
                                    _placeholder(project)
    return demo
