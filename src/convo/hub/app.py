"""The hub: one tabbed app, 21 slots, honest about which are empty."""

from __future__ import annotations

import gradio as gr

import convo.projects  # noqa: F401  (importing registers every shipped day)
from convo.core import WEEKS, Project, by_week, projects, shipped


def _progress_line() -> str:
    done = len(shipped())
    total = len(projects())
    filled = "#" * done + "." * (total - done)
    return f"**{done} of {total} days shipped**  \n`{filled}`"


FINDINGS = (
    (
        "96.7% → 0.0%",
        "Five-shot routing, same five examples",
        "Across ten withheld intents the spread is nearly the whole range and "
        "roughly bimodal — so the 52% average means almost nothing. Whether a "
        "handful of examples will do is predictable before you collect them.",
        "#2a78d6",
        "Day 03",
    ),
    (
        "10 → 0",
        "False confirmations, pass rate unchanged",
        "A small model emitted one malformed tool call in twenty — and ten "
        "well-formed, entirely invented bookings. Schema validation passed every "
        "one. A provenance guard fixed it, and the headline score never moved.",
        "#1baf7a",
        "Day 05",
    ),
    (
        "0.092",
        "Max confidence that accuracy hid",
        "A run at 74% accuracy looked mediocre but plausible. Its top softmax "
        "never cleared 0.092 across 5,500 rows, so every threshold above 0.1 "
        "refused everything. Accuracy alone cannot show that.",
        "#eb6834",
        "Day 02",
    ),
)


def _findings() -> str:
    cards = "".join(
        f'<div style="flex:1 1 230px;min-width:0;padding:14px 16px;'
        f'border:1px solid rgba(128,128,128,0.25);border-radius:6px">'
        f'<div style="font-size:0.68rem;letter-spacing:0.05em;text-transform:uppercase;'
        f'opacity:0.55">{where}</div>'
        f'<div style="font-size:1.75rem;font-weight:600;line-height:1.15;color:{colour};'
        f'margin:2px 0 4px">{figure}</div>'
        f'<div style="font-size:0.85rem;font-weight:600;margin-bottom:6px">{headline}</div>'
        f'<div style="font-size:0.8rem;opacity:0.72;line-height:1.5">{body}</div></div>'
        for figure, headline, body, colour, where in FINDINGS
    )
    return f'<div style="display:flex;gap:12px;flex-wrap:wrap">{cards}</div>'


def _landing() -> None:
    gr.Markdown(
        "# Conversational AI portfolio\n"
        "A booking assistant built four different ways — a fine-tuned classifier, "
        "an embedding router, a deterministic state machine and a tool-calling "
        "agent — then measured against each other on one eval suite, with a CI "
        "gate that blocks on a safety regression."
    )

    gr.Markdown("### Three things it found")
    gr.Markdown("_Every number below came out of a run in this repo, not from a blog post._")
    gr.HTML(_findings())

    gr.Markdown(
        "### What runs where\n"
        "Training happens on a laptop and produces small files. This app reads "
        "those files and carries **no ML framework at all** — which is why the "
        "image is 588 MB and can run free, and why every slider here is instant "
        "rather than re-running a model.\n\n"
        "```\n"
        "laptop (torch)  ──►  committed artifacts (9 MB)  ──►  this hub (no torch)\n"
        "```"
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
