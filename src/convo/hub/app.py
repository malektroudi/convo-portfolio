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


def _landing() -> None:
    gr.Markdown(
        "# Conversational AI portfolio\n"
        "Twenty-one projects behind one hub: a grounded, evaluated, escalating "
        "support agent, built twice — once open-source, once Google-native — with "
        "the numbers for both."
    )
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
        with gr.Tabs():
            with gr.Tab("Overview"):
                _landing()
            for project in projects():
                with gr.Tab(project.label):
                    if project.builder is not None:
                        project.builder()
                    else:
                        _placeholder(project)
    return demo
