"""Day 6's tab: the scoreboard CI actually blocks on.

Three dimensions with three different bars, the diff against the committed
baseline, and every failing transcript one click away. The page is built from
the same `EvalRun` the gate reads, so what you see here is what fails the
build — not a prettier parallel report that drifts from it.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from convo.core.evals import EFFICIENCY, SAFETY, TASK, EvalRun
from convo.projects.day06_eval_harness.gate import THRESHOLDS, baseline_path
from convo.projects.day06_eval_harness.harness import engine_runner, run_suite
from convo.projects.day06_eval_harness.suite import SUITE, tags

DIMENSION_COLOUR = {TASK: "#2a78d6", SAFETY: "#1baf7a", EFFICIENCY: "#eb6834"}
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

DIMENSION_NOTE = {
    TASK: "Did it do what the guest asked, with the right details?",
    SAFETY: "Did it avoid lying about state, inventing facts or echoing secrets?",
    EFFICIENCY: "Did it get there without wasting turns or model calls?",
}


def _run() -> EvalRun:
    return run_suite("day06-engine", engine_runner, SUITE)


def _tiles(run: EvalRun) -> str:
    cells = []
    for dimension in (TASK, SAFETY, EFFICIENCY):
        rate = run.pass_rate_in(dimension)
        bar = THRESHOLDS[dimension]
        ok = rate >= bar
        cells.append(
            (
                dimension.title(),
                f"{rate:.0%}",
                f"bar {bar:.0%} · {'passing' if ok else 'BELOW BAR'}",
                DIMENSION_COLOUR[dimension] if ok else CRITICAL,
                DIMENSION_NOTE[dimension],
            )
        )
    tiles = "".join(
        f'<div style="flex:1 1 190px;min-width:0;padding:12px 14px;'
        f'border:1px solid rgba(128,128,128,0.25);border-radius:6px">'
        f'<div style="font-size:0.7rem;letter-spacing:0.04em;text-transform:uppercase;'
        f'opacity:0.65">{title}</div>'
        f'<div style="font-size:1.8rem;font-weight:600;line-height:1.2;color:{colour}">'
        f"{value}</div>"
        f'<div style="font-size:0.74rem;opacity:0.65">{note}</div>'
        f'<div style="font-size:0.72rem;opacity:0.5;margin-top:4px">{blurb}</div></div>'
        for title, value, note, colour, blurb in cells
    )
    return f'<div style="display:flex;gap:10px;flex-wrap:wrap">{tiles}</div>'


def _scoreboard(run: EvalRun) -> pd.DataFrame:
    rows = []
    for case in run.cases:
        row = {"scenario": case.case_id, "tests": ", ".join(case.tags)}
        for dimension in (TASK, SAFETY, EFFICIENCY):
            if not case.scored_in(dimension):
                row[dimension] = "—"
            else:
                row[dimension] = "pass" if case.passed_in(dimension) else "FAIL"
        row["notes"] = case.judge_rationale
        rows.append(row)
    return pd.DataFrame(rows)


def _baseline_diff(run: EvalRun) -> str:
    path = baseline_path("engine")
    if not path.exists():
        return (
            "<p style='opacity:0.6;font-size:0.85rem'>No committed baseline — "
            "the gate is running on thresholds alone.</p>"
        )
    diff = run.compare(EvalRun.load(path))
    if diff.clean:
        return (
            f"<p style='font-size:0.88rem'><strong style='color:{GOOD}'>No regression"
            f"</strong> against the committed baseline — {len(diff.unchanged)} cases "
            f"unchanged, {len(diff.fixes)} improved, {len(diff.added)} new.</p>"
        )
    lines = []
    if diff.regressions:
        lines.append(f"cases now failing: <code>{', '.join(diff.regressions)}</code>")
    for dimension in diff.regressed_dimensions():
        lines.append(
            f"<strong>{dimension}</strong> regressed on "
            f"<code>{', '.join(diff.by_dimension[dimension])}</code>"
        )
    body = "<br>".join(lines)
    return (
        f"<p style='font-size:0.88rem'><strong style='color:{CRITICAL}'>Regression"
        f"</strong> against the committed baseline.<br>{body}</p>"
    )


def _failures(run: EvalRun) -> str:
    if not run.failures:
        return (
            "<p style='opacity:0.6;font-size:0.85rem'>Nothing failing. Break a scorer "
            "or a bot and the offending transcript appears here.</p>"
        )
    blocks = []
    for case in run.failures:
        reasons = "".join(
            f"<li style='font-size:0.8rem'><code>{a.dimension}</code> "
            f"<strong>{a.name}</strong> — {a.detail}</li>"
            for a in case.assertions
            if not a.passed
        )
        lines = "".join(
            f"<div style='margin:2px 0'><span style='opacity:0.5;font-size:0.72rem;"
            f"display:inline-block;width:32px'>"
            f"{'user' if m['role'] == 'user' else 'bot'}</span>{m['content']}</div>"
            for m in case.transcript
        )
        blocks.append(
            f"<details style='margin:8px 0;border:1px solid rgba(128,128,128,0.25);"
            f"border-radius:6px;padding:10px 12px'>"
            f"<summary style='cursor:pointer;font-weight:600'>{case.case_id}</summary>"
            f"<ul style='margin:8px 0'>{reasons}</ul>"
            f"<div style='font-size:0.85rem;margin-top:6px'>{lines}</div></details>"
        )
    return "".join(blocks)


def build() -> None:
    run = _run()

    gr.Markdown(
        f"### Day 06 · Conversational eval harness\n"
        f"**{len(SUITE)} scripted conversations**, scored on three dimensions with three "
        f"different bars, judged against a committed baseline. This exact run is what "
        f"blocks CI — the gate reads the same `EvalRun` this page renders.\n\n"
        f"Day 5 is why the dimensions exist. A guard there eliminated ten false "
        f"confirmations and the pass rate did not move, because one number scores "
        f"*booked the wrong table* and *booked nothing* identically. Safety is held to "
        f"**{THRESHOLDS[SAFETY]:.0%}** and task to {THRESHOLDS[TASK]:.0%}: a bot that "
        f"fails to help is recoverable, a bot that lies is not.\n\n"
        f"Tags in the suite: `{'`, `'.join(tags())}`"
    )

    gr.HTML(_tiles(run))
    gr.Markdown("#### Against the committed baseline")
    gr.HTML(_baseline_diff(run))

    gr.Markdown("#### Every scenario")
    gr.Dataframe(_scoreboard(run), interactive=False, wrap=True)

    gr.Markdown(
        "#### Failing transcripts\n"
        "The harness is only useful if a failure tells you what to fix, so each one "
        "opens to the assertion that broke and the conversation that broke it."
    )
    gr.HTML(_failures(run))

    with gr.Accordion("How to run it yourself", open=False):
        gr.Code(
            "# score the suite and fail on a regression or a threshold breach\n"
            "uv run python -m convo.projects.day06_eval_harness.gate\n\n"
            "# after a deliberate, reviewed behaviour change\n"
            "uv run python -m convo.projects.day06_eval_harness.gate --update-baseline",
            language="shell",
            label="",
        )
