"""Day 18's tab: two evaluators on the same conversations, and what each one misses.

Day 6's harness asks **did it end up right** — did it book, is the date
correct, did it invent anything. ADK's `TrajectoryEvaluator` asks **did it go
the right way** — the tools it called, turn by turn, against the tools a
correct bot calls. Both are defensible. They are not the same question.

The answer this run gives is not "one of them is better". It is that each is
structurally blind to a class of failure the other catches, and the tab is
ordered to make that case: the agreement rate, then the failure only ADK sees,
then the failure only day 6 sees, then who the spec can actually adjudicate
(nobody), and finally what a free tier can even run.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from convo.core.evals import EFFICIENCY, SAFETY, TASK
from convo.projects.day18_eval_gate.adk_eval import WHY_NOT_ROUGE
from convo.projects.day18_eval_gate.compare import Adjudication, Agreement
from convo.projects.day18_eval_gate.results import Comparison, load
from convo.projects.day18_eval_gate.trajectory import BOOK_TOOL

GAP = "\n\n"
"""A blank line between Markdown paragraphs, as a constant.

Written this way because every scripted edit to this file that tried to put
an escaped newline inside an f-string had the escape eaten somewhere between
the shell and the editor, twice producing a file that would not parse."""

HARNESS = "#2a78d6"
ADK = "#9450d6"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
WARN = "#c98a16"

UNAVAILABLE = (
    ("TrajectoryEvaluator", "works", "deterministic, no model, no key"),
    ("FinalResponseMatchV1", "left out", "needs rouge_score; no reference replies exist"),
    ("FinalResponseMatchV2", "out of reach", "LLM-as-judge: a model call per case"),
    ("HallucinationsV1", "out of reach", "LLM-as-judge, same"),
    ("SafetyEvaluator", "cannot install", "imports agentplatform, not on PyPI"),
    ("LocalEvalService", "cannot install", "imports agentplatform"),
    ("MetricEvaluatorRegistry", "cannot install", "imports agentplatform"),
    ("multi_turn_* evaluators", "cannot install", "import agentplatform"),
)

MISSING = """
### Day 18 · Two evaluators, one suite

No comparison artifact found.

```bash
uv sync
uv pip install --no-deps google-adk && uv pip install --upgrade pydantic
# GOOGLE_API_KEY in .env
uv run python -m convo.projects.day18_eval_gate.build
```

That runs day 6's suite through day 16's ADK agent once, recording the tool
trajectory per turn, then scores it twice — day 6's assertions and ADK's
`TrajectoryEvaluator` — and stores both. Reload the hub afterwards.
"""


def _books(stored, which: str) -> int:
    turns = stored.actual if which == "actual" else stored.expected
    return sum(turn.count(BOOK_TOOL) for turn in turns)


def _tiles(c: Comparison) -> str:
    cells = [
        ("Day 6's harness", f"{c.harness.pass_rate:.1%}", "did it end up right", HARNESS),
        (
            "ADK, turn-aligned",
            f"{c.trajectory_pass_rate:.1%}",
            "right tools, right turns",
            CRITICAL if c.trajectory_pass_rate < 0.5 else WARN,
        ),
        ("ADK, turns collapsed", f"{c.flat_pass_rate:.1%}", "right tools, any turn", ADK),
        (
            "They agree on",
            f"{c.agreement_rate:.1%}",
            f"{len(c.disagreements)} of {len(c.rows)} disagree",
            GOOD if c.agreement_rate > 0.8 else WARN,
        ),
    ]
    if c.variance is not None:
        cells.append(
            (
                "Run-to-run",
                f"{c.variance:+.1%}",
                "same suite and model, vs day 16",
                GOOD if abs(c.variance) < 0.02 else WARN,
            )
        )
    tiles = "".join(
        f'<div style="flex:1 1 160px;min-width:0;padding:12px 14px;'
        f'border:1px solid rgba(128,128,128,0.25);border-radius:6px">'
        f'<div style="font-size:0.7rem;letter-spacing:0.04em;text-transform:uppercase;'
        f'opacity:0.65">{title}</div>'
        f'<div style="font-size:1.8rem;font-weight:600;line-height:1.2;color:{colour}">'
        f"{value}</div>"
        f'<div style="font-size:0.74rem;opacity:0.6">{note}</div></div>'
        for title, value, note, colour in cells
    )
    return f'<div style="display:flex;gap:10px;flex-wrap:wrap">{tiles}</div>'


def _writes_table(c: Comparison) -> str:
    rows = "".join(
        f"<tr><td style='padding:4px 12px 4px 0;font-family:ui-monospace,monospace;"
        f"font-size:0.84rem'>{t.scenario_id}</td>"
        f"<td style='padding:4px 12px;text-align:right;color:{CRITICAL};"
        f"font-variant-numeric:tabular-nums'>{_books(t, 'actual')}</td>"
        f"<td style='padding:4px 12px;text-align:right;opacity:0.7'>"
        f"{_books(t, 'expected')}</td>"
        f"<td style='padding:4px 12px;color:{GOOD}'>pass</td></tr>"
        for t in c.redundant_writes
    )
    return (
        "<div style='overflow-x:auto'><table style='border-collapse:collapse;"
        "font-size:0.92rem'><thead><tr style='border-bottom:1px solid "
        "rgba(128,128,128,0.35)'>"
        "<th style='text-align:left;padding:4px 12px 4px 0'>scenario</th>"
        "<th style='text-align:right;padding:4px 12px'>bookings written</th>"
        "<th style='text-align:right;padding:4px 12px'>should be</th>"
        "<th style='text-align:left;padding:4px 12px'>day 6 says</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def _agreement_table(c: Comparison) -> str:
    rows = "".join(
        f"<tr><td style='padding:5px 12px 5px 0'>{agreement.value}</td>"
        f"<td style='padding:5px 12px;text-align:right;"
        f"font-variant-numeric:tabular-nums'>{c.count(agreement)}</td></tr>"
        for agreement in Agreement
    )
    adjudged = "".join(
        f"<tr><td style='padding:5px 12px 5px 0'>{call.value}</td>"
        f"<td style='padding:5px 12px;text-align:right'>{len(c.adjudged(call))}</td></tr>"
        for call in Adjudication
    )
    return f"""
<div style="display:flex;gap:28px;flex-wrap:wrap">
<div style="overflow-x:auto">
<table style="border-collapse:collapse;font-size:0.92rem">
  <thead><tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
    <th style="text-align:left;padding:4px 12px 4px 0">Verdict pairs</th>
    <th style="text-align:right;padding:4px 12px">cases</th>
  </tr></thead><tbody>{rows}</tbody></table>
</div>
<div style="overflow-x:auto">
<table style="border-collapse:collapse;font-size:0.92rem">
  <thead><tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
    <th style="text-align:left;padding:4px 12px 4px 0">Who the spec backs</th>
    <th style="text-align:right;padding:4px 12px">cases</th>
  </tr></thead><tbody>{adjudged}</tbody></table>
</div>
</div>
"""


def _spec_table(c: Comparison) -> str:
    def cell(value: float) -> str:
        return (
            f"<td style='padding:5px 12px;text-align:right;"
            f"font-variant-numeric:tabular-nums'>{value:.1%}</td>"
        )

    return f"""
<div style="overflow-x:auto">
<table style="border-collapse:collapse;font-size:0.92rem">
  <thead><tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
    <th style="text-align:left;padding:4px 12px 4px 0">Matches the spec's verdict</th>
    <th style="text-align:right;padding:4px 12px;color:{HARNESS}">day 6's harness</th>
    <th style="text-align:right;padding:4px 12px;color:{ADK}">ADK trajectory</th>
  </tr></thead>
  <tbody><tr><td style="padding:5px 12px 5px 0">on the outcome</td>
    {cell(c.spec_agreement.get("harness", 0.0))}
    {cell(c.spec_agreement.get("adk", 0.0))}
  </tr></tbody>
</table>
</div>
<p style="font-size:0.8rem;opacity:0.7;margin-top:8px">
  <strong>Not a quality score, and it does not pick the gate.</strong> The spec
  verdict is about the outcome and ADK's metric is deliberately about the
  route, so it is penalised here for answering the question it was asked. An
  earlier version of this page used these two numbers to choose which
  evaluator should gate &mdash; which guaranteed day 6's would win, since day
  6's assertions and the spec read the same fields. The choice now rests on
  the adjudications above, where the spec can only convict an evaluator it
  actually contradicts.
</p>
"""


def _disagreements(c: Comparison) -> pd.DataFrame:
    rows = []
    for row in c.disagreements:
        stored = c.trajectory(row.scenario_id)
        rows.append(
            {
                "scenario": row.scenario_id,
                "day 6": "pass" if row.harness_ok else "FAIL",
                "ADK": "pass" if row.adk_ok else "FAIL",
                "score": f"{row.adk_score:.2f}",
                "same tools, other turn": "yes" if stored and stored.only_timing else "no",
                "why": row.adk_detail or row.harness_detail,
            }
        )
    return pd.DataFrame(rows)


def _trajectory_view(c: Comparison, scenario_id: str) -> str:
    stored = c.trajectory(scenario_id)
    row = c.row(scenario_id)
    if stored is None or row is None:
        return ""

    turns = max(len(stored.actual), len(stored.expected))
    cells = []
    for index in range(turns):
        got = stored.actual[index] if index < len(stored.actual) else ()
        want = stored.expected[index] if index < len(stored.expected) else ()
        colour = GOOD if got == want else CRITICAL
        cells.append(
            f"<tr><td style='padding:4px 10px 4px 0;opacity:0.5;font-size:0.8rem'>"
            f"turn {index}</td>"
            f"<td style='padding:4px 10px;font-family:ui-monospace,monospace;"
            f"font-size:0.84rem'>{', '.join(want) or '&mdash;'}</td>"
            f"<td style='padding:4px 10px;font-family:ui-monospace,monospace;"
            f"font-size:0.84rem;color:{colour}'>{', '.join(got) or '&mdash;'}</td></tr>"
        )
    verdicts = (
        f"<p style='font-size:0.88rem;margin:0 0 10px'>"
        f"<strong style='color:{HARNESS}'>day 6</strong>: "
        f"{'pass' if row.harness_ok else 'FAIL'}"
        f"{' &mdash; ' + row.harness_detail if row.harness_detail else ''}<br>"
        f"<strong style='color:{ADK}'>ADK</strong>: "
        f"{'pass' if row.adk_ok else 'FAIL'} at {row.adk_score:.2f}"
        f"{' &mdash; ' + row.adk_detail if row.adk_detail else ''}<br>"
        f"<span style='opacity:0.7'>the spec says the outcome was "
        f"{'correct' if row.spec_ok else 'wrong'}; {row.adjudication.value}</span>"
        f"</p>"
    )
    return (
        verdicts + "<div style='overflow-x:auto'><table style='border-collapse:collapse'>"
        "<thead><tr style='border-bottom:1px solid rgba(128,128,128,0.35)'>"
        "<th style='text-align:left;padding:3px 10px 3px 0;font-size:0.76rem;"
        "text-transform:uppercase;opacity:0.6'></th>"
        "<th style='text-align:left;padding:3px 10px;font-size:0.76rem;"
        "text-transform:uppercase;opacity:0.6'>day 4's engine would call</th>"
        "<th style='text-align:left;padding:3px 10px;font-size:0.76rem;"
        "text-transform:uppercase;opacity:0.6'>ADK called</th>"
        f"</tr></thead><tbody>{''.join(cells)}</tbody></table></div>"
    )


def _refresh(scenario_id: str):
    c = load()
    return "" if c is None else _trajectory_view(c, scenario_id)


def _availability() -> str:
    colour = {"works": GOOD, "left out": WARN, "out of reach": WARN, "cannot install": CRITICAL}
    rows = "".join(
        f"<tr><td style='padding:5px 12px 5px 0;font-family:ui-monospace,monospace;"
        f"font-size:0.84rem'>{name}</td>"
        f"<td style='padding:5px 12px;color:{colour[state]}'>{state}</td>"
        f"<td style='padding:5px 12px;opacity:0.75;font-size:0.88rem'>{why}</td></tr>"
        for name, state, why in UNAVAILABLE
    )
    return (
        "<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:0.92rem'>"
        "<thead><tr style='border-bottom:1px solid rgba(128,128,128,0.35)'>"
        "<th style='text-align:left;padding:4px 12px 4px 0'>ADK evaluator</th>"
        "<th style='text-align:left;padding:4px 12px'>here</th>"
        "<th style='text-align:left;padding:4px 12px'>why</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def _dimensions(c: Comparison) -> str:
    rows = "".join(
        f"<tr><td style='padding:5px 12px 5px 0'>{dimension}</td>"
        f"<td style='padding:5px 12px;text-align:right'>"
        f"{c.harness.pass_rate_in(dimension):.1%}</td></tr>"
        for dimension in (TASK, SAFETY, EFFICIENCY)
    )
    return (
        "<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:0.92rem'>"
        "<thead><tr style='border-bottom:1px solid rgba(128,128,128,0.35)'>"
        "<th style='text-align:left;padding:4px 12px 4px 0'>This run, day 6's dimensions</th>"
        "<th style='text-align:right;padding:4px 12px'>rate</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def build() -> None:
    c = load()
    if c is None:
        gr.Markdown(MISSING)
        return

    partial = (
        f" The free tier stopped the run after {c.scenarios} of {c.of_suite}, so "
        f"every figure here is over those {c.scenarios}."
        if c.partial
        else ""
    )
    gr.Markdown(
        f"### Day 18 · Two evaluators, one suite\n"
        f"Day 6's harness asks **did it end up right**. ADK's "
        f"`TrajectoryEvaluator` asks **did it go the right way** — the tools it "
        f"called, turn by turn. Both are defensible; they are not the same "
        f"question.\n\n"
        f"One run of day 16's agent on `{c.model}`, scored both ways.{partial} "
        f"The expected trajectory is **derived from day 4's state machine**, not "
        f"hand-written — the same oracle day 5's provenance guard used, and for "
        f"the same reason: it has no model in it, so it cannot be talked round. "
        f"It agrees with every scenario's own label, which the build checks "
        f"before spending a single model call."
    )

    gr.HTML(_tiles(c))

    if not c.rescorable:
        gr.Markdown(
            "> **Day 6's column predates the fix this day caused.** The bug is "
            "described below: `score_safety` read *“I'll get that table "
            "reserved for you”* as a claim that a booking existed. It wrongly "
            "failed `never_books_without_details` in this run, so day 6's "
            "figures above are one scenario pessimistic — **93.5%** rather "
            "than 90.3%, on both the overall rate and safety.\n\n"
            "> They are not simply recomputed here because this artifact kept "
            "the scored verdicts and not the outcomes behind them, which is "
            "the second thing this day fixed. `build --rescore` now re-applies "
            "the current scorers with no model call, and the build writes the "
            "outcomes it needs — so a re-run refreshes these numbers and "
            "removes this note. The free tier's daily quota is why that re-run "
            "has not happened yet; ADK's column and every conclusion below are "
            "unaffected, since the trajectory metric does not use day 6's "
            "scorers."
        )

    failures = sum(1 for t in c.trajectories if not t.passed)
    if c.only_timing and failures:
        gr.Markdown(
            f"#### First, the part that is only a difference of opinion\n"
            f"Turn-aligned matching fails **{failures} of {len(c.trajectories)}** "
            f"conversations, and **{c.only_timing}** of those "
            f"({c.timing_only_share:.0%}) are one thing: the right tools, in the "
            f"right order, one turn early.\n\n"
            f"Day 4's engine collects four slots, moves to a **confirm** page, "
            f"and books when the guest says yes. The ADK agent books the moment "
            f"it holds four slots — it never waits to be confirmed. Whether "
            f"*that* is a defect is a product decision nobody wrote down, which "
            f"is why the adjudication below calls it undecided.\n\n"
            f"What follows from it is not a judgement call."
        )

    if c.redundant_writes:
        writes = c.redundant_writes
        worst = max(writes, key=lambda t: _books(t, "actual"))
        gr.Markdown(
            f"#### The finding with consequences: it books more than once\n"
            f"**{len(writes)} of {len(c.trajectories)}** conversations call "
            f"`{BOOK_TOOL}` more times than a correct bot calls it. "
            f"`{worst.scenario_id}` calls it **{_books(worst, 'actual')} times** "
            f"where one is right.\n\n"
            f"These are the repair scenarios. The guest gives the details, the "
            f"agent books immediately, the guest corrects one of them — and the "
            f"agent books again. Day 4's engine cannot do this: it holds the "
            f"slots, confirms once, writes once.\n\n"
            f"**Every one of these passes day 6's suite.** The final reservation "
            f"is correct, and the final state is all an outcome assertion reads. "
            f"A real booking system would see a table held, released and "
            f"re-held — or, with no idempotency key, two reservations. That is "
            f"not a quibble about confirmation turns; it is a bot writing to a "
            f"database more often than it means to, and only one of these two "
            f"evaluators can see it."
        )
        gr.HTML(_writes_table(c))

    adk_only = [r for r in c.rows if r.agreement is Agreement.ADK_ONLY]
    if adk_only:
        names = ", ".join(f"`{r.scenario_id}`" for r in adk_only)
        gr.Markdown(
            f"#### And one case the other way, where day 6 is simply wrong\n"
            f"{names} is the only conversation ADK's trajectory passes and day "
            f"6 fails — and reading it is what turned day 18 from a tidy result "
            f"into a correction.\n\n"
            f"The agent called no tool, which is exactly what the expected "
            f"trajectory says. Day 6 failed it for *no false confirmation*, "
            f"which sounded like the bot had lied about a booking. It had not. "
            f"What it actually said was **“I'll get that table reserved for "
            f"you”** — a future-tense offer, with no booking claimed and none "
            f"made.\n\n"
            f"Day 6's check is a substring match on "
            f'`("booked", "confirmed", "reserved", "all set", "see '
            f'you")`. The word *reserved* appeared, the scenario must not '
            f"book, so it fired. **It is a false positive in my own safety "
            f"scorer**, and the trajectory metric was right.\n\n"
            f"That is still an argument for running both evaluators, just not "
            f"the one I expected to be making. One caught a bot writing to the "
            f"booking system too often; the other caught a scorer convicting a "
            f"bot that had done nothing wrong. Neither would have been found by "
            f"the other, and the only reason this one was found at all is that "
            f"a second opinion disagreed and the transcript got read."
        )

    gr.Markdown(
        "#### How often they agree, and who the spec can actually adjudicate\n"
        "Collapsing turn boundaries is what separates *wrong tools* from *right "
        "tools at the wrong time*. Without that split the headline is “ADK's "
        "evaluator rejects most of the suite”, which is true and useless."
    )
    gr.HTML(_agreement_table(c))

    undecided = len(c.adjudged(Adjudication.UNDECIDED))
    if undecided and undecided == len(c.disagreements):
        gr.Markdown(
            f"The spec settles **none** of the {len(c.disagreements)} "
            f"disagreements — all {undecided} come back undecided. Every one is "
            f"about the route, and the scenarios only declare outcomes.\n\n"
            f"That is a finding about the suite rather than the evaluators: "
            f"**day 6's suite is underspecified about process.** It says what "
            f"state a conversation must end in, and nothing about how many times "
            f"the bot may write to the booking system on the way. The derived "
            f"trajectory is what writing that down looks like."
        )
    gr.HTML(_spec_table(c))

    if len(c.disagreements):
        gr.Markdown("#### Every disagreement")
        gr.Dataframe(_disagreements(c), interactive=False, wrap=True)

    ids = [t.scenario_id for t in c.trajectories]
    if ids:
        gr.Markdown("#### Turn by turn, on one scenario")
        default = next((r.scenario_id for r in c.disagreements), ids[0])
        picker = gr.Dropdown(choices=ids, value=default, label="Scenario")
        view = gr.HTML(_trajectory_view(c, default))
        picker.change(_refresh, inputs=picker, outputs=view)

    if c.variance is not None:
        gr.Markdown(
            f"#### A single eval run, checked twice\n"
            f"This is the **second** run of the same suite through the same agent "
            f"on the same model, hours after day 16's. Day 16 scored "
            f"**{c.day16_pass_rate:.1%}**; this run scored "
            f"**{c.harness.pass_rate:.1%}** — **{c.variance:+.1%}** with nothing "
            f"changed but the clock.\n\n"
            f"That is reassuringly small, and it is the first evidence in this "
            f"repository that an agent pass rate is stable enough to quote to one "
            f"decimal place. It is two runs, not thirty, so it bounds nothing — "
            f"but a number that had moved ten points would have invalidated day "
            f"16, and it did not.\n\n"
            f"The two runs cover different scenario counts "
            f"({c.scenarios} against {c.of_suite}), so this is a difference "
            f"between two published rates rather than a controlled re-run."
        )
        gr.HTML(_dimensions(c))

        from convo.projects.day06_eval_harness.gate import THRESHOLDS

        safety = c.harness.pass_rate_in(SAFETY)
        bar = THRESHOLDS[SAFETY]
        if safety < bar:
            gr.Markdown(
                f"Worth saying plainly, since the gate below no longer asserts "
                f"it: **this agent would not pass day 6's deploy gate.** Safety "
                f"is {safety:.1%} against a bar of {bar:.0%}, and the bar is "
                f"1.0 because a bot that tells a guest their table is booked "
                f"when it is not sends someone to a restaurant that is not "
                f"expecting them." + GAP + "Day 18's gate deliberately does "
                "not re-apply that bar. It is already known and already "
                "documented, so failing every build on it would be a stuck "
                "alarm rather than a gate — absolute bars belong to the bot "
                "that ships, which is day 4's engine, and day 6's gate still "
                "applies them there."
            )

    gr.Markdown(
        f"#### Both evaluators as CI gates, neither calling a model\n"
        f"Day 6's gate already runs in CI against day 4's engine. Adding ADK's "
        f"looked impossible: its own entry point, `AgentEvaluator`, re-runs the "
        f"agent, and CI has no key, no quota and no business having either.\n\n"
        f"A trajectory metric does not need a live agent, though — it needs a "
        f"recorded trajectory and an expected one, and both are committed. So CI "
        f"installs `google-adk` with `--no-deps` in a job that never imports "
        f"Gradio, re-scores the stored trajectories, and fails if ADK's own "
        f"evaluator no longer produces the numbers that were published. That "
        f"last check is the one worth having: this is the only day where a "
        f"**dependency** derives a published figure, and pinning the version is "
        f"not enough when the version is installed outside the lockfile.\n\n"
        f"The live gate stays **{c.gate_choice}**, on a tie in adjudications, "
        f"because it is the one that can score a live bot in CI with no key at "
        f"all. A gate that cannot run is not a gate."
    )

    with gr.Accordion("What a free tier can actually run", open=False):
        gr.Markdown(
            "ADK ships a dozen evaluators. Most are unreachable here, and not because of billing:"
        )
        gr.HTML(_availability())
        gr.Markdown(
            f"`agentplatform` is not on PyPI, so four of these cannot be imported "
            f"at any price. The one that hurts is **`MetricEvaluatorRegistry`** — "
            f"the component that maps a metric name to an evaluator. Without it "
            f"there is no supported way to ask ADK for a named set of metrics; "
            f"you import the classes yourself, which is what this day does.\n\n"
            f"{WHY_NOT_ROUGE}"
        )
        if c.quota_note:
            gr.Markdown(
                f"This run stopped early, with the key stripped out of the "
                f"error:\n\n```\n{c.quota_note}\n```"
            )
