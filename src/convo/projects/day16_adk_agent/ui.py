"""Day 16's tab: the same bot a third time, and the question day 5 left open.

Day 5 scored 8.3% and could not say why. Two explanations fit that number
equally well — a 0.5B model is too small to hold a tool-calling loop together,
or tool-calling is a worse architecture for this job than day 4's state
machine. Day 5 varied both at once, so it separated neither.

This varies one. Same tools, same suite, same scorers, same business rules; a
production framework and a production model in place of the hand-rolled loop
and the 0.5B.

Two tables, not one, because the three runs are not all comparable on the same
terms — see `results.Scope`. Mixing them into a single table would be the
easier presentation and would be wrong.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from convo.core.evals import EFFICIENCY, SAFETY, TASK
from convo.projects.day16_adk_agent.results import Bot, Comparison, Scope, load

COLOURS = {
    "engine": "#2a78d6",
    "schema_only": "#eb6834",
    "grounded": "#1baf7a",
    "adk": "#9450d6",
}
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
WARN = "#c98a16"
DASH = "<span style='opacity:0.4'>not measured</span>"

DIMENSION_NOTE = {
    TASK: "Did it achieve what the guest asked for?",
    SAFETY: "Did it avoid inventing facts or lying about state?",
    EFFICIENCY: "Did it get there without wasting turns or calls?",
}

VERDICTS = {
    "model-bound": (
        "**Day 5's 8.3% was the model, not the architecture.** The architecture "
        "here is the one day 5 used — an LLM choosing tool calls against the "
        "same tools — and on a current model it scores within a few points of "
        "a state machine that cannot be wrong by construction. Day 5's number "
        "was evidence about a 0.5B model and should never have been read as "
        "evidence about tool-calling."
    ),
    "architecture-bound": (
        "**Day 5's 8.3% was not only the model.** A production framework on a "
        "production model still trails day 4's state machine by enough to "
        "matter. For a flow this constrained — four slots, fixed validators, a "
        "known happy path — the deterministic version is better at the job and "
        "costs nothing to run."
    ),
    "architecture-neutral": (
        "**The framework caught up entirely.** ADK on a current Gemini matches "
        "or beats day 4's state machine on the same suite, which makes day 5's "
        "8.3% a statement about a 0.5B model and nothing more."
    ),
    "incomparable": (
        "Day 5's runs are missing from the artifact, so the three-way "
        "comparison cannot be made. Only the ADK/state-machine table below is "
        "meaningful."
    ),
}

MISSING = """
### Day 16 · The booking bot in ADK

No run artifact found.

```bash
uv sync
uv pip install --no-deps google-adk && uv pip install --upgrade pydantic
# then put GOOGLE_API_KEY in .env — see the README
uv run python -m convo.projects.day16_adk_agent.build
```

That drives day 6's suite through an ADK agent on a free Gemini key and scores
it with day 6's unchanged scorers, beside day 4's state machine. Reload the hub
afterwards.
"""


def _pct(value: float | None) -> str:
    return DASH if value is None else f"{value:.1%}"


def _count(value: int | None) -> str:
    return DASH if value is None else str(value)


def _banner(c: Comparison) -> str:
    if not c.partial:
        return (
            f'<div style="padding:10px 14px;border-left:3px solid {GOOD};'
            f'background:rgba(12,163,12,0.07);font-size:0.88rem">'
            f"All <strong>{c.of_suite}</strong> scenarios ran — the free tier held. "
            f"Every ADK figure below is over the full suite, the same one day 4's "
            f"state machine is scored on."
            f"</div>"
        )
    return (
        f'<div style="padding:10px 14px;border-left:3px solid {WARN};'
        f'background:rgba(201,138,22,0.08);font-size:0.88rem">'
        f"<strong>Partial run.</strong> The free tier's daily quota stopped the "
        f"suite after <strong>{c.scenarios} of {c.of_suite}</strong> scenarios "
        f"({c.coverage:.0%}). Comparing a rate over {c.scenarios} scenarios with "
        f"one over {c.of_suite} would not be a comparison, so day 4's engine was "
        f"re-run over exactly the scenarios ADK reached and every figure below "
        f"shares that denominator."
        f"</div>"
    )


def _tiles(c: Comparison) -> str:
    cells: list[tuple[str, str, str, str]] = []
    for bot in c.ordered(Scope.FULL):
        calls = bot.model_calls(Scope.FULL)
        cells.append(
            (
                bot.label,
                _pct(bot.pass_rate(Scope.FULL)),
                "no model in it, nothing to call" if calls == 0 else f"{calls} model calls",
                COLOURS.get(bot.key, "#888"),
            )
        )
    gap = c.gap
    if gap is not None:
        cells.append(
            (
                "The gap",
                f"{gap:+.1%}",
                "state machine minus ADK",
                CRITICAL if gap > 0.05 else GOOD,
            )
        )
    adk_false = c.adk.false_confirmations(Scope.FULL)
    cells.append(
        (
            "False confirmations",
            _count(adk_false),
            "ADK told a guest a booking existed",
            GOOD if adk_false == 0 else CRITICAL,
        )
    )
    tiles = "".join(
        f'<div style="flex:1 1 170px;min-width:0;padding:12px 14px;'
        f'border:1px solid rgba(128,128,128,0.25);border-radius:6px">'
        f'<div style="font-size:0.7rem;letter-spacing:0.04em;text-transform:uppercase;'
        f'opacity:0.65">{title}</div>'
        f'<div style="font-size:1.8rem;font-weight:600;line-height:1.2;color:{colour}">'
        f"{value}</div>"
        f'<div style="font-size:0.74rem;opacity:0.6">{note}</div></div>'
        for title, value, note, colour in cells
    )
    return f'<div style="display:flex;gap:10px;flex-wrap:wrap">{tiles}</div>'


def _table(c: Comparison, scope: Scope) -> str:
    bots = c.ordered(scope)

    def cell(value: float | None) -> str:
        if value is None:
            return f"<td style='padding:5px 12px;text-align:right'>{DASH}</td>"
        colour = GOOD if value >= 0.95 else CRITICAL if value < 0.5 else WARN
        return (
            f"<td style='padding:5px 12px;text-align:right;"
            f"font-variant-numeric:tabular-nums;color:{colour}'>{value:.1%}</td>"
        )

    def plain(value: int | None, good_is_zero: bool = True) -> str:
        if value is None:
            return f"<td style='padding:5px 12px;text-align:right'>{DASH}</td>"
        colour = (GOOD if value == 0 else CRITICAL) if good_is_zero else "inherit"
        return f"<td style='padding:5px 12px;text-align:right;color:{colour}'>{value}</td>"

    headers = "".join(
        f"<th style='text-align:right;padding:4px 12px;color:"
        f"{COLOURS.get(bot.key, '#888')}'>{bot.short}</th>"
        for bot in bots
    )
    rows = [
        "<tr style='font-weight:600'><td style='padding:5px 12px 5px 0'>pass rate</td>"
        + "".join(cell(bot.pass_rate(scope)) for bot in bots)
        + "</tr>"
    ]
    for dimension in (TASK, SAFETY, EFFICIENCY):
        rows.append(
            f"<tr><td style='padding:5px 12px 5px 0' title='{DIMENSION_NOTE[dimension]}'>"
            f"{dimension}</td>"
            + "".join(cell(bot.dimension(scope, dimension)) for bot in bots)
            + "</tr>"
        )
    rows.append(
        "<tr style='border-top:1px solid rgba(128,128,128,0.2)'>"
        "<td style='padding:5px 12px 5px 0'>false confirmations</td>"
        + "".join(plain(bot.false_confirmations(scope)) for bot in bots)
        + "</tr>"
    )
    rows.append(
        "<tr><td style='padding:5px 12px 5px 0'>bookings missed</td>"
        + "".join(plain(bot.missed_bookings(scope)) for bot in bots)
        + "</tr>"
    )
    rows.append(
        "<tr><td style='padding:5px 12px 5px 0;opacity:0.7'>model calls</td>"
        + "".join(plain(bot.model_calls(scope), good_is_zero=False) for bot in bots)
        + "</tr>"
    )
    return f"""
<div style="overflow-x:auto">
<table style="border-collapse:collapse;font-size:0.92rem">
  <thead><tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
    <th style="text-align:left;padding:4px 12px 4px 0">
      over {len(c.ids(scope))} scenarios</th>{headers}
  </tr></thead>
  <tbody>{"".join(rows)}</tbody>
</table>
</div>
"""


def _failures(c: Comparison) -> str:
    rows = []
    for case in c.adk.failures(Scope.FULL):
        broken = "<br>".join(
            f"<span style='opacity:0.8'>{a.name}</span> "
            f"<span style='opacity:0.55;font-size:0.8rem'>({a.dimension})</span>: {a.detail}"
            for a in case.assertions
            if not a.passed
        )
        rows.append(
            f"<tr><td style='padding:7px 12px 7px 0;vertical-align:top;"
            f"font-family:ui-monospace,monospace;font-size:0.84rem'>{case.case_id}</td>"
            f"<td style='padding:7px 0;font-size:0.86rem'>{broken}</td></tr>"
        )
    if not rows:
        return "<p style='opacity:0.7'>Nothing failed.</p>"
    return (
        "<div style='overflow-x:auto'><table style='border-collapse:collapse'>"
        f"{''.join(rows)}</table></div>"
    )


def _what_the_framework_replaced() -> str:
    rows = [
        ("Tool schema", "hand-written JSON Schema per tool", "derived from the Python signature"),
        ("Call parsing", "three output formats, hand-parsed", "handled by the framework"),
        ("Retry loop", "bounded retries, written by hand", "built in"),
        ("Session state", "a dict threaded through the loop", "a session service"),
        ("Provenance guard", "day 5's, day 4's parsers as oracle", "no equivalent"),
    ]
    body = "".join(
        f"<tr><td style='padding:5px 12px 5px 0;font-weight:600'>{what}</td>"
        f"<td style='padding:5px 12px;opacity:0.8'>{day5}</td>"
        f"<td style='padding:5px 12px;opacity:0.8'>{adk}</td></tr>"
        for what, day5, adk in rows
    )
    return f"""
<div style="overflow-x:auto">
<table style="border-collapse:collapse;font-size:0.9rem">
  <thead><tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
    <th style="text-align:left;padding:4px 12px 4px 0"></th>
    <th style="text-align:left;padding:4px 12px;color:{COLOURS["schema_only"]}">day 5, by hand</th>
    <th style="text-align:left;padding:4px 12px;color:{COLOURS["adk"]}">day 16, ADK</th>
  </tr></thead>
  <tbody>{body}</tbody>
</table>
</div>
"""


def _trace(c: Comparison, scenario_id: str) -> str:
    def panel(bot: Bot) -> str:
        case = bot.case(scenario_id)
        colour = COLOURS.get(bot.key, "#888")
        if case is None:
            inner = "<p style='opacity:0.5'>this run did not cover this scenario</p>"
        else:
            word = "pass" if case.passed else "FAIL"
            vcolour = GOOD if case.passed else CRITICAL
            detail = "".join(
                f"<br><span style='opacity:0.7;font-size:0.78rem'>{a.name}: {a.detail}</span>"
                for a in case.assertions
                if not a.passed
            )
            lines = "".join(
                f"<div style='margin:3px 0'><span style='opacity:0.5;font-size:0.72rem;"
                f"display:inline-block;width:32px'>"
                f"{'user' if m['role'] == 'user' else 'bot'}</span>"
                f"<span style='font-weight:{'600' if m['role'] == 'user' else '400'}'>"
                f"{m['content']}</span></div>"
                for m in case.transcript
            )
            inner = (
                f"<div style='margin-bottom:8px'><strong style='color:{vcolour}'>"
                f"{word}</strong>{detail}</div>"
                f"<div style='font-size:0.85rem'>{lines}</div>"
            )
        return (
            f"<div style='flex:1 1 300px;min-width:0;border:1px solid rgba(128,128,128,0.25);"
            f"border-radius:6px;padding:12px 14px'>"
            f"<div style='font-weight:600;color:{colour};margin-bottom:6px'>{bot.label}</div>"
            f"{inner}</div>"
        )

    shown = [c.engine, c.adk]
    panels = "".join(panel(bot) for bot in shown)
    return f"<div style='display:flex;gap:14px;flex-wrap:wrap'>{panels}</div>"


def _refresh(scenario_id: str):
    c = load()
    return "" if c is None else _trace(c, scenario_id)


def _scoreboard(c: Comparison) -> pd.DataFrame:
    rows = []
    for scenario_id in c.ids(Scope.FULL):
        row: dict[str, str] = {"scenario": scenario_id}
        for bot in c.ordered(Scope.FULL):
            case = bot.case(scenario_id)
            row[bot.short] = "—" if case is None else ("pass" if case.passed else "FAIL")
        rows.append(row)
    return pd.DataFrame(rows)


def build() -> None:
    c = load()
    if c is None:
        gr.Markdown(MISSING)
        return

    gr.Markdown(
        f"### Day 16 · The booking bot in ADK\n"
        f"Day 4 built this bot as a state machine with no model in it. Day 5 "
        f"rebuilt it as a tool-calling agent around a 0.5B model and scored "
        f"**8.3%**, which said a great deal about the model and nothing "
        f"reliable about the architecture. This is the run that separates them: "
        f"Google's Agent Development Kit on `{c.model}`, holding **day 5's "
        f"tools** and **day 6's scorers** fixed so the only variable left is the "
        f"agent itself.\n\n"
        f"Nothing here is deployed and nothing is billed — ADK runs in-process "
        f"against a free Gemini key, which is the whole reason week 3 was "
        f"replanned."
    )

    gr.HTML(_banner(c))
    gr.HTML(_tiles(c))

    if not c.rescorable:
        gr.Markdown(
            "> **These figures predate a scorer fix.** Day 18 found day 6's "
            "`score_safety` reading *“I'll get that table reserved for you”* "
            "as a claim that a booking existed — a substring match that could "
            "not tell an offer from an assertion. It wrongly failed "
            "`never_books_without_details` in this run, so the numbers above "
            "are one scenario pessimistic: overall and safety should read "
            "**93.8%** and **96.9%**, not 90.6% and 93.8%. The three-way "
            "comparison and every conclusion drawn from it are unchanged.\n\n"
            "> The figures cannot simply be recomputed, because this artifact "
            "stored the scored verdicts and not the outcomes they came from — "
            "a design error day 18 also fixed. The build now writes the "
            "outcomes, so re-running it will refresh these numbers and remove "
            "this note."
        )

    gr.Markdown(
        f"#### ADK against the state machine, on all {len(c.ids(Scope.FULL))} scenarios\n"
        f"Both runs were scored by day 6's harness with every dimension tagged, "
        f"so this is the table that needs no caveat. Day 4's engine was re-run "
        f"here rather than read from day 5's stored comparison — it has no model "
        f"in it, so re-running it is deterministic and free."
    )
    gr.HTML(_table(c, Scope.FULL))

    gr.Markdown(
        "#### Where ADK actually fails\n"
        "Three scenarios, and they are all the **same mistake**: it booked a "
        "table when the correct behaviour was to book nothing. Not one of them "
        "is a malformed call, a bad parameter or a broken loop — the things day "
        "5 spent most of its code defending against, and the things the "
        "framework genuinely does handle.\n\n"
        "Read the last line of the third one next to the *provenance guard* row "
        "below. The agent invented a name to fill a required argument. That is "
        "the exact failure day 5 built a guard for, and the exact thing ADK has "
        "no equivalent of."
    )
    gr.HTML(_failures(c))

    if c.ids(Scope.DAY5):
        gr.Markdown(
            f"#### The three-way comparison, on the {len(c.ids(Scope.DAY5))} "
            f"scenarios day 5 ran\n"
            f"Day 5's suite was smaller and its assertions carry no dimension, "
            f"because day 6 — which invented the dimensions — came after it. So "
            f"its columns are a pass rate and nothing else, and the other two "
            f"bots are cut down to the same {len(c.ids(Scope.DAY5))} scenarios "
            f"to sit beside it. **Blank means never measured, not zero.**"
        )
        gr.HTML(_table(c, Scope.DAY5))
        gr.Markdown(VERDICTS[c.verdict])

        schema, grounded = c.bot("schema_only"), c.bot("grounded")
        if schema and grounded:
            gr.Markdown(
                f"The *false confirmations* and *bookings missed* rows are the "
                f"interesting pair. Day 5's guard took false confirmations from "
                f"**{schema.false_confirmations(Scope.DAY5)}** to "
                f"**{grounded.false_confirmations(Scope.DAY5)}** and pushed "
                f"bookings missed from **{schema.missed_bookings(Scope.DAY5)}** to "
                f"**{grounded.missed_bookings(Scope.DAY5)}** — it converted silent "
                f"wrong bookings into visible failures, which is the trade a "
                f"restaurant wants. ADK, with no guard at all, misses "
                f"**{c.adk.missed_bookings(Scope.DAY5)}** and confirms "
                f"**{c.adk.false_confirmations(Scope.DAY5)}** falsely. It is better "
                f"at the task and it is the only one of the three that will tell a "
                f"guest about a table nobody is holding."
            )

    gr.Markdown(
        "#### What the framework actually replaced\n"
        "Day 5 is the control for this, because day 5 hand-wrote every piece ADK "
        "provides."
    )
    gr.HTML(_what_the_framework_replaced())
    gr.Markdown(
        "The last row is the one worth arguing about. A framework removes the "
        "code you would have written anyway; it does not remove the code you had "
        "to invent. Day 5's provenance guard was invented, so it has no "
        "counterpart — and the failures above are where that shows."
    )

    gr.Markdown("#### Scenario by scenario")
    gr.Dataframe(_scoreboard(c), interactive=False, wrap=True)

    ids = c.ids(Scope.FULL)
    if ids:
        gr.Markdown("#### One scenario, both bots")
        picker = gr.Dropdown(choices=list(ids), value=ids[0], label="Scenario")
        trace = gr.HTML(_trace(c, ids[0]))
        picker.change(_refresh, inputs=picker, outputs=trace)

    with gr.Accordion("What the free tier cost in practice", open=False):
        rejected = (
            "\n".join(f"- `{r}`" for r in c.rejected_models)
            if c.rejected_models
            else "- none on this run; the first preferred model answered"
        )
        quota = (
            f"\n\nThis run stopped early. The error, with the key stripped out:\n\n"
            f"```\n{c.quota_note}\n```"
            if c.quota_note
            else ""
        )
        gr.Markdown(
            f"The key lists **{c.models_listed}** models advertising "
            f"`generateContent`, and cannot call all of them — a new free key "
            f"gets 404 *no longer available to new users* on some and 503 on "
            f"others. So the model is resolved by invoking candidates in "
            f"preference order until one answers, and the winner is written into "
            f"the artifact, because a name that works today 404s for whoever runs "
            f"this next.\n\n"
            f"Listed, then refused the call:\n\n{rejected}\n\n"
            f"The binding limit is not tokens and not requests per minute. It is "
            f"`GenerateRequestsPerDayPerProjectPerModel`, which for the preview "
            f"flash model is **20 a day** against a suite that needed "
            f"{c.adk.model_calls(Scope.FULL)} calls. Lite models carry a larger "
            f"allowance, which is why the resolver prefers them — defensible on "
            f"merit too, for a bot doing shallow reasoning under a latency "
            f"budget.{quota}"
        )
