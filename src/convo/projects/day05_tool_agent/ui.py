"""Day 5's tab: one suite, three configurations, and what each guard is worth.

The three columns are the argument. A state machine cannot emit a malformed
transition or invent a parameter. An agent with schema validation cannot emit a
malformed call but can invent one. An agent that also checks provenance can do
neither — at the cost of a guard that has to be written and maintained.
"""

from __future__ import annotations

import json

import gradio as gr
import pandas as pd

from convo.projects.day05_tool_agent.results import AgentPass, Comparison, load

# Categorical slots 1-3 from the reference palette; the first three validate
# all-pairs in both light and dark, which is what a three-series chart needs.
ENGINE_COLOUR = "#2a78d6"
SCHEMA_COLOUR = "#eb6834"
GROUNDED_COLOUR = "#1baf7a"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

ENGINE = "State machine (day 4)"

FAILURE_LABELS = {
    "unparseable": "No tool call found in the output",
    "unknown_tool": "Called a tool that does not exist",
    "invalid_args": "Arguments failed schema validation",
    "ungrounded": "Well-formed, but the guest never said it",
    "tool_error": "Schema-valid but refused by a business rule",
}

MISSING = """
### Day 05 · The same bot, tool-calling

No comparison artifact found.

```bash
uv sync --extra train
uv run python -m convo.projects.day05_tool_agent.benchmark
```

That runs day 4's state machine and the agent — twice, with and without the
provenance guard — over the same scripted conversations, scored by the same
assertions. Reload the hub afterwards.
"""


def _tiles(c: Comparison) -> str:
    cells = [
        (ENGINE, f"{c.engine.pass_rate:.0%}", "no model, no tokens", ENGINE_COLOUR),
        (
            c.schema_only.label,
            f"{c.schema_only.run.pass_rate:.0%}",
            f"{c.schema_only.stats['model_calls']} model calls",
            SCHEMA_COLOUR,
        ),
        (
            c.grounded.label,
            f"{c.grounded.run.pass_rate:.0%}",
            f"{c.grounded.stats['model_calls']} model calls",
            GROUNDED_COLOUR,
        ),
    ]
    cells += [
        (
            "False confirmations · schema only",
            str(c.schema_only.false_confirmations),
            "told a guest a wrong booking was made",
            CRITICAL,
        ),
        (
            "False confirmations · + provenance",
            str(c.grounded.false_confirmations),
            "the number the pass rate hides",
            GOOD,
        ),
    ]
    tiles = "".join(
        f'<div style="flex:1 1 180px;min-width:0;padding:12px 14px;'
        f'border:1px solid rgba(128,128,128,0.25);border-radius:6px">'
        f'<div style="font-size:0.7rem;letter-spacing:0.04em;text-transform:uppercase;'
        f'opacity:0.65">{title}</div>'
        f'<div style="font-size:1.8rem;font-weight:600;line-height:1.2;color:{colour}">'
        f"{value}</div>"
        f'<div style="font-size:0.74rem;opacity:0.6">{note}</div></div>'
        for title, value, note, colour in cells
    )
    return f'<div style="display:flex;gap:10px;flex-wrap:wrap">{tiles}</div>'


def _scoreboard(c: Comparison) -> pd.DataFrame:
    rows = []
    for scenario in c.scenarios:
        row = {"scenario": scenario["id"], "tests": ", ".join(scenario["tags"])}
        engine_case = c.case(c.engine, scenario["id"])
        row["state machine"] = "pass" if engine_case and engine_case.passed else "FAIL"
        for key in ("schema_only", "with_grounding"):
            case = c.case(c.passes[key].run, scenario["id"])
            row[c.passes[key].label] = "pass" if case and case.passed else "FAIL"
        rows.append(row)
    return pd.DataFrame(rows)


def _guard_table(c: Comparison) -> str:
    def column(agent_pass: AgentPass, key: str) -> str:
        total = max(agent_pass.stats["model_calls"], 1)
        count = agent_pass.stats.get(key, 0)
        return (
            f"<td style='padding:5px 12px;text-align:right;"
            f"font-variant-numeric:tabular-nums'>{count}"
            f"<span style='opacity:0.55'> · {count / total:.0%}</span></td>"
        )

    rows = "".join(
        f"<tr><td style='padding:5px 12px 5px 0'>{label}</td>"
        f"{column(c.schema_only, key)}{column(c.grounded, key)}</tr>"
        for key, label in FAILURE_LABELS.items()
    )
    schema_stats, grounded_stats = c.schema_only.stats, c.grounded.stats
    return f"""
<div style="overflow-x:auto">
<table style="border-collapse:collapse;font-size:0.92rem">
  <thead><tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
    <th style="text-align:left;padding:4px 12px 4px 0">What each guard caught</th>
    <th style="text-align:right;padding:4px 12px;color:{SCHEMA_COLOUR}">schema only</th>
    <th style="text-align:right;padding:4px 12px;color:{GROUNDED_COLOUR}">+ provenance</th>
  </tr></thead>
  <tbody>{rows}</tbody>
  <tfoot><tr style="border-top:1px solid rgba(128,128,128,0.35)">
    <td style="padding:5px 12px 5px 0;opacity:0.7">tokens in / out</td>
    <td style="padding:5px 12px;text-align:right;opacity:0.7">
      {schema_stats["tokens_in"]:,} / {schema_stats["tokens_out"]:,}</td>
    <td style="padding:5px 12px;text-align:right;opacity:0.7">
      {grounded_stats["tokens_in"]:,} / {grounded_stats["tokens_out"]:,}</td>
  </tr></tfoot>
</table>
</div>
<p style="font-size:0.8rem;opacity:0.7;margin-top:8px">
  The state machine's column would be all zeros: it has no output to parse, no
  arguments to validate and nothing to ground, because a slot is only ever filled
  by an extractor reading the guest's own words.
</p>
"""


def _trace(c: Comparison, scenario_id: str) -> str:
    scenario = c.scenario(scenario_id)

    def panel(title: str, colour: str, case) -> str:
        if case is None:
            body = "<p style='opacity:0.5'>not run</p>"
            verdict = ""
        else:
            word = "pass" if case.passed else "FAIL"
            vcolour = GOOD if case.passed else CRITICAL
            detail = (
                ""
                if case.passed
                else "<br>"
                + "<br>".join(
                    f"<span style='opacity:0.7;font-size:0.78rem'>{a.name}: {a.detail}</span>"
                    for a in case.assertions
                    if not a.passed
                )
            )
            verdict = (
                f"<div style='margin-bottom:8px'>"
                f"<strong style='color:{vcolour}'>{word}</strong>{detail}</div>"
            )
            lines = []
            for message in case.transcript:
                who = "user" if message["role"] == "user" else "bot"
                weight = "600" if who == "user" else "400"
                lines.append(
                    f"<div style='margin:3px 0'><span style='opacity:0.5;font-size:0.72rem;"
                    f"display:inline-block;width:32px'>{who}</span>"
                    f"<span style='font-weight:{weight}'>{message['content']}</span></div>"
                )
            note = (
                f"<div style='font-size:0.74rem;opacity:0.6;margin-top:8px'>"
                f"{case.judge_rationale}</div>"
                if case.judge_rationale
                else ""
            )
            body = f"<div style='font-size:0.85rem'>{''.join(lines)}</div>{note}"
        return (
            f"<div style='flex:1 1 280px;min-width:0;border:1px solid rgba(128,128,128,0.25);"
            f"border-radius:6px;padding:12px 14px'>"
            f"<div style='font-weight:600;color:{colour};margin-bottom:6px'>{title}</div>"
            f"{verdict}{body}</div>"
        )

    panels = (
        panel(ENGINE, ENGINE_COLOUR, c.case(c.engine, scenario_id))
        + panel(c.schema_only.label, SCHEMA_COLOUR, c.case(c.schema_only.run, scenario_id))
        + panel(c.grounded.label, GROUNDED_COLOUR, c.case(c.grounded.run, scenario_id))
    )
    return (
        f"<p style='font-size:0.85rem;opacity:0.75;margin-bottom:10px'>{scenario['note']}</p>"
        f"<div style='display:flex;gap:14px;flex-wrap:wrap'>{panels}</div>"
    )


def _refresh(scenario_id: str):
    c = load()
    return "" if c is None else _trace(c, scenario_id)


def build() -> None:
    c = load()
    if c is None:
        gr.Markdown(MISSING)
        return

    config = c.config
    gr.Markdown(
        f"### Day 05 · The same bot, tool-calling\n"
        f"Day 4's booking flow rebuilt as an LLM with tools, run over the **same "
        f"{len(c.scenarios)} scripted conversations** and scored by the **same "
        f"assertions**. Model: `{config['model']}`, greedy, "
        f"{config['max_retries']} retries per turn, ceiling of "
        f"{config['max_model_calls']} model calls per conversation. The tools enforce "
        f"the same business rules as day 4's validators, so neither side gets to look "
        f"better by being allowed more.\n\n"
        f"**Read the state machine's score with suspicion:** this suite was written "
        f"from day 4's semantics, so it passing everything is partly by construction. "
        f"The two agent columns are the measurement."
    )

    gr.HTML(_tiles(c))

    gr.Markdown(
        f"#### The pass rate is the wrong metric here\n"
        f"Both agent configurations score "
        f"**{c.schema_only.run.pass_rate:.0%}**. On that number alone the provenance "
        f"guard bought nothing. But false confirmations went from "
        f"**{c.schema_only.false_confirmations}** to "
        f"**{c.grounded.false_confirmations}**, and failures to book went from "
        f"{c.schema_only.missed_bookings} to {c.grounded.missed_bookings}.\n\n"
        f"The guard did not make the agent more accurate. It converted **silent wrong "
        f"bookings into visible failures** — a system that tells you it cannot help "
        f"instead of sending someone to a restaurant that is not expecting them. A "
        f"single pass rate scores those two outcomes identically, which is the "
        f"argument for day 6 scoring several dimensions rather than one.\n\n"
        f"It is not free: "
        f"{c.schema_only.stats['model_calls']} model calls became "
        f"{c.grounded.stats['model_calls']}, and "
        f"{c.schema_only.stats['wall_seconds'] / 60:.0f} minutes became "
        f"{c.grounded.stats['wall_seconds'] / 60:.0f}, because the model keeps "
        f"retrying and keeps failing to ground its answer."
    )

    gr.Markdown("#### Scenario by scenario")
    gr.Dataframe(_scoreboard(c), interactive=False, wrap=True)

    gr.Markdown(
        "#### What each guard caught\n"
        "Schema validation checks **shape**. It cannot tell a real booking from a "
        "confidently invented one, because inventing plausible data is not a schema "
        "violation. The provenance guard uses day 4's own parsers as an oracle: every "
        "argument must be traceable to something the guest actually said."
    )
    gr.HTML(_guard_table(c))

    gr.Markdown("#### Side by side on one scenario")
    picker = gr.Dropdown(choices=c.scenario_ids, value=c.scenario_ids[0], label="Scenario")
    trace = gr.HTML(_trace(c, c.scenario_ids[0]))
    picker.change(_refresh, inputs=picker, outputs=trace)

    with gr.Accordion("Run configuration", open=False):
        gr.Code(json.dumps(config, indent=2), language="json", label="")
