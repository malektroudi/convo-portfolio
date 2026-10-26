"""Day 15's tab: the containment gap, the SQL behind every number, and the gaps.

Each query's own leading comment is shown next to its result, because the
argument for a metric belongs beside the metric. A dashboard that shows
"containment: 92%" and nothing else is how the number in this tab got believed
for years.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from convo.projects.day15_analytics.results import Day15Results, load

GOOD = "#1baf7a"
BAD = "#d03b3b"
WARN = "#eb6834"

MISSING = """
### Day 15 · Conversation analytics

No analytics artifact found.

```bash
uv run python -m convo.projects.day15_analytics.build
```

That projects days 10-13's measured runs into an event schema and queries
them. Reload the hub afterwards.
"""


def _headline(results: Day15Results) -> str:
    row = results.containment
    if not row:
        return ""
    naive = row["containment_naive"]
    honest = row["containment_honest"]
    abandoned = row["abandoned_on_a_refusal"]
    return f"""
<div style="display:flex;gap:34px;flex-wrap:wrap;margin:14px 0 6px">
  <div><div style="font-size:0.72rem;opacity:0.6;text-transform:uppercase">
    containment, as reported</div>
    <div style="font-size:2rem;font-weight:650;color:{GOOD}">{naive:.0%}</div>
    <div style="font-size:0.78rem;opacity:0.6">no human was involved</div></div>
  <div><div style="font-size:0.72rem;opacity:0.6;text-transform:uppercase">
    containment, honestly</div>
    <div style="font-size:2rem;font-weight:650;color:{BAD}">{honest:.0%}</div>
    <div style="font-size:0.78rem;opacity:0.6">
      no human <em>and</em> the last turn answered</div></div>
  <div><div style="font-size:0.72rem;opacity:0.6;text-transform:uppercase">
    the difference</div>
    <div style="font-size:2rem;font-weight:650">{abandoned}</div>
    <div style="font-size:0.78rem;opacity:0.6">
      conversations that ended on &ldquo;I don't know&rdquo;<br>
      with nobody to escalate to</div></div>
</div>
<p style="font-size:0.9rem;max-width:68ch">
The naive definition counts a customer who gave up as a success, because
nobody was escalated to. One extra clause &mdash; the last thing the system
did was answer &mdash; moves the number
<strong>{naive:.0%} &rarr; {honest:.0%}</strong>. Both are computed from the
same rows by the same query, which is in the dropdown below.
</p>
"""


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _sql_block(results: Day15Results, name: str) -> tuple[str, pd.DataFrame]:
    from convo.projects.day15_analytics.warehouse import queries

    text = dict(queries()).get(name, "")
    body = "\n".join(line for line in text.splitlines() if not line.startswith("--")).strip()
    note = results.describe(name)
    return (
        f"<p style='font-size:0.86rem;opacity:0.75;max-width:70ch'>{note}</p>"
        f"<pre style='white-space:pre-wrap;font-size:0.8rem;padding:10px 12px;"
        f"border:1px solid rgba(128,128,128,0.3);border-radius:6px'>{body}</pre>",
        _frame(results.rows(name)),
    )


def _portability(results: Day15Results) -> str:
    if results.bigquery_results:
        count = len(results.disagreements)
        colour = GOOD if count == 0 else BAD
        verdict = (
            "Both engines returned the same answer to every query."
            if count == 0
            else f"<strong>{count} cell(s) disagreed.</strong>"
        )
        return (
            f"<p style='font-size:0.88rem;color:{colour}'>Run on DuckDB "
            f"<em>and</em> BigQuery. {verdict}</p>"
        )
    commands = "\n".join(results.load_commands)
    return f"""
<p style="font-size:0.88rem;max-width:70ch">
The figures here were produced by <strong>DuckDB</strong>, locally. The same
SQL runs unchanged on BigQuery &mdash; the only incompatibility is
<code>SAFE_DIVIDE</code>, which DuckDB gets as a two-line macro. The sandbox
needs no billing account, but it does need a Google sign-in, so that path is a
command rather than something this build can do on its own:
</p>
<pre style="white-space:pre-wrap;font-size:0.78rem;padding:10px 12px;
border:1px solid rgba(128,128,128,0.3);border-radius:6px">{commands}
uv run python -m convo.projects.day15_analytics.build --bigquery convo_analytics</pre>
<p style="font-size:0.84rem;opacity:0.7;max-width:70ch">
That last command runs both engines and records every cell where they
disagree, which is the only way the portability claim above means anything.
</p>
"""


def build() -> None:
    results = load()
    if results is None:
        gr.Markdown(MISSING)
        return

    gr.Markdown(
        f"### Day 15 · Conversation analytics\n"
        f"**{results.events} turn events** from **{results.conversations} conversations**, "
        f"projected out of days {', '.join(str(d) for d in results.sources)}'s measured "
        f"runs. These are not synthetic logs: every row is a turn a pipeline in this "
        f"repository actually processed.\n\n"
        f"The SQL is unremarkable. What decides whether any of it is *possible* is what "
        f"the system wrote down while the conversation was happening — so the schema is "
        f"the deliverable, and the queries are what it makes cheap."
    )

    gr.HTML(_headline(results))

    gr.Markdown("#### Every query, with its argument and its answer")
    names = results.names
    chosen = gr.Dropdown(names, value=names[0], label="query")
    sql = gr.HTML()
    table = gr.Dataframe(interactive=False, wrap=True)

    with gr.Accordion("No timestamps, and what that rules out", open=False):
        gr.Markdown(
            "Nothing in this repository recorded a wall-clock time per turn, so "
            "the schema has no timestamp — rather than one invented to make a "
            "chart possible. That rules out every trend, cohort and seasonality "
            "question, which is a large part of what *analytics* usually means.\n\n"
            "What survives is rates, distributions, segmentation and content "
            "gaps, and those are the queries here. A line going up and to the "
            "right would have been more impressive and would have been fiction."
        )

    with gr.Accordion("Running the same SQL on BigQuery", open=False):
        gr.HTML(_portability(results))

    def _refresh(name: str):
        return _sql_block(results, name)

    chosen.change(_refresh, inputs=chosen, outputs=[sql, table])
    sql.value, table.value = _refresh(names[0])
