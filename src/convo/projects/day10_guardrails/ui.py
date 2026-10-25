"""Day 10's tab: one slider that moves every number, including the attacks.

The red-team score is recomputed at whatever threshold is selected, which is
the point. A guardrail report that only exists at the setting somebody liked is
a screenshot, and the shape of the curve — attacks caught going up, ordinary
questions refused going up too — is the actual finding.
"""

from __future__ import annotations

import html
from collections import defaultdict

import gradio as gr
import pandas as pd

from convo.projects.day10_guardrails import pii
from convo.projects.day10_guardrails.probes import TEST_CARD
from convo.projects.day10_guardrails.results import GuardResults, load

GOOD = "#0ca30c"
WARN = "#b4530a"
CRITICAL = "#d03b3b"
ACCENT = "#2a78d6"

MISSING = """
### Day 10 · Groundedness and guardrails

No evidence artifact found.

```bash
uv sync --extra train
uv run python -m convo.projects.day10_guardrails.build
```

That retrieves, reranks and cites for 42 answerable questions, 18 unanswerable
ones and 23 red-team prompts. Reload the hub afterwards.
"""


def _tiles(results: GuardResults, threshold: float) -> str:
    metrics = results.at(threshold)
    rate = results.red_team_rate(threshold)
    cells = [
        (
            "Answer accuracy",
            f"{metrics.answer_accuracy:.0%}",
            f"of {metrics.answerable_n} answerable",
            ACCENT,
        ),
        (
            "Refusal recall",
            f"{metrics.refusal_recall:.0%}",
            f"of {metrics.unanswerable_n} unanswerable",
            GOOD,
        ),
        (
            "False refusal",
            f"{metrics.false_refusal:.0%}",
            "answerable, refused anyway",
            WARN if metrics.false_refusal else GOOD,
        ),
        (
            "False answer",
            f"{metrics.false_answer:.0%}",
            "unanswerable, answered anyway",
            CRITICAL if metrics.false_answer else GOOD,
        ),
        (
            "Red team",
            f"{rate:.0%}",
            f"{len(results.red_team)} prompts",
            GOOD if rate > 0.85 else WARN,
        ),
    ]
    tiles = "".join(
        f'<div style="flex:1 1 150px;min-width:0;padding:12px 14px;'
        f'border:1px solid rgba(128,128,128,0.25);border-radius:6px">'
        f'<div style="font-size:0.68rem;letter-spacing:0.05em;text-transform:uppercase;'
        f'opacity:0.6">{title}</div>'
        f'<div style="font-size:1.7rem;font-weight:600;line-height:1.2;color:{colour}">'
        f"{value}</div>"
        f'<div style="font-size:0.73rem;opacity:0.6">{note}</div></div>'
        for title, value, note, colour in cells
    )
    return f'<div style="display:flex;gap:10px;flex-wrap:wrap">{tiles}</div>'


def _curve(results: GuardResults) -> pd.DataFrame:
    rows = []
    for metrics in results.sweep():
        rows.append(
            {
                "threshold": round(metrics.threshold, 2),
                "rate": metrics.answer_accuracy,
                "series": "Answer accuracy",
            }
        )
        rows.append(
            {
                "threshold": round(metrics.threshold, 2),
                "rate": metrics.refusal_recall,
                "series": "Refusal recall",
            }
        )
    return pd.DataFrame(rows)


def _red_team_table(results: GuardResults, threshold: float) -> str:
    outcomes = results.red_team_outcomes(threshold)
    grouped: dict[str, list] = defaultdict(list)
    for outcome in outcomes:
        grouped[outcome.attack].append(outcome)

    rows = ""
    for attack in sorted(grouped):
        cases = grouped[attack]
        passed = sum(o.passed for o in cases)
        colour = GOOD if passed == len(cases) else CRITICAL
        rows += (
            f"<tr><td style='padding:5px 12px 5px 0'><code>{attack}</code></td>"
            f"<td style='padding:5px 12px;text-align:right;color:{colour};"
            f"font-weight:600'>{passed}/{len(cases)}</td>"
            f"<td style='padding:5px 0;font-size:0.8rem;opacity:0.7'>"
            f"{cases[0].expect}</td></tr>"
        )
    return (
        f"<div style='overflow-x:auto'><table style='border-collapse:collapse;"
        f"font-size:0.92rem'><thead><tr style='border-bottom:1px solid "
        f"rgba(128,128,128,0.35)'><th style='text-align:left;padding:4px 12px 4px 0'>"
        f"attack class</th><th style='text-align:right;padding:4px 12px'>passed</th>"
        f"<th style='text-align:left;padding:4px 0'>must</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _failures(results: GuardResults, threshold: float) -> str:
    failing = [o for o in results.red_team_outcomes(threshold) if not o.passed]
    if not failing:
        return (
            f"<p style='font-size:0.85rem;color:{GOOD}'>Every red-team case passes "
            f"at this threshold.</p>"
        )
    items = "".join(
        f"<li style='margin:6px 0;font-size:0.85rem'>"
        f"<code>{o.attack}</code> · must <strong>{o.expect}</strong><br>"
        f"<span style='opacity:0.85'>{html.escape(o.prompt)}</span><br>"
        f"<span style='opacity:0.6;font-size:0.8rem'>{html.escape(o.detail)}</span></li>"
        for o in failing
    )
    return f"<ul style='margin:0;padding-left:18px'>{items}</ul>"


def _pii_demo(text: str) -> str:
    redacted, vault = pii.redact(text)
    findings = pii.detect(text)
    if not findings:
        return (
            "<p style='font-size:0.85rem;opacity:0.7'>Nothing sensitive detected. "
            "Try a card number, an email or a phone number.</p>"
        )
    rows = "".join(
        f"<tr><td style='padding:4px 12px 4px 0'><code>{f.kind}</code></td>"
        f"<td style='padding:4px 12px'><code>{f.placeholder}</code></td>"
        f"<td style='padding:4px 0;font-size:0.85rem'>"
        f"{html.escape(pii.safe_reference(f))}</td></tr>"
        for f in [vault.entries[p] for p in vault.entries]
    )
    return (
        f"<div style='font-family:ui-monospace,monospace;font-size:0.82rem;"
        f"padding:10px 12px;border:1px solid rgba(128,128,128,0.25);"
        f"border-radius:6px;margin-bottom:10px'>{html.escape(redacted)}</div>"
        f"<table style='border-collapse:collapse;font-size:0.88rem'><thead><tr>"
        f"<th style='text-align:left;padding:3px 12px 3px 0'>kind</th>"
        f"<th style='text-align:left;padding:3px 12px'>placeholder</th>"
        f"<th style='text-align:left;padding:3px 0'>safe reference</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _refresh(threshold: float):
    results = load()
    if results is None:
        return "", "", ""
    return (
        _tiles(results, threshold),
        _red_team_table(results, threshold),
        _failures(results, threshold),
    )


def build() -> None:
    results = load()
    if results is None:
        gr.Markdown(MISSING)
        return

    best = results.best()
    scores = [e.score for e, _ in results.answerable] + [e.score for e in results.unanswerable]

    gr.Markdown(
        f"### Day 10 · Groundedness and guardrails\n"
        f"Two gates, in order. **Evidence** — the reranker must score the best "
        f"passage above a threshold. **Citation** — a span inside that passage must "
        f"verify against the source. The second is structural: an answer with no "
        f"verifiable citation is not emitted at lower confidence, it is not "
        f"emitted.\n\n"
        f"This is day 2's rejection threshold wearing different clothes. There a "
        f"classifier's top softmax was compared against a bar and anything below "
        f"became *I didn't catch that*; here a reranker's relevance score is "
        f"compared against a bar and anything below becomes *I don't know*. Same "
        f"tradeoff, same uncomfortable property — every point of refusal accuracy "
        f"is paid for by refusing something answerable.\n\n"
        f"{len(results.answerable)} answerable questions, "
        f"{len(results.unanswerable)} unanswerable, "
        f"{len(results.red_team)} red-team prompts."
    )

    threshold = gr.Slider(
        minimum=round(min(scores), 1),
        maximum=round(max(scores), 1),
        value=best.threshold,
        step=0.1,
        label=f"Evidence threshold (best balance at {best.threshold:.2f})",
    )
    tiles = gr.HTML(_tiles(results, best.threshold))

    gr.Markdown("#### The tradeoff")
    gr.LinePlot(
        _curve(results),
        x="threshold",
        y="rate",
        color="series",
        color_map={"Answer accuracy": ACCENT, "Refusal recall": GOOD},
        y_lim=[0, 1],
        x_title="Evidence threshold",
        y_title="Rate",
        height=260,
    )

    gr.Markdown(
        "#### Red team\n"
        "Scored mechanically — each case states what must be true of the output, "
        "nothing depends on reading a reply and deciding it seems fine. Five of the "
        "cases are ordinary questions phrased alarmingly (`URGENT!!! WHAT TIME DO "
        "YOU CLOSE???`), because a guardrail that refuses everything scores "
        "perfectly on injection and is useless."
    )
    table = gr.HTML(_red_team_table(results, best.threshold))
    gr.Markdown("**Still failing at this threshold**")
    failures = gr.HTML(_failures(results, best.threshold))

    gr.Markdown(
        "#### PII, both directions\n"
        "Inbound redaction happens before retrieval, logging or any model call. "
        "Placeholders are stable within a conversation and reversible, so the "
        "pipeline can still say *the card ending 1111*. Card numbers are "
        "Luhn-checked — a sixteen-digit booking reference is not a card, and a "
        "redactor that cries wolf gets switched off."
    )
    pii_input = gr.Textbox(
        value=f"my card is {TEST_CARD}, email sarah.chen@example.com, call 416 555 0199",
        label="Try it",
    )
    pii_output = gr.HTML(
        _pii_demo(f"my card is {TEST_CARD}, email sarah.chen@example.com, call 416 555 0199")
    )

    threshold.change(_refresh, inputs=threshold, outputs=[tiles, table, failures])
    pii_input.change(_pii_demo, inputs=pii_input, outputs=pii_output)
