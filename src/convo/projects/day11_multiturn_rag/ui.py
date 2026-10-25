"""Day 11's tab: a conversation replayed turn by turn, with the rewrite shown.

The headline number here is negative for one retriever, so the tab is built to
let a reader check that rather than take it on trust. Picking a conversation
replays it turn by turn: what was typed, what the rewriter made of it, which
rule fired and why, and where the right passage landed under each condition.

A rewriter that cannot be watched working is a rewriter nobody can debug.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from convo.projects.day11_multiturn_rag.results import (
    CONDITIONS,
    RETRIEVERS,
    Day11Results,
    load,
)

GOOD = "#1baf7a"
CRITICAL = "#d03b3b"
WARN = "#eb6834"

CONDITION_NOTE = {
    "raw": "the follow-up as typed — what single-turn RAG receives",
    "concat": "the previous turn glued on — the cheap fix, and the honest baseline",
    "rewrite": "the four rules, as designed before any result was seen",
    "oracle": "the hand-written standalone question — the ceiling",
}

MISSING = """
### Day 11 · Multi-turn RAG

No turns artifact found.

```bash
uv sync --extra train
uv run python -m convo.projects.day11_multiturn_rag.build
```

That replays twelve conversations under four conditions and three retrievers
and stores every ranking. Reload the hub afterwards.
"""


def _delta(value: float, baseline: float) -> str:
    delta = value - baseline
    if abs(delta) < 0.0005:
        return "<span style='opacity:0.5'>—</span>"
    colour = GOOD if delta > 0 else CRITICAL
    return f"<span style='color:{colour}'>{delta:+.3f}</span>"


def _condition_table(results: Day11Results, retriever: str) -> str:
    scores = {c: results.score(c, retriever) for c in CONDITIONS}
    raw = scores["raw"]
    rows = ""
    for condition in CONDITIONS:
        score = scores[condition]
        strong = "font-weight:600" if condition == "oracle" else ""
        rows += (
            f"<tr><td style='padding:6px 12px 6px 0;{strong}'><code>{condition}</code></td>"
            f"<td style='padding:6px 12px;text-align:right'>{score.hit_at_1:.0%}</td>"
            f"<td style='padding:6px 12px;text-align:right'>{score.hit_at_3:.0%}</td>"
            f"<td style='padding:6px 12px;text-align:right;{strong}'>{score.mrr:.3f}</td>"
            f"<td style='padding:6px 12px;text-align:right'>{_delta(score.mrr, raw.mrr)}</td>"
            f"<td style='padding:6px 0;font-size:0.8rem;opacity:0.65'>"
            f"{CONDITION_NOTE[condition]}</td></tr>"
        )

    gap = results.gap_closed(retriever)
    if gap is None:
        verdict = "The oracle matches doing nothing here, so there is no gap to close."
    elif gap < 0:
        verdict = (
            f"<span style='color:{CRITICAL};font-weight:600'>Rewriting makes this "
            f"retriever worse</span> — it moves {abs(gap):.0%} of the way from "
            f"doing nothing in the wrong direction, while the oracle shows the "
            f"improvement was there to take."
        )
    else:
        verdict = (
            f"Rewriting captures <span style='color:{GOOD};font-weight:600'>"
            f"{gap:.0%}</span> of the distance between doing nothing and a "
            f"hand-written standalone question."
        )

    return f"""
<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:0.9rem">
<thead><tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
  <th style="text-align:left;padding:4px 12px 4px 0">condition</th>
  <th style="text-align:right;padding:4px 12px">hit@1</th>
  <th style="text-align:right;padding:4px 12px">hit@3</th>
  <th style="text-align:right;padding:4px 12px">MRR</th>
  <th style="text-align:right;padding:4px 12px">vs raw</th>
  <th style="text-align:left;padding:4px 0"></th>
</tr></thead><tbody>{rows}</tbody></table></div>
<p style="font-size:0.86rem;margin-top:10px">{verdict}</p>
"""


def _replay(results: Day11Results, name: str, retriever: str) -> tuple[str, pd.DataFrame]:
    by_condition = {c: {r.key: r for r in results.turns(c, retriever)} for c in CONDITIONS}
    conversation = next(c for c in results.conversations if c["name"] == name)

    blocks = ""
    rows = []
    for index in range(len(conversation["turns"])):
        key = f"{name}:{index}"
        rewritten = by_condition["rewrite"][key]
        want = results.relevance[key]

        if index == 0:
            label = "<span style='opacity:0.6'>opening turn — standalone by construction</span>"
        elif rewritten.rewritten:
            label = (
                f"<code>{rewritten.strategy}</code> &nbsp;"
                f"<span style='opacity:0.7'>{rewritten.detail}</span><br>"
                f"<span style='opacity:0.85'>&rarr; {rewritten.query}</span>"
            )
        else:
            label = (
                f"<span style='color:{WARN}'>no rule fired</span> "
                f"<span style='opacity:0.6'>— sent as typed</span>"
            )

        blocks += (
            f"<div style='margin:0 0 10px;padding:9px 12px;border-left:2px solid "
            f"rgba(128,128,128,0.3)'>"
            f"<div style='font-weight:600;font-size:0.92rem'>{rewritten.text}</div>"
            f"<div style='font-size:0.82rem;margin-top:4px'>{label}</div></div>"
        )

        rows.append(
            {
                "turn": rewritten.text,
                **{
                    condition: _position(by_condition[condition][key].ranked, want)
                    for condition in CONDITIONS
                },
                "answer in": rewritten.doc_id,
            }
        )

    note = conversation["note"]
    header = f"<p style='font-size:0.84rem;opacity:0.7;margin-bottom:10px'>{note}</p>{blocks}"
    return header, pd.DataFrame(rows)


def _position(ranked: tuple[str, ...], want: set[str]) -> str:
    for position, chunk_id in enumerate(ranked, start=1):
        if chunk_id in want:
            return str(position)
    return "miss"


def _ablations(results: Day11Results) -> str:
    rows = ""
    previous: float | None = None
    for score in results.ablation_table():
        step = "—" if previous is None else f"{score.mrr - previous:+.4f}"
        colour = ""
        if previous is not None and abs(score.mrr - previous) < 1e-9:
            colour = f"color:{WARN}"
        rows += (
            f"<tr><td style='padding:5px 12px 5px 0'><code>{score.retriever}</code></td>"
            f"<td style='padding:5px 12px;text-align:right'>{score.hit_at_1:.0%}</td>"
            f"<td style='padding:5px 12px;text-align:right'>{score.mrr:.4f}</td>"
            f"<td style='padding:5px 12px;text-align:right;{colour}'>{step}</td></tr>"
        )
        previous = score.mrr
    return f"""
<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:0.9rem">
<thead><tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
  <th style="text-align:left;padding:4px 12px 4px 0">rules enabled</th>
  <th style="text-align:right;padding:4px 12px">hit@1</th>
  <th style="text-align:right;padding:4px 12px">MRR</th>
  <th style="text-align:right;padding:4px 12px">added</th>
</tr></thead><tbody>{rows}</tbody></table></div>
"""


def _depths(results: Day11Results) -> str:
    cells = ""
    for depth, score in results.depth_table():
        label = "all" if depth > 10 else str(depth)
        cells += (
            f"<tr><td style='padding:5px 12px 5px 0'>{label}</td>"
            f"<td style='padding:5px 12px;text-align:right'>{score.mrr:.4f}</td></tr>"
        )
    return f"""
<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:0.9rem">
<thead><tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
  <th style="text-align:left;padding:4px 12px 4px 0">turns remembered</th>
  <th style="text-align:right;padding:4px 12px">MRR</th>
</tr></thead><tbody>{cells}</tbody></table></div>
"""


def build() -> None:
    results = load()
    if results is None:
        gr.Markdown(MISSING)
        return

    names = [c["name"] for c in results.conversations]
    turns = sum(len(c["turns"]) for c in results.conversations)
    follow_ups = turns - len(results.conversations)

    gr.Markdown(
        f"### Day 11 · Multi-turn RAG\n"
        f"{len(results.conversations)} conversations, {turns} turns, of which "
        f"**{follow_ups} are follow-ups** that do not stand on their own — "
        f'*"is it free on Sundays"*, *"what about for a private event"*. Each is '
        f"retrieved four ways and scored against the same span labels day 9 uses.\n\n"
        f"Opening turns are scored separately and excluded from every table below: "
        f"they are standalone by construction, and averaging them in would hide the "
        f"effect under twelve questions no rewriter touches."
    )

    retriever = gr.Radio(
        RETRIEVERS,
        value="bm25",
        label="retriever",
        info="whether rewriting helps depends on what is downstream of it",
    )
    table = gr.HTML(_condition_table(results, "bm25"))

    gr.Markdown(
        "#### Replay a conversation\nThe number in each column is where the correct passage ranked."
    )
    chosen = gr.Dropdown(names, value=names[0], label="conversation")
    replay = gr.HTML()
    positions = gr.Dataframe(interactive=False, wrap=True)

    with gr.Accordion("What each rule bought, and how much memory it needed", open=False):
        gr.Markdown(
            "Cumulative, on BM25 and follow-ups only. A rule that adds nothing "
            "is marked — `substitute` rewrote one turn into a better-formed "
            "question that retrieved identically, and `carryover` never fired."
        )
        gr.HTML(_ablations(results))
        gr.Markdown(
            "`+guard` was added **after** reading the failures below, so it has "
            "seen its own test; it is reported apart from the headline for that "
            "reason. Memory depth, same slice — one turn is not enough and three "
            "is no better than two, on conversations that are only three turns long."
        )
        gr.HTML(_depths(results))

    def _refresh(name: str, which: str):
        header, frame = _replay(results, name, which)
        return _condition_table(results, which), header, frame

    for control in (chosen, retriever):
        control.change(_refresh, inputs=[chosen, retriever], outputs=[table, replay, positions])

    first = _refresh(names[0], "bm25")
    table.value, replay.value, positions.value = first
