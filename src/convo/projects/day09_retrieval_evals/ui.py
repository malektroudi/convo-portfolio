"""Day 9's tab: the lift table, and the per-question drill-down behind it.

An aggregate that nobody can take apart is a claim, not a measurement. Picking
a question shows every retriever's ordering side by side with the relevant
chunk marked, so a reader can see exactly where a win came from — or that it
came from somewhere unconvincing.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from convo.projects.day09_retrieval_evals.results import (
    RETRIEVER_NOTE,
    RETRIEVERS,
    Day9Results,
    load,
)

COLOUR = {
    "bm25": "#eb6834",
    "dense": "#2a78d6",
    "hybrid": "#6250d6",
    "hybrid+rerank": "#1baf7a",
}
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

MISSING = """
### Day 09 · Make retrieval measurably better

No rankings artifact found.

```bash
uv sync --extra train
uv run python -m convo.projects.day09_retrieval_evals.build
```

That scores four retrievers over three chunkings against 42 labelled
questions and stores every ranking. Reload the hub afterwards.
"""


def _lift_table(results: Day9Results, strategy: str) -> str:
    rows = ""
    scores = {r: results.strategies[strategy].score(r) for r in RETRIEVERS}
    baseline = scores["dense"]
    for retriever in RETRIEVERS:
        score = scores[retriever]
        delta = score.hit_at_1 - baseline.hit_at_1
        arrow = (
            f"<span style='color:{GOOD}'>+{delta:.0%}</span>"
            if delta > 0
            else f"<span style='color:{CRITICAL}'>{delta:.0%}</span>"
            if delta < 0
            else "<span style='opacity:0.5'>baseline</span>"
        )
        rows += (
            f"<tr><td style='padding:6px 12px 6px 0'>"
            f"<span style='display:inline-block;width:8px;height:8px;border-radius:50%;"
            f"background:{COLOUR[retriever]};margin-right:7px'></span>"
            f"<code>{retriever}</code></td>"
            f"<td style='padding:6px 12px;text-align:right;font-variant-numeric:tabular-nums'>"
            f"{score.hit_at_1:.0%}</td>"
            f"<td style='padding:6px 12px;text-align:right'>{arrow}</td>"
            f"<td style='padding:6px 12px;text-align:right;font-variant-numeric:tabular-nums'>"
            f"{score.hit_at_3:.0%}</td>"
            f"<td style='padding:6px 12px;text-align:right;font-variant-numeric:tabular-nums'>"
            f"{score.mrr:.2f}</td>"
            f"<td style='padding:6px 12px;text-align:right;font-variant-numeric:tabular-nums'>"
            f"{score.ndcg_at_5:.2f}</td>"
            f"<td style='padding:6px 0;text-align:right;font-variant-numeric:tabular-nums;"
            f"color:{CRITICAL if score.misses else 'inherit'}'>{len(score.misses)}</td></tr>"
        )
    return f"""
<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:0.92rem">
<thead><tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
  <th style="text-align:left;padding:4px 12px 4px 0">retriever</th>
  <th style="text-align:right;padding:4px 12px">hit@1</th>
  <th style="text-align:right;padding:4px 12px">vs dense</th>
  <th style="text-align:right;padding:4px 12px">hit@3</th>
  <th style="text-align:right;padding:4px 12px">MRR</th>
  <th style="text-align:right;padding:4px 12px">nDCG@5</th>
  <th style="text-align:right;padding:4px 0">never found</th>
</tr></thead><tbody>{rows}</tbody></table></div>
"""


def _split_table(results: Day9Results, strategy: str) -> str:
    strat = results.strategies[strategy]
    rows = ""
    for retriever in RETRIEVERS:
        easy = strat.score_subset(retriever, hard=False)
        hard = strat.score_subset(retriever, hard=True)
        drop = easy.hit_at_1 - hard.hit_at_1
        rows += (
            f"<tr><td style='padding:6px 12px 6px 0'><code>{retriever}</code></td>"
            f"<td style='padding:6px 12px;text-align:right'>{easy.hit_at_1:.0%}</td>"
            f"<td style='padding:6px 12px;text-align:right'>{hard.hit_at_1:.0%}</td>"
            f"<td style='padding:6px 12px;text-align:right;color:{CRITICAL}'>"
            f"−{drop:.0%}</td>"
            f"<td style='padding:6px 12px;text-align:right'>{easy.mrr:.2f}</td>"
            f"<td style='padding:6px 0;text-align:right'>{hard.mrr:.2f}</td></tr>"
        )
    return f"""
<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:0.92rem">
<thead><tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
  <th style="text-align:left;padding:4px 12px 4px 0">retriever</th>
  <th style="text-align:right;padding:4px 12px">hit@1 keyword</th>
  <th style="text-align:right;padding:4px 12px">hit@1 paraphrase</th>
  <th style="text-align:right;padding:4px 12px">drop</th>
  <th style="text-align:right;padding:4px 12px">MRR keyword</th>
  <th style="text-align:right;padding:4px 0">MRR paraphrase</th>
</tr></thead><tbody>{rows}</tbody></table></div>
"""


def _questions(results: Day9Results, strategy: str) -> list[str]:
    return [q.question for q in results.strategies[strategy].questions]


def _drilldown(results: Day9Results, strategy: str, question: str) -> str:
    strat = results.strategies[strategy]
    row = next((q for q in strat.questions if q.question == question), None)
    if row is None:
        return ""

    blocks = ""
    for retriever in RETRIEVERS:
        ranked = row.rankings.get(retriever, [])
        items = ""
        for position, chunk_id in enumerate(ranked[:5], start=1):
            good = chunk_id in row.relevant
            mark = "✓" if good else "·"
            colour = GOOD if good else "inherit"
            weight = "600" if good else "400"
            items += (
                f"<div style='font-size:0.78rem;font-family:ui-monospace,monospace;"
                f"color:{colour};font-weight:{weight};padding:1px 0'>"
                f"{mark} {position}. {chunk_id}</div>"
            )
        first = next((i for i, c in enumerate(ranked, start=1) if c in row.relevant), None)
        verdict = (
            f"<span style='color:{GOOD}'>rank {first}</span>"
            if first
            else f"<span style='color:{CRITICAL}'>never</span>"
        )
        blocks += (
            f"<div style='flex:1 1 220px;min-width:0;border:1px solid "
            f"rgba(128,128,128,0.25);border-radius:6px;padding:10px 12px'>"
            f"<div style='font-weight:600;color:{COLOUR[retriever]};font-size:0.85rem'>"
            f"{retriever}</div>"
            f"<div style='font-size:0.75rem;opacity:0.7;margin:2px 0 8px'>"
            f"found at {verdict}</div>{items}</div>"
        )

    tag = (
        " <span style='font-size:0.72rem;opacity:0.6'>(paraphrase — little shared "
        "vocabulary with the answer)</span>"
        if row.hard
        else ""
    )
    return (
        f"<p style='font-size:0.85rem;margin-bottom:10px'>Relevant chunk: "
        f"<code>{', '.join(sorted(row.relevant))}</code>{tag}</p>"
        f"<div style='display:flex;gap:12px;flex-wrap:wrap'>{blocks}</div>"
    )


def _strategy_summary(results: Day9Results) -> pd.DataFrame:
    rows = []
    for name, strat in results.strategies.items():
        best = strat.score("hybrid+rerank")
        dense = strat.score("dense")
        rows.append(
            {
                "chunking": name,
                "chunks": strat.chunks,
                "dense hit@1": f"{dense.hit_at_1:.0%}",
                "best hit@1": f"{best.hit_at_1:.0%}",
                "lift": f"+{best.hit_at_1 - dense.hit_at_1:.0%}",
                "best MRR": f"{best.mrr:.2f}",
            }
        )
    return pd.DataFrame(rows)


def _refresh(strategy: str, question: str):
    results = load()
    if results is None:
        return "", "", "", gr.update()
    questions = _questions(results, strategy)
    chosen = question if question in questions else questions[0]
    return (
        _lift_table(results, strategy),
        _split_table(results, strategy),
        _drilldown(results, strategy, chosen),
        gr.update(choices=questions, value=chosen),
    )


def _benchmark_block() -> str:
    """The same four retrievers on a corpus nobody here wrote.

    Day 9's numbers come from 42 questions I wrote about 14 articles I wrote.
    That is enough to show a mechanism and not enough to claim the stack is
    good. This runs the identical code over SciFact — 5,183 documents with
    relevance judgements made by domain experts — so the two can be read side
    by side.
    """
    from convo.benchmark.results import RETRIEVERS as B_RETRIEVERS
    from convo.benchmark.results import load as load_benchmark

    found = load_benchmark()
    if found is None:
        return (
            "<p style='font-size:0.88rem;opacity:0.7'>Not run yet:<br>"
            "<code>uv run python -m convo.benchmark.run</code></p>"
        )

    rows = ""
    for retriever in B_RETRIEVERS:
        mine = found.restaurant.get(retriever, {})
        cells = ""
        for depth in found.depths:
            score = found.scores(depth, retriever)
            cells += (
                f"<td style='padding:5px 12px;text-align:right'>{score['hit_at_1']:.0%}</td>"
                f"<td style='padding:5px 12px;text-align:right'>{score['mrr']:.3f}</td>"
            )
        own = (
            f"<td style='padding:5px 12px;text-align:right;opacity:0.65'>"
            f"{mine.get('hit_at_1', 0):.0%}</td>"
            f"<td style='padding:5px 12px;text-align:right;opacity:0.65'>"
            f"{mine.get('mrr', 0):.3f}</td>"
        )
        rows += (
            f"<tr><td style='padding:5px 12px 5px 0'><code>{retriever}</code></td>{own}{cells}</tr>"
        )

    depth_headers = "".join(
        f"<th colspan='2' style='text-align:center;padding:4px 12px'>SciFact, top {d}</th>"
        for d in found.depths
    )
    sub = "".join(
        "<th style='text-align:right;padding:2px 12px;font-weight:400'>hit@1</th>"
        "<th style='text-align:right;padding:2px 12px;font-weight:400'>MRR</th>"
        for _ in range(len(found.depths) + 1)
    )

    notes = []
    for depth in found.depths:
        delta = found.rerank_delta(depth)
        word = "adds" if delta > 0 else "costs"
        colour = GOOD if delta > 0 else CRITICAL
        notes.append(
            f"At top {depth} the cross-encoder <span style='color:{colour}'>{word} "
            f"{abs(delta):.3f} MRR</span> over fusion."
        )
    mine_delta = found.restaurant_rerank_delta()
    notes.append(f"On my own corpus it added {mine_delta:+.3f}." if mine_delta else "")

    return f"""
<p style="font-size:0.88rem;max-width:72ch">
<strong>{found.documents:,} documents, {found.queries} queries, {found.judgements}
judgements</strong> made by domain experts. Same <code>BM25</code>, same
<code>DenseIndex</code>, same fusion, same reranker &mdash; imported from days 8
and 9 rather than reimplemented, because the point is to test that code.
</p>
<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:0.88rem">
<thead>
<tr style="border-bottom:1px solid rgba(128,128,128,0.2)">
  <th style="padding:4px 12px 4px 0"></th>
  <th colspan="2" style="text-align:center;padding:4px 12px;opacity:0.65">
    my 14 articles</th>
  {depth_headers}
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
  <th style="text-align:left;padding:2px 12px 2px 0">retriever</th>{sub}
</tr>
</thead><tbody>{rows}</tbody></table></div>
<p style="font-size:0.86rem;opacity:0.78;max-width:72ch;margin-top:10px">
{" ".join(n for n in notes if n)}
</p>
"""


def build() -> None:
    results = load()
    if results is None:
        gr.Markdown(MISSING)
        return

    default_strategy = "structural"
    questions = _questions(results, default_strategy)
    notes = "\n".join(f"- `{r}` — {RETRIEVER_NOTE[r]}" for r in RETRIEVERS)

    gr.Markdown(
        f"### Day 09 · Make retrieval measurably better\n"
        f"Four retrievers over three chunkings, scored against **42 labelled "
        f"questions** with hit@k, MRR and nDCG.\n\n"
        f"Day 8 ended at 100% dense@3 and could not be improved on, because the "
        f"labels were document-level and there are only fourteen documents. Here "
        f"the label is a **quote from the source**, and chunk relevance is derived "
        f"from span overlap — so one label set works for every chunking, including "
        f"ones not written yet, and finding the right *chunk* is a real problem.\n\n"
        f"{notes}\n\n"
        f"Encoder `{results.encoder}`, reranker `{results.reranker}` over the top "
        f"{results.depth}, median **{results.rerank_ms_median:.0f} ms/query**."
    )

    strategy = gr.Radio(choices=list(results.strategies), value=default_strategy, label="Chunking")

    gr.Markdown("#### The lift")
    lift = gr.HTML(_lift_table(results, default_strategy))

    with gr.Accordion(
        "The same retrievers on a public benchmark, with labels I did not write",
        open=False,
    ):
        gr.HTML(_benchmark_block())

    gr.Markdown(
        "#### Where it stops working\n"
        "Ten of the 42 questions are deliberate paraphrases — phrased the way "
        "someone actually asks, sharing almost no vocabulary with the passage that "
        "answers them. The reranker reorders candidates retrieval already found; "
        "it cannot rescue a question where the right chunk never made the shortlist."
    )
    split = gr.HTML(_split_table(results, default_strategy))

    gr.Markdown("#### One question at a time")
    question = gr.Dropdown(choices=questions, value=questions[0], label="Question")
    drill = gr.HTML(_drilldown(results, default_strategy, questions[0]))

    gr.Markdown("#### Chunking against retrieval")
    gr.Dataframe(_strategy_summary(results), interactive=False, wrap=True)
    gr.Markdown(
        "_Fewer chunks is an easier retrieval problem — `fixed` has 27 candidates "
        "against `structural`'s 49 — so its higher score is partly granularity, not "
        "quality. The finer chunking is harder to retrieve from and produces the "
        "better citation, which day 8 measured._"
    )

    outputs = [lift, split, drill, question]
    strategy.change(_refresh, inputs=[strategy, question], outputs=outputs)
    question.change(_refresh, inputs=[strategy, question], outputs=outputs)
