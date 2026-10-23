"""Day 3's tab: when to fine-tune, and when not to bother.

Two thresholds, not one, because the scores are not on the same scale — the
classifier's confidence is a softmax probability and the router's is a cosine
similarity. Putting both on one slider would be the kind of quiet category
error that makes a comparison look rigorous and mean nothing.
"""

from __future__ import annotations

import json

import gradio as gr
import pandas as pd

from convo.projects.day03_embed_router import analysis as an
from convo.projects.day03_embed_router.results import Comparison, load

# Categorical slots 1 and 2 from the reference palette, validated against
# Gradio's light and dark surfaces: all six checks pass in both modes.
SERIES_ROUTER = "#2a78d6"
SERIES_CLASSIFIER = "#eb6834"

ROUTER = "Embedding router"
CLASSIFIER = "Fine-tuned classifier"

SEGMENT_LABEL = {
    an.KNOWN: "Known intents",
    an.NEW: "New intents",
    an.OUT_OF_SCOPE: "Out-of-scope",
}

MISSING = """
### Day 03 · Embedding router against the fine-tuned model

No comparison artifact found.

```bash
uv sync --extra train
uv run python -m convo.projects.day03_embed_router.benchmark
```

That withholds ten intents from the classifier, trains it on the remaining 140,
indexes all 150 for the router from five examples each of the new ones, and
scores both on the untouched test set. Reload the hub afterwards.
"""


def _segment_frame(c: Comparison, router_t: float, classifier_t: float) -> pd.DataFrame:
    rows = []
    for name, scores, threshold in (
        (ROUTER, c.router_scores, router_t),
        (CLASSIFIER, c.classifier_scores, classifier_t),
    ):
        for metrics in an.segment_report(scores, c.labels, c.held_out, threshold):
            rows.append(
                {
                    "segment": SEGMENT_LABEL[metrics.segment],
                    "correct": metrics.correct,
                    "approach": name,
                }
            )
    return pd.DataFrame(rows)


def _headline(c: Comparison, router_t: float, classifier_t: float) -> str:
    router = {
        r.segment: r for r in an.segment_report(c.router_scores, c.labels, c.held_out, router_t)
    }
    clf = {
        r.segment: r
        for r in an.segment_report(c.classifier_scores, c.labels, c.held_out, classifier_t)
    }

    def row(label: str, segment: str) -> str:
        r, k = router[segment].correct, clf[segment].correct
        lead = ROUTER if r > k else CLASSIFIER if k > r else "tie"
        colour = SERIES_ROUTER if r > k else SERIES_CLASSIFIER if k > r else "inherit"
        return (
            f"<tr><td style='padding:6px 12px 6px 0'>{label}"
            f"<span style='opacity:0.55'> · n={router[segment].n}</span></td>"
            f"<td style='padding:6px 12px;text-align:right;font-variant-numeric:tabular-nums'>"
            f"{r:.1%}</td>"
            f"<td style='padding:6px 12px;text-align:right;font-variant-numeric:tabular-nums'>"
            f"{k:.1%}</td>"
            f"<td style='padding:6px 0;color:{colour};font-weight:600'>{lead}</td></tr>"
        )

    return f"""
<div style="overflow-x:auto">
<table style="border-collapse:collapse;font-size:0.92rem">
  <thead><tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
    <th style="text-align:left;padding:4px 12px 4px 0">Segment</th>
    <th style="text-align:right;padding:4px 12px">{ROUTER}</th>
    <th style="text-align:right;padding:4px 12px">{CLASSIFIER}</th>
    <th style="text-align:left;padding:4px 0">Wins</th>
  </tr></thead>
  <tbody>
    {row("Known intents", an.KNOWN)}
    {row("New intents", an.NEW)}
    {row("Out-of-scope refused", an.OUT_OF_SCOPE)}
  </tbody>
</table>
</div>
"""


def _new_intent_frame(c: Comparison, router_t: float) -> pd.DataFrame:
    rows = an.per_new_intent_recall(c.router_scores, c.labels, c.held_out, c.class_names, router_t)
    if not rows:
        return pd.DataFrame({"new intent": [], "router recall": [], "test turns": []})
    return pd.DataFrame(
        {
            "new intent": [r[0] for r in rows],
            "router recall": [f"{r[1]:.0%}" for r in rows],
            "test turns": [r[2] for r in rows],
        }
    )


def _operational(c: Comparison) -> str:
    r = c.metadata["router"]
    k = c.metadata["classifier"]
    add = r["add_intent_s"]
    train_s = k["train_s"]
    speedup = int(train_s / add) if add > 0 else None

    def cell(value: str, emphasis: bool = False) -> str:
        weight = "600" if emphasis else "400"
        return (
            f"<td style='padding:6px 12px;text-align:right;font-weight:{weight};"
            f"font-variant-numeric:tabular-nums'>{value}</td>"
        )

    rows = [
        ("Fine-tuning required", "none", f"{train_s / 60:.0f} min", True),
        (
            "Add one new intent",
            f"{add * 1000:.0f} ms",
            f"{train_s / 60:.0f} min (full retrain)",
            True,
        ),
        ("Index / model build", f"{r['build_s'] * 1000:.0f} ms", f"{train_s / 60:.0f} min", False),
        (
            "Per-query latency",
            f"{r['embed_latency_ms'] + r['score_latency_ms']:.0f} ms",
            f"{k['latency_ms']:.0f} ms",
            False,
        ),
        (
            "Stored per intent",
            f"~{r['exemplars'] // max(c.metadata['canonical_classes'], 1)} vectors",
            "one output row",
            False,
        ),
    ]
    body = "".join(
        f"<tr><td style='padding:6px 12px 6px 0'>{label}</td>"
        f"{cell(router_value, emph)}{cell(clf_value, emph)}</tr>"
        for label, router_value, clf_value, emph in rows
    )
    note = (
        f"<p style='font-size:0.8rem;opacity:0.7;margin-top:8px'>Adding an intent is "
        f"<strong>{speedup:,}x</strong> faster on the router, and that is the real argument. "
        f"Note the latency row: the router is <em>slower</em> per query, because it runs the "
        f"same encoder forward pass and then searches an index. Its win is iteration speed, "
        f"not inference speed — anyone who tells you otherwise has not measured it.</p>"
        if speedup
        else ""
    )
    return f"""
<div style="overflow-x:auto">
<table style="border-collapse:collapse;font-size:0.92rem">
  <thead><tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
    <th style="text-align:left;padding:4px 12px 4px 0">Operational cost</th>
    <th style="text-align:right;padding:4px 12px">{ROUTER}</th>
    <th style="text-align:right;padding:4px 12px">{CLASSIFIER}</th>
  </tr></thead>
  <tbody>{body}</tbody>
</table>
</div>{note}
"""


def _refresh(router_t: float, classifier_t: float):
    c = load()
    if c is None:
        return "", pd.DataFrame(), pd.DataFrame()
    return (
        _headline(c, router_t, classifier_t),
        _segment_frame(c, router_t, classifier_t),
        _new_intent_frame(c, router_t),
    )


def build() -> None:
    c = load()
    if c is None:
        gr.Markdown(MISSING)
        return

    meta = c.metadata
    router_best = an.best_threshold_for_segments(c.router_scores, c.labels, c.held_out)
    clf_best = an.best_threshold_for_segments(c.classifier_scores, c.labels, c.held_out)

    gr.Markdown(
        f"### Day 03 · Embedding router against the fine-tuned model\n"
        f"Ten of the 150 intents were **withheld from the classifier entirely**, so it was "
        f"trained on {meta['known_classes']} classes and its output layer has no slot for "
        f"the other ten. The router indexes those same {meta['known_classes']} in full plus "
        f"**{meta['n_shot']} examples each** of the ten new ones, using the base encoder with "
        f"no fine-tuning at all. Both are scored on the untouched test set.\n\n"
        f"Asking a 150-class model about a 151st intent is not a hard question for it, it is "
        f"an impossible one — and the only honest way to show that is to make the new intents "
        f"genuinely absent from training.\n\n"
        f"Withheld: `{'`, `'.join(c.held_out_names)}`"
    )

    with gr.Row():
        router_t = gr.Slider(
            0.0,
            1.0,
            value=router_best,
            step=0.01,
            label="Router threshold (cosine similarity)",
            scale=1,
        )
        clf_t = gr.Slider(
            0.0,
            1.0,
            value=clf_best,
            step=0.01,
            label="Classifier threshold (softmax probability)",
            scale=1,
        )
    gr.Markdown(
        "_Two sliders, because the scores are not on the same scale. One slider would "
        "be a category error dressed up as a fair test._"
    )

    headline = gr.HTML(_headline(c, router_best, clf_best))

    gr.Markdown("#### Accuracy by segment")
    chart = gr.BarPlot(
        _segment_frame(c, router_best, clf_best),
        x="segment",
        y="correct",
        color="approach",
        color_map={ROUTER: SERIES_ROUTER, CLASSIFIER: SERIES_CLASSIFIER},
        y_lim=[0, 1],
        x_title="",
        y_title="Correct",
        height=300,
    )

    gr.Markdown(
        "#### The ten new intents, five examples each\n"
        "Few-shot routing is uneven in a way the average hides: a distinctive intent "
        "routes almost perfectly from five examples, while one that overlaps an existing "
        "intent barely routes at all. The classifier scores zero on every row here, by "
        "construction rather than by weakness."
    )
    new_intents = gr.Dataframe(_new_intent_frame(c, router_best), interactive=False, wrap=True)

    gr.Markdown("#### What each approach actually costs")
    gr.HTML(_operational(c))

    with gr.Accordion("Experiment setup", open=False):
        gr.Code(
            json.dumps({k: v for k, v in meta.items() if k != "class_names"}, indent=2),
            language="json",
            label="",
        )

    outputs = [headline, chart, new_intents]
    router_t.change(_refresh, inputs=[router_t, clf_t], outputs=outputs)
    clf_t.change(_refresh, inputs=[router_t, clf_t], outputs=outputs)
