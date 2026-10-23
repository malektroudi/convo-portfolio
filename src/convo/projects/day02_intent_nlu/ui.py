"""Day 2's tab: the rejection threshold as a product decision you can move.

Everything on screen recomputes from cached logits, so the slider is instant.
That is the point of the layout — the tradeoff is only legible if you can feel
it move.
"""

from __future__ import annotations

import gradio as gr
import numpy as np
import pandas as pd

from convo.projects.day02_intent_nlu import analysis as an
from convo.projects.day02_intent_nlu.results import Results, load

# Categorical slots 1 and 2 from the reference palette, validated with
# scripts/validate_palette.js against Gradio's light and dark surfaces:
# all six checks pass in both modes (worst adjacent CVD dE 24.7 / 26.8).
SERIES_IN_SCOPE = "#2a78d6"
SERIES_OOS = "#eb6834"

# Sequential blue ramp, 100 -> 700, for the confusion heatmap. One hue,
# light to dark; the lightest step is allowed to recede toward the surface.
BLUE_RAMP = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)
INK_ON_LIGHT = "#0b0b0b"
INK_ON_DARK = "#f0efec"
# Reserved status colour, never a series colour; paired with an icon and label.
STATUS_WARNING = "#fab219"
# Steps at this index and beyond are dark enough to need light ink.
INK_FLIP = 7

MISSING = """
### Day 02 · Intent classifier with an out-of-scope door

No trained artifact found, so there is nothing to analyse yet.

```bash
uv sync --extra train
uv run python -m convo.projects.day02_intent_nlu.train --save-model
```

That fine-tunes a small encoder on CLINC150's 150 in-scope intents, then writes
the test-set logits into the package. Reload the hub afterwards.
"""


def _stat_tiles(m: an.Metrics) -> str:
    cells = [
        ("In-scope accuracy", f"{m.in_scope_accuracy:.1%}", "correct, of in-scope turns"),
        ("Out-of-scope recall", f"{m.oos_recall:.1%}", "correctly refused"),
        ("False rejection", f"{m.false_rejection_rate:.1%}", "in-scope turns wrongly refused"),
        ("Overall", f"{m.overall_accuracy:.1%}", f"across all {m.in_scope_n + m.oos_n} turns"),
    ]
    tiles = "".join(
        f'<div style="flex:1 1 150px;min-width:0;padding:12px 14px;'
        f'border:1px solid rgba(128,128,128,0.25);border-radius:6px">'
        f'<div style="font-size:0.72rem;letter-spacing:0.05em;text-transform:uppercase;'
        f'opacity:0.65">{title}</div>'
        f'<div style="font-size:1.6rem;font-weight:600;line-height:1.25">{value}</div>'
        f'<div style="font-size:0.74rem;opacity:0.6">{note}</div></div>'
        for title, value, note in cells
    )
    return f'<div style="display:flex;gap:10px;flex-wrap:wrap">{tiles}</div>'


def _tradeoff_frame(results: Results) -> pd.DataFrame:
    rows = an.sweep(results.logits, results.labels)
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "threshold": [m.threshold for m in rows],
                    "rate": [m.in_scope_accuracy for m in rows],
                    "series": "In-scope accuracy",
                }
            ),
            pd.DataFrame(
                {
                    "threshold": [m.threshold for m in rows],
                    "rate": [m.oos_recall for m in rows],
                    "series": "Out-of-scope recall",
                }
            ),
        ],
        ignore_index=True,
    )
    return frame


def _heatmap(results: Results, threshold: float, n_intents: int) -> str:
    """A confusion matrix small enough to read, over the worst intents.

    A 150x150 matrix is a grey smear; the intents that actually fail are the
    ones worth looking at, and the last column is where rejections land.
    """
    worst = an.worst_intents(
        results.logits, results.labels, results.class_names, threshold, top_k=n_intents
    )
    if not worst:
        return "<p>No in-scope test examples to show.</p>"

    indices = [row[0] for row in worst]
    names = [row[1] for row in worst]
    matrix = an.confusion_submatrix(results.logits, results.labels, indices, threshold)
    supports = np.array([row[3] for row in worst], dtype=float)

    def cell(count: int, support: float, row_name: str, col_name: str) -> str:
        share = count / support if support else 0.0
        if count == 0:
            style = "background:transparent;opacity:0.35"
        else:
            step = min(int(share * len(BLUE_RAMP)), len(BLUE_RAMP) - 1)
            ink = INK_ON_DARK if step >= INK_FLIP else INK_ON_LIGHT
            style = f"background:{BLUE_RAMP[step]};color:{ink}"
        tip = f"true: {row_name} → predicted: {col_name} · {count} of {int(support)} ({share:.0%})"
        return (
            f'<td title="{tip}" style="{style};text-align:center;padding:5px 7px;'
            f'border-radius:3px;font-variant-numeric:tabular-nums;font-size:0.8rem">'
            f"{count or ''}</td>"
        )

    head = "".join(
        f'<th style="writing-mode:vertical-rl;transform:rotate(180deg);'
        f'font-weight:500;font-size:0.72rem;padding:4px 2px;white-space:nowrap">{n}</th>'
        for n in names
    )
    body = ""
    for slot, row_name in enumerate(names):
        cells = "".join(
            cell(int(matrix[slot, col]), supports[slot], row_name, names[col])
            for col in range(len(names))
        )
        rejected = cell(int(matrix[slot, len(names)]), supports[slot], row_name, "(rejected)")
        body += (
            f'<tr><th style="text-align:right;font-weight:500;font-size:0.78rem;'
            f'padding:4px 8px;white-space:nowrap">{row_name}</th>{cells}'
            f'<td style="width:8px"></td>{rejected}</tr>'
        )

    return f"""
<div style="overflow-x:auto">
  <table style="border-collapse:separate;border-spacing:2px;margin-top:4px">
    <thead><tr><th></th>{head}<th></th>
      <th style="writing-mode:vertical-rl;transform:rotate(180deg);font-weight:600;
                 font-size:0.72rem;padding:4px 2px">(rejected)</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</div>
<p style="font-size:0.78rem;opacity:0.65;margin-top:8px">
  Rows are true intents (the {len(names)} with the worst recall at this threshold), columns
  predictions. Colour is each cell's share of its row, so rows are comparable despite
  different support. Blank means zero. Hover a cell for counts.
</p>
"""


def _pairs_frame(results: Results, threshold: float) -> pd.DataFrame:
    pairs = an.confusion_pairs(
        results.logits,
        results.labels,
        results.class_names,
        threshold,
        top_k=15,
        include_rejections=True,
    )
    if not pairs:
        return pd.DataFrame({"true intent": [], "predicted": [], "turns": []})
    return pd.DataFrame(
        {
            "true intent": [p.true_intent for p in pairs],
            "predicted": [p.predicted_intent for p in pairs],
            "turns": [p.count for p in pairs],
        }
    )


def _refresh(threshold: float, n_intents: int):
    results = load()
    if results is None:
        empty = pd.DataFrame()
        return "", empty, "", empty
    m = an.metrics_at(results.logits, results.labels, threshold)
    return (
        _stat_tiles(m),
        _tradeoff_frame(results),
        _heatmap(results, threshold, int(n_intents)),
        _pairs_frame(results, threshold),
    )


def build() -> None:
    results = load()
    if results is None:
        gr.Markdown(MISSING)
        return

    recommended = an.best_threshold(results.logits, results.labels)
    meta = results.metadata

    gr.Markdown(
        f"### Day 02 · Intent classifier with an out-of-scope door\n"
        f"`{meta['model']}` fine-tuned on **{meta['dataset']}** — the 150 in-scope intents "
        f"only. Out-of-scope is never trained on; it is refused at inference by thresholding "
        f"the top softmax probability, because a client's real out-of-scope traffic never "
        f"looks like a curated OOS split.\n\n"
        f"That makes this slider a product decision rather than a hyperparameter. "
        f"Test set: **{results.metadata['test_in_scope']}** in-scope turns and "
        f"**{results.metadata['test_oos']}** out-of-scope."
    )

    confidence = an.confidence_stats(results.logits)
    if confidence.is_degenerate:
        gr.HTML(
            f'<div style="padding:12px 14px;border-radius:6px;'
            f'border:1px solid {STATUS_WARNING};background:rgba(250,178,25,0.12)">'
            f"<strong>⚠ This model is undertrained, so the threshold is a dead control.</strong>"
            f"<br>Top softmax confidence never exceeds "
            f"<code>{confidence.maximum:.3f}</code> across the test set "
            f"(median <code>{confidence.p50:.3f}</code>), so any threshold above that "
            f"rejects every turn. Train for more epochs at a higher learning rate "
            f"before reading anything below as a real tradeoff.</div>"
        )

    with gr.Row():
        threshold = gr.Slider(
            0.0,
            1.0,
            value=recommended.threshold,
            step=0.01,
            label="Rejection threshold",
            scale=4,
        )
        reset = gr.Button(f"Best balanced ({recommended.threshold:.2f})", scale=1)

    opening = an.metrics_at(results.logits, results.labels, recommended.threshold)
    tiles = gr.HTML(_stat_tiles(opening))

    gr.Markdown("#### What the threshold costs and buys")
    tradeoff = gr.LinePlot(
        _tradeoff_frame(results),
        x="threshold",
        y="rate",
        color="series",
        color_map={"In-scope accuracy": SERIES_IN_SCOPE, "Out-of-scope recall": SERIES_OOS},
        y_lim=[0, 1],
        x_title="Rejection threshold",
        y_title="Rate",
        height=260,
    )

    gr.Markdown("#### Where it actually fails")
    with gr.Row():
        n_intents = gr.Slider(5, 20, value=12, step=1, label="Intents shown (worst recall first)")
    heatmap = gr.HTML(_heatmap(results, recommended.threshold, 12))

    gr.Markdown("#### The mistakes worth fixing")
    pairs = gr.Dataframe(
        _pairs_frame(results, recommended.threshold),
        interactive=False,
        wrap=True,
    )

    with gr.Accordion("Training run and confidence health", open=False):
        gr.Code(
            "\n".join(f"{k:>16}: {v}" for k, v in meta.items() if k != "class_names")
            + f"\n{'classes':>16}: {len(meta['class_names'])}"
            + "\n\nmax-softmax confidence over the test set\n"
            + "\n".join(f"{k:>16}: {v}" for k, v in confidence.as_dict().items())
            + f"\n{'degenerate':>16}: {confidence.is_degenerate}",
            label="",
        )

    outputs = [tiles, tradeoff, heatmap, pairs]
    threshold.change(_refresh, inputs=[threshold, n_intents], outputs=outputs)
    n_intents.change(_refresh, inputs=[threshold, n_intents], outputs=outputs)
    reset.click(lambda: recommended.threshold, outputs=threshold)
