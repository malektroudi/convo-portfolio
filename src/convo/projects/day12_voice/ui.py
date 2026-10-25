"""Day 12's tab: the budget, the waterfall, and the endpointing slider.

The waterfall is the point. A table of four numbers tells you the total was
missed; a bar whose segments are drawn to scale tells you which segment to go
and fix, and makes the generated path's think stage look the way it actually
feels on a phone call.

The endpointing slider is live because its right answer is a product decision.
Moving it moves the dead air, the share of callers cut off mid-sentence, and
the word error rate together, which is the only honest way to present a knob
whose two costs pull in opposite directions.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from convo.projects.day12_voice.budget import BUDGET, DEADLINE_MS, UNCOMFORTABLE_MS
from convo.projects.day12_voice.results import PATHS, SAMPLE_WAV, Day12Results, load

GOOD = "#1baf7a"
WARN = "#eb6834"
CRITICAL = "#d03b3b"

STAGE_COLOUR = {
    "endpoint": "#8b8b8b",
    "stt": "#2a78d6",
    "think": "#d03b3b",
    "tts": "#1baf7a",
}

PATH_NOTE = {
    "cited": "days 8-11: retrieve, check the evidence, speak the verified span. No model.",
    "generated": "a 0.5B instruct model streaming tokens on CPU, given the same passage.",
}

MISSING = """
### Day 12 · Voice, with a latency budget

No latency artifact found.

```bash
uv sync --extra train --extra voice
uv run python -m convo.projects.day12_voice.build
```

That synthesises each question, transcribes it back with whisper-tiny.en,
answers it two ways and speaks the answer, recording where every millisecond
went. Reload the hub afterwards.
"""


def _waterfall(results: Day12Results, path: str, strategy: str) -> str:
    rows = results.stage_table(path, strategy)
    total = sum(r["measured_ms"] for r in rows)
    scale = max(total, DEADLINE_MS) * 1.05

    bar = ""
    for row in rows:
        width = row["measured_ms"] / scale * 100
        if width <= 0:
            continue
        bar += (
            f"<div title='{row['stage']}: {row['measured_ms']:.0f} ms' "
            f"style='width:{width}%;background:{STAGE_COLOUR[row['key']]};'></div>"
        )

    deadline_at = DEADLINE_MS / scale * 100
    verdict_colour = (
        GOOD if total <= DEADLINE_MS else (WARN if total <= UNCOMFORTABLE_MS else CRITICAL)
    )
    verdict = (
        "inside the budget"
        if total <= DEADLINE_MS
        else "over budget, still a conversation"
        if total <= UNCOMFORTABLE_MS
        else "past the point a caller assumes the line has dropped"
    )

    legend = " &nbsp; ".join(
        f"<span style='color:{STAGE_COLOUR[r['key']]}'>&#9632;</span> "
        f"{r['stage']} {r['measured_ms']:.0f}"
        for r in rows
    )

    return f"""
<div style="margin:6px 0 2px;font-size:0.82rem;opacity:0.7">{PATH_NOTE[path]}</div>
<div style="position:relative;margin:10px 0 6px">
  <div style="display:flex;height:26px;border-radius:4px;overflow:hidden">{bar}</div>
  <div style="position:absolute;top:-4px;bottom:-4px;left:{deadline_at}%;
              border-left:2px dashed rgba(128,128,128,0.9)"></div>
</div>
<div style="font-size:0.78rem;opacity:0.72">{legend}</div>
<p style="margin-top:10px;font-size:0.9rem">
  <strong style="color:{verdict_colour}">{total:.0f} ms</strong> to first audio
  against a {DEADLINE_MS:.0f} ms budget (dashed) — {verdict}.
  Worst stage: <strong>{results.worst_stage(path, strategy)["stage"]}</strong>,
  {results.worst_stage(path, strategy)["over_ms"]:.0f} ms over.
</p>
"""


def _budget_table(results: Day12Results, path: str, strategy: str) -> str:
    rows = ""
    for row in results.stage_table(path, strategy):
        colour = GOOD if row["over_ms"] <= 0 else (WARN if row["ratio"] < 2 else CRITICAL)
        rows += (
            f"<tr><td style='padding:5px 12px 5px 0'>{row['stage']}</td>"
            f"<td style='padding:5px 12px;text-align:right;opacity:0.6'>"
            f"{row['target_ms']:.0f}</td>"
            f"<td style='padding:5px 12px;text-align:right'>{row['measured_ms']:.0f}</td>"
            f"<td style='padding:5px 12px;text-align:right;color:{colour}'>"
            f"{row['ratio']:.2f}&times;</td>"
            f"<td style='padding:5px 0;font-size:0.78rem;opacity:0.62'>"
            f"{next(s.why for s in BUDGET if s.key == row['key'])}</td></tr>"
        )
    return f"""
<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:0.88rem">
<thead><tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
  <th style="text-align:left;padding:4px 12px 4px 0">stage</th>
  <th style="text-align:right;padding:4px 12px">budget</th>
  <th style="text-align:right;padding:4px 12px">measured</th>
  <th style="text-align:right;padding:4px 12px"></th>
  <th style="text-align:left;padding:4px 0">why it is in the budget</th>
</tr></thead><tbody>{rows}</tbody></table></div>
"""


def _token_gap(results: Day12Results) -> str:
    gaps = results.token_vs_sentence()
    rows = ""
    for path in PATHS:
        if path not in gaps:
            continue
        value = gaps[path]
        colour = CRITICAL if value["gap_ms"] > 500 else GOOD
        rows += (
            f"<tr><td style='padding:5px 12px 5px 0'><code>{path}</code></td>"
            f"<td style='padding:5px 12px;text-align:right'>{value['first_token_ms']:.0f}</td>"
            f"<td style='padding:5px 12px;text-align:right'>{value['first_sentence_ms']:.0f}</td>"
            f"<td style='padding:5px 12px;text-align:right;color:{colour};font-weight:600'>"
            f"{value['gap_ms']:+.0f}</td></tr>"
        )
    return f"""
<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:0.88rem">
<thead><tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
  <th style="text-align:left;padding:4px 12px 4px 0">reply path</th>
  <th style="text-align:right;padding:4px 12px">first token</th>
  <th style="text-align:right;padding:4px 12px">first sentence</th>
  <th style="text-align:right;padding:4px 12px">gap</th>
</tr></thead><tbody>{rows}</tbody></table></div>
"""


def _endpoint_panel(results: Day12Results, hangover: float) -> str:
    table = results.sweep_table()
    row = min(table, key=lambda r: abs(r["hangover_ms"] - hangover))
    clean = results.best_hangover()

    cut_colour = GOOD if row["truncated"] == 0 else CRITICAL
    note = (
        f"Every caller finished their sentence. The shortest setting that manages "
        f"that is <strong>{clean['hangover_ms']:.0f} ms</strong>, which is "
        f"{clean['hangover_ms'] / DEADLINE_MS:.0%} of the entire budget spent "
        f"before anything is computed."
        if row["truncated"] == 0
        else f"<strong style='color:{CRITICAL}'>{row['truncated']} of {row['of']} "
        f"callers were cut off mid-sentence.</strong> Their transcripts are missing "
        f"the second half of the question, so the word error rate on them rises to "
        f"{row['wer_hesitant']:.0%} — the turn is not slow, it is wrong."
    )

    return f"""
<div style="display:flex;gap:28px;flex-wrap:wrap;margin:6px 0 10px">
  <div><div style="font-size:0.72rem;opacity:0.6;text-transform:uppercase">dead air</div>
    <div style="font-size:1.35rem;font-weight:600">{row["wait_ms"]:.0f} ms</div></div>
  <div><div style="font-size:0.72rem;opacity:0.6;text-transform:uppercase">cut off</div>
    <div style="font-size:1.35rem;font-weight:600;color:{cut_colour}">
      {row["truncated"]}/{row["of"]}</div></div>
  <div><div style="font-size:0.72rem;opacity:0.6;text-transform:uppercase">WER, hesitant</div>
    <div style="font-size:1.35rem;font-weight:600">{row["wer_hesitant"]:.0%}</div></div>
  <div><div style="font-size:0.72rem;opacity:0.6;text-transform:uppercase">WER, fluent</div>
    <div style="font-size:1.35rem;font-weight:600">{row["wer_fluent"]:.0%}</div></div>
</div>
<p style="font-size:0.88rem">{note}</p>
"""


def _turns_frame(results: Day12Results, path: str, strategy: str) -> pd.DataFrame:
    rows = []
    for turn in results.select(path, strategy):
        rows.append(
            {
                "question": turn.label,
                "heard": turn.heard,
                "endpoint": round(turn.stage_ms("endpoint")),
                "stt": round(turn.stage_ms("stt")),
                "think": round(turn.stage_ms("think")),
                "tts": round(turn.stage_ms("tts")),
                "total": round(turn.total_ms),
                "met": "yes" if turn.met else "no",
            }
        )
    return pd.DataFrame(rows)


def build() -> None:
    results = load()
    if results is None:
        gr.Markdown(MISSING)
        return

    meta = results.meta
    gr.Markdown(
        f"### Day 12 · Voice, with a latency budget\n"
        f"Real audio end to end: **{meta['tts_voice']}** speaks the question, "
        f"**{meta['stt_model']}** transcribes it, one of two paths answers, and the "
        f"answer is spoken back. Every number below was measured on a laptop CPU.\n\n"
        f"The deadline is **{DEADLINE_MS:.0f} ms from the caller stopping speaking to "
        f"the first audio sample** — not to the end of the reply. What a person "
        f"experiences is the silence they sit through, and nothing after the first "
        f"sample is inside it.\n\n"
        f"*The input audio is synthetic, so the word error rates here are a floor "
        f"rather than a result. The latency is the measurement.*"
    )

    with gr.Row():
        path = gr.Radio(list(PATHS), value="cited", label="reply path")
        strategy = gr.Radio(
            ["stream", "wait"],
            value="stream",
            label="TTS strategy",
            info="speak the first sentence as soon as it exists, or wait for the whole reply",
        )

    waterfall = gr.HTML(_waterfall(results, "cited", "stream"))
    table = gr.HTML(_budget_table(results, "cited", "stream"))

    gr.Markdown(
        "#### Time-to-first-token is the wrong metric for voice\n"
        "A token cannot be spoken. What gates the audio is the first *complete "
        "sentence*, and on a streamed model the two are seconds apart — so a "
        "dashboard tracking TTFT would show this pipeline looking healthy."
    )
    gr.HTML(_token_gap(results))

    gr.Markdown(
        "#### Endpointing: the stage whose target is a product decision\n"
        "Nothing is computed while the endpointer waits. Every millisecond is "
        "dead air — and shortening it does not save time, it cuts people off."
    )
    hangover = gr.Slider(
        100,
        1200,
        value=results.meta["default_hangover_ms"],
        step=100,
        label="silence held before the turn is called finished (ms)",
    )
    endpoint_panel = gr.HTML(_endpoint_panel(results, results.meta["default_hangover_ms"]))

    with gr.Accordion("Every measured turn", open=False):
        turns = gr.Dataframe(_turns_frame(results, "cited", "stream"), interactive=False, wrap=True)

    if SAMPLE_WAV.exists():
        gr.Markdown("#### What the caller actually hears")
        gr.Audio(str(SAMPLE_WAV), label="a spoken reply from the cited path")

    def _refresh(which_path: str, which_strategy: str):
        return (
            _waterfall(results, which_path, which_strategy),
            _budget_table(results, which_path, which_strategy),
            _turns_frame(results, which_path, which_strategy),
        )

    for control in (path, strategy):
        control.change(_refresh, inputs=[path, strategy], outputs=[waterfall, table, turns])
    hangover.change(
        lambda value: _endpoint_panel(results, value), inputs=hangover, outputs=endpoint_panel
    )
