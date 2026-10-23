"""Day 1's tab: proof the shared plumbing works end to end.

Rather than an empty placeholder, this runs a synthetic conversational turn
through tracing, cost accounting and the eval types, and shows what each one
produced. Every later day reuses exactly this machinery.
"""

from __future__ import annotations

import json
import time
from typing import Any

import gradio as gr

from convo.core import Assertion, CaseResult, CostMeter, EvalRun, ship, span, trace


def _fake_turn(user_text: str) -> tuple[dict[str, Any], CostMeter, EvalRun]:
    """One instrumented turn with no model behind it. The plumbing is the demo."""
    meter = CostMeter()
    # A placeholder price so the demo shows a number. Replace with real prices
    # in prices.toml before any figure from this repo goes in front of anyone.
    meter.register_price("demo-model", input_per_mtok=1.0, output_per_mtok=5.0)

    run = EvalRun(suite="day01-smoke")

    with trace() as active:
        with span("nlu", component="stub"):
            time.sleep(0.012)
            intent = "book_appointment" if "book" in user_text.lower() else "fallback"
            confidence = 0.91 if intent != "fallback" else 0.22

        with span("retrieve", k=3):
            time.sleep(0.02)

        with span("generate", model="demo-model"):
            time.sleep(0.03)
            reply = (
                "Happy to book that. Which day suits you?"
                if intent == "book_appointment"
                else "I didn't catch that — could you rephrase?"
            )
            meter.record(
                "demo-model",
                input_tokens=len(user_text.split()) * 4 + 120,
                output_tokens=len(reply.split()) * 4,
                label="turn-reply",
            )

        payload = active.as_dict()

    run.add(
        CaseResult.from_assertions(
            "intent-is-resolved",
            [
                Assertion("intent predicted", intent != ""),
                Assertion(
                    "confidence above rejection threshold",
                    confidence >= 0.5,
                    f"confidence={confidence:.2f}, threshold=0.50",
                ),
                Assertion("reply is non-empty", bool(reply.strip())),
            ],
            transcript=[
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": reply},
            ],
            latency_ms=payload["total_ms"],
        )
    )
    payload["reply"] = reply
    payload["intent"] = intent
    payload["confidence"] = confidence
    return payload, meter, run


def _run(user_text: str) -> tuple[str, list[list[Any]], str, str]:
    if not user_text.strip():
        return "Type something first.", [], "", ""
    payload, meter, run = _fake_turn(user_text)

    waterfall = [[row["name"], row["start_ms"], row["duration_ms"]] for row in payload["spans"]]
    cost_lines = [f"**Tokens:** {meter.total_tokens}", f"**Cost:** ${meter.total_usd:.6f}"]
    if meter.unpriced:
        cost_lines.append(f"**Unpriced models:** {sorted(meter.unpriced)}")
    cost_lines.append("")
    cost_lines.append("_Placeholder price. Fill prices.toml before quoting any figure._")

    summary = (
        f"**Reply:** {payload['reply']}\n\n"
        f"**Intent:** `{payload['intent']}` at {payload['confidence']:.2f} confidence\n\n"
        f"**Total:** {payload['total_ms']} ms across {len(payload['spans'])} spans"
    )
    return summary, waterfall, "\n\n".join(cost_lines), json.dumps(run.as_dict(), indent=2)


def build() -> None:
    gr.Markdown(
        "### The plumbing, running\n"
        "No model behind this yet. It pushes one synthetic turn through the three "
        "primitives in `convo.core` — tracing, cost accounting and the eval types — "
        "so that when day 2 arrives there is nothing left to invent."
    )
    with gr.Row():
        with gr.Column(scale=2):
            box = gr.Textbox(
                label="A user turn",
                value="I'd like to book an appointment",
                elem_id="day01-input",
            )
            go = gr.Button("Run one turn", variant="primary")
            summary = gr.Markdown()
        with gr.Column(scale=3):
            spans = gr.Dataframe(
                headers=["span", "offset ms", "duration ms"],
                label="Trace",
                interactive=False,
                wrap=True,
            )
            cost = gr.Markdown()
    with gr.Accordion("Eval run as JSON (the shape day 6 consumes)", open=False):
        evals = gr.Code(language="json", label="")

    go.click(_run, inputs=box, outputs=[summary, spans, cost, evals])
    box.submit(_run, inputs=box, outputs=[summary, spans, cost, evals])


ship(1, build)
