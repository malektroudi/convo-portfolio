"""Day 4's tab: the conversation on the left, the machine's state on the right.

The inspector is the point. A chat transcript alone tells you nothing about why
a bot did something; showing the page, the form, the last transition and what
the turn extracted makes the mechanism legible — and it is the same record the
day-6 eval harness will assert against.
"""

from __future__ import annotations

from datetime import date

import gradio as gr

from convo.projects.day04_dialog_engine import flow as fl
from convo.projects.day04_dialog_engine.engine import Session, TurnResult, new_session, turn
from convo.projects.day04_dialog_engine.extract import Reference

# Reserved status colours, each paired with a word so colour never carries the
# meaning alone.
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"
ACCENT = "#2a78d6"

PAGE_ORDER = (fl.COLLECT, fl.CONFIRM, fl.AMEND, fl.BOOKED, fl.HANDOFF)

SCRIPTS: dict[str, list[str]] = {
    "Happy path": ["a table for 2 tomorrow at 8pm", "my name is chen", "yes"],
    "Repair at confirmation": [
        "a table for 4",
        "friday",
        "7:30pm",
        "my name is sarah",
        "no, make it 8pm",
        "yes",
    ],
    "Digression mid-form": [
        "a table for 4",
        "do you have parking?",
        "friday",
        "are you wheelchair accessible?",
        "7pm",
        "under lee",
        "yes",
    ],
    "Repeated failure": ["hmm", "uhh", "???"],
}


def _dot(colour: str) -> str:
    return (
        f"<span style='display:inline-block;width:7px;height:7px;border-radius:50%;"
        f"background:{colour};margin-right:6px;vertical-align:middle'></span>"
    )


def _render_pages(session: Session) -> str:
    rows = []
    for name in PAGE_ORDER:
        page = session.flow.page(name)
        current = name == session.page
        colour = CRITICAL if name == fl.HANDOFF else GOOD if name == fl.BOOKED else ACCENT
        mark = _dot(colour) if current else _dot("transparent")
        weight = "600" if current else "400"
        opacity = "1" if current else "0.5"
        label = "here" if current else ""
        rows.append(
            f"<tr style='opacity:{opacity}'>"
            f"<td style='padding:3px 8px 3px 0;white-space:nowrap;font-weight:{weight}'>"
            f"{mark}<code>{name}</code></td>"
            f"<td style='padding:3px 8px;font-size:0.78rem'>{page.description}</td>"
            f"<td style='padding:3px 0;font-size:0.72rem;color:{ACCENT};font-weight:600'>"
            f"{label}</td></tr>"
        )
    return (
        "<div style='font-size:0.85rem'><strong>Pages</strong>"
        f"<table style='border-collapse:collapse;margin-top:6px;width:100%'>"
        f"{''.join(rows)}</table></div>"
    )


def _render_slots(session: Session, result: TurnResult | None) -> str:
    pending = result.pending_slot if result else None
    rejected = dict(result.rejected) if result else {}
    repaired = set(result.repaired) if result else set()

    rows = []
    for name in session.flow.page(fl.COLLECT).slots:
        spec = session.flow.spec(name)
        if name in rejected:
            status, colour = "rejected", CRITICAL
            value = f"<em style='opacity:0.7'>{rejected[name]}</em>"
        elif name in session.slots:
            status = "repaired" if name in repaired else "filled"
            colour = WARNING if name in repaired else GOOD
            value = f"<strong>{spec.display(session.slots[name])}</strong>"
        elif name == pending:
            status, colour = "asking", WARNING
            value = "<span style='opacity:0.55'>waiting on this</span>"
        else:
            status, colour = "empty", "transparent"
            value = "<span style='opacity:0.4'>—</span>"
        rows.append(
            f"<tr><td style='padding:4px 8px 4px 0;white-space:nowrap'>{_dot(colour)}"
            f"{spec.label}</td>"
            f"<td style='padding:4px 8px;font-size:0.85rem'>{value}</td>"
            f"<td style='padding:4px 0;font-size:0.7rem;opacity:0.65;white-space:nowrap'>"
            f"{status}</td></tr>"
        )
    return (
        "<div style='font-size:0.85rem;margin-top:14px'><strong>Form</strong>"
        f"<table style='border-collapse:collapse;margin-top:6px;width:100%'>"
        f"{''.join(rows)}</table></div>"
    )


def _render_turn(result: TurnResult | None) -> str:
    if result is None:
        return (
            "<div style='font-size:0.85rem;margin-top:14px'><strong>Last turn</strong>"
            "<p style='opacity:0.55;font-size:0.8rem'>Say something to see the "
            "transition, what was extracted, and which slots moved.</p></div>"
        )

    def line(label: str, value: str, colour: str = "inherit") -> str:
        return (
            f"<tr><td style='padding:3px 10px 3px 0;opacity:0.6;white-space:nowrap;"
            f"font-size:0.75rem'>{label}</td>"
            f"<td style='padding:3px 0;font-size:0.8rem;color:{colour}'>{value}</td></tr>"
        )

    found = ", ".join(result.extracted.found) or "<em style='opacity:0.6'>nothing</em>"
    rows = [
        line("transition", f"<code>{result.transition}</code>"),
        line("page", f"<code>{result.page_before}</code> → <code>{result.page_after}</code>"),
        line("extracted", found),
    ]
    if result.filled:
        rows.append(line("filled", ", ".join(result.filled), GOOD))
    if result.repaired:
        rows.append(line("repaired", ", ".join(result.repaired), WARNING))
    if result.rejected:
        rows.append(line("rejected", ", ".join(slot for slot, _ in result.rejected), CRITICAL))
    if result.digression:
        rows.append(line("digression", f"<code>{result.digression}</code>", ACCENT))
    if result.reprompts:
        rows.append(line("failed prompts", str(result.reprompts), WARNING))
    rows.append(line("turn time", f"{result.latency_ms:.2f} ms"))

    return (
        "<div style='font-size:0.85rem;margin-top:14px'><strong>Last turn</strong>"
        f"<table style='border-collapse:collapse;margin-top:6px'>{''.join(rows)}</table>"
        "</div>"
    )


def _inspector(session: Session, result: TurnResult | None) -> tuple[str, str, str]:
    return _render_pages(session), _render_slots(session, result), _render_turn(result)


def _fresh() -> Session:
    return new_session(Reference(today=date.today()))


def _messages(session: Session) -> list[dict[str, str]]:
    out: list[dict[str, str]] = [{"role": "assistant", "content": session.flow.greeting}]
    for item in session.history:
        out.append({"role": "user", "content": item.user_text})
        out.append({"role": "assistant", "content": item.reply})
    return out


def _send(text: str, session: Session):
    if not text.strip():
        last = session.history[-1] if session.history else None
        return _messages(session), session, "", *_inspector(session, last)
    result = turn(session, text)
    return _messages(session), session, "", *_inspector(session, result)


def _reset():
    session = _fresh()
    return _messages(session), session, "", *_inspector(session, None)


def _run_script(name: str):
    session = _fresh()
    result = None
    for utterance in SCRIPTS[name]:
        result = turn(session, utterance)
    return _messages(session), session, "", *_inspector(session, result)


def build() -> None:
    today = date.today()
    gr.Markdown(
        f"### Day 04 · Deterministic dialogue engine\n"
        f"A restaurant booking as an explicit state machine: pages, a form of typed "
        f"parameters, per-slot validators, a confirmation step, and routes between them. "
        f"**No model anywhere** — intent and entity handling are patterns and parsers.\n\n"
        f"This is the same shape Conversational Agents (formerly Dialogflow CX) gives you, "
        f"written out so the mechanism is visible rather than drawn in a console. Day 19 "
        f"rebuilds this exact flow in the managed product and compares the two.\n\n"
        f'The two cases worth trying: a **repair** ("no, make it 8pm" at the confirmation '
        f"must change the time and re-confirm, not dump you in a menu) and a **digression** "
        f'("do you have parking?" mid-form must be answered and then resume at the same '
        f"slot). Today is {today.strftime('%A %d %B %Y')}, so relative dates resolve from there."
    )

    state = gr.State(_fresh)

    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(
                value=[{"role": "assistant", "content": fl.BOOKING_FLOW.greeting}],
                type="messages",
                height=400,
                label="Conversation",
            )
            box = gr.Textbox(
                placeholder="a table for 4 on friday at 7:30pm…",
                label="",
                submit_btn=True,
            )
            gr.Markdown("_Replay a scripted conversation:_")
            script_buttons: list[tuple[str, gr.Button]] = []
            with gr.Row():
                for script_name in SCRIPTS:
                    script_buttons.append((script_name, gr.Button(script_name, size="sm")))
            reset = gr.Button("Start a new conversation", size="sm")
        with gr.Column(scale=2):
            pages_view = gr.HTML(_render_pages(_fresh()))
            slots_view = gr.HTML(_render_slots(_fresh(), None))
            turn_view = gr.HTML(_render_turn(None))

    outputs = [chat, state, box, pages_view, slots_view, turn_view]
    box.submit(_send, inputs=[box, state], outputs=outputs)
    reset.click(_reset, outputs=outputs)
    for script_name, button in script_buttons:
        button.click(_run_script, inputs=gr.State(script_name), outputs=outputs)
