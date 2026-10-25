"""Day 13's tab: the policy table, a conversation replayed, and the packet.

The packet comparison is the part worth looking at. Three ways of writing the
same handoff, scored on what the receiving agent would have to ask the
customer to repeat — which is the thing people actually hate about being
transferred, and the thing an escalation metric built on precision and recall
cannot see at all.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from convo.projects.day13_escalation.results import PACKET_STYLES, Day13Results, load

GOOD = "#1baf7a"
WARN = "#eb6834"
CRITICAL = "#d03b3b"

MISSING = """
### Day 13 · Escalation and agent assist

No escalation artifact found.

```bash
uv sync --extra train
uv run python -m convo.projects.day13_escalation.build
```

That replays sixteen conversations through days 4 and 8-10 for real, scores
every policy against them and builds the handoff packets. Reload the hub
afterwards.
"""


def _policy_table(results: Day13Results) -> str:
    best = results.best_policy()
    rows = ""
    for row in results.policy_rows():
        is_best = row["policy"] == best["policy"]
        weight = "font-weight:600" if is_best else ""
        late_colour = (
            GOOD if row["turns_late"] <= 1 else (WARN if row["turns_late"] <= 6 else CRITICAL)
        )
        waste_colour = GOOD if row["wasted_handoffs"] <= 1 else CRITICAL
        rows += (
            f"<tr><td style='padding:5px 12px 5px 0;{weight}'><code>{row['policy']}</code></td>"
            f"<td style='padding:5px 12px;text-align:right'>{row['recall']:.0%}</td>"
            f"<td style='padding:5px 12px;text-align:right'>{row['precision']:.0%}</td>"
            f"<td style='padding:5px 12px;text-align:right'>"
            f"{row['on_time']}/{row['of_needed']}</td>"
            f"<td style='padding:5px 12px;text-align:right;color:{late_colour};{weight}'>"
            f"{row['turns_late']}</td>"
            f"<td style='padding:5px 12px;text-align:right;color:{waste_colour}'>"
            f"{row['wasted_handoffs']}</td></tr>"
        )
    return f"""
<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:0.9rem">
<thead><tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
  <th style="text-align:left;padding:4px 12px 4px 0">policy</th>
  <th style="text-align:right;padding:4px 12px">recall</th>
  <th style="text-align:right;padding:4px 12px">precision</th>
  <th style="text-align:right;padding:4px 12px">on time</th>
  <th style="text-align:right;padding:4px 12px">turns late</th>
  <th style="text-align:right;padding:4px 12px">wasted</th>
</tr></thead><tbody>{rows}</tbody></table></div>
<p style="font-size:0.84rem;opacity:0.72;margin-top:8px">
<strong>Turns late</strong> is the headline: exchanges a customer spent stuck
after the point a person should have taken over, summed across the set. A
conversation never escalated contributes everything after that point, so
abandoning somebody is not rewarded with a missing row.
</p>
"""


def _packet_table(results: Day13Results) -> str:
    rows = ""
    for row in results.packet_summary():
        colour = GOOD if row["re_asks"] <= 2 else (WARN if row["re_asks"] <= 6 else CRITICAL)
        rows += (
            f"<tr><td style='padding:5px 12px 5px 0'><code>{row['style']}</code></td>"
            f"<td style='padding:5px 12px;text-align:right'>{row['complete']:.0%}</td>"
            f"<td style='padding:5px 12px;text-align:right;color:{colour};font-weight:600'>"
            f"{row['re_asks']}</td>"
            f"<td style='padding:5px 12px;text-align:right'>"
            f"{row['grounded']}/{row['of']}</td>"
            f"<td style='padding:5px 12px;text-align:right'>{row['words']}</td>"
            f"<td style='padding:5px 0;font-size:0.8rem;opacity:0.65'>{row['note']}</td></tr>"
        )
    return f"""
<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:0.9rem">
<thead><tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
  <th style="text-align:left;padding:4px 12px 4px 0">packet</th>
  <th style="text-align:right;padding:4px 12px">complete</th>
  <th style="text-align:right;padding:4px 12px">re-asks</th>
  <th style="text-align:right;padding:4px 12px">grounded</th>
  <th style="text-align:right;padding:4px 12px">words</th>
  <th style="text-align:left;padding:4px 0"></th>
</tr></thead><tbody>{rows}</tbody></table></div>
"""


def _ablation(results: Day13Results) -> str:
    full = results.policies.get("signals", {})
    rows = ""
    for name, row in results.ablation.items():
        delta = row["recall"] - full.get("recall", 0)
        colour = WARN if abs(delta) < 1e-9 else GOOD
        label = "earns nothing" if abs(delta) < 1e-9 else f"{delta:+.0%} recall"
        rows += (
            f"<tr><td style='padding:5px 12px 5px 0'><code>{name}</code></td>"
            f"<td style='padding:5px 12px;text-align:right'>{row['recall']:.0%}</td>"
            f"<td style='padding:5px 12px;text-align:right'>{row['turns_late']}</td>"
            f"<td style='padding:5px 12px;text-align:right;color:{colour}'>{label}</td></tr>"
        )
    return f"""
<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:0.88rem">
<thead><tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
  <th style="text-align:left;padding:4px 12px 4px 0">full policy, minus one signal</th>
  <th style="text-align:right;padding:4px 12px">recall</th>
  <th style="text-align:right;padding:4px 12px">turns late</th>
  <th style="text-align:right;padding:4px 12px">vs full</th>
</tr></thead><tbody>{rows}</tbody></table></div>
"""


def _replay(results: Day13Results, name: str) -> tuple[str, pd.DataFrame]:
    conversation = results.by_name(name)
    if conversation is None:
        return "", pd.DataFrame()

    want = conversation.escalate_at
    header = (
        f"<p style='font-size:0.84rem;opacity:0.72'>{conversation.note}</p>"
        f"<p style='font-size:0.86rem'>Label: "
        + (
            f"<strong>a person should take over at exchange {want + 1}</strong>"
            if want is not None
            else "<strong>the bot should finish this itself</strong>"
        )
        + "</p>"
    )

    rows = []
    for index, exchange in enumerate(conversation.exchanges):
        fired = [s for s in results.signals_at(name, index) if s["fired"]]
        rows.append(
            {
                "#": index + 1,
                "customer": exchange.user,
                "answered": "yes" if exchange.answered else "REFUSED",
                "evidence": round(exchange.evidence_score, 2),
                "reprompts": exchange.reprompts,
                "signals": ", ".join(f"{s['name']}*" if s["binding"] else s["name"] for s in fired)
                or "—",
            }
        )
    return header, pd.DataFrame(rows)


def _packet_view(results: Day13Results, name: str, style: str) -> str:
    found = results.packet_for(name, style)
    if found is None:
        return (
            "<p style='font-size:0.88rem;opacity:0.7'>This conversation was not "
            "escalated, so there is no packet.</p>"
        )
    packet, score = found["packet"], found["score"]
    missing = [n for n in score["needs"] if not n["met"]]
    note = (
        f"<span style='color:{GOOD}'>Nothing to re-ask.</span>"
        if not missing
        else f"<span style='color:{CRITICAL}'>The agent would have to ask again for: "
        + ", ".join(n["want"] for n in missing)
        + "</span>"
    )
    body = "\n".join(
        [
            f"Topic: {packet['topic']}",
            f"Why a person: {packet['reason']}",
            *(
                ["Already given: " + ", ".join(f"{k} = {v}" for k, v in packet["slots"].items())]
                if packet["slots"]
                else []
            ),
            *(["Already tried: " + "; ".join(packet["tried"])] if packet["tried"] else []),
            *(
                ["What the system found: " + "; ".join(packet["evidence"])]
                if packet["evidence"]
                else []
            ),
            *(
                ["Transcript:"] + [f"  customer: {t}" for t in packet["transcript"]]
                if packet["transcript"]
                else []
            ),
        ]
    )
    return (
        f"<pre style='white-space:pre-wrap;font-size:0.84rem;padding:10px 12px;"
        f"border:1px solid rgba(128,128,128,0.3);border-radius:6px'>{body}</pre>"
        f"<p style='font-size:0.86rem'>{packet['words']} words · "
        f"{score['complete']:.0%} complete · {note}</p>"
    )


def build() -> None:
    results = load()
    if results is None:
        gr.Markdown(MISSING)
        return

    best = results.best_policy()
    counter = results.counterfactual

    gr.Markdown(
        f"### Day 13 · Escalation and agent assist\n"
        f"{len(results.conversations)} conversations — **{len(results.needing)} need a "
        f"person, {len(results.resolving)} do not**. Each is replayed through the real "
        f"stack: `faq` turns go through days 8-10's retrieval, reranker and both "
        f"guardrail gates, `booking` turns through day 4's engine. So a refusal in the "
        f"record is one day 10's thresholds produced, not one I asserted.\n\n"
        f"The label is a **turn index, not a flag**. Handing over on the last exchange "
        f"of a four-turn disaster is not a success, and a yes/no label cannot say so."
    )

    gr.HTML(_policy_table(results))
    gr.Markdown(
        f"Best by turns late: **`{best['policy']}`**. The intuitive design — make weak "
        f"signals agree before acting — is `signals`, and it is worse on every axis "
        f"than letting any one of them fire."
        + (
            f"\n\n`{counter['policy']}` reaches {counter['recall']:.0%} recall: the only "
            f'change is adding the phrase *"put me through"* to day 4\'s list of ways '
            f"to ask for a human. One ordinary sentence was the difference, which is "
            f"the case against pattern lists rather than a fix for this one."
            if counter
            else ""
        )
    )

    gr.Markdown(
        "#### What the agent receives\n"
        "Scored on **re-asks**: facts the customer would have to repeat because the "
        "handoff did not carry them. Every packet is also checked for invented "
        "facts — a summary that fabricates a party size is worse than no summary, "
        "because the agent will act on it."
    )
    gr.HTML(_packet_table(results))

    gr.Markdown(
        "#### Replay a conversation\nA `*` marks a binding signal — one that escalates alone."
    )
    names = [c.name for c in results.conversations]
    chosen = gr.Dropdown(names, value=names[0], label="conversation")
    header = gr.HTML()
    frame = gr.Dataframe(interactive=False, wrap=True)

    style = gr.Radio(list(PACKET_STYLES), value="full", label="handoff packet")
    packet = gr.HTML()

    with gr.Accordion("What each signal earns", open=False):
        gr.Markdown(
            "Each row is the full policy with one signal removed. Two of the six "
            "change nothing and are kept at zero rather than deleted — a table "
            "where every row earns its place has usually had the losing rows cut."
        )
        gr.HTML(_ablation(results))

    def _refresh(name: str, which: str):
        head, table = _replay(results, name)
        return head, table, _packet_view(results, name, which)

    for control in (chosen, style):
        control.change(_refresh, inputs=[chosen, style], outputs=[header, frame, packet])

    header.value, frame.value, packet.value = _refresh(names[0], "full")
