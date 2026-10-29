"""Day 20's tab: day 13's handoff across a real agent boundary, and what it loses.

Day 13 built the half of escalation nobody builds — a packet the human
receives, scored on whether every fact traces to something said, whether the
facts worth not repeating are present, and how long it is. All three were
measured inside one process.

Day 20's brief said: carry that across an agent boundary over A2A, "with the
provenance guard intact". The guard does not cross, and the reason is not a
bad serialisation. The tab is ordered to make that case: the two honest ways
to send a packet, what each loses, the check that fails on uncorrupted data,
and then the price of the only fix.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from convo.projects.day20_a2a_handoff.results import Results, load

GAP = "\n\n"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
WARN = "#c98a16"

MISSING = (
    "No day 20 artifact. Run "
    "`python -m convo.projects.day20_a2a_handoff.build` — it needs no key and "
    "no quota, and takes about a second."
)

CHECKS = (
    ("integrity", "bytes arrived as sent", "transport did its job"),
    (
        "payload_hash_matches",
        "hash of the decoded payload matches",
        "the standard integrity pattern",
    ),
    ("reconstructed", "a Packet equal to the one sent rebuilds", "needed to re-check anything"),
    ("provenance_recheckable", "day 13's grounding check can be re-run", "the brief"),
)


def _checks_table(results: Results) -> pd.DataFrame:
    rows = []
    for field, what, why in CHECKS:
        row = {"the receiver can establish": what, "": why}
        for carrier in results.carriers:
            row[carrier] = results.tally(carrier, field)
        rows.append(row)
    return pd.DataFrame(rows)


def _crossings_table(results: Results) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "conversation": c.conversation,
                "carrier": c.carrier,
                "bytes": c.wire_bytes,
                "words": c.sent_words,
                "rebuilt": "yes" if c.reconstructed else "no",
                "hash ok": "yes" if c.payload_hash_matches else "NO",
                "complete, sender": f"{c.complete_sender:.0%}",
                "complete, receiver": (
                    "cannot compute"
                    if c.complete_receiver is None
                    else f"{c.complete_receiver:.0%}"
                ),
                "+history": f"x{c.history_multiplier}",
            }
            for c in results.crossings
        ]
    )


def build() -> None:
    gr.Markdown("## A2A carrying day 13's handoff packet")
    results = load()
    if results is None:
        gr.Markdown(MISSING)
        return

    gr.Markdown(
        f"Day 13's policy escalates **{results.packets}** of its "
        f"{results.of_conversations} conversations, so there are "
        f"{results.packets} packets to carry. Each one goes to a real A2A agent "
        f"— discovered from its own card, over `SendMessage` on protocol "
        f"{results.protocol}, `a2a-sdk {results.sdk}` — twice: once as the text a "
        "person reads, once as structure a program can check."
        + GAP
        + "Everything on this page costs nothing. No model, no key, no quota, no "
        "port: the desk is an ASGI app in the same process. A protocol question "
        "does not need a model to answer, which makes this the only day in week 3 "
        "that re-runs in a second."
    )

    gr.Markdown("### Neither way of sending it is clean")
    gr.Dataframe(_checks_table(results), interactive=False, wrap=True)
    gr.Markdown(
        f"<p style='color:{CRITICAL}'><b>{results.any_clean} of "
        f"{len(results.crossings)} crossings pass every check.</b></p>"
        + "The two carriers fail in complementary ways, which is the useful shape "
        "of the result rather than a disappointment."
        + GAP
        + "**Text** is what most systems send, because it is what the agent wants to "
        "read. It arrives intact and cannot be parsed back: `Packet.text` joins "
        'slots on `", "` and `tried` on `"; "`, so a value containing either is '
        "indistinguishable from two entries. A receiver holding it can check "
        "nothing." + GAP + "**Structure** rebuilds exactly — and fails the hash."
    )

    gr.Markdown("### The integrity check that fails on data nothing corrupted")
    gr.Markdown(
        "A2A's `Part.data` is a `google.protobuf.Value`, which has one numeric kind "
        "and it is a double. So the packet's word count goes out as "
        f"`{results.drift_fields and results.drift_fields[0]}: 237` and arrives as "
        "`237.0`."
        + GAP
        + "The dict still compares **equal**, because `237 == 237.0` in Python. The "
        "test you would naturally write passes. But the standard integrity pattern "
        "— sender hashes the payload, receiver re-hashes what it decoded — "
        f"**fails on {results.tally('data', 'payload_hash_matches')} of the "
        "structured crossings**, because the two serialise to different JSON."
        + GAP
        + "Three checks, three answers, on the same uncorrupted packet: the bytes are "
        "intact, the value is equal, the hash says corrupted. The lesson is narrow "
        "and worth having — hash the wire bytes, or a canonical form that pins "
        "number types, never a re-serialised decode."
    )

    gr.Markdown("### What the brief asked for, and why it cannot be delivered")
    gr.Markdown(
        "Day 13's grounding check is `ungrounded_claims(packet, history)`. The "
        "history is the conversation. The receiving agent does not have it, and A2A "
        "has no reason to send it."
        + GAP
        + f"So provenance is re-checkable on **{results.tally('data', 'provenance_recheckable')}** "
        "of the structured crossings and the same for text. Not because the packet "
        "arrived damaged — the structured one arrives perfect — but because the "
        "guard needs an input the protocol does not carry. Checking the packet "
        "against its own transcript would be circular: a sender that invents a slot "
        "can invent the line supporting it."
        + GAP
        + "**A provenance guard that lives in the sender's process is not a property "
        "the receiver holds. It is a claim the receiver takes on trust.** That is "
        "true of any agent-to-agent protocol, not a fault in this one, and it is the "
        "thing I would want said out loud before building on one."
    )

    gr.Markdown("### The price of the only honest fix")
    gr.Markdown(
        "Send the conversation as well and the guard runs: it agrees with the sender "
        f"on **{results.grounded_with_history}** packets. It costs "
        f"**x{results.mean_multiplier}** the words, on average."
        + GAP
        + "Which spends day 13's third property to buy its first. Day 13 reported "
        'length without a threshold because "too long" has no defensible one — and '
        "that was the right call, because here is the trade it refused to prejudge. "
        "An agent reads this while a customer waits."
    )

    gr.Markdown("### Every crossing")
    gr.Dataframe(_crossings_table(results), interactive=False, wrap=True)

    with gr.Accordion("The desk, and two things that cost a debugging round", open=False):
        replies = results.replies
        gr.Markdown(
            f"The receiving agent advertises `{results.card['name']}`, accepting "
            f"{', '.join(results.card['accepts'])} over "
            f"{', '.join(results.card['interfaces'])}. It does the least convincing "
            "thing possible on purpose — reports what it received and what it could "
            "establish — because an agent that summarised would need a quota and "
            "would make the crossing harder to read."
            + GAP
            + "What it actually said:"
            + GAP
            + "\n".join(f"- `{carrier}` → *{reply}*" for carrier, reply in sorted(replies.items()))
        )
        gr.Markdown(
            "**The method is `SendMessage`, not `message/send`.** Protocol 1.0 names "
            "its JSON-RPC methods in PascalCase; protocol 0.3 — which is what the "
            "published examples use — names them `message/send`, and "
            "`create_jsonrpc_routes` does not enable 0.3 compatibility by default. A "
            "client written from the documentation gets `-32601 Method not found` "
            "from a correctly configured server, and the error mentions no versions."
            + GAP
            + "**And the version goes in a header.** With the method fixed, the server "
            "answered `Version mismatch: actual='0.3', expected='1.0'` — the request "
            "declared no version, the server assumed the old one, then rejected it for "
            "being the old one. `A2A-Version: 1.0` is the whole fix."
            + GAP
            + "**`supported_interfaces` is required and the SDK only warns.** A card "
            'without it serves happily here and is not a valid card — "may not verify '
            "or interoperate across SDKs and could be rejected in a future major "
            'release". Serving fine while being invalid is the worse of the two states.'
        )
