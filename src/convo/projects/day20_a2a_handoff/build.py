"""Day 20's artifact: day 13's packets across a real A2A boundary.

    uv run python -m convo.projects.day20_a2a_handoff.build

**Costs nothing and calls no model.** Day 13's packets are built by a
deterministic policy from committed conversations, the crossing is protobuf
serialisation, and the transport is an ASGI app in this process. The whole day
re-runs in about a second, which is the first time that has been true of a
week-3 day and is a consequence of the subject rather than a virtue: a protocol
question does not need a model to answer.

Two things are measured and they answer different questions.

`verify.cross` is the measurement: for each packet and each carrier, what can
the far side establish? It uses protobuf's own `SerializeToString`, which is
what A2A puts on a gRPC wire and encodes its JSON transport from.

`send.deliver` is the check on that measurement's premise: the same packet
goes out through `SendMessage` to an agent discovered from its own card, over
the SDK's JSON-RPC routes. If the fields the desk receives differ from the ones
`cross` predicts, then `cross` is measuring something other than A2A and
nothing in the artifact is worth reading. The build fails in that case rather
than reporting both.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from convo.core import get_logger, setup
from convo.projects.day13_escalation import policy as policies
from convo.projects.day13_escalation.conversations import CONVERSATIONS, Conversation
from convo.projects.day13_escalation.packet import Packet, build_packet
from convo.projects.day20_a2a_handoff.desk import DESK_NAME, agent_card
from convo.projects.day20_a2a_handoff.send import deliver_all
from convo.projects.day20_a2a_handoff.verify import Crossing, cross, provenance_with_history
from convo.projects.day20_a2a_handoff.wire import CARRIERS, DATA

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_FILE = ARTIFACT_DIR / "a2a.json"

log = get_logger("day20.build")


def handoffs() -> list[tuple[Conversation, Packet]]:
    """Day 13's packets, from day 13's own policy. Nothing re-implemented here.

    `signals_policy()` is the policy day 13 published as its best, and the
    conversations where it escalates are the ones that have a packet at all.
    Building the packet any other way would make day 20 a measurement of my
    re-implementation.
    """
    policy = policies.signals_policy()
    found = []
    for conversation in CONVERSATIONS:
        handoff = policy(conversation.exchanges)
        if handoff is not None:
            found.append((conversation, build_packet(conversation, handoff)))
    return found


def confirm_transport(pairs: list[tuple[Conversation, Packet]]) -> list[dict]:
    """Send each packet for real, and check it matches what `cross` predicted.

    The premise of the whole artifact is that protobuf serialisation is what
    A2A does to a packet. This is where that stops being an assumption.
    """
    # One flat list of (conversation, packet, carrier), so the deliveries, the
    # replies and the predictions stay index-aligned under `strict=True`.
    plan = [
        (conversation, packet, carrier) for conversation, packet in pairs for carrier in CARRIERS
    ]
    results = asyncio.run(deliver_all([(carrier, packet) for _, packet, carrier in plan]))

    report: list[dict] = []
    for (conversation, packet, carrier), delivered in zip(plan, results, strict=True):
        if not delivered.arrived:
            raise SystemExit(
                f"the {carrier} handoff for {conversation.name} did not arrive "
                f"(status {delivered.status}); nothing downstream is measurable"
            )
        if delivered.card_name != DESK_NAME:
            raise SystemExit(f"discovered {delivered.card_name!r}, expected {DESK_NAME!r}")

        if carrier == DATA:
            if not cross(conversation, packet, carrier).reconstructed:
                raise SystemExit("cross() says the data carrier cannot reconstruct")
            # The float is the expectation, not a concession: protobuf's Value
            # has one numeric kind. If the desk ever reports an int here, the
            # drift finding has stopped being true and the write-up is wrong.
            expected = {**packet.as_dict(), "words": float(packet.words)}
            if delivered.seen_fields != expected:
                raise SystemExit(
                    "the desk received different fields from the ones cross() measured "
                    f"for {conversation.name}; cross() is not measuring A2A"
                )

        report.append(
            {
                "conversation": conversation.name,
                "carrier": carrier,
                "status": delivered.status,
                "discovered": delivered.card_name,
                "accepts": list(delivered.accepts),
                "reply": delivered.reply,
            }
        )
    return report


def main(argv: list[str] | None = None) -> int:
    setup()
    pairs = handoffs()
    if not pairs:
        raise SystemExit("day 13's policy escalates nothing; there is no packet to carry")

    crossings: list[Crossing] = []
    for conversation, packet in pairs:
        for carrier in CARRIERS:
            crossings.append(cross(conversation, packet, carrier))

    transport = confirm_transport(pairs)

    with_history = {
        conversation.name: provenance_with_history(conversation, packet)
        for conversation, packet in pairs
    }

    payload = {
        "packets": len(pairs),
        "of_conversations": len(CONVERSATIONS),
        "carriers": list(CARRIERS),
        "protocol": "1.0",
        "sdk": _sdk_version(),
        "card": {
            "name": agent_card().name,
            "accepts": list(agent_card().default_input_modes),
            "interfaces": [i.protocol_binding for i in agent_card().supported_interfaces],
        },
        "crossings": [c.as_dict() for c in crossings],
        "transport": transport,
        "provenance_with_history": {
            name: {"grounded": grounded, "problems": problems}
            for name, (grounded, problems) in with_history.items()
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_FILE.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    by_carrier = {carrier: [c for c in crossings if c.carrier == carrier] for carrier in CARRIERS}
    log.info(
        "artifact written",
        extra={
            "packets": len(pairs),
            "crossings": len(crossings),
            "reconstructed": {k: sum(c.reconstructed for c in v) for k, v in by_carrier.items()},
            "payload_hash_matches": {
                k: sum(c.payload_hash_matches for c in v) for k, v in by_carrier.items()
            },
            "provenance_recheckable": sum(c.provenance_recheckable for c in crossings),
        },
    )
    return 0


def _sdk_version() -> str:
    import importlib.metadata as meta

    try:
        return meta.version("a2a-sdk")
    except Exception:  # pragma: no cover - the dependency is declared
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
