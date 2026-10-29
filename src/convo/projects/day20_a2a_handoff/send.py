"""The bot's side: discover the desk from its card, then send the handoff.

Deliberately the whole A2A sequence rather than a direct call into the
handler — fetch the agent card from `/.well-known/agent-card.json`, read what
the desk accepts, then `message/send` — because "the packet crossed an agent
boundary" means the receiver was found the way A2A says to find one. A
function call into `HandoffDesk.execute` would prove serialisation worked and
nothing about the protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from google.protobuf import json_format

from convo.projects.day13_escalation.packet import Packet
from convo.projects.day20_a2a_handoff.desk import BASE, CARD_URL, RPC_URL, Received, build_app
from convo.projects.day20_a2a_handoff.wire import to_message

METHOD_SEND = "SendMessage"
"""The method name a2a-sdk 1.3.0 actually serves.

Protocol 1.0 names its JSON-RPC methods in PascalCase after the RPC —
`SendMessage`, `GetTask`. Protocol 0.3, which is what the published examples
and nearly every tutorial use, names them `message/send` and `tasks/get`.
`create_jsonrpc_routes` takes `enable_v0_3_compat` and it defaults to False.

So a client written from the documentation gets `-32601 Method not found`
from a server that is configured correctly, and the error says nothing about
versions. Cost me a round of debugging; `METHOD_SEND_LEGACY` and the test
beside it exist so it costs the next reader nothing."""

METHOD_SEND_LEGACY = "message/send"

VERSION_HEADER = "A2A-Version"
PROTOCOL_VERSION = "1.0"
"""Protocol 1.0 requires the version in a header, and defaults to 0.3 without it.

The second thing that cost a round of debugging. With the method name fixed
the server answered `Version mismatch: actual='0.3', expected='1.0'` — the
request had declared no version, the server assumed the old one, and then
rejected it for being the old one. Sending `A2A-Version: 1.0` is the whole
fix, and no example I started from includes it."""


@dataclass(frozen=True)
class Delivered:
    """What came back, and what the desk turned out to have seen."""

    card_name: str
    accepts: tuple[str, ...]
    reply: str
    seen_carrier: str
    seen_fields: dict[str, Any] | None
    status: int

    @property
    def arrived(self) -> bool:
        return self.status == 200 and bool(self.seen_carrier)


async def deliver_all(pairs: list[tuple[str, Packet]]) -> list[Delivered]:
    """Many handoffs to one desk, over one client.

    One desk serving every handoff is both the realistic shape and the quiet
    one: a fresh app per packet leaves the SDK's event queue to be garbage
    collected mid-dispatch, which it logs about once per call.
    """
    import httpx

    seen = Received()
    app = build_app(seen)
    out: list[Delivered] = []

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE) as client:
        card = await client.get(CARD_URL)
        card.raise_for_status()
        described = card.json()
        for carrier, packet in pairs:
            out.append(await _send_one(client, described, packet, carrier, seen))
    return out


async def deliver(packet: Packet, carrier: str) -> Delivered:
    """One real A2A exchange, in this process, over ASGI."""
    return (await deliver_all([(carrier, packet)]))[0]


async def _send_one(client, described, packet: Packet, carrier: str, seen: Received) -> Delivered:
    message = to_message(packet, carrier)
    request = {
        "jsonrpc": "2.0",
        "id": f"handoff-{carrier}",
        "method": METHOD_SEND,
        "params": {"message": json.loads(json_format.MessageToJson(message))},
    }
    response = await client.post(RPC_URL, json=request, headers={VERSION_HEADER: PROTOCOL_VERSION})

    reply = ""
    if response.status_code == 200:
        body = response.json()
        # `result.message.parts`, not `result.parts`: SendMessage answers with
        # a oneof of a Message or a Task, so the reply is nested under whichever
        # it chose. Reading `result.parts` returns nothing and looks like an
        # agent that said nothing.
        message = (body.get("result") or {}).get("message") or {}
        for part in message.get("parts") or []:
            if part.get("text"):
                reply = part["text"]

    return Delivered(
        card_name=described.get("name", ""),
        accepts=tuple(described.get("defaultInputModes") or ()),
        reply=reply,
        seen_carrier=seen.carrier,
        seen_fields=seen.fields,
        status=response.status_code,
    )
