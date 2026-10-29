"""A real A2A agent on the far side of the handoff, served in this process.

`wire.py` measures what survives serialisation, which is the part that decides
the answer. This exists so the answer is about A2A and not about my reading of
A2A: the packet goes out through `message/send` to an agent discovered from its
own card, over the SDK's JSON-RPC routes, through a Starlette app driven by
httpx's ASGI transport.

No socket, no port, no second process, nothing to leave running — which is why
it can be a test rather than a demo. The one thing it does not exercise is the
network, and the network is not where a packet loses its provenance.

The desk agent does the least convincing thing possible on purpose: it reports
what it received and what it could establish about it. An agent that
summarised, or asked a model to triage, would make the crossing harder to read
and would need a quota this day does not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from convo.projects.day20_a2a_handoff.wire import DATA, TEXT

DESK_NAME = "human_handoff_desk"
RPC_URL = "/a2a"
CARD_URL = "/.well-known/agent-card.json"
BASE = "http://desk.invalid"
"""A hostname that cannot resolve, because nothing here should reach a network.

If this ever starts making real requests, the failure is immediate and loud
rather than a quiet dependency on something outside the test.
"""


def agent_card():
    """The desk's A2A card: what it is, and what it will accept.

    `default_input_modes` is the part worth reading. The desk advertises both
    `text/plain` and `application/json`, because day 20's whole question is
    which of those two a handoff should use — and a card that advertised only
    one would be answering the question in the discovery document.
    """
    from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
    from a2a.utils.constants import PROTOCOL_VERSION_CURRENT, TransportProtocol

    card = AgentCard(
        name=DESK_NAME,
        description=(
            "Receives an escalation handoff from the booking bot and reports what it "
            "can establish about it without access to the original conversation."
        ),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["application/json"],
    )
    # REQUIRED, and the SDK only warns when it is missing: "allowed and does
    # not raise, but it may not verify or interoperate across SDKs and could
    # be rejected in a future major release". A card without it serves fine
    # here and is not a valid card, which is the worst of the two states to
    # ship. It declares where the agent is and which binding it speaks.
    card.supported_interfaces.append(
        AgentInterface(
            url=f"{BASE}{RPC_URL}",
            protocol_binding=TransportProtocol.JSONRPC.value,
            protocol_version=PROTOCOL_VERSION_CURRENT,
        )
    )
    card.skills.append(
        AgentSkill(
            id="receive_handoff",
            name="Receive a handoff",
            description=(
                "Accept a day 13 escalation packet and report integrity, whether the "
                "packet could be reconstructed, and whether its provenance can be "
                "re-checked. The last is always no: re-checking needs the conversation."
            ),
            tags=["escalation", "handoff"],
        )
    )
    return card


@dataclass
class Received:
    """What the desk saw, recorded for the test to read back."""

    carrier: str = ""
    text: str = ""
    fields: dict[str, Any] | None = None


class HandoffDesk:
    """An `AgentExecutor` that accepts a handoff and answers about it."""

    def __init__(self, seen: Received) -> None:
        self.seen = seen

    async def execute(self, context, event_queue) -> None:
        from a2a.types import Message, Part, Role

        from convo.projects.day20_a2a_handoff.wire import from_data_part

        incoming = context.message
        part = incoming.parts[0] if incoming.parts else Part()

        if part.HasField("data"):
            self.seen.carrier = DATA
            self.seen.fields = from_data_part(part)
            # Cleared, not left: one desk serves every handoff, so a text part
            # arriving after a data part would otherwise report the previous
            # one's fields and look like a text carrier that recovered them.
            self.seen.text = ""
            answer = f"structured handoff for {self.seen.fields.get('conversation', '?')}"
        else:
            self.seen.carrier = TEXT
            self.seen.text = part.text
            self.seen.fields = None
            # No branch here can rebuild the packet, and saying so in the reply
            # is the honest output: the desk has prose and cannot check it.
            answer = "text handoff received; fields not recoverable"

        reply = Message(message_id="desk-reply", role=Role.ROLE_AGENT)
        reply.parts.append(Part(text=answer))
        await event_queue.enqueue_event(reply)

    async def cancel(self, context, event_queue) -> None:  # pragma: no cover - never called
        raise NotImplementedError("the desk has nothing long-running to cancel")


def build_app(seen: Received):
    """A Starlette app serving the desk's card and its JSON-RPC endpoint."""
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
    from a2a.server.tasks import InMemoryTaskStore
    from starlette.applications import Starlette

    card = agent_card()
    handler = DefaultRequestHandler(
        agent_executor=HandoffDesk(seen),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    routes = create_agent_card_routes(card, card_url=CARD_URL) + create_jsonrpc_routes(
        handler, rpc_url=RPC_URL
    )
    return Starlette(routes=routes)
