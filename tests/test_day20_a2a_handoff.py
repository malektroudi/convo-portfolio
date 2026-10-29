"""Day 20: what crosses an A2A boundary, and what the obvious test would miss.

Nothing here calls a model or opens a socket. The transport tests drive a real
A2A agent through httpx's ASGI transport, in this process, which is why they
can be tests rather than a demo.

The tests that matter most are the ones asserting a *failure*: the hash that
does not match, the packet that cannot be rebuilt, the guard that cannot run.
Each of those is a published claim, and a test that only checked the happy
path would let any of them quietly stop being true.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from convo.projects.day13_escalation import policy as policies
from convo.projects.day13_escalation.conversations import CONVERSATIONS
from convo.projects.day13_escalation.packet import Packet, build_packet
from convo.projects.day20_a2a_handoff import wire
from convo.projects.day20_a2a_handoff.verify import cross, digest, provenance_with_history


@pytest.fixture(scope="module")
def pairs():
    policy = policies.signals_policy()
    found = []
    for conversation in CONVERSATIONS:
        handoff = policy(conversation.exchanges)
        if handoff is not None:
            found.append((conversation, build_packet(conversation, handoff)))
    assert found, "day 13's policy escalates nothing"
    return found


@pytest.fixture(scope="module")
def one(pairs):
    return pairs[0]


class TestTheTextCarrierCannotBeCheckedOnArrival:
    """What most systems send, and what it costs the receiver."""

    def test_it_arrives_intact(self, one):
        conversation, packet = one
        assert cross(conversation, packet, wire.TEXT).integrity

    def test_it_cannot_be_rebuilt(self, one):
        conversation, packet = one
        assert not cross(conversation, packet, wire.TEXT).reconstructed

    def test_the_receiver_cannot_compute_completeness(self, one):
        """None, not 0.0 — day 5's lesson about unmeasured dimensions."""
        conversation, packet = one
        assert cross(conversation, packet, wire.TEXT).complete_receiver is None

    def test_recover_declines_rather_than_guessing(self, one):
        _, packet = one
        message = wire.to_message(packet, wire.TEXT)
        _, arrived = wire.round_trip(message)
        assert wire.recover(arrived, wire.TEXT) is None

    def test_the_rendering_is_genuinely_ambiguous(self):
        """Not a limitation of my parser — there is nothing to parse.

        Two different packets render to the same `tried` section, so no
        reader could recover which one was sent.
        """
        base = dict(conversation="c", topic="t", reason="r", slots={}, transcript=(), evidence=())
        one_item = Packet(**base, tried=("a; b",))
        two_items = Packet(**base, tried=("a", "b"))
        assert one_item.text == two_items.text
        assert one_item != two_items


class TestTheDataCarrierRebuildsAndFailsTheHash:
    def test_it_rebuilds_exactly(self, one):
        conversation, packet = one
        assert cross(conversation, packet, wire.DATA).reconstructed

    def test_the_receiver_agrees_on_completeness(self, one):
        conversation, packet = one
        crossing = cross(conversation, packet, wire.DATA)
        assert crossing.complete_agrees is True

    def test_the_payload_hash_does_not_match(self, one):
        """The standard integrity pattern, failing on an uncorrupted packet."""
        conversation, packet = one
        assert not cross(conversation, packet, wire.DATA).payload_hash_matches

    def test_the_bytes_are_intact_even_so(self, one):
        """Three checks, three answers, one packet. This is the one that passes."""
        conversation, packet = one
        assert cross(conversation, packet, wire.DATA).integrity

    def test_the_dict_compares_equal_which_is_why_this_hides(self, one):
        """The test a careful person would write, passing."""
        conversation, packet = one
        assert cross(conversation, packet, wire.DATA).equal_by_value

    def test_the_type_drifted_anyway(self, one):
        _, packet = one
        sent = packet.as_dict()
        message = wire.to_message(packet, wire.DATA)
        _, arrived = wire.round_trip(message)
        received = wire.recover(arrived, wire.DATA)
        drift = wire.type_drift(sent, received)
        assert [d.field for d in drift] == ["words"]
        assert (drift[0].sent, drift[0].received) == ("int", "float")

    def test_an_int_is_what_went_out_and_a_float_is_what_came_back(self, one):
        _, packet = one
        message = wire.to_message(packet, wire.DATA)
        _, arrived = wire.round_trip(message)
        received = wire.recover(arrived, wire.DATA)
        assert isinstance(packet.as_dict()["words"], int)
        assert isinstance(received["words"], float)

    def test_the_two_digests_differ_for_that_reason_alone(self, one):
        """Pinning the cause: equalise the type and the hashes agree."""
        _, packet = one
        sent = packet.as_dict()
        message = wire.to_message(packet, wire.DATA)
        _, arrived = wire.round_trip(message)
        received = wire.recover(arrived, wire.DATA)
        assert digest(sent) != digest(received)
        assert digest({**sent, "words": float(sent["words"])}) == digest(received)


class TestTheGuardDoesNotCross:
    """The brief said "with the provenance guard intact". It is not."""

    def test_no_carrier_can_re_run_it(self, pairs):
        for conversation, packet in pairs:
            for carrier in wire.CARRIERS:
                assert not cross(conversation, packet, carrier).provenance_recheckable

    def test_the_guard_needs_the_conversation(self):
        """Why, in one line: it is in the signature."""
        from convo.projects.day13_escalation.packet import ungrounded_claims

        assert "history" in ungrounded_claims.__code__.co_varnames[:2]

    def test_it_runs_and_agrees_once_the_history_crosses(self, pairs):
        for conversation, packet in pairs:
            grounded, problems = provenance_with_history(conversation, packet)
            assert grounded
            assert problems == 0

    def test_and_that_costs_words(self, pairs):
        for conversation, packet in pairs:
            crossing = cross(conversation, packet, wire.DATA)
            assert crossing.history_multiplier > 1.0

    def test_nothing_is_fully_verifiable(self, pairs):
        """The headline. If a crossing ever goes clean, the write-up is wrong."""
        for conversation, packet in pairs:
            for carrier in wire.CARRIERS:
                assert not cross(conversation, packet, carrier).clean


class TestARealA2AExchange:
    """Discovered from a card, sent over SendMessage, in this process."""

    def test_the_card_is_served_and_names_the_desk(self, one):
        from convo.projects.day20_a2a_handoff.desk import DESK_NAME
        from convo.projects.day20_a2a_handoff.send import deliver

        _, packet = one
        delivered = asyncio.run(deliver(packet, wire.DATA))
        assert delivered.card_name == DESK_NAME
        assert "application/json" in delivered.accepts

    def test_the_structured_handoff_arrives_with_every_field(self, one):
        from convo.projects.day20_a2a_handoff.send import deliver

        _, packet = one
        delivered = asyncio.run(deliver(packet, wire.DATA))
        assert delivered.arrived
        assert set(delivered.seen_fields) == set(packet.as_dict())

    def test_the_desk_sees_the_float_too(self, one):
        """The drift is A2A's, not an artefact of how `cross` serialises."""
        from convo.projects.day20_a2a_handoff.send import deliver

        _, packet = one
        delivered = asyncio.run(deliver(packet, wire.DATA))
        assert isinstance(delivered.seen_fields["words"], float)

    def test_the_text_handoff_arrives_unparseable(self, one):
        from convo.projects.day20_a2a_handoff.send import deliver

        _, packet = one
        delivered = asyncio.run(deliver(packet, wire.TEXT))
        assert delivered.arrived
        assert delivered.seen_fields is None

    def test_one_desk_does_not_leak_state_between_handoffs(self, one):
        """A text handoff after a structured one must not report its fields."""
        from convo.projects.day20_a2a_handoff.send import deliver_all

        _, packet = one
        results = asyncio.run(deliver_all([(wire.DATA, packet), (wire.TEXT, packet)]))
        assert results[0].seen_fields is not None
        assert results[1].seen_fields is None

    def test_the_documented_method_name_is_not_the_served_one(self, one):
        """`message/send` is what every example uses and gets -32601 here."""
        import httpx
        from google.protobuf import json_format

        from convo.projects.day20_a2a_handoff.desk import BASE, RPC_URL, Received, build_app
        from convo.projects.day20_a2a_handoff.send import (
            METHOD_SEND,
            METHOD_SEND_LEGACY,
            PROTOCOL_VERSION,
            VERSION_HEADER,
        )

        _, packet = one
        message = wire.to_message(packet, wire.DATA)
        params = {"message": json.loads(json_format.MessageToJson(message))}

        async def ask(method: str) -> dict:
            app = build_app(Received())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url=BASE
            ) as client:
                response = await client.post(
                    RPC_URL,
                    json={"jsonrpc": "2.0", "id": "t", "method": method, "params": params},
                    headers={VERSION_HEADER: PROTOCOL_VERSION},
                )
                return response.json()

        assert "error" in asyncio.run(ask(METHOD_SEND_LEGACY))
        assert "result" in asyncio.run(ask(METHOD_SEND))

    def test_the_version_header_is_required(self, one):
        """Without it the server assumes 0.3 and rejects the request for being 0.3."""
        import httpx
        from google.protobuf import json_format

        from convo.projects.day20_a2a_handoff.desk import BASE, RPC_URL, Received, build_app
        from convo.projects.day20_a2a_handoff.send import METHOD_SEND

        _, packet = one
        message = wire.to_message(packet, wire.DATA)

        async def ask() -> dict:
            app = build_app(Received())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url=BASE
            ) as client:
                response = await client.post(
                    RPC_URL,
                    json={
                        "jsonrpc": "2.0",
                        "id": "t",
                        "method": METHOD_SEND,
                        "params": {"message": json.loads(json_format.MessageToJson(message))},
                    },
                )
                return response.json()

        assert "error" in asyncio.run(ask())

    def test_the_card_declares_an_interface(self, one):
        """REQUIRED, and the SDK only warns when it is missing."""
        from convo.projects.day20_a2a_handoff.desk import agent_card

        assert list(agent_card().supported_interfaces)


class TestTheArtifactAndItsLoader:
    def test_the_loader_needs_no_sdk(self):
        """The site and the tab must render without the dependency.

        Day 19 took Pages down by having the site import a framework the job
        does not install. This day's dependency does resolve, so the risk is
        lower and the boundary is kept anyway.
        """
        import importlib
        import sys
        from importlib.abc import MetaPathFinder

        class Block(MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "a2a" or name.startswith("a2a."):
                    raise ImportError(f"blocked: {name}")
                return None

        blocker = Block()
        sys.meta_path.insert(0, blocker)
        cached = {n: m for n, m in sys.modules.items() if n == "a2a" or n.startswith("a2a.")}
        for name in cached:
            del sys.modules[name]
        for name in [n for n in sys.modules if n.startswith("convo.projects.day20")]:
            del sys.modules[name]
        try:
            results = importlib.import_module("convo.projects.day20_a2a_handoff.results")
            importlib.import_module("convo.projects.day20_a2a_handoff.ui")
            assert results.load() is not None or not results.ARTIFACT_FILE.exists()
            data = importlib.import_module("convo.site.data")
            summary, _, tables, caveat = data._day20()
            assert summary and tables and caveat
        finally:
            sys.meta_path.remove(blocker)
            sys.modules.update(cached)

    def test_the_stored_figures_are_the_ones_measured_now(self, pairs):
        """Re-derive every crossing and compare. Free, so there is no excuse."""
        from convo.projects.day20_a2a_handoff.results import load

        load.cache_clear()
        stored = load()
        if stored is None:
            pytest.skip("day 20 has not been built")
        fresh = {
            (c.name, carrier): cross(c, p, carrier) for c, p in pairs for carrier in wire.CARRIERS
        }
        assert len(stored.crossings) == len(fresh)
        for row in stored.crossings:
            now = fresh[(row.conversation, row.carrier)]
            assert row.reconstructed == now.reconstructed
            assert row.payload_hash_matches == now.payload_hash_matches
            assert row.provenance_recheckable == now.provenance_recheckable
            assert row.wire_bytes == now.wire_bytes

    def test_the_headline_is_zero(self):
        from convo.projects.day20_a2a_handoff.results import load

        load.cache_clear()
        stored = load()
        if stored is None:
            pytest.skip("day 20 has not been built")
        assert stored.any_clean == 0
