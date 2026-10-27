"""Which tools each agent can reach, derived from the topology. No model.

This is day 19's spine, and the only part of it that is a proof rather than a
measurement. Everything else here needs a model run and a day's quota; this
needs neither, runs in CI, and is what makes the claim in `agents.py`
falsifiable instead of rhetorical.

The claim is that once the router transfers to `handoff_desk`, booking is out
of scope for the rest of the conversation. That is a statement about a graph,
so it is checked as one. Edges are the transfers ADK permits:

    router        -> each of its sub_agents          always
    a sub_agent   -> the router                      unless transfer to parent is disallowed
    a sub_agent   -> each of its peers               unless transfer to peers is disallowed

`reachable_from("handoff_desk")` is then `{"handoff_desk"}`, and the tools
reachable from it are the union of the tools held by that set, which is empty.
The conclusion follows from the topology rather than from my reading of two
constructor flags — which matters, because reading a flag and asserting what
it implies is exactly the mistake that cost day 18 three red builds.

The audit checks the opposite direction too. A topology where nothing can
reach `book_table` would satisfy the headline check perfectly and be a broken
bot, so `book_table` must be reachable from the router. A guarantee that only
holds because the feature is missing is not a guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from convo.projects.day06_eval_harness.suite import REF
from convo.projects.day19_multi_agent.agents import (
    BOOK_TOOL,
    BOOKER,
    EVERY_AGENT,
    HANDOFF,
    POLICY,
    POLICY_TOOL,
    ROUTER,
    WRAPPER_FOR,
    Booking,
    build_agent,
)


def tool_names(agent: Any) -> tuple[str, ...]:
    """The executed tool names an agent holds, read off the live object.

    Read from the constructed agent rather than from a table in this file:
    a table would be a second description of the topology, and the two would
    eventually disagree. `__name__` is the wrapper's, so it is mapped back
    through `WRAPPER_FOR` to the day 5 tool that actually runs.
    """
    advertised = {WRAPPER_FOR[tool]: tool for tool in WRAPPER_FOR}
    found: list[str] = []
    for tool in agent.tools or []:
        name = getattr(tool, "__name__", None) or getattr(tool, "name", "")
        if name in advertised:
            found.append(advertised[name])
        elif name:
            found.append(name)
    return tuple(found)


@dataclass(frozen=True)
class Topology:
    """The agent graph, flattened into something answerable."""

    tools: dict[str, tuple[str, ...]]
    edges: dict[str, tuple[str, ...]]

    @property
    def agents(self) -> tuple[str, ...]:
        return tuple(sorted(self.tools))

    def reachable_from(self, start: str) -> frozenset[str]:
        """Every agent control can end up at, starting from `start`."""
        seen = {start}
        stack = [start]
        while stack:
            for nxt in self.edges.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return frozenset(seen)

    def tools_reachable_from(self, start: str) -> frozenset[str]:
        """Every tool that can still execute, once control is at `start`."""
        return frozenset(
            tool for agent in self.reachable_from(start) for tool in self.tools.get(agent, ())
        )

    def holders_of(self, tool: str) -> tuple[str, ...]:
        return tuple(a for a in self.agents if tool in self.tools.get(a, ()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "tools": {a: list(t) for a, t in sorted(self.tools.items())},
            "edges": {a: list(e) for a, e in sorted(self.edges.items())},
            "reachable_from_handoff": sorted(self.reachable_from(HANDOFF)),
            "tools_reachable_from_handoff": sorted(self.tools_reachable_from(HANDOFF)),
            "tools_reachable_from_router": sorted(self.tools_reachable_from(ROUTER)),
        }


def topology(model: str = "none") -> Topology:
    """Read the graph off a constructed agent tree, with no key and no call.

    `model` is a string ADK stores and never resolves until a run, so the
    default is deliberately not a real model id: nothing here may depend on
    one existing.
    """
    root = build_agent(REF, Booking(), model, require_key=False)
    subs = list(root.sub_agents or [])

    tools = {root.name: tool_names(root)}
    edges: dict[str, tuple[str, ...]] = {root.name: tuple(s.name for s in subs)}

    for sub in subs:
        tools[sub.name] = tool_names(sub)
        out: list[str] = []
        if not sub.disallow_transfer_to_parent:
            out.append(root.name)
        if not sub.disallow_transfer_to_peers:
            out.extend(peer.name for peer in subs if peer.name != sub.name)
        edges[sub.name] = tuple(out)

    return Topology(tools=tools, edges=edges)


def audit(graph: Topology | None = None) -> list[str]:
    """Complaints about the topology. Empty means the claim holds.

    Returned rather than asserted so the gate, the tests and the tab can each
    decide what to do with them, and so a failure names the property that
    broke instead of a line number.
    """
    graph = graph or topology()
    complaints: list[str] = []

    missing = [name for name in EVERY_AGENT if name not in graph.tools]
    if missing:
        complaints.append("agents missing from the tree: " + ", ".join(missing))
        return complaints

    holders = graph.holders_of(BOOK_TOOL)
    if holders != (BOOKER,):
        complaints.append(
            f"{BOOK_TOOL} must be held by {BOOKER} alone, found: {', '.join(holders) or 'nobody'}"
        )
    if graph.holders_of(POLICY_TOOL) != (POLICY,):
        complaints.append(f"{POLICY_TOOL} must be held by {POLICY} alone")

    if graph.tools.get(HANDOFF):
        complaints.append(f"{HANDOFF} holds tools: {', '.join(graph.tools[HANDOFF])}")

    escape = graph.reachable_from(HANDOFF) - {HANDOFF}
    if escape:
        complaints.append(
            f"control can leave {HANDOFF} for: {', '.join(sorted(escape))} "
            "— the booking tool is reachable again"
        )

    stranded = graph.tools_reachable_from(HANDOFF)
    if stranded:
        complaints.append(f"tools still reachable from {HANDOFF}: {', '.join(sorted(stranded))}")

    # The other direction. A bot that can never book would pass every check
    # above and be useless, so the guarantee has to cost something real.
    from_router = graph.tools_reachable_from(ROUTER)
    for tool in (BOOK_TOOL, POLICY_TOOL):
        if tool not in from_router:
            complaints.append(f"{tool} is not reachable from {ROUTER} at all")

    return complaints
