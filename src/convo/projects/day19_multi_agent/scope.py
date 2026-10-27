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

The graph is **derived** from the real agent objects, which needs the
framework, and then **committed** so that everything downstream — the site
build, the container, the tab — reads an artifact instead of importing ADK.
That split is not tidiness: the first version of this module skipped it, the
site called `derive()`, and the Pages build went red on a job that
deliberately does not install the framework. `reproduces()` re-derives in the
one job that has it and fails if the committed graph has drifted.
The conclusion follows from the topology rather than from my reading of two
constructor flags — which matters, because reading a flag and asserting what
it implies is exactly the mistake that cost day 18 three red builds.

The audit checks the opposite direction too. A topology where nothing can
reach `book_table` would satisfy the headline check perfectly and be a broken
bot, so `book_table` must be reachable from the router. A guarantee that only
holds because the feature is missing is not a guarantee.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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


def derive(model: str = "none") -> Topology:
    """Read the graph off a constructed agent tree, with no key and no call.

    Needs `google-adk`, because it builds the real agent objects rather than
    describing them — which is the point, and also why it cannot be what the
    site and the tab call. Use `committed()` there.

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


ARTIFACT_FILE = Path(__file__).parent / "artifacts" / "topology.json"


def save(graph: Topology | None = None) -> Topology:
    """Commit the derived graph, so the proof can be read without the framework."""
    graph = graph or derive()
    ARTIFACT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_FILE.write_text(json.dumps(graph.as_dict(), indent=1), encoding="utf-8")
    return graph


def load() -> Topology | None:
    """The committed graph, or None. Needs no framework and no key.

    This exists because the first version of day 19 did not have it, and the
    Pages build went red: the site called `derive()`, which imports ADK, and
    the site is built in a job that deliberately does not install it. Day 18's
    CI even has a leak guard naming `google.adk` for exactly this reason, and
    I walked into it anyway.

    The fix is this repository's own rule, applied to a proof rather than a
    measurement: derive it once where the framework exists, commit the result,
    and let every reader read the artifact. `reproduces()` is what stops the
    artifact drifting from the code.
    """
    if not ARTIFACT_FILE.exists():
        return None
    payload = json.loads(ARTIFACT_FILE.read_text(encoding="utf-8"))
    return Topology(
        tools={a: tuple(t) for a, t in payload["tools"].items()},
        edges={a: tuple(e) for a, e in payload["edges"].items()},
    )


def committed() -> Topology:
    """The committed graph, or a clear error. For the site, the tab and tests."""
    found = load()
    if found is None:
        raise SystemExit(
            f"no committed topology at {ARTIFACT_FILE}; run "
            "`python -m convo.projects.day19_multi_agent.build --write-topology`"
        )
    return found


def reproduces() -> list[str]:
    """Whether the committed graph still matches the one the code builds.

    Day 18's reproducibility gate, applied to a topology instead of a score.
    A committed proof that nothing re-derives is just a file, and the two
    could drift the first time a constructor flag changed.
    """
    stored = load()
    if stored is None:
        return [f"no committed topology at {ARTIFACT_FILE}"]
    fresh = derive()
    complaints: list[str] = []
    if stored.tools != fresh.tools:
        complaints.append(f"tools differ: committed {stored.tools}, derived {fresh.tools}")
    if stored.edges != fresh.edges:
        complaints.append(f"edges differ: committed {stored.edges}, derived {fresh.edges}")
    return complaints


def audit(graph: Topology | None = None) -> list[str]:
    """Complaints about the topology. Empty means the claim holds.

    Defaults to the *committed* graph, so the audit runs anywhere — the site
    build, the container, a laptop without the framework. Pass `derive()` to
    audit what the code builds right now.
    """
    graph = graph if graph is not None else committed()
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
