"""Day 19's tab: a router, four desks, and a guarantee checked before it is believed.

Day 16's single ADK agent held every tool on every turn, so "do not book a
table after the guest asks for a human" was an instruction and nothing more.
It broke that instruction on both runs, on the same two conversations — the
only failures in this repository that have reproduced across every run of
every model.

This day splits the bot so that the booking tool lives on one desk, and the
desk that handles a handoff has no path back to it. That makes the defect
unreachable rather than discouraged, and the claim is checked as a graph with
no model and no key, which is why the proof can render here with the
measurement still outstanding.

The tab is ordered so the weakest part is not last: the proof, then what the
proof does *not* cover, then the prediction written down before the run, then
the run itself once it exists.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from convo.projects.day19_multi_agent import scope
from convo.projects.day19_multi_agent.agents import (
    BOOK_TOOL,
    BOOKER,
    EVERY_AGENT,
    HANDOFF,
    POLICY_TOOL,
    ROUTER,
)
from convo.projects.day19_multi_agent.results import (
    PENDING_TARGETS,
    Results,
    handoff_scenarios,
    load,
    load_others,
)

GAP = "\n\n"
"""A blank line between Markdown paragraphs, as a constant.

Day 18's reason, unchanged: an escaped newline inside an f-string did not
survive the shell twice, producing a file that would not parse."""

GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
WARN = "#c98a16"

DESKS = (
    (ROUTER, "none", "Reads the turn and transfers. Holds nothing."),
    ("slot_collector", "none", "Collects the four details. Cannot book."),
    (BOOKER, "book_a_table", "The only holder of the booking tool."),
    ("policy_desk", "restaurant_policy", "Answers FAQs, then hands back."),
    (HANDOFF, "none", "Terminal. No transfer out, so no tool is reachable."),
)


def _topology_table(graph: scope.Topology) -> pd.DataFrame:
    rows = []
    for name, advertised, note in DESKS:
        reachable = sorted(graph.tools_reachable_from(name))
        rows.append(
            {
                "desk": name,
                "holds": advertised,
                "tools still reachable from here": ", ".join(reachable) or "none",
                "": note,
            }
        )
    return pd.DataFrame(rows)


def _handoff_table(results: Results | None) -> pd.DataFrame:
    """The five scenarios the day is about, with day 16's verdict beside them."""
    from convo.projects.day19_multi_agent.build import day16_corrected, day16_stale

    corrected = day16_corrected()
    stale = day16_stale()
    rows = []
    for scenario in handoff_scenarios():
        day16 = corrected.get(scenario, "not run")
        row = {
            "scenario": scenario,
            "day 16, one agent": day16,
            "target": "yes" if scenario in PENDING_TARGETS else "",
        }
        if results is None:
            row["day 19, router"] = "not run"
            row["booked after the handoff"] = "—"
        else:
            verdict = results.verdicts.get(scenario)
            found = next((r for r in results.rows if r.scenario_id == scenario), None)
            row["day 19, router"] = (
                "not run" if verdict is None else ("pass" if verdict else "fail")
            )
            row["booked after the handoff"] = (
                "—" if found is None else ("yes" if found.late_booking else "no")
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if stale:
        frame.attrs["note"] = "day 16's column applies the known scorer correction"
    return frame


def _proof() -> None:
    graph = scope.load()

    gr.Markdown("### The guarantee, as a graph")
    gr.Markdown(
        "Day 4's engine makes `HANDOFF` a terminal page: once the guest asks for a "
        "person, no transition leads anywhere, so booking is not something the engine "
        "can go on to do. ADK has an exact equivalent, and this is a check that it "
        "holds — not a reading of two constructor flags." + GAP + "Transfers are the "
        "edges: a router reaches its desks, a desk reaches the router and its peers "
        "unless that is disallowed. The question is then what is reachable."
    )

    if graph is None:
        gr.Markdown(
            f"<p style='color:{WARN}'>The committed topology is missing. It is derived "
            "from the real agent objects, which needs `google-adk`, and then committed "
            "so this tab and the static site can read it without the framework — run "
            "`build --write-topology` to regenerate it.</p>"
        )
    else:
        complaints = scope.audit(graph)
        stranded = sorted(graph.tools_reachable_from(HANDOFF))
        from_router = sorted(graph.tools_reachable_from(ROUTER))
        gr.Dataframe(_topology_table(graph), interactive=False, wrap=True)

        colour = GOOD if not complaints and not stranded else CRITICAL
        verdict = "holds" if not complaints and not stranded else "FAILS: " + "; ".join(complaints)
        gr.Markdown(
            f"<p style='color:{colour}'><b>Reachable from <code>{HANDOFF}</code>:</b> "
            f"{', '.join(sorted(graph.reachable_from(HANDOFF)))} — tools: "
            f"{', '.join(stranded) or '<b>none</b>'}. The guarantee {verdict}.</p>"
            + GAP
            + f"And from `{ROUTER}`, so the bot can still do its job: "
            f"{', '.join(from_router)}. A topology where nothing reaches `{BOOK_TOOL}` "
            "would pass the check above and be a broken bot, so the audit asserts both "
            "directions." + GAP + "This table is read from a committed artifact, not "
            "built here: deriving it imports the framework, and the first version of "
            "this day did that in the static site build, which runs where the framework "
            "is deliberately absent. CI re-derives it and fails if the two have drifted."
        )

    gr.Markdown("### What the proof does not cover")
    gr.Markdown(
        f"The router can still send a *never mind* turn to `{BOOKER}` instead of "
        f"`{HANDOFF}`, and then the booking happens. **The defect is not "
        "eliminated.**" + GAP + "What changes is its shape: from *the agent ignored "
        "its instructions*, which is invisible until someone reads the transcript, to "
        "*the router made one wrong, logged, attributable decision*. That is a better "
        "failure to own, and it is a smaller claim than the word guarantee usually "
        "carries. Worth saying before the numbers, not after."
    )


def _prediction() -> None:
    gr.Markdown("### Written down before the run")
    gr.Markdown(
        "Day 16's single agent failed exactly two of these five on both of its runs: "
        f"**{PENDING_TARGETS[0]}** and **{PENDING_TARGETS[1]}**. Those are the targets."
        + GAP
        + "A third — `never_books_without_details` — is recorded as a failure in day "
        "16's artifact and was not one: day 6's safety scorer convicted the bot for "
        '"I\'ll get that table reserved for you", a future-tense offer with no tool '
        "call behind it. Day 18 found that and fixed the scorer. The column below "
        "applies the correction, because crediting the router with fixing a "
        "conversation the single agent never broke would be the most flattering "
        "possible error."
    )
    gr.Dataframe(_handoff_table(load()), interactive=False, wrap=True)
    gr.Markdown(
        "**Five scenarios, not six.** The suite has six conversations that must not "
        "book, and `refuses_a_rule_override` is not in the table because it never "
        "reaches a handoff: the guest demands a table for 50, the engine re-prompts, "
        "and `book_table`'s own validator rejects the party size. Day 19's topology "
        "does nothing for that case — the tool does. Two mechanisms, two different "
        "refusals, and the tidy version of this sentence was wrong until a test said so."
    )


def _cost() -> None:
    gr.Markdown("### What it costs, and why that is measured too")
    gr.Markdown(
        "`transfer_to_agent` is an ordinary function call in ADK's stream, so routing "
        "spends model calls the single agent never spent. Those are counted into day "
        "6's efficiency dimension exactly as day 16's were — changing the counting "
        "rule per bot would make the column meaningless — and counted again on their "
        "own, so the write-up can say how much of any gap is the price of the "
        "topology rather than the agent being worse at its job." + GAP + "The "
        "per-utterance cap went from 6 rounds to 10 for the same reason: a routed "
        "turn spends one on the transfer before the specialist does anything, and "
        "keeping 6 would have cut day 19's conversations off earlier than day 16's "
        "and shown up as a behavioural difference."
    )


def _pending() -> None:
    gr.Markdown(
        f"<p style='color:{WARN}'><b>The run has not happened.</b> The free tier "
        "allows 500 requests per model per day and one pass of day 6's 32 scenarios "
        "costs 162 — measured from day 18's stored outcomes. The day's allowance was "
        "already spent when this was written, so day 19 is the half that costs "
        "nothing: the topology proof above, both oracles cross-checked against each "
        "other, and the prediction below.</p>"
        + GAP
        + "Nothing here is a result. There is no artifact, so there is no figure, "
        "which is the rule this repository runs on working as intended rather than "
        "an omission."
    )
    gr.Markdown(
        "```bash\n"
        "uv run python -m convo.projects.day19_multi_agent.build --preflight  # free\n"
        "uv run python -m convo.projects.day19_multi_agent.build             # 162 calls\n"
        "```"
    )


def _results(results: Results) -> None:
    fixed = results.fixed_what_day16_broke()
    broke = results.broke_what_day16_held()

    gr.Markdown("### The run")
    gr.Markdown(
        f"Model `{results.model}`, {results.scenarios} of {results.of_suite} scenarios."
        + (
            f" Excluded as unavailable: {', '.join(results.unavailable)}."
            if results.unavailable
            else ""
        )
        + GAP
        + f"Day 6's pass rate **{results.run.pass_rate:.1%}**, day 4's engine on the "
        f"same scenarios {results.engine.pass_rate:.1%}. Routing accuracy "
        + (
            f"**{results.routing_accuracy:.1%}**"
            if results.routing_accuracy is not None
            else "not measured"
        )
        + f", handoffs held {results.handoff_held}/{len(results.handoff_rows)}, "
        f"{results.total_transfers} transfers across the suite."
    )

    if results.model_drift:
        gr.Markdown(
            f"<p style='color:{CRITICAL}'><b>Cross-model comparison.</b> This run used "
            f"`{results.model}` and day 16's used `{results.baseline_model}`, so the "
            "difference below is topology <i>and</i> model, not topology. The build "
            "refuses this by default; it was asked for.</p>"
        )

    if results.scope_breaches:
        gr.Markdown(
            f"<p style='color:{CRITICAL}'><b>Scope breach.</b> "
            f"`{BOOK_TOOL}` executed under an agent that does not hold it, on: "
            f"{', '.join(r.scenario_id for r in results.scope_breaches)}. The static "
            "audit says no path leads there, so the framework does something the "
            "topology does not describe — and the proof above is wrong.</p>"
        )
    else:
        gr.Markdown(
            f"<p style='color:{GOOD}'>No scope breach: every `{BOOK_TOOL}` execution "
            f"was under `{BOOKER}`, which is what the static audit predicts.</p>"
        )

    gr.Markdown("#### Against day 16, on the targets")
    gr.Dataframe(_handoff_table(results), interactive=False, wrap=True)
    gr.Markdown(
        (
            f"Fixed what day 16 broke: **{', '.join(fixed)}**."
            if fixed
            else "**Fixed nothing day 16 broke.**"
        )
        + " "
        + (
            f"<span style='color:{WARN}'>Broke what day 16 held: "
            f"{', '.join(broke)}</span> — on the model-call budget alone. Every "
            "task and safety assertion in it passes: the bot refuses the "
            "impossible party politely and books nothing. What the split costs "
            "is calls, not correctness."
            if broke
            else "Broke nothing day 16 held."
        )
    )

    if results.late_bookings:
        gr.Markdown("#### Bookings written after the handoff")
        gr.Dataframe(
            pd.DataFrame(
                [
                    {
                        "scenario": r.scenario_id,
                        "handoff expected at turn": r.expected.index(HANDOFF),
                        "booked at turn": r.late_booking_turn,
                        "desks that answered": " | ".join(" ".join(a) for a in r.actual),
                    }
                    for r in results.late_bookings
                ]
            ),
            interactive=False,
            wrap=True,
        )

    with gr.Accordion("Every scenario, turn by turn", open=False):
        gr.Dataframe(
            pd.DataFrame(
                [
                    {
                        "scenario": r.scenario_id,
                        "expected": " -> ".join(r.expected),
                        "actual": " | ".join(" ".join(a) for a in r.actual),
                        "misroutes": len(r.misroutes),
                        "routing": (
                            "—" if r.routing_accuracy is None else f"{r.routing_accuracy:.0%}"
                        ),
                    }
                    for r in results.rows
                ]
            ),
            interactive=False,
            wrap=True,
        )


def _cross_model(controlled: Results) -> None:
    """Runs on another model, on the scenarios both of them reached.

    A separate section with its own heading, because it answers a different
    question: whether day 19's result is a property of the topology or of one
    small model. Merging it into the table above would make a larger model's
    score look like evidence for the architecture.
    """
    others = load_others()
    if not others:
        return

    gr.Markdown("### The same topology on a different model")
    for other in others:
        shared = controlled.scenario_ids & other.scenario_ids
        if reason := controlled.too_thin_to_compare(other):
            gr.Markdown(
                f"**`{other.model}` — not answerable.** It reached "
                f"{other.scenario_ids and len(other.scenario_ids)} of {other.of_suite} "
                f"scenarios before its daily allowance ran out, and {reason}."
                + GAP
                + "The reason is the finding. The free tier's per-day allowance is **not "
                "the same for every model**: `gemini-3.5-flash-lite` allows 500 requests "
                "a day and `gemini-3.5-flash` allows **20**. Twenty requests buys two or "
                "three conversations of this suite, and the five handoff scenarios sit at "
                "positions 12 to 20, so they cannot be reached at all."
                + GAP
                + "So the cross-model question — is day 19's result architectural or "
                "particular to one small model — **cannot be answered on a free tier.** "
                "No rate is shown here, because a percentage over one conversation would "
                "be the most misleading honest number in this repository."
            )
            continue
        mine, theirs = controlled.rates_on(shared), other.rates_on(shared)
        rows = [
            {
                "": label,
                f"{controlled.model} (day 16's)": _rate(mine[key]),
                other.model: _rate(theirs[key]),
            }
            for key, label in (
                ("pass_rate", "pass rate"),
                ("task", "task"),
                ("safety", "safety"),
                ("efficiency", "efficiency"),
            )
        ]
        gr.Markdown(
            f"`{other.model}`, {other.scenarios} of {other.of_suite} scenarios, "
            f"compared on the **{len(shared)}** both runs reached." + GAP + "Handoffs held "
            f"{other.handoff_held}/{len(other.handoff_rows)}, routing accuracy "
            + (
                f"{other.routing_accuracy:.1%}"
                if other.routing_accuracy is not None
                else "not measured"
            )
            + f", {other.total_transfers} transfers."
        )
        gr.Dataframe(pd.DataFrame(rows), interactive=False, wrap=True)

    gr.Markdown(
        "**What this can and cannot settle.** A different model is a different "
        "model: a better score here is not evidence that the topology did it, and a "
        "worse one is not evidence against. What it can show is whether the "
        "*structural* results carry over — whether the handoff desk still holds, "
        "whether anything reaches `book_table` that should not, and whether the "
        "routing errors land in the same places. Those are claims about the "
        "topology, and they are the ones worth reading across models."
    )


def _rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def build() -> None:
    gr.Markdown("## Multi-agent against one state machine")
    gr.Markdown(
        "Day 16 built this bot as one ADK agent holding every tool, and it booked a "
        "table after the guest said *never mind* and after the guest asked for a "
        "human — on both runs. Those two are the only failures here that have "
        "reproduced across every run of every model." + GAP + "So the question is "
        "narrow and answerable: **does putting the booking tool out of reach fix "
        "them?** Not does multi-agent feel more professional."
    )

    _proof()
    _prediction()
    _cost()

    results = load()
    if results is None:
        _pending()
    else:
        _results(results)
        _cross_model(results)

    with gr.Accordion("The desks, and the one variable this cannot control", open=False):
        gr.Markdown(
            "Day 5's tools and day 6's scorers are unchanged, as in day 16. What day "
            "19 adds that cannot be held fixed is one role sentence per desk — five "
            "short prompts that did not exist on day 16. So a difference in results "
            "is a difference in *topology plus those sentences*, not topology alone."
            + GAP
            + f"Agents: {', '.join(f'`{a}`' for a in EVERY_AGENT)}. Executed tools "
            f"`{BOOK_TOOL}` and `{POLICY_TOOL}`, which are day 5's names — the model "
            "is offered wrappers, and keying the measurement to the executed tool "
            "means renaming a wrapper cannot quietly change what was measured."
        )
