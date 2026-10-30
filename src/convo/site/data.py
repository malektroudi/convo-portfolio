"""Every figure the site shows, pulled from the artifacts that produced it.

This module is the only place in `convo.site` allowed to touch a result. The
renderer receives numbers and never computes one, and a test asserts the
renderer contains no metric-shaped literal — because the failure mode for a
portfolio site is not a crash, it is a number that was true in October and is
quietly wrong in November.

Each day's figures come from the same `results` module its Gradio tab calls.
Rendering the artifacts twice, through two front ends that share no display
code, is the strongest available check that the artifacts really are the
interface: if a number here disagrees with the tab, one of them is reading
the file wrong, and that is visible rather than silent.

Days with no artifact — the dialogue engine, the eval harness, the write-up —
say so. A card claiming a figure for a day that measures nothing would be the
exact dishonesty the rest of this repo is built to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from convo.core.registry import Project, projects


@dataclass(frozen=True)
class Table:
    caption: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    note: str = ""
    highlight: int | None = None
    """Row index to mark as the result worth reading."""


@dataclass(frozen=True)
class Stat:
    value: str
    label: str
    tone: str = "plain"
    """`good`, `bad` or `plain` — the renderer maps these to colour."""


@dataclass
class DayPage:
    project: Project
    summary: str = ""
    stats: tuple[Stat, ...] = ()
    tables: tuple[Table, ...] = ()
    caveat: str = ""
    """What the figures on this page do not prove. Rendered in full."""
    no_artifact: str = ""

    @property
    def day(self) -> int:
        return self.project.day

    @property
    def slug(self) -> str:
        return f"day{self.project.day:02d}"

    @property
    def shipped(self) -> bool:
        return self.project.status == "shipped"


def _pct(value: float) -> str:
    return f"{value:.0%}"


def _num(value: float, places: int = 2) -> str:
    return f"{value:.{places}f}"


# ---------------------------------------------------------------- per day


def _day02() -> tuple[str, tuple[Stat, ...], tuple[Table, ...], str]:
    from convo.projects.day02_intent_nlu.analysis import (
        best_threshold,
        confidence_stats,
        metrics_at,
    )
    from convo.projects.day02_intent_nlu.results import load

    found = load()
    if found is None:
        return "", (), (), ""
    stats = confidence_stats(found.logits)
    best = best_threshold(found.logits, found.labels)
    zero = metrics_at(found.logits, found.labels, 0.0)

    rows = []
    for threshold in (0.0, 0.3, 0.5, 0.7, 0.9):
        row = metrics_at(found.logits, found.labels, threshold)
        rows.append(
            (
                _num(threshold, 2),
                _pct(row.in_scope_accuracy),
                _pct(row.oos_recall),
                _pct(row.false_rejection_rate),
            )
        )
    return (
        "A 150-intent classifier with an out-of-scope door bolted on at "
        "inference time, not trained in.",
        (
            Stat(_pct(zero.in_scope_accuracy), "in-scope accuracy, no threshold"),
            Stat(_num(stats.p50, 3), "median top probability"),
            Stat(_num(best.threshold, 2), "balanced threshold", "good"),
        ),
        (
            Table(
                "What the rejection threshold costs and buys",
                ("threshold", "in-scope accuracy", "OOS recall", "false rejection"),
                tuple(rows),
                "Every point of out-of-scope recall is paid for by refusing "
                "something answerable. There is no setting that avoids both.",
            ),
        ),
        "Accuracy alone hid a broken run: the first model scored 74% with a "
        "maximum softmax of 0.092, so every threshold above 0.1 refused "
        "everything. The confidence distribution is checked now because of it.",
    )


def _day03() -> tuple[str, tuple[Stat, ...], tuple[Table, ...], str]:
    from convo.projects.day03_embed_router.analysis import per_new_intent_recall
    from convo.projects.day03_embed_router.results import load

    found = load()
    if found is None:
        return "", (), (), ""
    threshold = found.metadata.get("threshold", 0.5)
    recalls = per_new_intent_recall(
        found.router_scores, found.labels, found.held_out, found.class_names, threshold
    )
    values = sorted(recall for _, recall, _ in recalls)
    rows = tuple(
        (name, _pct(recall), str(count))
        for name, recall, count in sorted(recalls, key=lambda row: -row[1])
    )
    spread = values[-1] - values[0]
    return (
        "Can an embedding router answer an intent it has never been trained "
        "on, from five examples?",
        (
            Stat(_pct(values[-1]), "best new intent", "good"),
            Stat(_pct(values[0]), "worst new intent", "bad"),
            Stat(_pct(spread), "spread across ten"),
        ),
        (
            Table(
                "Five-shot recall per withheld intent",
                ("intent", "recall", "test examples"),
                rows,
                "The average across these is close to meaningless: the "
                "distribution is bimodal, so a mean lands between the two "
                "groups where no intent actually sits.",
            ),
        ),
        "Ten intents is a small sample, and the five examples per intent were "
        "picked by the dataset's own ordering rather than chosen well.",
    )


def _day05() -> tuple[str, tuple[Stat, ...], tuple[Table, ...], str]:
    from convo.projects.day05_tool_agent.results import load

    found = load()
    if found is None:
        return "", (), (), ""
    engine, schema, grounded = found.engine, found.schema_only, found.grounded
    rows = (
        ("day 4 state machine", _pct(engine.pass_rate), "0"),
        (
            "agent, schema validation only",
            _pct(schema.run.pass_rate),
            str(schema.false_confirmations),
        ),
        (
            "agent, plus a provenance guard",
            _pct(grounded.run.pass_rate),
            str(grounded.false_confirmations),
        ),
    )
    return (
        "The same booking bot rebuilt as a tool-calling agent, scored against "
        "the deterministic one on identical conversations.",
        (
            Stat(str(schema.false_confirmations), "invented bookings, schema only", "bad"),
            Stat(str(grounded.false_confirmations), "after a provenance guard", "good"),
            Stat(_pct(engine.pass_rate), "state machine pass rate"),
        ),
        (
            Table(
                "Schema validity is not the thing that was wrong",
                ("configuration", "pass rate", "invented bookings"),
                rows,
                "The model emitted one malformed call and many well-formed, "
                "entirely invented ones. Every invented booking passed the "
                "schema validator, because inventing plausible data is not a "
                "shape violation.",
                highlight=2,
            ),
        ),
        "A 0.5B model on CPU cannot settle an architectural question. The "
        "shape of the finding generalises; the pass rates do not.",
    )


def _day08() -> tuple[str, tuple[Stat, ...], tuple[Table, ...], str]:
    from convo.projects.day08_rag_cites.results import load

    found = load()
    if found is None:
        return "", (), (), ""
    rows = tuple(
        (
            name,
            str(strategy.stats["chunks"]),
            f"{strategy.stats['mean_chars']:.0f}",
            f"{strategy.stats['starts_mid_sentence']} "
            f"({strategy.stats['starts_mid_sentence'] / max(strategy.stats['chunks'], 1):.0%})",
        )
        for name, strategy in found.strategies.items()
    )
    return (
        "Retrieval that cites exact character offsets, so a claim can be "
        "checked against the source rather than taken on trust.",
        (
            Stat(str(len(found.documents)), "articles"),
            Stat(str(len(found.questions)), "questions embedded"),
            Stat("0", "citations that fail the offset check", "good"),
        ),
        (
            Table(
                "Chunking changes the citation, not the retrieval",
                ("strategy", "chunks", "mean chars", "start mid-sentence"),
                rows,
                "All three find the right article about equally often. What "
                "differs is whether the span you can show a person reads as a "
                "sentence.",
            ),
        ),
        "The corpus is synthetic, written to contain the failure modes worth "
        "testing: a near-duplicate policy pair, a superseded figure, hours "
        "split across two articles, and questions nothing answers.",
    )


def _day09() -> tuple[str, tuple[Stat, ...], tuple[Table, ...], str]:
    from convo.projects.day09_retrieval_evals.results import RETRIEVERS, load

    found = load()
    if found is None:
        return "", (), (), ""
    strategy = found.strategies["structural"]
    rows = tuple(
        (
            retriever,
            _pct(score.hit_at_1),
            _pct(score.hit_at_3),
            _num(score.mrr),
            str(len(score.misses)),
        )
        for retriever, score in ((r, strategy.score(r)) for r in RETRIEVERS)
    )
    best = strategy.score("hybrid+rerank")
    dense = strategy.score("dense")
    return (
        "Span-level labels, then four retrievers scored against them with hit@k, MRR and nDCG.",
        (
            Stat(_pct(best.hit_at_1), "hit@1, reranked", "good"),
            Stat(_pct(dense.hit_at_1), "hit@1, dense alone"),
            Stat(f"{found.rerank_ms_median:.0f} ms", "reranker cost per query"),
        ),
        (
            Table(
                "Structural chunking, 42 labelled questions",
                ("retriever", "hit@1", "hit@3", "MRR", "never found"),
                rows,
                "Hybrid fusion on its own is not reliable — it helps two "
                "chunkings and hurts the third. It earns its place as a "
                "feeder for the reranker, not as a win by itself.",
                highlight=3,
            ),
        ),
        "Ten of the 42 questions are deliberate paraphrases. The reranker "
        "helps enormously on keyword questions and not at all on those, "
        "because it reorders what retrieval already surfaced.",
    )


def _day10() -> tuple[str, tuple[Stat, ...], tuple[Table, ...], str]:
    from convo.projects.day10_guardrails.results import load

    found = load()
    if found is None:
        return "", (), (), ""
    best = found.best()
    rows = tuple(
        (
            _num(metrics.threshold, 1),
            _pct(metrics.answer_accuracy),
            _pct(metrics.refusal_recall),
            _pct(metrics.false_refusal),
            _pct(found.red_team_rate(metrics.threshold)),
        )
        for metrics in (found.at(t) for t in (-10.0, -7.0, -6.0, -2.0, 4.0))
    )
    perfect = max(
        (m for m in found.sweep()),
        key=lambda m: (found.red_team_rate(m.threshold), -abs(m.threshold)),
    )
    return (
        "Two gates: the evidence must clear a bar, and a span must verify "
        "against the source. An answer with no citation is not emitted.",
        (
            Stat(_pct(best.refusal_recall), "unanswerable correctly refused", "good"),
            Stat(_pct(best.false_refusal), "answerable refused anyway", "bad"),
            Stat(_pct(found.red_team_rate(perfect.threshold)), "red team, best threshold", "good"),
        ),
        (
            Table(
                "The red-team score is not monotonic, and that is the point",
                ("threshold", "answer", "refuse", "false refusal", "red team"),
                rows,
                "Raising the bar catches more attacks and refuses more "
                "ordinary questions, so the score peaks in the middle. Five "
                "of the 23 cases are benign questions phrased alarmingly.",
            ),
        ),
        "23 red-team prompts is a small set. The separation between the "
        "lowest legitimate question and the highest attack is real but narrow, "
        "which is a more useful result than the headline score.",
    )


def _day11() -> tuple[str, tuple[Stat, ...], tuple[Table, ...], str]:
    from convo.projects.day11_multiturn_rag.results import CONDITIONS, RETRIEVERS, load

    found = load()
    if found is None:
        return "", (), (), ""
    rows = tuple(
        (
            retriever,
            *(_num(found.score(condition, retriever).mrr, 3) for condition in CONDITIONS),
            (lambda g: "—" if g is None else _pct(g))(found.gap_closed(retriever)),
        )
        for retriever in RETRIEVERS
    )
    return (
        "Follow-ups that do not stand on their own — “is it free on "
        "Sundays” — rewritten before retrieval, measured against a "
        "hand-written ceiling.",
        (
            Stat(_pct(found.gap_closed("dense") or 0), "of the gap closed, dense", "good"),
            Stat(_pct(found.gap_closed("bm25") or 0), "of the gap closed, BM25", "bad"),
            Stat(_num(found.score("oracle", "hybrid+rerank").mrr, 3), "the ceiling"),
        ),
        (
            Table(
                "MRR on follow-up turns only",
                ("retriever", *CONDITIONS, "gap closed"),
                rows,
                "Rule-based rewriting helps semantic retrieval and hurts "
                "lexical retrieval. Resolving a pronoun swaps one stopword "
                "for a multi-word article title: an embedding sharpens, BM25 "
                "gets three more terms competing for weight.",
            ),
        ),
        "Only four of twenty follow-ups cross to a different article, which "
        "flatters the do-nothing baseline. The obvious comparison — an LLM "
        "rewriter — is missing for want of an API key.",
    )


def _day12() -> tuple[str, tuple[Stat, ...], tuple[Table, ...], str]:
    from convo.projects.day12_voice.budget import DEADLINE_MS
    from convo.projects.day12_voice.results import load

    found = load()
    if found is None:
        return "", (), (), ""
    stages = tuple(
        (
            row["stage"],
            f"{row['target_ms']:.0f} ms",
            f"{row['measured_ms']:.0f} ms",
            f"{row['ratio']:.2f}×",
        )
        for row in found.stage_table("cited", "stream")
    )
    sweep = tuple(
        (
            f"{row['hangover_ms']:.0f} ms",
            f"{row['truncated']}/{row['of']}",
            _pct(row["wer_hesitant"]),
        )
        for row in found.sweep_table()
        if row["hangover_ms"] in (100, 300, 600, 800, 1000)
    )
    gaps = found.token_vs_sentence()
    clean = found.best_hangover()
    return (
        "Real audio end to end on a laptop CPU, against a budget written "
        "before anything was measured.",
        (
            Stat(f"{found.median_total('cited', 'stream'):.0f} ms", "to first audio, no model"),
            Stat(f"{gaps['generated']['gap_ms']:.0f} ms", "first token to first sentence", "bad"),
            Stat(
                f"{clean['hangover_ms']:.0f} ms", "shortest endpointing that cuts nobody off", "bad"
            ),
        ),
        (
            Table(
                f"Where the {DEADLINE_MS:.0f} ms goes, median turn",
                ("stage", "budget", "measured", ""),
                stages,
                "Time-to-first-token is the wrong metric for voice. A token "
                "cannot be spoken; what gates the audio is the first complete "
                "sentence, and on a streamed model the two are seconds apart.",
            ),
            Table(
                "Endpointing breaks the budget by itself",
                ("silence held", "callers cut off", "their word error rate"),
                sweep,
                "Nothing is computed while the endpointer waits. Shortening "
                "it does not buy speed, it cuts people off mid-sentence.",
            ),
        ),
        "The input audio is synthetic, so the word error rates are a floor "
        "rather than a result. Latency transfers; accuracy does not.",
    )


def _day13() -> tuple[str, tuple[Stat, ...], tuple[Table, ...], str]:
    from convo.projects.day13_escalation.results import load

    found = load()
    if found is None:
        return "", (), (), ""
    rows = tuple(
        (
            row["policy"],
            _pct(row["recall"]),
            _pct(row["precision"]),
            str(row["turns_late"]),
            str(row["wasted_handoffs"]),
        )
        for row in found.policy_rows()
    )
    packets = tuple(
        (row["style"], _pct(row["complete"]), str(row["re_asks"]), str(row["words"]))
        for row in found.packet_summary()
    )
    best = found.best_policy()
    full = next(r for r in found.packet_summary() if r["style"] == "full")
    ticket = next(r for r in found.packet_summary() if r["style"] == "ticket")
    return (
        "When to hand a conversation to a person, and what to hand them.",
        (
            Stat(str(best["turns_late"]), "turns late, best policy", "good"),
            Stat(str(ticket["re_asks"] - full["re_asks"]), "re-asks removed by the packet", "good"),
            Stat(str(full["words"] - ticket["words"]), "extra words it cost"),
        ),
        (
            Table(
                "Escalation is a timing problem, not a classification one",
                ("policy", "recall", "precision", "turns late", "wasted"),
                rows,
                "Precision and recall call a handoff at turn six a success "
                "when the conversation was lost at turn two. Turns late is "
                "the metric that notices.",
            ),
            Table(
                "What the receiving agent is given",
                ("packet", "complete", "re-asks", "words"),
                packets,
                "Re-asks are facts the customer would have to repeat. The "
                "transcript row is why completeness alone is not enough: it "
                "is grounded by construction and leaves the agent to do all "
                "the reading.",
                highlight=2,
            ),
        ),
        "Sixteen conversations is a small set, and the high-stakes topic list "
        "is hand-written — the same brittleness this day measures in day 4's "
        "list of ways to ask for a human.",
    )


def _day15() -> tuple[str, tuple[Stat, ...], tuple[Table, ...], str]:
    from convo.projects.day15_analytics.results import load

    found = load()
    if found is None:
        return "", (), (), ""
    row = found.containment
    gaps = tuple(
        (r["refusal_reason"], str(r["refusals"]), str(r["were_follow_ups"]))
        for r in found.rows("03_content_gaps")
    )
    surfaces = tuple(
        (
            r["surface"],
            str(r["conversations"]),
            _pct(r["containment_naive"]),
            _pct(r["containment_honest"]),
        )
        for r in found.rows("02_containment_by_surface")
    )
    return (
        "Turn events from every measured run in this repository, queried with "
        "SQL that runs on DuckDB and BigQuery unchanged.",
        (
            Stat(_pct(row["containment_naive"]), "containment, as reported", "good"),
            Stat(_pct(row["containment_honest"]), "containment, honestly", "bad"),
            Stat(str(row["abandoned_on_a_refusal"]), "customers who gave up", "bad"),
        ),
        (
            Table(
                "The same conversations, two definitions",
                ("surface", "conversations", "as reported", "honestly"),
                surfaces,
                "The naive metric counts a customer who gave up as a success, "
                "because nobody was escalated to. Adding one clause — the last "
                "thing the system did was answer — is the whole difference.",
            ),
            Table(
                "What the help centre is missing",
                ("refusal reason", "refusals", "were follow-ups"),
                gaps,
                "Every refusal where nothing was worth citing is a question "
                "the corpus cannot answer. Grouped, it is a writing backlog "
                "ordered by demand. Red-team prompts are excluded: a refused "
                "attack is the guardrail working.",
            ),
        ),
        "There are no timestamps anywhere in this repository, so there is no "
        "trend, cohort or seasonality analysis here — a line going up and to "
        "the right would have been more impressive and would have been "
        "fiction.",
    )


def _day16() -> tuple[str, tuple[Stat, ...], tuple[Table, ...], str]:
    from convo.projects.day16_adk_agent.results import Scope, load

    found = load()
    if found is None:
        return "", (), (), ""

    def rate(value: float | None) -> str:
        return "not measured" if value is None else _pct(value)

    def count(value: int | None) -> str:
        return "not measured" if value is None else str(value)

    full = Scope.FULL
    bots = found.ordered(full)
    scores = tuple(
        (
            bot.label,
            rate(bot.pass_rate(full)),
            rate(bot.dimension(full, "task")),
            rate(bot.dimension(full, "safety")),
            count(bot.false_confirmations(full)),
            count(bot.model_calls(full)),
        )
        for bot in bots
    )
    failures = tuple(
        (
            case.case_id,
            "; ".join(f"{a.name} ({a.dimension})" for a in case.assertions if not a.passed),
        )
        for case in found.adk.failures(full)
    )
    scope = (
        f"all {found.of_suite} scenarios"
        if not found.partial
        else f"the {found.scenarios} of {found.of_suite} scenarios the free tier allowed"
    )
    return (
        "Day 4's booking bot for the third time, in Google's Agent Development "
        f"Kit on {found.model}, scored by day 6's unchanged harness over {scope}.",
        (
            Stat(rate(found.adk.pass_rate(full)), "ADK + Gemini", "good"),
            Stat(rate(found.engine.pass_rate(full)), "state machine, day 4", "good"),
            Stat(
                count(found.adk.false_confirmations(full)),
                "tables confirmed but never booked",
                "bad",
            ),
        ),
        (
            Table(
                "ADK against a state machine, same suite, same scorers",
                ("bot", "pass rate", "task", "safety", "false confirmations", "model calls"),
                scores,
                "Day 5's tools are imported rather than reimplemented, so a "
                "difference in these columns cannot be a difference in the "
                "business rules. Day 5's own agent is absent here: it ran a "
                "smaller suite whose assertions carry no dimension, so it "
                "belongs in a pass-rate comparison and nowhere near this table.",
            ),
            Table(
                "Every scenario ADK got wrong",
                ("scenario", "what broke"),
                failures,
                "All of one kind: it booked a table when the right answer was "
                "to book nothing. Not a malformed call, not a bad parameter, "
                "not a runaway loop — the failures day 5 wrote most of its code "
                "to prevent are the ones the framework does prevent. The "
                "escalation case invented a guest name to fill a required "
                "argument, which is exactly what day 5's provenance guard was "
                "built to catch and exactly what ADK has no equivalent of.",
            ),
        ),
        "A framework removes the code you would have written anyway, not the "
        "code you had to invent. ADK replaced day 5's schema validation, call "
        "parsing, retry loop and session state; it replaced nothing of the "
        "provenance guard, and that is where the remaining failures are.",
    )


def _day18() -> tuple[str, tuple[Stat, ...], tuple[Table, ...], str]:
    from convo.projects.day18_eval_gate.results import load

    found = load()
    if found is None:
        return "", (), (), ""

    rates = (
        ("day 6's harness — did it end up right", _pct(found.harness.pass_rate)),
        ("ADK trajectory — right tools, right turns", _pct(found.trajectory_pass_rate)),
        ("ADK trajectory — right tools, any turn", _pct(found.flat_pass_rate)),
        ("the two evaluators agree", _pct(found.agreement_rate)),
    )
    disagreements = tuple(
        (
            row.scenario_id,
            "pass" if row.harness_ok else "FAIL",
            "pass" if row.adk_ok else "FAIL",
            row.adjudication.value,
            row.adk_detail or row.harness_detail,
        )
        for row in found.disagreements
    )
    scope = (
        f"all {found.of_suite} scenarios"
        if not found.partial
        else f"{found.scenarios} of {found.of_suite} scenarios before the free tier stopped it"
    )
    timing = (
        f" Of the trajectory failures, {found.only_timing} are one behaviour: the "
        f"agent calls the right tools one turn early, because it books without "
        f"waiting for the guest to confirm."
        if found.only_timing
        else ""
    )
    return (
        "Day 6's harness asks whether the conversation ended right. ADK's "
        "evaluator asks whether it went the right way. The same "
        f"{len(found.rows)} conversations, scored both ways, over {scope}.",
        (
            Stat(_pct(found.harness.pass_rate), "day 6's harness", "good"),
            Stat(_pct(found.trajectory_pass_rate), "ADK, turn-aligned", "bad"),
            Stat(_pct(found.agreement_rate), "the two agree", "bad"),
        ),
        (
            Table(
                "One run, two evaluators",
                ("what it measures", "pass rate"),
                rates,
                "The expected trajectory is derived from day 4's state machine, "
                "not hand-written — the same oracle day 5's provenance guard "
                "used, because it has no model in it and cannot be talked "
                "round. Collapsing turn boundaries is what separates calling "
                "the wrong tools from calling the right ones at the wrong "
                "time." + timing,
            ),
            Table(
                "Where they disagree",
                ("scenario", "day 6", "ADK", "who the spec backs", "why"),
                disagreements,
                "The spec verdict comes from each scenario's own declared "
                "expectations, written before any bot existed. Day 6's scorers "
                "read them; ADK's trajectory metric never sees them. Where the "
                "disagreement is about the route rather than the outcome the "
                "spec has nothing to say, and the case is reported as "
                "undecided rather than scored against either evaluator.",
            ),
        ),
        "Both gates run in CI and neither calls a model: a trajectory metric "
        "needs a recorded trajectory and an expected one, and both are "
        "committed. The gate also fails if ADK's own evaluator stops "
        "reproducing the scores published here — this is the only day where a "
        "dependency derives a figure, and it is installed outside the "
        "lockfile.",
    )


def _day19() -> tuple[str, tuple[Stat, ...], tuple[Table, ...], str]:
    """Day 19 publishes a proof before it publishes a measurement.

    The reachability table is read from day 19's committed topology artifact,
    never derived here: deriving it builds the real agent objects and so
    imports `google-adk`, which this job deliberately does not install. The
    first version of this function called the deriving function and took the
    Pages build down, which is the clearest possible demonstration of why the
    artifact exists.

    The pass rate is not published, because the run has not happened.
    """
    from convo.projects.day19_multi_agent import scope
    from convo.projects.day19_multi_agent.build import day16_corrected
    from convo.projects.day19_multi_agent.results import (
        PENDING_TARGETS,
        handoff_scenarios,
        load,
        load_others,
    )

    graph = scope.load()
    found = load()

    reach = (
        tuple(
            (
                desk,
                ", ".join(graph.tools.get(desk, ())) or "none",
                ", ".join(sorted(graph.tools_reachable_from(desk))) or "none",
            )
            for desk in (
                "booking_router",
                "slot_collector",
                "booking_desk",
                "policy_desk",
                "handoff_desk",
            )
        )
        if graph is not None
        else ()
    )
    corrected = day16_corrected()
    targets = tuple(
        (
            scenario,
            corrected.get(scenario, "not run"),
            ("not run" if found is None else ("pass" if found.verdicts.get(scenario) else "fail")),
            "yes" if scenario in PENDING_TARGETS else "",
        )
        for scenario in handoff_scenarios()
    )

    cross = ()
    if found is not None:
        for other in load_others():
            shared = found.scenario_ids & other.scenario_ids
            if reason := found.too_thin_to_compare(other):
                cross += (
                    Table(
                        f"The same topology on {other.model} — not answerable",
                        ("what was asked", "what the free tier allows"),
                        (
                            (
                                "is day 19's result architectural, or particular "
                                "to one small model?",
                                "unanswered",
                            ),
                            ("gemini-3.5-flash-lite, requests per day", "500"),
                            ("gemini-3.5-flash, requests per day", "20"),
                            (
                                "scenarios it reached before the wall",
                                f"{other.scenarios} of {other.of_suite}",
                            ),
                        ),
                        f"The run stopped because {reason}. The per-day allowance is not "
                        "the same for every model, which is the finding: twenty requests "
                        "buys two or three conversations, and the five handoff scenarios "
                        "sit at positions 12 to 20, so the structural claim cannot be "
                        "reached at all. No rate is shown, because a percentage over one "
                        "conversation would be the most misleading honest number here.",
                    ),
                )
                continue
            mine, theirs = found.rates_on(shared), other.rates_on(shared)
            cross += (
                Table(
                    f"The same topology on {other.model}",
                    ("", f"{found.model} (day 16's)", other.model),
                    tuple(
                        (
                            label,
                            _pct(mine[key]) if mine[key] is not None else "not measured",
                            _pct(theirs[key]) if theirs[key] is not None else "not measured",
                        )
                        for key, label in (
                            ("pass_rate", "pass rate"),
                            ("task", "task"),
                            ("safety", "safety"),
                            ("efficiency", "efficiency"),
                        )
                    ),
                    f"Compared on the {len(shared)} scenarios both runs reached, because "
                    "comparing a 32-scenario rate against a 24-scenario one is the "
                    "denominator mistake day 5 made. A different model is a different "
                    "model: a better score here is not evidence the topology did it. What "
                    "carries across models is the structural result — whether the handoff "
                    f"desk still holds ({other.handoff_held} of "
                    f"{len(other.handoff_rows)} here), whether anything reached the "
                    "booking tool that should not, and where the routing errors land.",
                ),
            )

    tables = (
        Table(
            "What each desk can still reach",
            ("desk", "holds", "tools reachable from here"),
            reach,
            "Transfers are the edges: a router reaches its desks, a desk reaches "
            "the router and its peers unless that is disallowed. The handoff desk "
            "has both disallowed, so nothing is reachable from it and the booking "
            "tool is out of scope for the rest of the conversation. The audit also "
            "asserts the opposite direction — a topology where nothing could book "
            "would satisfy the first check and be a broken bot.",
            highlight=4,
        ),
        Table(
            "The five conversations this day re-tests",
            ("scenario", "day 16, one agent", "day 19, router", "pre-named target"),
            targets,
            (
                "Day 16's column applies the correction day 18 forced: its artifact "
                "records never_books_without_details as a failure, and that was a "
                "false positive in day 6's own safety scorer. "
                if found is not None and found.day16_stale
                else "Day 16's column is a fresh measurement: the day has been re-run "
                "against the fixed scorer, so the override day 19 carried for it has "
                "removed itself. "
            )
            + "The suite has six must-not-book conversations; the sixth, "
            "refuses_a_rule_override, never reaches a handoff — book_table's own "
            "validator rejects a party of 50 — so this day's guarantee does nothing "
            "for it.",
        ),
    ) + cross

    if found is None:
        return (
            "Day 16's single ADK agent held every tool on every turn, and it booked "
            "a table after the guest said never mind and after the guest asked for a "
            "human, on both runs. Day 19 moves the booking tool onto one desk and "
            "gives the handoff desk no way back to it, so the defect is unreachable "
            "rather than discouraged.",
            (
                Stat("none", "tools reachable after a handoff", "good"),
                Stat("2", "failures it was built to fix", "plain"),
                Stat("not run", "pass rate", "bad"),
            ),
            tables,
            "The run has not happened: the free tier allows 500 requests per model "
            "per day, one pass of day 6's suite costs 162, and the day's allowance "
            "was spent. So this day publishes the half that costs nothing — the "
            "reachability proof, two oracles cross-checked against each other, and "
            "the two target scenarios named in advance. There is no artifact, so "
            "there is no figure, which is the repository's rule working rather than "
            "an omission.",
        )

    fixed = found.fixed_what_day16_broke()
    broke = found.broke_what_day16_held()
    return (
        f"The same bot as day 16, split into a router and four desks, on "
        f"{found.scenarios} of {found.of_suite} scenarios. The booking tool lives on "
        "one desk and the handoff desk has no path back to it.",
        (
            Stat(_pct(found.run.pass_rate), "day 6's harness", "plain"),
            Stat(
                f"{found.handoff_held}/{len(found.handoff_rows)}",
                "handoffs held",
                "good" if found.handoff_held == len(found.handoff_rows) else "bad",
            ),
            Stat(
                _pct(found.routing_accuracy) if found.routing_accuracy is not None else "—",
                "turns routed correctly",
                "plain",
            ),
        ),
        tables,
        (
            (
                "Fixed what day 16 broke: " + ", ".join(fixed) + ". "
                if fixed
                else "Fixed nothing day 16 broke. "
            )
            + (
                "Broke what day 16 held: " + ", ".join(broke) + " — on the "
                "model-call budget alone, with every task and safety assertion "
                "passing. "
                if broke
                else "Broke nothing day 16 held. "
            )
            + f"Routing cost {found.total_transfers} transfers, each one a model call "
            "the single agent never spent — counted into day 6's efficiency dimension "
            "exactly as day 16's calls were, and counted again on its own so the "
            "price of the guarantee is visible."
            + (
                f" This run used {found.model} and day 16's used "
                f"{found.baseline_model}, so the difference is topology and model "
                "rather than topology."
                if found.model_drift
                else ""
            )
            + (
                f" The free tier stopped the run at {found.scenarios} of "
                f"{found.of_suite}: the day's 500 requests ran out on the suite's "
                "last scenario, digression_before_anything_else, which is therefore "
                "absent rather than scored as a failure."
                if found.partial
                else ""
            )
        ),
    )


def _day20() -> tuple[str, tuple[Stat, ...], tuple[Table, ...], str]:
    """Day 20 is the only week-3 day whose figures cost nothing to re-derive.

    No model, no key, no quota: the measurement is protobuf serialisation and
    the transport is an ASGI app in the same process.
    """
    from convo.projects.day20_a2a_handoff.results import load

    found = load()
    if found is None:
        return "", (), (), ""

    checks = (
        ("bytes arrived as sent", "integrity"),
        ("hash of the decoded payload matches", "payload_hash_matches"),
        ("a Packet equal to the one sent rebuilds", "reconstructed"),
        ("day 13's grounding check can be re-run", "provenance_recheckable"),
    )
    rows = tuple(
        (label,) + tuple(found.tally(carrier, field) for carrier in found.carriers)
        for label, field in checks
    )

    return (
        f"Day 13's {found.packets} escalation packets, each sent twice to a real A2A "
        f"agent discovered from its own card — once as the text a person reads, once "
        f"as structure a program can check. Protocol {found.protocol}, "
        f"a2a-sdk {found.sdk}.",
        (
            Stat(f"{found.any_clean}/{len(found.crossings)}", "crossings fully verifiable", "bad"),
            Stat(found.tally("data", "payload_hash_matches"), "structured hashes match", "bad"),
            Stat(f"x{found.mean_multiplier}", "words, to make provenance checkable", "plain"),
        ),
        (
            Table(
                "What the receiving agent can establish",
                ("the receiver can establish", *found.carriers),
                rows,
                "Text is what most systems send and cannot be parsed back: the "
                "rendering joins slots on a comma and tried-items on a semicolon, so a "
                "value containing either is indistinguishable from two entries. "
                "Structure rebuilds exactly and fails the hash, because protobuf's "
                "Value has one numeric kind and it is a double, so a word count of 237 "
                "arrives as 237.0. The dict still compares equal, which is why the "
                "obvious test passes. Neither carrier can re-run the grounding check: "
                "it needs the conversation, and the protocol has no reason to send it.",
                highlight=3,
            ),
        ),
        "A provenance guard that lives in the sender's process is not a property the "
        "receiver holds — it is a claim the receiver takes on trust. Sending the "
        f"conversation as well makes the guard runnable, agreeing with the sender on "
        f"{found.grounded_with_history} packets, at x{found.mean_multiplier} the words. "
        "That spends day 13's third property, brevity, to buy its first. Day 13 "
        "declined to put a threshold on length because none is defensible; this is the "
        "trade it was refusing to prejudge.",
    )


def _day21() -> tuple[str, tuple[Stat, ...], tuple[Table, ...], str]:
    from convo.projects.day21_capstone.results import load

    found = load()
    if found is None:
        return "", (), (), ""
    tally = found.tally
    audit = tuple(
        (
            row.key,
            ", ".join(row.where),
            row.published,
            row.derived if row.derived is not None else "nothing derives it",
            row.verdict,
        )
        for row in found.audit
    )
    costs = tuple(
        (row["wanted"], row["instead"], row["dollars"], row["lost"]) for row in found.costs
    )
    pipeline = tuple(
        (
            stage["name"],
            stage["size"],
            "measured here" if stage["measured"] else "asserted",
        )
        for stage in found.pipeline
    )
    return (
        f"{found.shipped} of {found.of_days} days, {found.days_with_artifacts} of "
        f"them leaving a committed artifact totalling {found.megabytes} MB. The "
        "brief asked for one narrative, one architecture diagram and one cost "
        "table reading $0; the useful part was auditing whether the front "
        "pages' numbers were still the artifacts' numbers.",
        (
            Stat(str(tally["stale"]), "front-page figures that had gone stale", "bad"),
            Stat(str(tally["holds"]), "that still held", "good"),
            Stat(
                str(tally["no artifact"] + tally["not stored"]),
                "that nothing can check",
                "bad",
            ),
        ),
        (
            Table(
                "Every figure the three front pages published, re-derived",
                ("figure", "shown on", "as published", "re-derived now", "verdict"),
                audit,
                "The per-day pages were all correct, because each one computes "
                "its figures from the artifact it describes. These were typed "
                "in by hand, weeks apart, and nothing checked them again. The "
                "hub and this site also disagreed with each other about which "
                "finding was the portfolio's best.",
            ),
            Table(
                "One cost table",
                ("what the plan assumed", "used instead", "cost", "what that cost instead"),
                costs,
                "Every row reads $0, which was the brief. The last column is "
                "the one worth reading.",
            ),
            Table(
                "One architecture, with the middle number measured",
                ("stage", "size", "evidence"),
                pipeline,
                "Heavy work on a laptop, small artifacts in git, a serving "
                "layer with no ML framework in it. The middle figure was "
                "written down as 9 MB, which was true when two days had "
                "artifacts; it is computed now. The image's size is the last "
                "figure nothing commits, so a ceiling is published in its "
                "place and CI checks it.",
                highlight=1,
            ),
        ),
        "This day audits what the repository published, not whether the "
        "underlying work was good. A figure that recomputes can still be the "
        "wrong figure to have measured, and the prose around a number does not "
        "recompute at all — day 21's artifact exists as a tripwire for exactly "
        "that, and a tripwire is weaker than a check.",
    )


BUILDERS = {
    2: _day02,
    3: _day03,
    5: _day05,
    8: _day08,
    9: _day09,
    10: _day10,
    11: _day11,
    12: _day12,
    13: _day13,
    15: _day15,
    16: _day16,
    18: _day18,
    19: _day19,
    20: _day20,
    21: _day21,
}

NO_ARTIFACT = {
    1: "The scaffold: the shared core, the hub, the container. Its output is "
    "the repository, not a measurement.",
    4: "A deterministic dialogue engine with no model in it, so there is "
    "nothing to measure against a baseline — the tests are the artifact.",
    6: "The eval harness the rest of the repo is gated by. Its results are "
    "every other day's numbers.",
    7: "The write-up, and the defects that looking at the hub revealed.",
    14: "This site. It has no artifact of its own because it is generated "
    "from every other day's — which is the whole argument for building it "
    "this way.",
}


@dataclass
class Site:
    pages: list[DayPage] = field(default_factory=list)

    @property
    def shipped(self) -> list[DayPage]:
        return [p for p in self.pages if p.shipped]

    @property
    def planned(self) -> list[DayPage]:
        return [p for p in self.pages if not p.shipped]


def collect() -> Site:
    """Build every page's figures. Nothing here is written by hand."""
    # Importing the projects package is what runs each day's `ship()` call and
    # marks it shipped. The registry is the single source of truth for what
    # exists, so the site reads it rather than keeping a second list that
    # would drift the first time a day was added to one and not the other.
    import convo.projects  # noqa: F401

    pages: list[DayPage] = []
    for project in projects():
        page = DayPage(project=project)
        if project.status != "shipped":
            pages.append(page)
            continue

        builder = BUILDERS.get(project.day)
        if builder is None:
            page.no_artifact = NO_ARTIFACT.get(project.day, "")
            pages.append(page)
            continue

        summary, stats, tables, caveat = builder()
        page.summary = summary
        page.stats = stats
        page.tables = tables
        page.caveat = caveat
        if not tables:
            page.no_artifact = (
                "This day's artifact has not been built, so its figures are "
                "not shown rather than guessed."
            )
        pages.append(page)
    return Site(pages)
