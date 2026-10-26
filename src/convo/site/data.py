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
