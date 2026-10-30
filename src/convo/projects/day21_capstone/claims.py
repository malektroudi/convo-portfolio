"""The portfolio's headline claims, each with the code that recomputes it.

Every per-day page in this repository computes its figures from the artifact
that produced them, and a test asserts the renderer holds no metric-shaped
literal. Two pages were exempt from that discipline and nobody noticed: the
Gradio hub's overview and the static site's index each carried three findings
typed in by hand, written weeks apart, never checked again.

They are the first thing a reader sees. They were also the only numbers in the
repository that nothing re-derived — and, it turns out, the only wrong ones.

So this module makes them data. A claim carries the sentence it wants to make
and, where one exists, a resolver that recomputes the figure from a committed
artifact. `WAS_PUBLISHED` keeps a frozen record of what the two pages said
before day 21, so the drift is reportable rather than quietly corrected, and
`audit()` is what reports it.

Four kinds of number can appear on a page like this, and the difference is the
whole point of the day:

`DERIVED`
    Recomputed from a committed artifact every time the page is built. Anyone
    with the repository can falsify it, and a stale one is impossible.

`PINNED`
    The artifact is committed, but regenerating it costs model quota, so a
    rerun might land somewhere else. The committed file is the evidence; the
    figure is honest about what it is.

`TESTIMONY`
    True, recorded in prose, with no artifact behind it — usually because the
    run that produced it was overwritten by the run that fixed it. Day 2's
    0.092 is the portfolio's best finding and is in this category, which is an
    uncomfortable thing to discover on the last day.

`UNSTORED`
    A build step measures it and nothing commits it, so the page can go stale
    without anything failing. The serving image's size is the only one left,
    and day 21 replaces it with a weaker figure that is checked.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from convo.projects.day21_capstone.ledger import IMAGE_CEILING_MB as _IMAGE_CEILING_MB

DERIVED = "derived"
PINNED = "pinned"
TESTIMONY = "testimony"
UNSTORED = "unstored"

KINDS = (DERIVED, PINNED, TESTIMONY, UNSTORED)

# What each kind means, in one line, rendered next to the figure rather than
# kept in a legend the reader has to go and find.
MEANS = {
    DERIVED: "recomputed from a committed artifact on every build",
    PINNED: "committed, but a rerun costs quota and could differ",
    TESTIMONY: "true, written down, no artifact behind it",
    UNSTORED: "a build measures it; nothing commits it",
}

Resolver = Callable[[], "str | None"]
"""Returns the figure, or None when the artifact it needs is absent."""


@dataclass(frozen=True)
class Claim:
    """One finding worth putting on a front page.

    `figure` is only a fallback for the kinds that have no resolver. For
    `DERIVED` and `PINNED` claims the published figure is whatever `resolve`
    returns, so this dataclass cannot carry a stale one.
    """

    key: str
    day: int
    headline: str
    body: str
    kind: str
    resolve: Resolver | None = None
    figure: str = ""
    lead: bool = False
    """Shown as a card on both front pages. The flag lives here so the two
    front ends cannot disagree about what the portfolio's headline is, which
    is precisely what they did."""

    def value(self) -> str | None:
        """What to put on the page."""
        if self.resolve is None:
            return self.figure or None
        return self.resolve()

    def derived(self) -> str | None:
        """What a committed artifact says, or None when nothing derives it.

        Deliberately not `value()`. A `TESTIMONY` claim's figure is the
        literal that was typed in, so checking it against itself would make
        the audit report that the one unfalsifiable number in the portfolio
        holds — which is the exact failure the audit exists to catch.
        """
        return None if self.resolve is None else self.resolve()


# ------------------------------------------------------------------ resolvers
#
# Each one imports inside the function. A claim whose day has no artifact
# returns None and is reported as unavailable rather than taking the module
# down, and `convo.site` can import this without paying for fourteen days of
# artifact loading to render one page.


def _pct(value: float) -> str:
    return f"{value:.0%}"


def _day02_committed_run() -> str | None:
    from convo.projects.day02_intent_nlu.analysis import confidence_stats, metrics_at
    from convo.projects.day02_intent_nlu.results import load

    found = load()
    if found is None:
        return None
    stats = confidence_stats(found.logits)
    accuracy = metrics_at(found.logits, found.labels, 0.0).in_scope_accuracy
    return f"{stats.maximum:.3f} / {_pct(accuracy)}"


def _day03_recalls() -> list[float] | None:
    from convo.projects.day03_embed_router.analysis import per_new_intent_recall
    from convo.projects.day03_embed_router.results import load

    found = load()
    if found is None:
        return None
    return [
        recall
        for _, recall, _ in per_new_intent_recall(
            found.router_scores,
            found.labels,
            found.held_out,
            found.class_names,
            found.metadata.get("threshold", 0.5),
        )
    ]


def _day03_spread() -> str | None:
    recalls = _day03_recalls()
    if recalls is None:
        return None
    ordered = sorted(recalls)
    return f"{_pct(ordered[-1])} to {_pct(ordered[0])}"


def _day03_average() -> str | None:
    recalls = _day03_recalls()
    if recalls is None:
        return None
    return _pct(sum(recalls) / len(recalls))


def _day05_false_confirmations() -> str | None:
    from convo.projects.day05_tool_agent.results import load

    found = load()
    if found is None:
        return None
    return f"{found.schema_only.false_confirmations} to {found.grounded.false_confirmations}"


def _day05_pass_rates() -> str | None:
    """The other half of day 5, and the reason its headline was not news.

    Both agent configurations pass the same share of day 6's suite. The guard
    removed ten invented bookings and added ten missed ones, which is a real
    improvement that no single number shows.
    """
    from convo.projects.day05_tool_agent.results import load

    found = load()
    if found is None:
        return None
    before = found.schema_only.run.pass_rate
    after = found.grounded.run.pass_rate
    return f"{_pct(before)} to {_pct(after)}"


def _day12_token_to_sentence() -> str | None:
    from convo.projects.day12_voice.results import load

    found = load()
    if found is None:
        return None
    gaps = found.token_vs_sentence()["generated"]
    return f"{gaps['first_token_ms']:.0f} ms to {gaps['first_sentence_ms']:.0f} ms"


def _artifacts_total() -> str | None:
    """The middle number of the architecture, measured rather than remembered."""
    from convo.projects.day21_capstone.ledger import committed_bytes

    return f"{committed_bytes() / 1024 / 1024:.1f} MB"


def _day15_containment() -> str | None:
    from convo.projects.day15_analytics.results import load

    found = load()
    if found is None:
        return None
    row = found.containment
    return f"{_pct(row['containment_naive'])} to {_pct(row['containment_honest'])}"


def _day18_agreement() -> str | None:
    from convo.projects.day18_eval_gate.results import load

    found = load()
    if found is None:
        return None
    return _pct(found.agreement_rate)


def _day19_reachable() -> str | None:
    """Read from the committed topology, which needs no framework and no key."""
    from convo.projects.day19_multi_agent import scope

    graph = scope.load()
    if graph is None:
        return None
    return str(len(graph.tools_reachable_from(scope.HANDOFF)))


def _day19_coverage() -> str | None:
    from convo.projects.day19_multi_agent.results import load

    found = load()
    if found is None:
        return None
    return f"{found.scenarios} of {found.of_suite}"


def _day20_clean() -> str | None:
    from convo.projects.day20_a2a_handoff.results import load

    found = load()
    if found is None:
        return None
    return f"{found.any_clean} of {len(found.crossings)}"


# --------------------------------------------------------------------- claims

CLAIMS: tuple[Claim, ...] = (
    Claim(
        key="artifacts-total",
        day=1,
        headline="Everything the serving layer reads, in one number",
        body=(
            "Heavy work on a laptop, small files in git, a serving layer with "
            "no ML framework in it. This figure is the bargain, and it was "
            "written into three pages by hand when two days had artifacts."
        ),
        kind=DERIVED,
        resolve=_artifacts_total,
    ),
    Claim(
        key="image-size",
        day=1,
        headline="The serving image, as a ceiling rather than a size",
        body=(
            "The only machine that can measure the image is the one building "
            "it, and nothing commits what it finds. So the figure published "
            "here is a ceiling: a number that is in a file, that CI checks on "
            "every push, and that fails loudly when the image grows into it. "
            "The gate answered on its first run — the image is 630 MB, "
            "where three pages had said 588. An unstored figure is not a "
            "softer case than a stale one; it is a stale one with the evidence "
            "removed."
        ),
        kind=UNSTORED,
        figure=f"under {_IMAGE_CEILING_MB} MB",
    ),
    Claim(
        key="day02-max-confidence",
        day=2,
        headline="Accuracy hid a model that could never be thresholded",
        body=(
            "A first training run reached 74% in-scope accuracy — mediocre but "
            "plausible — with a maximum softmax confidence of 0.092 across all "
            "5,500 test rows. Every rejection threshold above 0.1 therefore "
            "refused everything, and accuracy alone cannot show that. It is the "
            "best finding in this portfolio and the artifact behind it no longer "
            "exists: the run that fixed the problem overwrote it."
        ),
        kind=TESTIMONY,
        figure="0.092",
        lead=True,
    ),
    Claim(
        key="day02-committed-run",
        day=2,
        headline="What the committed run actually says",
        body=(
            "Maximum confidence and in-scope accuracy, from the artifact that is "
            "in the repository. Both are healthy, which is exactly why the "
            "finding above cannot be checked against it."
        ),
        kind=DERIVED,
        resolve=_day02_committed_run,
    ),
    Claim(
        key="day03-spread",
        day=3,
        headline="Five-shot routing, from perfect to nothing on the same five examples",
        body=(
            "Across ten intents the fine-tuned classifier never saw, an "
            "embedding router's recall spans the whole range and is roughly "
            "bimodal — so the average lands between the two groups, where no "
            "intent actually sits. Whether a handful of examples will do is "
            "predictable before anyone collects them."
        ),
        kind=DERIVED,
        resolve=_day03_spread,
        lead=True,
    ),
    Claim(
        key="day03-average",
        day=3,
        headline="The average of those ten, which is the number not to quote",
        body=(
            "Reported here because the front page quoted it for weeks as though it meant something."
        ),
        kind=DERIVED,
        resolve=_day03_average,
    ),
    Claim(
        key="day05-false-confirmations",
        day=5,
        headline="Invented bookings, every one of which passed schema validation",
        body=(
            "A 0.5B model emitted one malformed tool call and ten well-formed, "
            "entirely fictional reservations. The validator accepted all ten, "
            "because inventing plausible data is not a shape violation. A "
            "provenance guard — confirm only what a tool actually returned — "
            "removed them."
        ),
        kind=DERIVED,
        resolve=_day05_false_confirmations,
        lead=True,
    ),
    Claim(
        key="day05-pass-rates",
        day=5,
        headline="And the pass rate did not move",
        body=(
            "The guard traded ten invented bookings for ten missed ones, so the "
            "headline score is identical before and after the most important fix "
            "in week 1. A suite reporting one number would have called it a "
            "no-op."
        ),
        kind=DERIVED,
        resolve=_day05_pass_rates,
    ),
    Claim(
        key="day12-token-to-sentence",
        day=12,
        headline="Time-to-first-token is the wrong metric for voice",
        body=(
            "A token cannot be spoken. What gates the audio is the first "
            "complete sentence, and on the generated path the two are a second "
            "and a half apart — so a dashboard tracking time-to-first-token "
            "reports a healthy number while the caller sits in silence."
        ),
        kind=DERIVED,
        resolve=_day12_token_to_sentence,
        lead=True,
    ),
    Claim(
        key="day15-containment",
        day=15,
        headline="Containment, as reported and as it happened",
        body=(
            "The usual definition counts a conversation with no human in it as a "
            "success, which silently scores every customer who gave up as a win. "
            "Adding one clause — the last thing the system did was answer — is "
            "the entire difference, and the SQL runs unchanged on DuckDB and "
            "BigQuery."
        ),
        kind=DERIVED,
        resolve=_day15_containment,
        lead=True,
    ),
    Claim(
        key="day18-agreement",
        day=18,
        headline="Two evaluators, one suite, almost no agreement",
        body=(
            "ADK's trajectory metric and day 6's harness score the same stored "
            "conversations and reach the same verdict on roughly a fifth of "
            "them. Neither is broken: they measure different things, and a team "
            "that swapped one for the other would read the difference as a "
            "regression."
        ),
        kind=PINNED,
        resolve=_day18_agreement,
        lead=True,
    ),
    Claim(
        key="day19-reachable",
        day=19,
        headline="Tools reachable from the handoff desk",
        body=(
            "Day 16's single agent booked a table after the guest asked for a "
            "human. Day 19 moves booking onto its own desk and gives the handoff "
            "desk no transfer back, so the defect is unreachable rather than "
            "discouraged — and that is a property of a committed graph, "
            "checkable with no model, no key and no quota."
        ),
        kind=DERIVED,
        resolve=_day19_reachable,
    ),
    Claim(
        key="day19-coverage",
        day=19,
        headline="And the run behind it is one scenario short",
        body=(
            "The free tier allows 500 requests per model per day. A full pass "
            "reports 213 model calls and spends the whole allowance, so the "
            "day's requests ran out on the suite's last scenario — which is "
            "absent rather than scored as a failure. Safety is 31 of 31, task "
            "30 of 31, and both efficiency failures are call-budget failures "
            "with every task and safety assertion passing."
        ),
        kind=PINNED,
        resolve=_day19_coverage,
    ),
    Claim(
        key="day20-clean",
        day=20,
        headline="Crossings that pass every check a receiver could run",
        body=(
            "Day 13's handoff packets over real A2A, as text and as structure. "
            "Text arrives intact and cannot be parsed back; structure rebuilds "
            "exactly and fails the standard hash-the-payload check, because "
            "protobuf has one numeric kind and it is a double. Neither carrier "
            "can re-run the provenance guard at all."
        ),
        kind=DERIVED,
        resolve=_day20_clean,
        lead=True,
    ),
)


def leads() -> tuple[Claim, ...]:
    return tuple(c for c in CLAIMS if c.lead)


def by_key(key: str) -> Claim:
    for claim in CLAIMS:
        if claim.key == key:
            return claim
    raise KeyError(key)


# ------------------------------------------------------- what used to be shown


@dataclass(frozen=True)
class WasPublished:
    """A figure the landing pages carried before day 21, kept verbatim.

    Frozen on purpose. This is evidence about what the repository claimed, not
    a figure to keep current — and the whole finding is that nothing was
    keeping it current.
    """

    key: str
    where: tuple[str, ...]
    figure: str
    note: str = ""


WAS_PUBLISHED: tuple[WasPublished, ...] = (
    WasPublished(
        "day02-max-confidence",
        ("hub", "site", "readme"),
        "0.092",
        "The lead card on both pages and the third finding in the README. No "
        "committed artifact contains it.",
    ),
    WasPublished(
        "day02-committed-run",
        ("hub", "site", "readme"),
        "0.092 / 74%",
        "All three described the discarded run as though it were the one in the repository.",
    ),
    WasPublished(
        "day03-spread",
        ("hub", "readme"),
        "97% to 0%",
        "The hub's lead card and the README's first finding, both reading "
        "96.7%. Correct when written; day 3's artifact was rebuilt afterwards "
        "and nothing rechecked it.",
    ),
    WasPublished(
        "day03-average",
        ("hub", "readme"),
        "52%",
        "Quoted beside it to argue that the average is meaningless, which it "
        "is, at a different value.",
    ),
    WasPublished("day05-false-confirmations", ("hub", "site", "readme"), "10 to 0"),
    WasPublished(
        "day05-pass-rates",
        ("site", "readme"),
        "8% to 8%",
        "Both asserted that the score never moved without showing it. It did not move.",
    ),
    WasPublished("day12-token-to-sentence", ("site",), "688 ms to 2205 ms"),
    WasPublished(
        "artifacts-total",
        ("hub", "readme"),
        "9.0 MB",
        "The middle box of the architecture diagram, in the hub's overview and "
        "in the README's mermaid. True when days 2 and 3 were the only days "
        "with an artifact.",
    ),
    WasPublished(
        "image-size",
        ("hub", "site", "readme"),
        "588 MB",
        "Quoted in six places. Nothing in the repository records the size of "
        "the image, so no rebuild could ever have corrected it.",
    ),
)
"""Only the figures the three front pages published — the hub's overview,
the site's index and the README's head. Every per-day page has always computed
its own, and the audit found nothing wrong with any of them."""


@dataclass(frozen=True)
class Checked:
    """One previously published figure, next to what the artifact says now."""

    key: str
    where: tuple[str, ...]
    published: str
    derived: str | None
    kind: str
    note: str

    @property
    def verdict(self) -> str:
        """Four outcomes, and three of them are not failures.

        `stale` is the only one that says a page was wrong. `no artifact` and
        `not stored` say a figure cannot be checked at all, which is a
        different and in one case more interesting complaint.
        """
        if self.derived is None:
            if self.kind == TESTIMONY:
                return "no artifact"
            if self.kind == UNSTORED:
                return "not stored"
            return "unavailable"
        return "holds" if self.derived == self.published else "stale"

    @property
    def holds(self) -> bool:
        return self.verdict == "holds"


def audit() -> tuple[Checked, ...]:
    """Re-derive every figure the landing pages published, and compare.

    A `TESTIMONY` claim has no resolver, so it comes back as `no artifact`
    rather than as a mismatch: the number is not wrong, it is unfalsifiable,
    and those are different complaints.
    """
    rows = []
    for was in WAS_PUBLISHED:
        claim = by_key(was.key)
        rows.append(
            Checked(
                key=was.key,
                where=was.where,
                published=was.figure,
                derived=claim.derived(),
                kind=claim.kind,
                note=was.note,
            )
        )
    return tuple(rows)


VERDICTS = ("holds", "stale", "no artifact", "not stored", "unavailable")


def tally(rows: tuple[Checked, ...] | None = None) -> dict[str, int]:
    rows = audit() if rows is None else rows
    counts = dict.fromkeys(VERDICTS, 0)
    for row in rows:
        counts[row.verdict] += 1
    return counts
