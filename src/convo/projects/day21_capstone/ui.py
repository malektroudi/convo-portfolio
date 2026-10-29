"""Day 21's tab: the close-out, and an audit of what the front pages claimed.

The brief asked for one narrative, one architecture diagram and one cost table
reading $0. All three are here. What was not in the brief is the thing that
turned out to be worth building: a check that the numbers on the front page
are the numbers in the artifacts.

They were not. Three of the seven figures the two landing pages carried had
gone stale, and the single most prominent number in the portfolio has no
artifact behind it at all. Every per-day page was correct, because every
per-day page computes. The front pages were typed.

The tab reads day 21's own artifact and imports no other day, so it draws
without loading fourteen result files.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from convo.projects.day21_capstone.results import Results, load

GAP = "\n\n"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
WARN = "#c98a16"

MISSING = (
    "No day 21 artifact. Run "
    "`python -m convo.projects.day21_capstone.build` — it needs no key and no "
    "quota, and takes about a second."
)

TONE = {"holds": GOOD, "stale": CRITICAL, "no artifact": WARN, "unavailable": WARN}

KIND_NOTE = {
    "derived": "Recomputed from a committed artifact every time this page is built.",
    "pinned": "Committed, but regenerating it costs model quota and a rerun could land elsewhere.",
    "testimony": "True, written down, and nothing in the repository can check it.",
    "unstored": "A build step measures it and nothing commits it.",
}


def _claims_table(results: Results) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "day": c["day"],
                "figure": c["figure"] if c["figure"] is not None else "no artifact",
                "finding": c["headline"],
                "kind": c["kind"],
                "front page": "lead" if c["lead"] else "",
            }
            for c in results.claims
        ]
    )


def _audit_table(results: Results) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "figure": row.key,
                "shown on": ", ".join(row.where),
                "as published": row.published,
                "re-derived now": row.derived if row.derived is not None else "nothing derives it",
                "verdict": row.verdict,
            }
            for row in results.audit
        ]
    )


def _inventory_table(results: Results) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "day": row["day"],
                "title": row["title"],
                "artifact": (
                    f"{row['files']} file{'s' if row['files'] != 1 else ''}"
                    if row["files"]
                    else "none"
                ),
                "KB": round(row["bytes"] / 1024) if row["bytes"] else "",
                "rebuilt in CI": row["rebuilt_in_ci"] or "",
                "shipped": "yes" if row["shipped"] else "NO",
            }
            for row in results.inventory
        ]
    )


def _costs_table(results: Results) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "what the plan assumed": row["wanted"],
                "which needs": row["needs"],
                "used instead": row["instead"],
                "cost": row["dollars"],
                "what that cost instead": row["lost"],
            }
            for row in results.costs
        ]
    )


def _cards(results: Results) -> str:
    cells = []
    for claim in results.leads:
        figure = claim["figure"] if claim["figure"] is not None else "no artifact"
        colour = WARN if claim["kind"] == "testimony" else "#2a78d6"
        cells.append(
            f'<div style="flex:1 1 220px;min-width:0;padding:14px 16px;'
            f'border:1px solid rgba(128,128,128,0.25);border-radius:6px">'
            f'<div style="font-size:0.68rem;letter-spacing:0.05em;'
            f'text-transform:uppercase;opacity:0.55">Day {claim["day"]:02d} '
            f"&middot; {claim['kind']}</div>"
            f'<div style="font-size:1.5rem;font-weight:600;line-height:1.15;'
            f'color:{colour};margin:2px 0 4px">{figure}</div>'
            f'<div style="font-size:0.85rem;font-weight:600;margin-bottom:6px">'
            f"{claim['headline']}</div>"
            f'<div style="font-size:0.8rem;opacity:0.72;line-height:1.5">'
            f"{claim['body']}</div></div>"
        )
    return f'<div style="display:flex;gap:12px;flex-wrap:wrap">{"".join(cells)}</div>'


def _pipeline(results: Results) -> str:
    lines = []
    for stage in results.pipeline:
        mark = "measured here" if stage["measured"] else "asserted"
        lines.append(f"- **{stage['name']}** — {stage['size']} _({mark})_  \n  {stage['note']}")
    return "\n".join(lines)


def build() -> None:
    gr.Markdown("## Close it as one product")
    results = load()
    if results is None:
        gr.Markdown(MISSING)
        return

    tally = results.tally
    gr.Markdown(
        f"**{results.shipped} of {results.of_days} days shipped**, "
        f"{results.days_with_artifacts} of them leaving a committed artifact, "
        f"{results.megabytes} MB in total. The brief for today asked for one "
        "narrative, one architecture diagram and one cost table reading $0. "
        "All three are below."
        + GAP
        + "The useful part was not in the brief. This repository's rule is that "
        "every published number must be re-derivable from a committed artifact, "
        "and a test enforces it on the renderer. Three surfaces were exempt and "
        "nobody noticed: the hub's overview, the static site's index and the "
        "README's head, whose findings were typed in by hand."
        + GAP
        + f"<p style='color:{CRITICAL}'><b>Of the {sum(tally.values())} figures "
        f"those three carried, {tally['holds']} still hold, "
        f"{tally['stale']} had gone stale, {tally['no artifact']} has no "
        f"artifact behind it at all and {tally['not stored']} was never "
        "recorded anywhere.</b></p>"
        + "Every per-day page was correct, because every per-day page computes "
        "its figures. The front pages were the only ones that did not, and they "
        "were the only ones that were wrong."
    )

    gr.Markdown("### The audit")
    gr.Dataframe(_audit_table(results), interactive=False, wrap=True)
    gr.Markdown(
        "The two pages also disagreed with each other. The hub led on days 3, 5 "
        "and 2; the site led on days 2, 5 and 12. Nothing was reconciling them, "
        "because each had its own hand-written list — so the portfolio had two "
        "different answers to the question of what it had found."
        + GAP
        + "Both now render the same claims from the same module, resolved at "
        "build time. The `lead` flag lives with the claim rather than with "
        "either front end, which is what makes a disagreement impossible rather "
        "than unlikely."
    )

    gr.Markdown("### Four kinds of number, and why the distinction is the day")
    kinds = results.kinds()
    gr.Markdown(
        "\n".join(
            f"- **{kind}** ({kinds.get(kind, 0)} of {len(results.claims)} claims) — {note}"
            for kind, note in KIND_NOTE.items()
            if kinds.get(kind)
        )
        + GAP
        + "The uncomfortable one is `testimony`. Day 2's maximum softmax of "
        "**0.092** is the best finding in this portfolio — a model at 74% "
        "accuracy that no rejection threshold could ever use — and the run that "
        "fixed it overwrote the artifact that proved it. It is true, it is "
        "written down in three places, and nothing here can check it."
        + GAP
        + "It stays on the front page, labelled. Deleting it would hide the most "
        "useful thing week 1 produced; presenting it as a measurement alongside "
        "figures that recompute would be the quiet dishonesty this repository "
        "was built to avoid. Naming the difference is the only honest third "
        "option, and the lesson is cheap to state: **a finding is only as "
        "durable as the artifact you keep from the run that produced it.**"
    )

    gr.Markdown("### The findings worth leading with")
    gr.HTML(_cards(results))

    gr.Markdown("### Every claim this portfolio makes")
    gr.Dataframe(_claims_table(results), interactive=False, wrap=True)

    gr.Markdown("### One architecture, with the middle number measured")
    gr.Markdown(
        "```\n"
        "laptop (torch)  ---->  committed artifacts  ---->  serving layer (no torch)\n"
        "```\n" + _pipeline(results) + GAP + "The middle figure is the bargain. It was written "
        "down as 9 MB, which was right when two days had artifacts and has been "
        f"wrong for six of them; it is {results.megabytes} MB and is now "
        "computed. The serving image's size is the last number here that "
        "nothing commits, so day 21 publishes a ceiling instead — "
        f"**under {results.image_ceiling_mb} MB** — which CI checks on every "
        "push. A weaker claim that is checked beats a precise one that is not."
        + GAP
        + f"CI also refuses to find {', '.join(f'`{m}`' for m in results.blocked_in_image)} "
        "inside the built image. That list is the architecture stated as a "
        "prohibition, and it is the reason the serving layer can be free."
    )

    gr.Markdown("### One cost table")
    gr.Dataframe(_costs_table(results), interactive=False, wrap=True)
    gr.Markdown(
        "Every row reads $0, which was the brief. The last column is the one "
        "worth reading: $0 bought a portfolio that cannot measure a cold start, "
        "has never compared itself against the managed product the role is "
        "about, and ran its best week-3 day on three quarters of its own test "
        "suite." + GAP + f"Of {results.days_with_artifacts} days with an artifact, "
        f"**{results.days_rebuilt_in_ci}** have one that CI regenerates from "
        "scratch. The rest are trusted. That ratio is the honest version of the "
        "word *reproducible*, and it is low because a model call is not free."
    )

    gr.Markdown("### What is missing, written down on purpose")
    gr.Markdown(
        "\n".join(f"- **{gap['what']}** — {gap['why']}" for gap in results.gaps)
        + GAP
        + "The registry has always marked day 17 unshipped, so the hub and the "
        "site have been showing the gap this whole time. Nothing explained it "
        "until now, which is the difference between a page that is not lying "
        "and a page that is telling you something."
    )

    gr.Markdown("### Where the days ended up")
    gr.Dataframe(_inventory_table(results), interactive=False, wrap=True)
