"""What the twenty-one days cost, and what the repository can still prove.

Three things live here, and they are the three the day's brief asked for.

`inventory()` walks the registry and the artifact directories: which days
shipped, which committed a file, how big it is, and whether a CI step
regenerates it from scratch rather than trusting what was committed. That last
column is the one that matters, and only three days can claim it.

`pipeline()` is the architecture diagram as numbers instead of a picture. The
whole repository is one bargain — heavy work on a laptop, small artifacts in
git, a serving layer with no ML framework in it — and the bargain is only
worth anything if the middle number stays small. It is measured here rather
than written down, because the version written down said 9 MB and had been
wrong for six days.

`COSTS` is the cost table. Every row reads $0, which is the brief, and the
column next to it is the one worth reading: what each free substitute cannot
do. A portfolio that shows the first column and hides the second is an
advertisement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from convo.core.registry import projects

PROJECTS_DIR = Path(__file__).resolve().parent.parent

# Days whose artifact a CI step rebuilds from scratch on every push, rather
# than reading the committed copy. Everything else is trusted, and the
# distinction is the honest version of "reproducible".
REBUILT_IN_CI = {19: "topology re-derived from the real agent tree", 20: "the whole day, in ~1s"}

# The modules CI refuses to find inside the serving image. Kept here so the
# page quotes the gate rather than a memory of it.
BLOCKED_IN_IMAGE = ("torch", "transformers", "onnxruntime", "piper", "duckdb", "google.adk")

# The serving image is the one headline figure in this repository that nothing
# commits, so day 21 publishes a ceiling instead of a size: a number that is in
# a file, that CI checks on every push, and that fails loudly when the image
# grows into it. A weaker claim that is checked beats a precise one that is
# not.
IMAGE_CEILING_MB = 700


# Artifacts git stores as text. `.gitattributes` sets `* text=auto`, so these
# are LF in the index and CRLF in a Windows working tree.
TEXT_SUFFIXES = (".json", ".ndjson", ".csv", ".txt", ".md", ".sql")


@dataclass(frozen=True)
class DayRow:
    day: int
    title: str
    shipped: bool
    files: int
    bytes_: int
    rebuilt_in_ci: str = ""

    @property
    def has_artifact(self) -> bool:
        return self.files > 0


def _artifact_dir(day: int) -> Path | None:
    found = sorted(PROJECTS_DIR.glob(f"day{day:02d}_*/artifacts"))
    return found[0] if found else None


def size_of(path: Path) -> int:
    """Bytes as git stores them, not as this checkout happens to hold them.

    `st_size` is the wrong measurement for a figure that gets committed. This
    repository is written on Windows and built on Linux, `.gitattributes` sets
    `* text=auto`, and so every JSON artifact is one byte per line longer here
    than in the index. Day 20's file is 11,779 bytes on my laptop and 11,271
    in CI.

    That is enough to make the total differ between the machine that writes
    the artifact and the machine that checks it, which would have failed
    `--check` on every push with a message about drift and nothing drifting.
    Collapsing CRLF in the files git treats as text makes the figure a
    property of the repository rather than of the checkout. Binary artifacts
    are measured as they are, because a `.npz` may contain those two bytes
    for reasons of its own.
    """
    if path.suffix.lower() in TEXT_SUFFIXES:
        return len(path.read_bytes().replace(b"\r\n", b"\n"))
    return path.stat().st_size


SELF = 21
"""Day 21 is left out of its own inventory, and not for tidiness.

The artifact this module feeds records the total size of every committed
artifact. Counting its own file in that total makes the figure a fixpoint:
writing it changes the number it reports, so the next rebuild writes a
different file, and a CI step that rebuilds and diffs can never agree with
what is committed. A summary that counts itself as evidence does not settle.
"""


def inventory() -> tuple[DayRow, ...]:
    """Every day in the plan, with whatever evidence it left behind.

    Day 21 is absent; `SELF` says why.
    """
    rows = []
    for project in projects():
        if project.day == SELF:
            continue
        directory = _artifact_dir(project.day)
        files = sorted(p for p in directory.rglob("*") if p.is_file()) if directory else []
        rows.append(
            DayRow(
                day=project.day,
                title=project.title,
                shipped=project.status == "shipped",
                files=len(files),
                bytes_=sum(size_of(p) for p in files),
                rebuilt_in_ci=REBUILT_IN_CI.get(project.day, ""),
            )
        )
    return tuple(rows)


def committed_bytes() -> int:
    """Every committed artifact except this day's own. See `SELF`."""
    return sum(row.bytes_ for row in inventory())


def progress() -> tuple[int, int]:
    """Days shipped, days planned. Day 21 counts here — it is a day, not
    evidence, and the fixpoint problem is only about bytes."""
    all_days = projects()
    return sum(p.status == "shipped" for p in all_days), len(all_days)


@dataclass(frozen=True)
class Stage:
    name: str
    size: str
    note: str
    measured: bool
    """False when the figure is asserted rather than computed here. The two
    unmeasured stages are the ones outside the repository, which is not a
    coincidence: what git holds is what can be checked."""


def pipeline() -> tuple[Stage, ...]:
    """The repository's one architectural idea, with the middle number real."""
    total = committed_bytes()
    rows = [row for row in inventory() if row.has_artifact]
    biggest = max(rows, key=lambda r: r.bytes_) if rows else None
    return (
        Stage(
            "laptop",
            "~90 MB of weights",
            "A fine-tuned MiniLM, a reranker, whisper-tiny.en and a piper "
            "voice, none of them in git. Asserted, not measured: this is the "
            "one stage the repository cannot see, because `models/` is "
            "ignored.",
            measured=False,
        ),
        Stage(
            "committed artifacts",
            f"{total / 1024 / 1024:.1f} MB across {len(rows)} days",
            (
                f"Logits, score matrices, timings, traces and run outcomes. The "
                f"largest single day is day {biggest.day} at "
                f"{biggest.bytes_ / 1024 / 1024:.1f} MB."
                if biggest
                else "No artifacts are committed."
            ),
            measured=True,
        ),
        Stage(
            "serving image",
            f"under {IMAGE_CEILING_MB} MB",
            "A ceiling rather than a size. CI builds the image, asserts it is "
            "below this, and separately refuses to find "
            f"{', '.join(BLOCKED_IN_IMAGE)} inside it.",
            measured=False,
        ),
    )


@dataclass(frozen=True)
class Cost:
    """One thing the plan assumed, and what happened instead."""

    wanted: str
    needs: str
    instead: str
    dollars: str
    lost: str
    """What the substitute cannot do. The column that makes the table worth
    printing."""


COSTS: tuple[Cost, ...] = (
    Cost(
        "Cloud Run for the hub",
        "a billing account, even inside the free tier",
        "a static site on GitHub Pages, plus a container CI builds and boots",
        "$0",
        "No request-time Python, so the site is a reading rather than a demo, "
        "and nothing here measures cold starts or behaviour under load.",
    ),
    Cost(
        "Vertex AI for the agent",
        "a billing account",
        "an AI Studio key against the Gemini free tier",
        "$0",
        "500 requests per model per day, and one pass of day 6's suite costs "
        "162. Day 19 ran 24 of 32 scenarios.",
    ),
    Cost(
        "Dialogflow CX",
        "a billing account",
        "a hand-written state machine on day 4, then ADK on days 16 and 19",
        "$0",
        "No comparison against the managed product the role is actually "
        "about. The framework comparison is ADK against my own code.",
    ),
    Cost(
        "BigQuery for analytics",
        "a billing account",
        "DuckDB running the same SQL unchanged",
        "$0",
        "The SQL is portable and was never run on BigQuery, so nothing here "
        "proves it scales or costs what it would cost.",
    ),
    Cost(
        "Managed speech-to-text and text-to-speech",
        "per-minute billing",
        "whisper-tiny.en and a piper voice, both on CPU",
        "$0",
        "The input audio is synthetic, which flatters the recogniser: a TTS "
        "voice has no accent, no background and no hesitation.",
    ),
    Cost(
        "A hosted model for the tool-calling agent",
        "per-token billing",
        "Qwen 0.5B on a laptop CPU in week 1",
        "$0",
        "A 0.5B model cannot settle an architectural question. The shape of "
        "day 5's finding generalises; its pass rates do not.",
    ),
    Cost(
        "A GPU for fine-tuning",
        "a paid instance or a Colab subscription",
        "a free Kaggle session and a laptop",
        "$0",
        "One seed, one configuration, no hyperparameter search, so none of "
        "the training results carry an error bar.",
    ),
)


@dataclass(frozen=True)
class Gap:
    """Something the portfolio does not have, written down on purpose."""

    what: str
    why: str


GAPS: tuple[Gap, ...] = (
    Gap(
        "Day 17 was never built",
        "ADK sessions and memory against day 11's rewriter and buffer. It "
        "needed no billing account and no quota — it was skipped for time, "
        "and until today nothing in the repository said so. The registry "
        "showed it unshipped, which is why this is a gap rather than a lie.",
    ),
    Gap(
        "Day 19 covers 24 of 32 scenarios",
        "The eight that did not run are the digression-heavy ones, where a "
        "router is most likely to fail. Its pass rate is an indication, not a "
        "verified result, and the page says so.",
    ),
    Gap(
        "The cross-model comparison is unanswerable here",
        "The free tier allows 500 requests a day on flash-lite and 20 on "
        "flash. A controlled comparison needs 162 on each, so one of the two "
        "arms cannot be run at all.",
    ),
    Gap(
        "Model call counts exclude retries",
        "The `model_calls` column on five days counts scored calls, not "
        "spend. Day 19's run cost roughly 190 calls against a reported 162. "
        "Fixing it would redefine a published column on five artifacts, so it "
        "is recorded instead.",
    ),
    Gap(
        "Nothing here has met a real user",
        "Every conversation in every suite was written by me, which means "
        "every failure mode measured is one I thought of. That is the "
        "limitation a portfolio cannot engineer around.",
    ),
)
