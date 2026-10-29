"""Day 21's artifact: the close-out, and a tripwire under the front page.

    uv run python -m convo.projects.day21_capstone.build
    uv run python -m convo.projects.day21_capstone.build --check

**Costs nothing and calls no model.** Everything here is read out of artifacts
that already exist, so CI rebuilds it on every push rather than trusting the
committed copy — the same arrangement day 20 earned, for the same reason.

The two front pages do **not** read this file. They call `claims.leads()` and
resolve each figure live, which is the whole point of the day: a front page
that recomputes cannot go stale, and the one that did not went stale in three
places. So what is this artifact for?

It is a tripwire for the half a resolver cannot check. Every claim carries a
sentence of prose around its number — "a second and a half apart", "ten
well-formed, entirely fictional reservations" — and prose does not recompute.
`--check` rebuilds the payload and diffs it against the committed one, so a
rebuilt artifact that moves a figure fails CI and the sentence beside it gets
read again by a person. That is the only mechanism available: the alternative
is a number that updates itself while the paragraph explaining it quietly
becomes false.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from convo.core import get_logger, setup
from convo.projects.day21_capstone import claims, ledger

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_FILE = ARTIFACT_DIR / "capstone.json"

log = get_logger("day21.build")


def payload() -> dict:
    """Everything day 21 publishes, as plain data.

    Importing this module pulls in every day's `results`, because a claim
    resolves against the artifact that produced it. That is fine here and
    deliberately not fine in `results.py`.
    """
    import convo.projects  # noqa: F401  (registers every shipped day)

    shipped, of_days = ledger.progress()
    audited = claims.audit()
    return {
        "shipped": shipped,
        "of_days": of_days,
        "committed_bytes": ledger.committed_bytes(),
        "image_ceiling_mb": ledger.IMAGE_CEILING_MB,
        "blocked_in_image": list(ledger.BLOCKED_IN_IMAGE),
        "claims": [
            {
                "key": c.key,
                "day": c.day,
                "headline": c.headline,
                "body": c.body,
                "kind": c.kind,
                "figure": c.value(),
                "lead": c.lead,
            }
            for c in claims.CLAIMS
        ],
        "audit": [
            {
                "key": r.key,
                "where": list(r.where),
                "published": r.published,
                "derived": r.derived,
                "kind": r.kind,
                "note": r.note,
                "verdict": r.verdict,
            }
            for r in audited
        ],
        "tally": claims.tally(audited),
        "inventory": [
            {
                "day": r.day,
                "title": r.title,
                "shipped": r.shipped,
                "files": r.files,
                "bytes": r.bytes_,
                "rebuilt_in_ci": r.rebuilt_in_ci,
            }
            for r in ledger.inventory()
        ],
        "pipeline": [asdict(s) for s in ledger.pipeline()],
        "costs": [asdict(c) for c in ledger.COSTS],
        "gaps": [asdict(g) for g in ledger.GAPS],
    }


def _serialise(data: dict) -> str:
    return json.dumps(data, indent=1)


def differences(stored: dict, fresh: dict) -> list[str]:
    """Where the committed artifact and a fresh rebuild disagree.

    Reported key by key rather than as one failed comparison, because the
    useful output of a drift check is which figure moved.
    """
    complaints = []
    for key in sorted(set(stored) | set(fresh)):
        if key not in stored:
            complaints.append(f"{key}: missing from the committed artifact")
        elif key not in fresh:
            complaints.append(f"{key}: no longer produced")
        elif stored[key] != fresh[key]:
            if key == "claims":
                for was, now in zip(stored[key], fresh[key], strict=False):
                    if was.get("figure") != now.get("figure"):
                        complaints.append(
                            f"{now['key']}: {was.get('figure')!r} -> {now.get('figure')!r}"
                        )
                if len(stored[key]) != len(fresh[key]):
                    complaints.append(f"claims: {len(stored[key])} stored, {len(fresh[key])} now")
            else:
                complaints.append(f"{key}: changed")
    return complaints


def main(argv: list[str] | None = None) -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and fail if the committed artifact disagrees",
    )
    args = parser.parse_args(argv)

    fresh = payload()

    if args.check:
        if not ARTIFACT_FILE.exists():
            log.error("no committed artifact to check", extra={"path": str(ARTIFACT_FILE)})
            return 1
        stored = json.loads(ARTIFACT_FILE.read_text(encoding="utf-8"))
        complaints = differences(stored, fresh)
        if complaints:
            for complaint in complaints:
                log.error("drift", extra={"what": complaint})
            return 1
        log.info(
            "the front page still matches the artifacts",
            extra={"claims": len(fresh["claims"]), **fresh["tally"]},
        )
        return 0

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_FILE.write_text(_serialise(fresh), encoding="utf-8")
    log.info(
        "artifact written",
        extra={
            "shipped": f"{fresh['shipped']}/{fresh['of_days']}",
            "claims": len(fresh["claims"]),
            "megabytes": round(fresh["committed_bytes"] / 1024 / 1024, 1),
            **fresh["tally"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
