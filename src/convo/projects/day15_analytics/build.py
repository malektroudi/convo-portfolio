"""Project the measured runs into events, then query them.

    uv run python -m convo.projects.day15_analytics.build

By default this queries the events locally with DuckDB, because the committed
numbers have to come from something this repository can actually run. The same
SQL runs on BigQuery with:

    uv run python -m convo.projects.day15_analytics.build --bigquery convo_analytics

which needs a Google sign-in and a project — the sandbox needs no billing
account, but it is still an account, so that path is a documented command
rather than something this build performs on its own. When it is available the
build runs *both* engines and records whether they agreed, which is the only
way the portability claim means anything.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from convo.core import get_logger, setup
from convo.projects.day15_analytics import warehouse as wh
from convo.projects.day15_analytics.emit import collect
from convo.projects.day15_analytics.events import SCHEMA_VERSION, write_ndjson

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
EVENTS_FILE = ARTIFACT_DIR / "turns.ndjson"

log = get_logger("day15.build")


def _disagreements(local: list[wh.QueryResult], remote: list[wh.QueryResult]) -> list[dict]:
    """Where the two engines returned different answers to the same SQL.

    Compared cell by cell with a tolerance on floats, because the two engines
    compute `APPROX_QUANTILE` with different sketches and are not obliged to
    agree to the last digit on an approximate function.
    """
    problems = []
    for left, right in zip(local, remote, strict=True):
        if len(left.rows) != len(right.rows):
            problems.append({"query": left.name, "why": "different row counts"})
            continue
        for index, (a, b) in enumerate(zip(left.rows, right.rows, strict=True)):
            for key in a:
                first, second = a.get(key), b.get(key)
                if isinstance(first, (int, float)) and isinstance(second, (int, float)):
                    if abs(float(first) - float(second)) > 0.01:
                        problems.append(
                            {
                                "query": left.name,
                                "row": index,
                                "column": key,
                                "duckdb": first,
                                "bigquery": second,
                            }
                        )
                elif str(first) != str(second):
                    problems.append(
                        {
                            "query": left.name,
                            "row": index,
                            "column": key,
                            "duckdb": first,
                            "bigquery": second,
                        }
                    )
    return problems


def main(argv: list[str] | None = None) -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bigquery",
        metavar="DATASET",
        help="also load and query a BigQuery dataset, and compare the answers",
    )
    args = parser.parse_args(argv)

    events = collect()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_ndjson(events, EVENTS_FILE)
    log.info(
        "events emitted",
        extra={
            "events": len(events),
            "conversations": len({e.conversation_id for e in events}),
            "kilobytes": EVENTS_FILE.stat().st_size // 1024,
        },
    )

    local = wh.run_duckdb(EVENTS_FILE)
    log.info("queried", extra={"engine": "duckdb", "queries": len(local)})

    remote: list[wh.QueryResult] = []
    disagreements: list[dict] = []
    if args.bigquery:
        if not wh.bq_available():
            raise SystemExit(
                "bq has no active account or project. Run `gcloud auth login` and "
                "`gcloud config set project <your-sandbox-project>` first — the "
                "BigQuery sandbox needs no billing account, but it does need a "
                "project."
            )
        for command in wh.load_commands(args.bigquery, EVENTS_FILE):
            log.info("bq", extra={"command": " ".join(command)})
            subprocess.run(command, check=False, timeout=600)
        remote = wh.run_bigquery(args.bigquery)
        disagreements = _disagreements(local, remote)
        log.info(
            "queried",
            extra={
                "engine": "bigquery",
                "queries": len(remote),
                "disagreements": len(disagreements),
            },
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "events": len(events),
        "conversations": len({e.conversation_id for e in events}),
        "sources": sorted({e.source_day for e in events}),
        "engine": "bigquery+duckdb" if remote else "duckdb",
        "bigquery_available": wh.bq_available(),
        "descriptions": wh.describe_all(),
        "results": [r.as_dict() for r in local],
        "bigquery_results": [r.as_dict() for r in remote],
        "disagreements": disagreements,
        "load_commands": [" ".join(c) for c in wh.load_commands("convo_analytics", EVENTS_FILE)],
    }

    out = ARTIFACT_DIR / "analytics.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log.info(
        "artifact written",
        extra={"path": str(out), "kilobytes": max(out.stat().st_size // 1024, 1)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
