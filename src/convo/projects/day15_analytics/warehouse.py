"""Run the same SQL text on DuckDB locally or on BigQuery, unchanged.

The queries live in `sql/` as files rather than as strings in Python, for one
reason: the thing shipped to BigQuery has to be the thing that was tested, and
a file can be read, diffed and pasted into a console. A query assembled from
f-strings cannot.

Portability is not a claim here, it is a constraint the queries were written
under, and its total cost is **one macro**. BigQuery has `SAFE_DIVIDE`;
DuckDB does not, so DuckDB gets a two-line definition of it and every query
runs on both engines as written. Everything else the analytics needs —
`QUALIFY`, `COUNTIF`, `APPROX_QUANTILE`, `UNNEST` over a repeated field —
already means the same thing in both.

BigQuery is reached through the `bq` CLI rather than a client library, so the
project takes no new runtime dependency to support a path most people running
this will never use. The sandbox needs no billing account, but it does need a
Google sign-in, which is why the committed numbers come from DuckDB and the
BigQuery path is a documented command rather than something this build can
run on its own.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from convo.core import get_logger
from convo.projects.day15_analytics.events import BIGQUERY_SCHEMA

SQL_DIR = Path(__file__).parent / "sql"
TABLE_PLACEHOLDER = "{table}"

# DuckDB's whole compatibility gap with these queries, in two lines.
SAFE_DIVIDE = """
CREATE OR REPLACE MACRO SAFE_DIVIDE(a, b) AS
  CASE WHEN b IS NULL OR b = 0 THEN NULL ELSE a / b END;
"""

log = get_logger("day15.warehouse")


@dataclass(frozen=True)
class QueryResult:
    name: str
    sql: str
    rows: list[dict]
    engine: str

    @property
    def one(self) -> dict:
        return self.rows[0] if self.rows else {}

    def as_dict(self) -> dict:
        return {"name": self.name, "engine": self.engine, "rows": self.rows}


def queries() -> list[tuple[str, str]]:
    """Every query file, in filename order. The numbers are the order."""
    return [(p.stem, p.read_text(encoding="utf-8")) for p in sorted(SQL_DIR.glob("*.sql"))]


def _describe(sql: str) -> str:
    """The leading comment block, which is where each query argues for itself."""
    lines = []
    for line in sql.splitlines():
        if not line.startswith("--"):
            break
        lines.append(line.lstrip("- ").rstrip())
    return " ".join(x for x in lines if x)


def run_duckdb(ndjson: Path) -> list[QueryResult]:
    """Query the event file in place. No server, no load step, no state."""
    import duckdb

    connection = duckdb.connect()
    connection.execute(SAFE_DIVIDE)
    # read_json_auto reads the same newline-delimited file `bq load` takes, so
    # both engines see byte-identical input and any disagreement downstream is
    # about the query rather than about how the rows got in.
    # DuckDB will not take a bound parameter inside CREATE VIEW, so the path
    # is inlined. It is a build-controlled temporary file rather than
    # anything a user supplies, and the quote doubling is belt and braces.
    location = str(ndjson).replace("'", "''")
    connection.execute(
        "CREATE VIEW turns AS SELECT * FROM "
        f"read_json_auto('{location}', format='newline_delimited')"
    )

    results = []
    for name, sql in queries():
        statement = sql.replace(TABLE_PLACEHOLDER, "turns")
        cursor = connection.execute(statement)
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, record, strict=True)) for record in cursor.fetchall()]
        results.append(QueryResult(name, sql, _jsonable(rows), "duckdb"))
    connection.close()
    return results


def _jsonable(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        clean = {}
        for key, value in row.items():
            if isinstance(value, float):
                clean[key] = round(value, 6)
            elif value is None or isinstance(value, bool | int | str):
                clean[key] = value
            else:
                clean[key] = str(value)
        out.append(clean)
    return out


# ---------------------------------------------------------------- BigQuery


def bq_available() -> bool:
    """Whether `bq` is installed *and* has an account and a project.

    All three are needed, and the usual failure is the third: the SDK
    installs, authenticates, and then every query fails because no project is
    configured. Checked separately so the build can say which part is missing.
    """
    try:
        account = subprocess.run(
            ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        project = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(account) and bool(project) and project != "(unset)"


def bq_schema_arg() -> str:
    """The schema in `bq load` syntax, generated from the dataclass."""
    return ",".join(f"{name}:{kind}" for name, kind, _ in BIGQUERY_SCHEMA)


def load_commands(dataset: str, ndjson: Path) -> list[list[str]]:
    """The exact commands that put these events in a sandbox dataset.

    Returned rather than run, so the README and the tab can show the real
    thing instead of an approximation of it, and so a reader can see there is
    no magic in it: make a dataset, load a file, query it.
    """
    return [
        ["bq", "--location=US", "mk", "--dataset", dataset],
        [
            "bq",
            "load",
            "--source_format=NEWLINE_DELIMITED_JSON",
            "--replace",
            f"{dataset}.turns",
            str(ndjson),
            bq_schema_arg(),
        ],
    ]


def run_bigquery(dataset: str) -> list[QueryResult]:
    """Run every query against a loaded BigQuery dataset."""
    results = []
    for name, sql in queries():
        statement = sql.replace(TABLE_PLACEHOLDER, f"`{dataset}.turns`")
        completed = subprocess.run(
            ["bq", "query", "--use_legacy_sql=false", "--format=json", statement],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{name}: {completed.stderr.strip()[:400]}")
        results.append(QueryResult(name, sql, json.loads(completed.stdout or "[]"), "bigquery"))
    return results


def describe_all() -> dict[str, str]:
    return {name: _describe(sql) for name, sql in queries()}
