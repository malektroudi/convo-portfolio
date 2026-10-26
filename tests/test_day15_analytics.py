"""Day 15: the event schema, the SQL, and the two engines agreeing.

The queries themselves are tested by running them, on a tiny hand-built event
set where the right answer can be worked out on paper. That matters more than
it sounds: a containment query that silently counts conversations twice will
produce a plausible number on 155 real rows and nobody will ever notice.

The schema gets the strictest tests, because it is the part that cannot be
fixed retrospectively — rows written without a column stay written without it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from convo.projects.day15_analytics import warehouse as wh
from convo.projects.day15_analytics.emit import collect
from convo.projects.day15_analytics.events import (
    BIGQUERY_SCHEMA,
    CHAT,
    FIELD_NAMES,
    REFUSAL_REASONS,
    VOICE,
    TurnEvent,
    write_ndjson,
)

duckdb = pytest.importorskip("duckdb", reason="day 15's analytics extra not installed")


def event(cid, index=0, answered=True, escalated=False, **kwargs) -> TurnEvent:
    base = dict(
        conversation_id=cid,
        turn_index=index,
        surface=CHAT,
        source_day=13,
        question_chars=20,
        is_follow_up=index > 0,
        answered=answered,
        escalated=escalated,
    )
    base.update(kwargs)
    return TurnEvent(**base)


@pytest.fixture
def run(tmp_path):
    """Query a hand-built event set and return results by query name."""

    def _run(events):
        path = tmp_path / "turns.ndjson"
        write_ndjson(events, path)
        return {r.name: r.rows for r in wh.run_duckdb(path)}

    return _run


class TestTheSchema:
    def test_the_two_schemas_describe_the_same_fields(self):
        """The dataclass and the BigQuery types must not drift apart.

        They are written in two places because `bq load` needs types the
        dataclass does not carry; a test is what keeps them one schema.
        """
        assert {name for name, _, _ in BIGQUERY_SCHEMA} == set(FIELD_NAMES)

    def test_it_carries_no_user_text(self):
        """Day 10 redacts PII; a warehouse is the last place to undo that."""
        for name in FIELD_NAMES:
            assert "text" not in name, name
            assert name not in {"question", "reply", "utterance"}, name

    def test_refusal_reasons_are_a_closed_set(self):
        for row in collect():
            if row.refusal_reason:
                assert row.refusal_reason in REFUSAL_REASONS, row.refusal_reason

    def test_a_missing_measurement_is_null_not_zero(self):
        """A zero latency would drag every average that touched it."""
        events = collect()
        if not events:
            pytest.skip("no artifacts built")
        untimed = [e for e in events if e.latency_ms is None]
        assert untimed, "days 10/11/13 never timed their turns; that should be null"
        assert all(e.latency_ms is None or e.latency_ms > 0 for e in events)

    def test_there_is_no_timestamp(self):
        """Nothing here recorded wall-clock time, so nothing may pretend to.

        A synthetic timestamp would make trend charts possible and every one
        of them fiction.
        """
        assert not [n for n in FIELD_NAMES if "time" in n or n == "ts" or "date" in n]


class TestTheEvents:
    def test_every_event_traces_to_a_measured_run(self):
        events = collect()
        if not events:
            pytest.skip("no artifacts built")
        assert {e.source_day for e in events} <= {10, 11, 12, 13}

    def test_voice_rows_are_the_only_timed_ones(self):
        events = collect()
        if not events:
            pytest.skip("no artifacts built")
        for row in events:
            if row.latency_ms is not None:
                assert row.surface == VOICE, row

    def test_no_negative_latency_survives_into_the_warehouse(self):
        """The defect day 15 found in day 12, pinned from this side too.

        An interrupted caller is not a fast response, and a negative duration
        in a warehouse corrupts every mean computed over it.
        """
        for row in collect():
            assert row.latency_ms is None or row.latency_ms > 0, row

    def test_turn_indexes_within_a_conversation_are_contiguous(self):
        events = collect()
        if not events:
            pytest.skip("no artifacts built")
        seen: dict[str, list[int]] = {}
        for row in events:
            seen.setdefault(row.conversation_id, []).append(row.turn_index)
        for cid, indexes in seen.items():
            assert sorted(indexes) == list(range(len(indexes))), cid


class TestContainment:
    def test_a_resolved_conversation_counts_both_ways(self, run):
        rows = run([event("a", 0, answered=True)])
        out = rows["01_containment"][0]
        assert out["containment_naive"] == 1.0
        assert out["containment_honest"] == 1.0

    def test_an_abandoned_conversation_splits_the_two_metrics(self, run):
        """The whole finding, on one conversation where it can be checked.

        Nobody escalated, so the naive metric calls it contained. The last
        thing that happened was a refusal, so the customer left with nothing.
        """
        rows = run([event("a", 0, answered=True), event("a", 1, answered=False)])
        out = rows["01_containment"][0]
        assert out["containment_naive"] == 1.0
        assert out["containment_honest"] == 0.0
        assert out["abandoned_on_a_refusal"] == 1

    def test_only_the_last_turn_decides(self, run):
        """A refusal mid-conversation that is then resolved is not abandonment."""
        rows = run([event("a", 0, answered=False), event("a", 1, answered=True)])
        assert rows["01_containment"][0]["containment_honest"] == 1.0

    def test_an_escalated_conversation_is_contained_by_neither(self, run):
        rows = run([event("a", 0, answered=True, escalated=True)])
        out = rows["01_containment"][0]
        assert out["containment_naive"] == 0.0
        assert out["containment_honest"] == 0.0

    def test_conversations_are_counted_once_however_many_turns(self, run):
        rows = run([event("a", i, answered=True) for i in range(5)])
        assert rows["01_containment"][0]["conversations"] == 1

    def test_surfaces_are_reported_separately(self, run):
        rows = run(
            [
                event("a", 0, answered=True),
                event("b", 0, answered=False, surface=VOICE),
            ]
        )
        by_surface = {r["surface"]: r for r in rows["02_containment_by_surface"]}
        assert by_surface[CHAT]["containment_honest"] == 1.0
        assert by_surface[VOICE]["containment_honest"] == 0.0


class TestTheOtherQueries:
    def test_red_team_refusals_are_not_counted_as_content_gaps(self, run):
        """A refused attack is the guardrail working, not a missing article."""
        rows = run(
            [
                event("a", 0, answered=False, refusal_reason=REFUSAL_REASONS[0]),
                event(
                    "b",
                    0,
                    answered=False,
                    refusal_reason=REFUSAL_REASONS[0],
                    tags=("red_team",),
                ),
            ]
        )
        total = sum(r["refusals"] for r in rows["03_content_gaps"])
        assert total == 1

    def test_untimed_turns_are_excluded_rather_than_zero_filled(self, run):
        rows = run(
            [
                event("a", 0, surface=VOICE, latency_ms=1000.0),
                event("b", 0, surface=VOICE, latency_ms=None),
            ]
        )
        out = rows["04_latency"][0]
        assert out["timed_turns"] == 1
        assert out["mean_ms"] == 1000

    def test_the_mean_is_never_below_the_median(self, run):
        """The assertion that found day 12's negative latencies."""
        rows = run(
            [
                event(f"c{i}", 0, surface=VOICE, latency_ms=float(x))
                for i, x in enumerate((900, 950, 1000, 1200))
            ]
        )
        out = rows["04_latency"][0]
        assert out["mean_ms"] >= out["p50_ms"] * 0.9

    def test_follow_ups_are_split_from_openers(self, run):
        rows = run([event("a", 0), event("a", 1, answered=False)])
        by_flag = {r["is_follow_up"]: r for r in rows["05_follow_ups"]}
        assert by_flag[False]["answer_rate"] == 1.0
        assert by_flag[True]["answer_rate"] == 0.0


class TestPortability:
    def test_every_query_uses_the_table_placeholder(self):
        """A hard-coded table name would run on one engine only."""
        for name, sql in wh.queries():
            assert wh.TABLE_PLACEHOLDER in sql, name

    def test_every_query_explains_itself_first(self):
        for name, sql in wh.queries():
            assert sql.startswith("--"), name
            assert len(wh.describe_all()[name]) > 60, name

    def test_no_query_uses_a_duckdb_only_construct(self):
        """The queries are BigQuery SQL that DuckDB happens to accept.

        Writing them the other way round would make the BigQuery path a
        rewrite rather than a copy.
        """
        banned = ("read_json_auto", "read_csv", "list_value", "::", "SIMILAR TO")
        for name, sql in wh.queries():
            for token in banned:
                assert token not in sql, f"{name} uses {token}"

    def test_safe_divide_is_the_whole_compatibility_shim(self):
        assert "SAFE_DIVIDE" in wh.SAFE_DIVIDE
        assert wh.SAFE_DIVIDE.count("CREATE OR REPLACE MACRO") == 1

    def test_the_bigquery_load_command_matches_the_schema(self):
        commands = wh.load_commands("ds", Path("turns.ndjson"))
        schema_arg = commands[-1][-1]
        assert schema_arg.count(":") == len(BIGQUERY_SCHEMA)
        assert "NEWLINE_DELIMITED_JSON" in " ".join(commands[-1])


class TestResults:
    def test_it_loads_or_says_it_has_not_been_built(self):
        from convo.projects.day15_analytics.results import load

        found = load()
        if found is None:
            pytest.skip("analytics artifact not built")
        assert found.events > 0
        assert len(found.results) == len(wh.queries())

    def test_the_honest_metric_is_lower_than_the_reported_one(self):
        from convo.projects.day15_analytics.results import load

        found = load()
        if found is None:
            pytest.skip("analytics artifact not built")
        row = found.containment
        assert row["containment_honest"] <= row["containment_naive"]
        assert found.gap > 0, "the whole day is about this gap being real"

    def test_the_engines_agreed_where_both_ran(self):
        from convo.projects.day15_analytics.results import load

        found = load()
        if found is None:
            pytest.skip("analytics artifact not built")
        if not found.bigquery_results:
            pytest.skip("BigQuery path not run; needs a signed-in project")
        assert found.disagreements == []
