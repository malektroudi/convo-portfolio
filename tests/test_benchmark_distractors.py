"""The corpus-size sweep: that it varies one thing, and concludes from data.

This experiment is only worth anything if the corpus size is genuinely the
only thing that moves. Three ways it could silently fail to be, each of which
would produce a believable curve:

- re-embedding per condition, so a changed vector masquerades as changed
  difficulty
- sampling fresh queries per condition, so the question set moves with the
  corpus
- dropping judged documents when truncating, so small corpora look hard
  because the answer is absent rather than because the distractors are good

All three are tested here, because none of them raises an error.
"""

from __future__ import annotations

import json

import pytest

from convo.benchmark import distractors as dz


class TestOnlyTheCorpusSizeVaries:
    def test_the_corpus_is_embedded_once(self):
        """Embedding per size would let a changed vector look like difficulty."""
        import inspect

        source = inspect.getsource(dz.run)
        assert source.count("embed([c.text for c in") == 1
        # The per-size loop slices the one embedding rather than making another.
        assert "vectors[:size]" in source

    def test_the_query_sample_is_drawn_once_and_before_measuring(self):
        import inspect

        source = inspect.getsource(dz.run)
        sample_at = source.index("chosen = sorted(chosen[:QUERIES])")
        loop_at = source.index("for size in sizes:")
        assert sample_at < loop_at
        assert source.count("rng.shuffle(chosen)") == 1

    def test_the_query_sample_is_seeded(self):
        assert isinstance(dz.SEED, int)
        import inspect

        assert "random.Random(SEED)" in inspect.getsource(dz.run)

    def test_judged_documents_come_first_so_truncation_keeps_them(self):
        """A small corpus must be hard, not unanswerable.

        Dropping the answers would make every small corpus look difficult and
        the result would read as the opposite of what it is.
        """
        import inspect

        source = inspect.getsource(dz.run)
        assert "all_chunks = judged + fillers" in source
        assert "chunks = all_chunks[:size]" in source

    def test_a_size_below_the_judged_set_is_skipped_not_truncated(self):
        import inspect

        assert "if size < len(judged):" in inspect.getsource(dz.run)

    def test_the_smallest_size_matches_day_nines_corpus(self):
        """The first point is SciFact rebuilt at the restaurant corpus's scale."""
        assert min(dz.SIZES) == 49
        assert dz.QUERIES == 42

    def test_it_uses_day_nines_reranker_and_depth(self):
        from convo.projects.day09_retrieval_evals.build import CROSS_ENCODER

        assert dz.CROSS_ENCODER == CROSS_ENCODER
        assert dz.DEPTH == 10


class TestTheVerdictCanSayItFailed:
    """An inconclusive experiment reported as a result is worse than none.

    The verdict checks three ways this run can fail to answer its own
    question before offering any conclusion, and all three produce a
    plausible-looking table rather than an error.
    """

    @staticmethod
    def _rows(deltas, headroom=0.18, captured=None):
        sizes = [49, 250, 1000, 5183][: len(deltas)]
        captured = captured or [d / headroom for d in deltas]
        return [
            {
                "documents": size,
                "delta_mrr": delta,
                "headroom": headroom if i else headroom,
                "captured": cap,
            }
            for i, (size, delta, cap) in enumerate(zip(sizes, deltas, captured, strict=True))
        ]

    def test_a_decaying_lift_supports_the_hypothesis(self):
        rows = self._rows([0.09, 0.06, 0.03, 0.001], captured=[0.5, 0.33, 0.17, 0.01])
        assert "supports" in dz.verdict(rows, queries=300)

    def test_a_solved_smallest_corpus_is_called_out(self):
        """If fusion already nails the small corpus it is not day 9 at scale."""
        rows = self._rows([0.09, 0.06, 0.03, 0.001], captured=[0.5, 0.33, 0.17, 0.01])
        rows[0]["headroom"] = 0.02
        verdict = dz.verdict(rows, queries=300)
        assert "inconclusive" in verdict
        assert "headroom" in verdict

    def test_a_non_monotonic_lift_is_called_out(self):
        rows = self._rows([0.02, 0.09, 0.03, 0.001], captured=[0.11, 0.5, 0.17, 0.01])
        verdict = dz.verdict(rows, queries=300)
        assert "inconclusive" in verdict
        assert "does not decay" in verdict

    def test_an_underpowered_run_is_called_out(self):
        """At 42 queries a 0.012 MRR 'lift' is one query moving one rank."""
        rows = self._rows([0.012, 0.008, 0.015, -0.013], captured=[0.6, 0.4, 0.15, -0.39])
        verdict = dz.verdict(rows, queries=42)
        assert "inconclusive" in verdict
        assert "three queries out of 42" in verdict

    def test_it_returns_different_conclusions_for_different_data(self):
        good = self._rows([0.09, 0.06, 0.03, 0.001], captured=[0.5, 0.33, 0.17, 0.01])
        bumpy = self._rows([0.02, 0.09, 0.03, 0.001], captured=[0.11, 0.5, 0.17, 0.01])
        weak = self._rows([0.012, 0.008, 0.015, -0.013], captured=[0.6, 0.4, 0.15, -0.39])
        verdicts = {
            dz.verdict(good, queries=300),
            dz.verdict(bumpy, queries=300),
            dz.verdict(weak, queries=42),
        }
        assert len(verdicts) == 3

    def test_the_band_widens_as_the_query_count_falls(self):
        from convo.benchmark.rerankers import material

        assert material(42) > material(300)
        assert material(300) == pytest.approx(0.01)


class TestTheStoredRun:
    @pytest.fixture(scope="class")
    def payload(self):
        path = dz.ARTIFACT_DIR / "distractors.json"
        if not path.exists():
            pytest.skip("corpus-size sweep not run")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_the_question_set_is_identical_at_every_size(self, payload):
        for row in payload["rows"]:
            assert row["fusion"]["queries"] == payload["queries"]
            assert row["reranked"]["queries"] == payload["queries"]

    def test_every_judged_document_survives_every_size(self, payload):
        assert min(payload["sizes"]) >= payload["judged_documents"]

    def test_the_sizes_actually_span_a_range(self, payload):
        assert max(payload["sizes"]) >= 20 * min(payload["sizes"])

    def test_retrieval_gets_harder_as_documents_are_added(self, payload):
        """The premise of the whole experiment.

        If fusion scored the same at 49 and 5,183 documents, the corpus size
        would not be changing the difficulty and nothing downstream would
        mean anything.
        """
        first, last = payload["rows"][0], payload["rows"][-1]
        assert last["fusion"]["mrr"] < first["fusion"]["mrr"]

    def test_every_metric_is_a_rate(self, payload):
        for row in payload["rows"]:
            for block in (row["fusion"], row["reranked"]):
                for key in ("hit_at_1", "hit_at_5", "mrr", "ndcg_at_5"):
                    assert 0.0 <= block[key] <= 1.0, (key, block[key])

    def test_the_stored_verdict_still_follows_from_the_stored_rows(self, payload):
        assert payload["verdict"] == dz.verdict(payload["rows"], payload["queries"])
