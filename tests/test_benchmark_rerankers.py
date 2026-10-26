"""The reranker comparison: that it is controlled, and that its conclusion is
derived rather than asserted.

The experiment exists to decide between two explanations, so the thing worth
testing hardest is that it *can* decide — that the shortlist is genuinely held
fixed, that the three models differ along the axes claimed, and that the
verdict is computed from the numbers rather than typed next to them.

A comparison whose conclusion is a hand-written string would survive any
result, including the opposite one.
"""

from __future__ import annotations

import pytest

from convo.benchmark import rerankers as rr


class TestTheComparisonIsControlled:
    def test_the_shortlist_is_built_once_and_shared(self):
        """Every model must reorder identical candidates.

        Re-running retrieval per model would leave a confound in a comparison
        whose whole purpose is removing one.
        """
        import inspect

        source = inspect.getsource(rr.main)
        # The shortlists are constructed in the loop header, once per corpus,
        # and `rerank_all` receives that object rather than rebuilding it.
        assert "for reranker in RERANKERS" in source
        assert source.count("rerank_all(shortlists, reranker)") == 1
        assert "scifact_shortlists" in source and source.count("scifact_shortlists") == 1

    def test_rerank_all_never_retrieves(self):
        import inspect

        source = inspect.getsource(rr.rerank_all)
        for forbidden in ("BM25", "DenseIndex", "reciprocal_rank_fusion", "search("):
            assert forbidden not in source, forbidden

    def test_every_model_sees_the_same_depth(self):
        assert rr.DEPTH == 10


class TestTheModelsSeparateTheExplanations:
    def test_there_are_at_least_three(self):
        assert len(rr.RERANKERS) >= 3

    def test_two_share_training_data_and_differ_in_size(self):
        """The capacity control. Without it, a win for the third model could
        just as easily be explained by its being bigger."""
        ms_marco = [r for r in rr.RERANKERS if r.trained_on == "MS MARCO"]
        assert len(ms_marco) >= 2
        assert len({r.params_m for r in ms_marco}) == len(ms_marco)

    def test_one_is_trained_on_something_else(self):
        """The domain test."""
        assert any(r.trained_on != "MS MARCO" for r in rr.RERANKERS)

    def test_the_baseline_model_is_the_one_day_nine_used(self):
        from convo.projects.day09_retrieval_evals.build import CROSS_ENCODER

        assert rr.RERANKERS[0].model_id == CROSS_ENCODER


class TestTheVerdictIsDerived:
    """The conclusion must follow from the numbers, whatever they turn out to
    be. These feed it fabricated results and check it says the right thing."""

    @staticmethod
    def _results(l6: float, l12: float, bge: float, base: float = 0.70) -> dict:
        def row(label, mrr):
            return {"label": label, "scores": {"mrr": mrr}}

        return {
            "scifact": {
                "baseline": {"mrr": base},
                "rerankers": [
                    row("MiniLM-L6", base + l6),
                    row("MiniLM-L12", base + l12),
                    row("bge-base", base + bge),
                ],
            }
        }

    def test_broad_training_helps_where_capacity_does_not(self):
        verdict = rr.verdict(self._results(l6=-0.02, l12=-0.01, bge=+0.04))
        assert "domain hypothesis" in verdict
        assert "supports" in verdict

    def test_capacity_alone_rescuing_it_contradicts_the_hypothesis(self):
        verdict = rr.verdict(self._results(l6=-0.02, l12=+0.03, bge=+0.002))
        assert "contradicts" in verdict

    def test_both_helping_isolates_neither(self):
        """Branch order must not pick a winner when both explanations move."""
        verdict = rr.verdict(self._results(l6=-0.02, l12=+0.03, bge=+0.05))
        assert verdict.startswith("mixed")

    def test_nothing_helping_materially_says_so(self):
        verdict = rr.verdict(self._results(l6=-0.02, l12=+0.001, bge=+0.002))
        assert "buys nothing" in verdict

    def test_a_difference_too_small_to_mean_anything_is_not_a_rescue(self):
        """The bug this band exists for.

        The first real run returned +0.0008 for the capacity control. Treating
        any delta above zero as a rescue would have reported that as "more
        capacity was enough" — a conclusion resting on about a quarter of one
        query out of 300 moving up a rank.
        """
        verdict = rr.verdict(self._results(l6=-0.013, l12=+0.0008, bge=+0.04))
        assert "supports the domain hypothesis" in verdict
        assert rr.MATERIAL > 0.0008

    def test_it_does_not_always_say_the_same_thing(self):
        """The test that would catch a hard-coded conclusion."""
        verdicts = {
            rr.verdict(self._results(-0.02, -0.01, +0.04)),
            rr.verdict(self._results(-0.02, +0.03, +0.002)),
            rr.verdict(self._results(-0.02, +0.001, +0.002)),
            rr.verdict(self._results(-0.02, +0.03, +0.05)),
        }
        assert len(verdicts) == 4


class TestTheStoredRun:
    @pytest.fixture(scope="class")
    def payload(self):
        import json

        path = rr.ARTIFACT_DIR / "rerankers.json"
        if not path.exists():
            pytest.skip("reranker comparison not run")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_it_is_not_a_truncated_run(self, payload):
        assert not payload["truncated"]

    def test_both_corpora_were_measured(self, payload):
        assert set(payload["corpora"]) == {"restaurant", "scifact"}

    def test_every_model_ran_on_every_corpus(self, payload):
        labels = {r["label"] for r in payload["rerankers"]}
        for corpus in payload["corpora"].values():
            assert {r["label"] for r in corpus["rerankers"]} == labels

    def test_it_reproduces_day_nines_published_number(self, payload):
        """The harness is only trustworthy if it agrees where the answer is
        already known. Day 9 published hybrid 0.7035 and reranked 0.7897."""
        restaurant = payload["corpora"]["restaurant"]
        assert restaurant["baseline"]["mrr"] == pytest.approx(0.7035, abs=0.0005)
        l6 = next(r for r in restaurant["rerankers"] if r["label"] == "MiniLM-L6")
        assert l6["scores"]["mrr"] == pytest.approx(0.7897, abs=0.0005)

    def test_the_stored_verdict_matches_a_fresh_reading(self, payload):
        """A conclusion saved to disk must still follow from the saved data."""
        assert payload["verdict"] == rr.verdict(payload["corpora"])

    def test_every_metric_is_a_rate(self, payload):
        for corpus in payload["corpora"].values():
            for row in [corpus["baseline"], *[r["scores"] for r in corpus["rerankers"]]]:
                for key in ("hit_at_1", "mrr", "ndcg_at_5"):
                    assert 0.0 <= row[key] <= 1.0, (key, row[key])
