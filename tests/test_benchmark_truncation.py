"""The truncation experiment: one variable, full query set, derived verdict.

This is the third attempt at explaining why day 9's reranker helps on the
restaurant corpus and not on SciFact. The first two failed in instructive
ways — the domain hypothesis was refuted by a model that should have rescued
it, and the corpus-size sweep turned out to rest on a false premise — so the
tests here lean on the thing both of those lacked: a check that the premise
is true before the conclusion is read.
"""

from __future__ import annotations

import json

import pytest

from convo.benchmark import truncation as tr


class TestThePremise:
    """Each of these would silently invalidate the experiment."""

    def test_it_uses_day_nines_own_reranker(self):
        from convo.projects.day09_retrieval_evals.build import CROSS_ENCODER

        assert tr.CROSS_ENCODER == CROSS_ENCODER

    def test_the_narrow_window_is_the_one_day_nine_shipped(self):
        """If 256 were not day 9's default, this would test nothing about it."""
        from convo.projects.day09_retrieval_evals.build import MAX_LENGTH

        assert MAX_LENGTH == 256
        assert min(tr.WINDOWS) == MAX_LENGTH

    def test_day_nines_default_is_safe_for_its_own_corpus(self):
        """The default is only a trap for corpora longer than the one it was
        chosen for, which is exactly how it survived unnoticed."""
        pytest.importorskip("transformers", reason="needs the train extra")
        from convo.projects.day08_rag_cites.chunking import chunk_corpus, load_corpus
        from convo.projects.day09_retrieval_evals.build import MAX_LENGTH

        lengths = tr.token_lengths([c.text for c in chunk_corpus(load_corpus(), "structural")])
        assert lengths[f"over_{MAX_LENGTH}"] == 0.0

    def test_the_wide_window_is_within_the_model(self):
        # Needs the tokenizer, which the serving environment does not carry.
        pytest.importorskip("transformers", reason="needs the train extra")
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(tr.CROSS_ENCODER)
        assert max(tr.WINDOWS) <= config.max_position_embeddings

    def test_only_the_window_varies(self):
        """Same shortlists, same model, same queries across conditions."""
        import inspect

        source = inspect.getsource(tr.run)
        assert source.count("scifact_shortlists()") == 1
        assert source.count("restaurant_shortlists()") == 1
        assert "for window in windows" in source
        assert "rerank_all(shortlists, reranker, max_length=window)" in source

    def test_the_reranker_window_is_actually_plumbed_through(self):
        """A parameter that is accepted and ignored would produce two
        identical rows and read as 'truncation is not the explanation'."""
        import inspect

        from convo.benchmark.rerankers import rerank_all

        source = inspect.getsource(rerank_all)
        assert "max_length=max_length" in source


class TestTheVerdictIsDerived:
    @staticmethod
    def _corpora(narrow: float, wide: float, over: float = 0.70, queries: int = 300):
        return {
            "scifact": {
                "queries": queries,
                "lengths": {"over_256": over, "over_512": 0.30},
                "windows": [
                    {"max_length": 256, "delta_mrr": narrow},
                    {"max_length": 512, "delta_mrr": wide},
                ],
            }
        }

    def test_a_rescue_is_called_a_rescue(self):
        verdict = tr.verdict(self._corpora(narrow=-0.013, wide=+0.05))
        assert "truncation explains it" in verdict
        assert "70%" in verdict

    def test_a_material_but_insufficient_gain_says_so(self):
        verdict = tr.verdict(self._corpora(narrow=-0.06, wide=-0.02))
        assert "does not fully explain" in verdict

    def test_no_movement_rules_truncation_out(self):
        verdict = tr.verdict(self._corpora(narrow=-0.013, wide=-0.012))
        assert "not the explanation" in verdict

    def test_it_returns_different_conclusions_for_different_data(self):
        verdicts = {
            tr.verdict(self._corpora(-0.013, +0.05)),
            tr.verdict(self._corpora(-0.06, -0.02)),
            tr.verdict(self._corpora(-0.013, -0.012)),
        }
        assert len(verdicts) == 3

    def test_the_band_comes_from_the_query_count(self):
        """The mistake the previous experiment made, not repeated here."""
        from convo.benchmark.rerankers import material

        # At 300 queries a 0.02 gain is material; at 42 it is not.
        assert "explains it" in tr.verdict(self._corpora(-0.001, 0.02, queries=300))
        assert "not the explanation" in tr.verdict(self._corpora(-0.001, 0.02, queries=42))
        assert material(42) > 0.02 > material(300)


class TestTheStoredRun:
    @pytest.fixture(scope="class")
    def payload(self):
        path = tr.ARTIFACT_DIR / "truncation.json"
        if not path.exists():
            pytest.skip("truncation experiment not run")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_it_used_the_full_query_set(self, payload):
        """42 queries is what made the previous run unreadable."""
        assert payload["corpora"]["scifact"]["queries"] >= 300

    def test_the_restaurant_corpus_is_not_truncated_at_all(self, payload):
        """The control. If 256 tokens clipped these chunks too, the two
        corpora would not differ in the way this experiment assumes."""
        assert payload["corpora"]["restaurant"]["lengths"]["over_256"] == 0.0

    def test_most_scifact_documents_exceed_the_narrow_window(self, payload):
        assert payload["corpora"]["scifact"]["lengths"]["over_256"] > 0.5

    def test_widening_cannot_change_the_restaurant_result(self, payload):
        """Nothing there is clipped, so the two windows must agree exactly.

        A difference here would mean the window is changing something other
        than how much text the model sees, and the whole reading would be
        unsafe.
        """
        rows = payload["corpora"]["restaurant"]["windows"]
        assert len({r["scores"]["mrr"] for r in rows}) == 1

    def test_every_metric_is_a_rate(self, payload):
        for corpus in payload["corpora"].values():
            for row in corpus["windows"]:
                for key in ("hit_at_1", "hit_at_5", "mrr", "ndcg_at_5"):
                    assert 0.0 <= row["scores"][key] <= 1.0, key

    def test_the_stored_verdict_still_follows_from_the_numbers(self, payload):
        assert payload["verdict"] == tr.verdict(payload["corpora"])
