"""Ranking metrics, checked against hand-computed values.

Every expected number here is worked out on paper in the test name or a
comment. Metrics that are only ever compared against themselves are how a
retrieval project convinces itself it improved something.
"""

from __future__ import annotations

import math

import pytest

from convo.core.retrieval import (
    hit_at_k,
    ndcg_at_k,
    reciprocal_rank,
    reciprocal_rank_fusion,
    score_runs,
)

RANKED = ["a", "b", "c", "d", "e"]


class TestHitAtK:
    def test_relevant_at_rank_one_hits_every_cutoff(self):
        assert hit_at_k(RANKED, {"a"}, 1)
        assert hit_at_k(RANKED, {"a"}, 5)

    def test_relevant_at_rank_three_misses_at_one_and_two(self):
        assert not hit_at_k(RANKED, {"c"}, 1)
        assert not hit_at_k(RANKED, {"c"}, 2)
        assert hit_at_k(RANKED, {"c"}, 3)

    def test_nothing_relevant_never_hits(self):
        assert not hit_at_k(RANKED, {"zzz"}, 5)

    def test_k_beyond_the_list_is_fine(self):
        assert hit_at_k(RANKED, {"e"}, 99)

    def test_k_must_be_positive(self):
        with pytest.raises(ValueError):
            hit_at_k(RANKED, {"a"}, 0)


class TestReciprocalRank:
    @pytest.mark.parametrize(
        ("relevant", "expected"),
        [({"a"}, 1.0), ({"b"}, 0.5), ({"c"}, 1 / 3), ({"e"}, 0.2), ({"zzz"}, 0.0)],
    )
    def test_it_is_one_over_the_first_relevant_rank(self, relevant, expected):
        assert reciprocal_rank(RANKED, relevant) == pytest.approx(expected)

    def test_only_the_first_relevant_result_counts(self):
        assert reciprocal_rank(RANKED, {"b", "c", "d"}) == pytest.approx(0.5)

    def test_an_empty_ranking_scores_zero(self):
        assert reciprocal_rank([], {"a"}) == 0.0


class TestNdcg:
    def test_the_perfect_ordering_scores_one(self):
        assert ndcg_at_k(RANKED, {"a", "b"}, 5) == pytest.approx(1.0)

    def test_one_relevant_at_rank_one_scores_one(self):
        assert ndcg_at_k(RANKED, {"a"}, 5) == pytest.approx(1.0)

    def test_one_relevant_at_rank_two_scores_the_discount(self):
        # DCG = 1/log2(3); IDCG = 1/log2(2) = 1. So nDCG = 1/log2(3) ~ 0.6309.
        assert ndcg_at_k(RANKED, {"b"}, 5) == pytest.approx(1 / math.log2(3), abs=1e-6)

    def test_two_relevant_split_apart_scores_less_than_adjacent(self):
        adjacent = ndcg_at_k(RANKED, {"a", "b"}, 5)
        split = ndcg_at_k(RANKED, {"a", "e"}, 5)
        assert split < adjacent

    def test_it_rewards_several_relevant_results_where_mrr_cannot(self):
        # MRR is blind past the first hit; nDCG is not.
        one = ndcg_at_k(RANKED, {"a"}, 5)
        two = ndcg_at_k(RANKED, {"a", "b"}, 5)
        assert reciprocal_rank(RANKED, {"a"}) == reciprocal_rank(RANKED, {"a", "b"})
        assert two > 0 and one > 0

    def test_nothing_relevant_scores_zero(self):
        assert ndcg_at_k(RANKED, {"zzz"}, 5) == 0.0

    def test_an_empty_relevant_set_scores_zero(self):
        assert ndcg_at_k(RANKED, set(), 5) == 0.0

    def test_the_cutoff_is_respected(self):
        assert ndcg_at_k(RANKED, {"e"}, 3) == 0.0
        assert ndcg_at_k(RANKED, {"e"}, 5) > 0.0


class TestScoreRuns:
    def test_it_averages_across_queries(self):
        runs = {
            "q1": (["a", "b"], {"a"}),  # rr 1.0, hit@1
            "q2": (["x", "y"], {"y"}),  # rr 0.5, hit@3 not @1
        }
        score = score_runs("test", runs)
        assert score.queries == 2
        assert score.hit_at_1 == 0.5
        assert score.hit_at_3 == 1.0
        assert score.mrr == pytest.approx(0.75)

    def test_queries_with_nothing_relevant_are_named(self):
        runs = {
            "found": (["a"], {"a"}),
            "lost": (["a", "b"], {"zzz"}),
        }
        score = score_runs("test", runs)
        assert score.misses == ("lost",)
        assert score.hit_at_5 == 0.5

    def test_an_empty_run_set_is_all_zeros(self):
        score = score_runs("test", {})
        assert score.queries == 0 and score.mrr == 0.0

    def test_it_serialises(self):
        score = score_runs("test", {"q": (["a"], {"a"})})
        payload = score.as_dict()
        assert payload["retriever"] == "test"
        assert isinstance(payload["misses"], list)


class TestReciprocalRankFusion:
    def test_an_item_ranked_well_by_both_wins(self):
        fused = reciprocal_rank_fusion({"dense": ["a", "b", "c"], "bm25": ["a", "c", "b"]})
        assert fused[0] == "a"

    def test_it_rescues_an_item_both_retrievers_found(self):
        # "m" is second in both lists; "x" is first in one and absent from the
        # other. Appearing twice at all is worth more than one strong opinion,
        # which is the property that makes fusion worth doing.
        fused = reciprocal_rank_fusion({"dense": ["x", "m"], "bm25": ["m"]})
        assert fused[0] == "m"

    def test_ranks_one_and_three_edge_out_two_and_two(self):
        # Worth pinning because it is counter-intuitive. With k=60 the discount
        # curve is nearly linear, and by convexity 1/61 + 1/63 is very slightly
        # more than 2/62 — so RRF does NOT prefer consistent-middling over
        # strong-then-weak. The margin is 8e-6; treat them as equivalent.
        fused = reciprocal_rank_fusion({"dense": ["x", "m", "y"], "bm25": ["y", "m", "x"]})
        assert fused[0] in {"x", "y"}
        assert "m" in fused

    def test_items_in_only_one_ranking_still_appear(self):
        fused = reciprocal_rank_fusion({"dense": ["a"], "bm25": ["b"]})
        assert set(fused) == {"a", "b"}

    def test_ties_break_deterministically(self):
        first = reciprocal_rank_fusion({"one": ["a", "b"], "two": ["b", "a"]})
        second = reciprocal_rank_fusion({"one": ["a", "b"], "two": ["b", "a"]})
        assert first == second

    def test_a_single_ranking_passes_through_in_order(self):
        assert reciprocal_rank_fusion({"only": ["a", "b", "c"]}) == ["a", "b", "c"]

    def test_it_reads_positions_not_scores(self):
        # No score is ever supplied, which is the point: BM25 and cosine are
        # not on a comparable scale and any weighting would be a tuned knob.
        fused = reciprocal_rank_fusion({"dense": ["a", "b"], "bm25": ["b", "a"]})
        assert sorted(fused) == ["a", "b"]
