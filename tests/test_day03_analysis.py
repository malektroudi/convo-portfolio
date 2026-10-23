"""Segment scoring, with a fixture that makes the headline result explicit.

Four classes; class 3 is withheld. The classifier's scores have a hard zero in
column 3 (it has no such output), the router has real scores there. Every
expected rate below is countable by hand from the table in the comment.
"""

from __future__ import annotations

import numpy as np

from convo.projects.day03_embed_router import analysis as an

NAMES = ["known_a", "known_b", "known_c", "new_d"]
HELD_OUT = [3]

#  row | label | note
#   0  |   0   | known, both should get it
#   1  |   1   | known, both should get it
#   2  |   3   | NEW intent
#   3  |   3   | NEW intent
#   4  |  OOS  | should be refused
LABELS = np.array([0, 1, 3, 3, -1])

# Router: real scores in every column, including the new intent.
ROUTER = np.array(
    [
        [0.90, 0.10, 0.10, 0.10],
        [0.10, 0.85, 0.10, 0.10],
        [0.10, 0.10, 0.10, 0.80],  # routes the new intent correctly
        [0.10, 0.10, 0.40, 0.35],  # confuses the new intent with known_c
        [0.30, 0.25, 0.20, 0.20],  # nothing scores high: refusable
    ],
    dtype=np.float32,
)

# Classifier: column 3 is structurally zero — it was never given that class.
CLASSIFIER = np.array(
    [
        [0.95, 0.03, 0.02, 0.0],
        [0.02, 0.93, 0.05, 0.0],
        [0.20, 0.30, 0.50, 0.0],  # cannot be right; best it can do is be wrong
        [0.60, 0.20, 0.20, 0.0],
        [0.40, 0.35, 0.25, 0.0],
    ],
    dtype=np.float32,
)


class TestSegmentMasks:
    def test_splits_the_set_three_ways_without_overlap(self):
        masks = an.segment_masks(LABELS, HELD_OUT)
        assert masks[an.KNOWN].tolist() == [True, True, False, False, False]
        assert masks[an.NEW].tolist() == [False, False, True, True, False]
        assert masks[an.OUT_OF_SCOPE].tolist() == [False, False, False, False, True]

    def test_every_row_lands_in_exactly_one_segment(self):
        masks = an.segment_masks(LABELS, HELD_OUT)
        total = sum(m.astype(int) for m in masks.values())
        assert (total == 1).all()

    def test_no_held_out_intents_means_an_empty_new_segment(self):
        masks = an.segment_masks(LABELS, [])
        assert not masks[an.NEW].any()


class TestSegmentReport:
    def _by_name(self, rows):
        return {r.segment: r for r in rows}

    def test_the_classifier_scores_zero_on_the_withheld_intent(self):
        rows = self._by_name(an.segment_report(CLASSIFIER, LABELS, HELD_OUT, 0.5))
        assert rows[an.NEW].n == 2
        assert rows[an.NEW].correct == 0.0

    def test_the_router_routes_some_of_the_withheld_intent(self):
        rows = self._by_name(an.segment_report(ROUTER, LABELS, HELD_OUT, 0.5))
        assert rows[an.NEW].n == 2
        assert rows[an.NEW].correct == 0.5  # row 2 right, row 3 confused

    def test_both_handle_the_known_intents(self):
        for scores in (ROUTER, CLASSIFIER):
            rows = self._by_name(an.segment_report(scores, LABELS, HELD_OUT, 0.5))
            assert rows[an.KNOWN].n == 2
            assert rows[an.KNOWN].correct == 1.0

    def test_out_of_scope_correct_means_refused(self):
        rows = self._by_name(an.segment_report(ROUTER, LABELS, HELD_OUT, 0.5))
        assert rows[an.OUT_OF_SCOPE].n == 1
        assert rows[an.OUT_OF_SCOPE].correct == 1.0
        assert rows[an.OUT_OF_SCOPE].rejected == 1.0

    def test_a_low_threshold_accepts_the_out_of_scope_turn(self):
        rows = self._by_name(an.segment_report(ROUTER, LABELS, HELD_OUT, 0.1))
        assert rows[an.OUT_OF_SCOPE].correct == 0.0

    def test_a_high_threshold_rejects_everything_including_the_good_rows(self):
        rows = self._by_name(an.segment_report(ROUTER, LABELS, HELD_OUT, 0.99))
        assert rows[an.KNOWN].correct == 0.0
        assert rows[an.KNOWN].rejected == 1.0
        assert rows[an.OUT_OF_SCOPE].correct == 1.0

    def test_an_empty_segment_reports_zeroes_rather_than_dividing_by_zero(self):
        rows = self._by_name(an.segment_report(ROUTER, LABELS, [], 0.5))
        assert rows[an.NEW].n == 0
        assert rows[an.NEW].correct == 0.0

    def test_all_three_segments_are_always_present_and_ordered(self):
        rows = an.segment_report(ROUTER, LABELS, HELD_OUT, 0.5)
        assert [r.segment for r in rows] == [an.KNOWN, an.NEW, an.OUT_OF_SCOPE]


class TestBestThresholdForSegments:
    def test_picks_a_threshold_in_the_unit_interval(self):
        t = an.best_threshold_for_segments(ROUTER, LABELS, HELD_OUT)
        assert 0.0 <= t <= 1.0

    def test_the_chosen_threshold_beats_the_extremes(self):
        t = an.best_threshold_for_segments(ROUTER, LABELS, HELD_OUT)

        def mean_correct(threshold: float) -> float:
            rows = an.segment_report(ROUTER, LABELS, HELD_OUT, threshold)
            return sum(r.correct for r in rows) / len(rows)

        assert mean_correct(t) >= mean_correct(0.0)
        assert mean_correct(t) >= mean_correct(1.0)

    def test_it_weights_the_small_new_segment_equally(self):
        # The new segment is 2 rows against 2 known and 1 oos; equal weighting
        # means a threshold that routes row 2 is preferred.
        t = an.best_threshold_for_segments(ROUTER, LABELS, HELD_OUT)
        rows = {r.segment: r for r in an.segment_report(ROUTER, LABELS, HELD_OUT, t)}
        assert rows[an.NEW].correct > 0.0


class TestPerNewIntentRecall:
    def test_reports_each_withheld_intent_with_its_support(self):
        rows = an.per_new_intent_recall(ROUTER, LABELS, HELD_OUT, NAMES, 0.5)
        assert rows == [("new_d", 0.5, 2)]

    def test_the_classifier_is_zero_across_the_board(self):
        rows = an.per_new_intent_recall(CLASSIFIER, LABELS, HELD_OUT, NAMES, 0.5)
        assert [r[1] for r in rows] == [0.0]

    def test_sorted_worst_first(self):
        labels = np.array([2, 2, 3, 3])
        scores = np.array(
            [
                [0.0, 0.0, 0.9, 0.1],  # intent 2 right
                [0.0, 0.0, 0.9, 0.1],  # intent 2 right
                [0.0, 0.0, 0.9, 0.1],  # intent 3 wrong
                [0.0, 0.0, 0.1, 0.9],  # intent 3 right
            ],
            dtype=np.float32,
        )
        rows = an.per_new_intent_recall(scores, labels, [2, 3], NAMES, 0.5)
        assert [r[0] for r in rows] == ["new_d", "known_c"]
        assert rows[0][1] == 0.5
        assert rows[1][1] == 1.0

    def test_an_intent_with_no_test_rows_is_absent(self):
        rows = an.per_new_intent_recall(ROUTER, LABELS, [0, 3], NAMES, 0.5)
        assert [r[0] for r in rows] == ["new_d", "known_a"]
