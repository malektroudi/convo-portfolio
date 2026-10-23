"""Hand-built logits with arithmetic you can check on paper.

softmax(log(p)) == p when p sums to 1, so the fixture states the probabilities
it wants and the expected metrics are countable by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from convo.projects.day02_intent_nlu import analysis as an

CLASS_NAMES = ["c0", "c1", "c2"]

# row | label | probabilities        | top | conf | note
#  0  |   0   | 0.90 0.05 0.05       |  0  | 0.90 | correct
#  1  |   1   | 0.60 0.30 0.10       |  0  | 0.60 | confuses c1 with c0
#  2  |   2   | 0.20 0.20 0.60       |  2  | 0.60 | correct
#  3  |  OOS  | 0.40 0.35 0.25       |  0  | 0.40 | out-of-scope
#  4  |  OOS  | 0.34 0.33 0.33       |  0  | 0.34 | out-of-scope, unsure
#  5  |   0   | 0.50 0.30 0.20       |  0  | 0.50 | correct but low confidence
PROBS = np.array(
    [
        [0.90, 0.05, 0.05],
        [0.60, 0.30, 0.10],
        [0.20, 0.20, 0.60],
        [0.40, 0.35, 0.25],
        [0.34, 0.33, 0.33],
        [0.50, 0.30, 0.20],
    ]
)
LOGITS = np.log(PROBS)
LABELS = np.array([0, 1, 2, an.OOS, an.OOS, 0])


class TestSoftmax:
    def test_rows_sum_to_one(self):
        out = an.softmax(np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]]))
        assert np.allclose(out.sum(axis=1), 1.0)

    def test_survives_large_logits_without_overflow(self):
        out = an.softmax(np.array([[1000.0, 1001.0]]))
        assert np.isfinite(out).all()
        assert out.argmax() == 1

    def test_recovers_the_probabilities_it_was_built_from(self):
        assert np.allclose(an.softmax(LOGITS), PROBS, atol=1e-9)


class TestPredict:
    def test_threshold_zero_accepts_everything(self):
        predicted, confidence = an.predict(LOGITS, 0.0)
        assert (predicted != an.OOS).all()
        assert np.allclose(confidence, [0.90, 0.60, 0.60, 0.40, 0.34, 0.50])

    def test_rejections_appear_as_the_oos_sentinel(self):
        predicted, _ = an.predict(LOGITS, 0.55)
        assert predicted.tolist() == [0, 0, 2, an.OOS, an.OOS, an.OOS]

    def test_threshold_above_one_rejects_everything(self):
        predicted, _ = an.predict(LOGITS, 1.01)
        assert (predicted == an.OOS).all()


class TestMetrics:
    def test_at_zero_nothing_is_rejected_so_oos_recall_is_zero(self):
        m = an.metrics_at(LOGITS, LABELS, 0.0)
        assert m.in_scope_accuracy == 0.75  # rows 0, 2, 5 correct of 4 in-scope
        assert m.oos_recall == 0.0
        assert m.false_rejection_rate == 0.0
        assert m.overall_accuracy == 0.5  # (3 + 0) / 6
        assert (m.in_scope_n, m.oos_n) == (4, 2)

    def test_a_well_placed_threshold_buys_oos_recall_for_free(self):
        m = an.metrics_at(LOGITS, LABELS, 0.45)
        assert m.in_scope_accuracy == 0.75
        assert m.oos_recall == 1.0
        assert m.false_rejection_rate == 0.0
        assert m.overall_accuracy == pytest.approx(0.8333, abs=1e-4)
        assert m.balanced == 0.875

    def test_too_high_a_threshold_starts_costing_in_scope_accuracy(self):
        m = an.metrics_at(LOGITS, LABELS, 0.55)
        assert m.in_scope_accuracy == 0.5  # row 5 is now wrongly rejected
        assert m.false_rejection_rate == 0.25
        assert m.oos_recall == 1.0

    def test_a_rejected_in_scope_turn_counts_as_wrong_not_abstained(self):
        strict = an.metrics_at(LOGITS, LABELS, 1.01)
        assert strict.in_scope_accuracy == 0.0
        assert strict.false_rejection_rate == 1.0
        assert strict.oos_recall == 1.0

    def test_an_all_oos_set_reports_zero_in_scope_rather_than_dividing_by_zero(self):
        m = an.metrics_at(LOGITS[3:5], np.array([an.OOS, an.OOS]), 0.5)
        assert m.in_scope_n == 0
        assert m.in_scope_accuracy == 0.0
        assert m.oos_recall == 1.0


class TestSweepAndBest:
    def test_sweep_covers_the_unit_interval(self):
        rows = an.sweep(LOGITS, LABELS)
        assert len(rows) == 101
        assert rows[0].threshold == 0.0
        assert rows[-1].threshold == 1.0

    def test_oos_recall_never_decreases_as_the_threshold_rises(self):
        recalls = [m.oos_recall for m in an.sweep(LOGITS, LABELS)]
        assert all(b >= a for a, b in zip(recalls[:-1], recalls[1:], strict=True))

    def test_false_rejection_never_decreases_either(self):
        rates = [m.false_rejection_rate for m in an.sweep(LOGITS, LABELS)]
        assert all(b >= a for a, b in zip(rates[:-1], rates[1:], strict=True))

    def test_best_balanced_threshold_lands_in_the_free_window(self):
        best = an.best_threshold(LOGITS, LABELS)
        assert 0.35 < best.threshold <= 0.50
        assert best.balanced == 0.875

    def test_unknown_objective_is_rejected(self):
        with pytest.raises(ValueError, match="unknown objective"):
            an.best_threshold(LOGITS, LABELS, objective="vibes")


class TestConfidenceStats:
    def test_summarises_the_top_probability_distribution(self):
        stats = an.confidence_stats(LOGITS)
        assert stats.maximum == 0.9
        assert stats.p50 == pytest.approx(0.55, abs=1e-4)
        assert 0.3 < stats.mean < 0.7

    def test_a_sharp_model_is_not_degenerate(self):
        assert an.confidence_stats(LOGITS).is_degenerate is False

    def test_a_flat_model_is_flagged_as_degenerate(self):
        # 150 classes, all logits equal: every top probability is 1/150.
        flat = np.zeros((20, 150))
        stats = an.confidence_stats(flat)
        assert stats.maximum == pytest.approx(1 / 150, abs=1e-4)
        assert stats.is_degenerate is True

    def test_the_real_failure_mode_is_caught(self):
        # Mirrors the first training run: top probability never clears 0.1.
        rng = np.random.default_rng(0)
        barely_trained = rng.normal(0, 0.05, size=(200, 150))
        stats = an.confidence_stats(barely_trained)
        assert stats.maximum < 0.1
        assert stats.is_degenerate is True


class TestConfusionPairs:
    def test_names_the_pair_the_model_actually_mixes_up(self):
        pairs = an.confusion_pairs(LOGITS, LABELS, CLASS_NAMES, 0.0)
        assert len(pairs) == 1
        assert (pairs[0].true_intent, pairs[0].predicted_intent) == ("c1", "c0")
        assert pairs[0].count == 1

    def test_rejections_are_excluded_unless_asked_for(self):
        without = an.confusion_pairs(LOGITS, LABELS, CLASS_NAMES, 0.55)
        assert [p.predicted_intent for p in without] == ["c0"]

        with_rejects = an.confusion_pairs(
            LOGITS, LABELS, CLASS_NAMES, 0.55, include_rejections=True
        )
        assert any(p.is_rejection for p in with_rejects)
        assert {p.true_intent for p in with_rejects} == {"c0", "c1"}

    def test_ordered_by_count_and_capped_at_top_k(self):
        labels = np.array([1, 1, 1, 2, 0])
        logits = np.log(
            np.array(
                [
                    [0.8, 0.1, 0.1],  # 1 -> 0
                    [0.8, 0.1, 0.1],  # 1 -> 0
                    [0.8, 0.1, 0.1],  # 1 -> 0
                    [0.7, 0.2, 0.1],  # 2 -> 0
                    [0.9, 0.05, 0.05],  # correct
                ]
            )
        )
        pairs = an.confusion_pairs(logits, labels, CLASS_NAMES, 0.0, top_k=1)
        assert len(pairs) == 1
        assert (pairs[0].true_intent, pairs[0].count) == ("c1", 3)


class TestWorstIntents:
    def test_ranks_by_recall_and_skips_unsupported_classes(self):
        rows = an.worst_intents(LOGITS, LABELS, CLASS_NAMES, 0.45)
        assert [r[1] for r in rows] == ["c1", "c0", "c2"]
        assert rows[0][2] == 0.0  # c1 is never predicted correctly
        assert rows[1][3] == 2  # c0 has two test examples

    def test_a_class_with_no_examples_is_absent(self):
        rows = an.worst_intents(LOGITS[:1], LABELS[:1], CLASS_NAMES, 0.0)
        assert [r[1] for r in rows] == ["c0"]


class TestConfusionSubmatrix:
    def test_rows_sum_to_support_with_a_rejection_column(self):
        matrix = an.confusion_submatrix(LOGITS, LABELS, [0, 1, 2], 0.45)
        assert matrix.shape == (3, 4)
        assert matrix[0].tolist() == [2, 0, 0, 0]
        assert matrix[1].tolist() == [1, 0, 0, 0]
        assert matrix[2].tolist() == [0, 0, 1, 0]

    def test_rejections_land_in_the_final_column(self):
        matrix = an.confusion_submatrix(LOGITS, LABELS, [0, 1, 2], 0.55)
        assert matrix[0].tolist() == [1, 0, 0, 1]
        assert matrix.sum() == 4  # the four in-scope rows

    def test_a_subset_ignores_predictions_outside_it(self):
        matrix = an.confusion_submatrix(LOGITS, LABELS, [1], 0.0)
        assert matrix.shape == (1, 2)
        assert matrix.sum() == 0  # c1's single row is predicted as c0, outside the subset
