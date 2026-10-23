"""Router tests over hand-placed vectors, so every similarity is obvious.

Vectors are axis-aligned in 3-D: cosine similarity to its own axis is 1.0 and
to any other axis is 0.0, which makes the expected scores exact rather than
approximate.
"""

from __future__ import annotations

import numpy as np
import pytest

from convo.core import classification as cls
from convo.projects.day03_embed_router.router import EmbeddingRouter, l2_normalize

NAMES = ["x_intent", "y_intent", "z_intent"]
X, Y, Z = np.eye(3, dtype=np.float32)

# Two examples per intent: the axis itself, and a slightly tilted variant.
TRAIN = np.array([X, X + 0.1 * Y, Y, Y + 0.1 * Z, Z, Z + 0.1 * X], dtype=np.float32)
TRAIN_LABELS = np.array([0, 0, 1, 1, 2, 2])


def _router(strategy: str = "knn", k: int = 1) -> EmbeddingRouter:
    return EmbeddingRouter.fit(TRAIN, TRAIN_LABELS, NAMES, strategy=strategy, k=k)


class TestNormalize:
    def test_rows_become_unit_length(self):
        out = l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0)

    def test_a_zero_vector_stays_zero_rather_than_nan(self):
        out = l2_normalize(np.array([[0.0, 0.0], [1.0, 1.0]]))
        assert np.isfinite(out).all()
        assert np.allclose(out[0], 0.0)

    def test_dot_product_of_normalized_rows_is_cosine(self):
        a = l2_normalize(np.array([[1.0, 1.0]]))
        assert (a @ a.T).item() == pytest.approx(1.0)


class TestFit:
    def test_indexes_every_example(self):
        r = _router()
        assert r.n_exemplars == 6
        assert r.n_classes == 3
        assert r.examples_per_class().tolist() == [2, 2, 2]

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="differ in length"):
            EmbeddingRouter.fit(TRAIN, TRAIN_LABELS[:3], NAMES)

    def test_oos_exemplars_are_rejected(self):
        bad = np.array([0, -1, 1, 1, 2, 2])
        with pytest.raises(ValueError, match="in-scope"):
            EmbeddingRouter.fit(TRAIN, bad, NAMES)

    def test_unknown_strategy_is_rejected(self):
        with pytest.raises(ValueError, match="unknown strategy"):
            EmbeddingRouter.fit(TRAIN, TRAIN_LABELS, NAMES, strategy="magic")

    def test_k_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="k must be"):
            EmbeddingRouter.fit(TRAIN, TRAIN_LABELS, NAMES, k=0)


class TestScores:
    def test_each_axis_query_routes_to_its_own_intent(self):
        r = _router()
        scores = r.scores(np.array([X, Y, Z]))
        assert scores.shape == (3, 3)
        assert scores.argmax(axis=1).tolist() == [0, 1, 2]
        assert np.allclose(np.diag(scores), 1.0, atol=1e-5)

    def test_an_unrelated_query_scores_low_everywhere(self):
        r = _router()
        orthogonal = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        # Equidistant from all three axes: cos = 1/sqrt(3) ~ 0.577 to each.
        scores = r.scores(orthogonal)
        assert scores.max() < 0.7

    def test_centroid_and_knn_agree_on_the_easy_case(self):
        knn = _router("knn").scores(np.array([X, Y, Z]))
        centroid = _router("centroid").scores(np.array([X, Y, Z]))
        assert knn.argmax(axis=1).tolist() == centroid.argmax(axis=1).tolist()

    def test_knn_max_beats_centroid_on_a_tilted_exemplar(self):
        # A query sitting exactly on the tilted example: max-similarity finds it
        # at 1.0, while the centroid has been pulled away by the other example.
        query = l2_normalize(np.array([X + 0.1 * Y]))
        assert _router("knn").scores(query)[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert _router("centroid").scores(query)[0, 0] < 1.0

    def test_mean_of_top_k_is_lower_than_the_max(self):
        query = np.array([X])
        top1 = _router("knn", k=1).scores(query)[0, 0]
        top2 = _router("knn", k=2).scores(query)[0, 0]
        assert top2 < top1

    def test_an_empty_index_scores_zero_without_crashing(self):
        r = EmbeddingRouter(class_names=NAMES)
        scores = r.scores(np.array([X, Y]))
        assert scores.shape == (2, 3)
        assert (scores == 0).all()

    def test_chunking_matches_a_single_pass(self, monkeypatch):
        import convo.projects.day03_embed_router.router as mod

        queries = l2_normalize(np.random.default_rng(1).normal(size=(40, 3)))
        full = _router().scores(queries)
        monkeypatch.setattr(mod, "QUERY_CHUNK", 7)
        chunked = _router().scores(queries)
        assert np.allclose(full, chunked, atol=1e-6)


class TestAddIntent:
    def test_a_new_intent_needs_no_retraining_and_is_routable(self):
        r = _router()
        new_axis = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        # Re-index in 4-D so there is a genuinely unseen direction.
        padded = np.hstack([TRAIN, np.zeros((len(TRAIN), 1), np.float32)])
        r = EmbeddingRouter.fit(padded, TRAIN_LABELS, list(NAMES))
        index = r.add_intent("w_intent", new_axis)

        assert index == 3
        assert r.n_classes == 4
        assert r.class_names[-1] == "w_intent"
        scores = r.scores(new_axis)
        assert scores.argmax(axis=1).tolist() == [3]
        assert scores[0, 3] == pytest.approx(1.0, abs=1e-5)

    def test_adding_an_intent_leaves_the_existing_ones_routing_the_same(self):
        padded = np.hstack([TRAIN, np.zeros((len(TRAIN), 1), np.float32)])
        queries = np.hstack([np.array([X, Y, Z]), np.zeros((3, 1), np.float32)])

        r = EmbeddingRouter.fit(padded, TRAIN_LABELS, list(NAMES))
        before = r.scores(queries)[:, :3]
        r.add_intent("w_intent", np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
        after = r.scores(queries)[:, :3]

        assert np.allclose(before, after, atol=1e-6)

    def test_a_duplicate_intent_name_is_rejected(self):
        r = _router()
        with pytest.raises(ValueError, match="already present"):
            r.add_intent("x_intent", np.array([X]))

    def test_a_dimension_mismatch_is_rejected(self):
        r = _router()
        with pytest.raises(ValueError, match="dimension"):
            r.add_intent("w_intent", np.zeros((1, 9), dtype=np.float32))


class TestScoresFeedTheSharedMetrics:
    def test_router_scores_drop_into_core_classification_unchanged(self):
        r = _router()
        scores = r.scores(np.array([X, Y, Z, [1.0, 1.0, 1.0]], dtype=np.float32))
        labels = np.array([0, 1, 2, cls.OOS])

        # At a threshold above the orthogonal query's 0.577, the OOS row is
        # refused and all three real ones still route correctly.
        m = cls.metrics_at(scores, labels, 0.8)
        assert m.in_scope_accuracy == 1.0
        assert m.oos_recall == 1.0
        assert m.in_scope_n == 3 and m.oos_n == 1
