"""The experiment's bookkeeping, where an off-by-one would silently fake a result.

These are the parts that decide what each approach is allowed to see. If
`restrict_train` leaked a held-out intent, or `expand_to_canonical` put a
column in the wrong place, the headline comparison would be wrong in a way no
amount of staring at the UI would reveal.
"""

from __future__ import annotations

import numpy as np

from convo.projects.day02_intent_nlu.data import Split
from convo.projects.day03_embed_router import benchmark as bm

NAMES = [f"intent_{i}" for i in range(10)]
TRAIN = Split(
    texts=[f"utterance {i}" for i in range(30)],
    labels=np.array([i % 10 for i in range(30)]),  # 3 examples per intent
)


class TestChooseHeldOut:
    def test_is_deterministic_for_a_seed(self):
        assert bm.choose_held_out(150, 10, 13) == bm.choose_held_out(150, 10, 13)

    def test_returns_distinct_sorted_indices_in_range(self):
        out = bm.choose_held_out(150, 10, 7)
        assert len(out) == len(set(out)) == 10
        assert out == sorted(out)
        assert all(0 <= i < 150 for i in out)

    def test_a_different_seed_gives_a_different_sample(self):
        assert bm.choose_held_out(150, 10, 1) != bm.choose_held_out(150, 10, 2)


class TestRestrictTrain:
    def test_drops_held_out_intents_entirely(self):
        known = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 and 9 withheld
        split, mapping = bm.restrict_train(TRAIN, known)

        assert len(split.texts) == 24  # 8 intents x 3 examples
        assert set(mapping) == set(known)
        assert split.labels.max() == 7
        assert len(set(split.labels.tolist())) == 8

    def test_relabels_contiguously_from_zero(self):
        known = [3, 7, 9]
        split, mapping = bm.restrict_train(TRAIN, known)

        assert mapping == {3: 0, 7: 1, 9: 2}
        assert sorted(set(split.labels.tolist())) == [0, 1, 2]

    def test_texts_and_labels_stay_aligned(self):
        known = [4, 5]
        split, mapping = bm.restrict_train(TRAIN, known)
        inverse = {local: canonical for canonical, local in mapping.items()}

        for text, local in zip(split.texts, split.labels, strict=True):
            original_row = int(text.split()[-1])
            assert TRAIN.labels[original_row] == inverse[int(local)]


class TestFewShotIndices:
    def test_takes_exactly_n_shot_rows_per_held_out_intent(self):
        shots = bm.few_shot_indices(TRAIN, [8, 9], n_shot=2, seed=13)
        assert set(shots) == {8, 9}
        assert all(len(rows) == 2 for rows in shots.values())

    def test_the_rows_really_belong_to_that_intent(self):
        shots = bm.few_shot_indices(TRAIN, [3, 6], n_shot=3, seed=5)
        for canonical, rows in shots.items():
            assert all(TRAIN.labels[row] == canonical for row in rows)

    def test_caps_at_the_available_examples(self):
        shots = bm.few_shot_indices(TRAIN, [1], n_shot=99, seed=13)
        assert len(shots[1]) == 3  # only three exist

    def test_is_deterministic_for_a_seed(self):
        a = bm.few_shot_indices(TRAIN, [2, 5], 2, seed=13)
        b = bm.few_shot_indices(TRAIN, [2, 5], 2, seed=13)
        assert a == b


class TestExpandToCanonical:
    def test_scatters_columns_to_their_canonical_positions(self):
        known = [1, 3, 4]
        local = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]], dtype=np.float32)
        out = bm.expand_to_canonical(local, known, n_canonical=6)

        assert out.shape == (2, 6)
        assert out[0, 1] == np.float32(0.7)
        assert out[0, 3] == np.float32(0.2)
        assert out[0, 4] == np.float32(0.1)

    def test_held_out_columns_are_exactly_zero(self):
        known = [1, 3, 4]
        local = np.array([[0.7, 0.2, 0.1]], dtype=np.float32)
        out = bm.expand_to_canonical(local, known, n_canonical=6)

        for held in (0, 2, 5):
            assert out[0, held] == 0.0

    def test_a_held_out_intent_can_never_be_the_argmax(self):
        known = [0, 1]
        local = np.array([[0.4, 0.6]], dtype=np.float32)
        out = bm.expand_to_canonical(local, known, n_canonical=4)
        assert int(out.argmax()) in known

    def test_row_mass_is_preserved(self):
        known = [2, 5, 7]
        local = np.array([[0.5, 0.3, 0.2]], dtype=np.float32)
        out = bm.expand_to_canonical(local, known, n_canonical=10)
        assert out.sum() == np.float32(1.0)
