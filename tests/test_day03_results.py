"""Loader guards. The shared-class-space invariant is the one that matters:
if the two score matrices ever disagreed on shape, the comparison would be
silently meaningless rather than loudly broken.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from convo.projects.day03_embed_router import results as res


@pytest.fixture(autouse=True)
def _clear_cache():
    res.load.cache_clear()
    yield
    res.load.cache_clear()


def _point_at(monkeypatch, tmp_path):
    monkeypatch.setattr(res, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(res, "COMPARISON_FILE", tmp_path / "comparison.npz")
    monkeypatch.setattr(res, "METADATA_FILE", tmp_path / "metadata.json")


def _write(tmp_path, router, classifier, class_names, held_out=(2,)):
    np.savez_compressed(
        tmp_path / "comparison.npz",
        router_scores=router,
        classifier_scores=classifier,
        labels=np.array([0, 1, 2, -1], dtype=np.int64),
        held_out=np.array(held_out, dtype=np.int64),
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps({"class_names": list(class_names), "router": {}, "classifier": {}}),
        encoding="utf-8",
    )


def test_missing_artifact_is_none_not_an_exception(monkeypatch, tmp_path):
    _point_at(monkeypatch, tmp_path)
    assert res.load() is None


def test_loads_a_well_formed_comparison(monkeypatch, tmp_path):
    _point_at(monkeypatch, tmp_path)
    scores = np.zeros((4, 3), dtype=np.float32)
    _write(tmp_path, scores, scores.copy(), ["a", "b", "c"])

    loaded = res.load()
    assert loaded is not None
    assert loaded.n_test == 4
    assert loaded.held_out == [2]
    assert loaded.held_out_names == ["c"]


def test_disagreeing_score_shapes_fail_loudly(monkeypatch, tmp_path):
    _point_at(monkeypatch, tmp_path)
    _write(
        tmp_path,
        np.zeros((4, 3), dtype=np.float32),
        np.zeros((4, 5), dtype=np.float32),
        ["a", "b", "c"],
    )
    with pytest.raises(ValueError, match="must share one class space"):
        res.load()


def test_a_class_count_mismatch_fails_loudly(monkeypatch, tmp_path):
    _point_at(monkeypatch, tmp_path)
    scores = np.zeros((4, 7), dtype=np.float32)
    _write(tmp_path, scores, scores.copy(), ["a", "b", "c"])
    with pytest.raises(ValueError, match="artifact mismatch"):
        res.load()
