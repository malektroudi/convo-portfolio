"""The loader has to behave when training has never been run — CI depends on it."""

from __future__ import annotations

import json

import numpy as np
import pytest

from convo.projects.day02_intent_nlu import results as res


@pytest.fixture(autouse=True)
def _clear_cache():
    res.load.cache_clear()
    yield
    res.load.cache_clear()


def _write(tmp_path, logits, labels, class_names):
    np.savez_compressed(tmp_path / "test_logits.npz", logits=logits, labels=labels)
    (tmp_path / "metadata.json").write_text(
        json.dumps({"model": "m", "dataset": "d", "class_names": class_names}), encoding="utf-8"
    )


def _point_at(monkeypatch, tmp_path):
    monkeypatch.setattr(res, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(res, "LOGITS_FILE", tmp_path / "test_logits.npz")
    monkeypatch.setattr(res, "METADATA_FILE", tmp_path / "metadata.json")


def test_missing_artifact_is_none_not_an_exception(monkeypatch, tmp_path):
    _point_at(monkeypatch, tmp_path)
    assert res.load() is None


def test_metadata_without_logits_is_still_none(monkeypatch, tmp_path):
    _point_at(monkeypatch, tmp_path)
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    assert res.load() is None


def test_loads_a_well_formed_artifact(monkeypatch, tmp_path):
    _point_at(monkeypatch, tmp_path)
    _write(
        tmp_path,
        np.zeros((4, 3), dtype=np.float32),
        np.array([0, 1, 2, -1], dtype=np.int64),
        ["a", "b", "c"],
    )
    loaded = res.load()
    assert loaded is not None
    assert loaded.n_test == 4
    assert loaded.class_names == ["a", "b", "c"]
    assert loaded.logits.shape == (4, 3)


def test_a_class_count_mismatch_fails_loudly(monkeypatch, tmp_path):
    _point_at(monkeypatch, tmp_path)
    _write(
        tmp_path,
        np.zeros((2, 5), dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
        ["a", "b", "c"],
    )
    with pytest.raises(ValueError, match="artifact mismatch"):
        res.load()
