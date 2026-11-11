"""Tests for the logit-averaging ensemble."""
from __future__ import annotations

import numpy as np
import pytest

from src.ensemble import ensemble_runs


def _run(path, value, tier=True):
    path.mkdir()
    for split in ("val", "test"):
        np.save(path / f"{split}_logits.npy", np.full((4, 14), value, dtype=np.float32))
        if tier:
            np.save(path / f"{split}_tier_logits.npy", np.full((4, 2), value, dtype=np.float32))


def test_ensemble_averages_logits_and_keeps_tier_only_when_all_have_it(tmp_path) -> None:
    _run(tmp_path / "a", 1.0)
    _run(tmp_path / "b", 3.0, tier=False)
    report = ensemble_runs([tmp_path / "a", tmp_path / "b"], tmp_path / "out")
    assert np.allclose(np.load(tmp_path / "out" / "test_logits.npy"), 2.0)
    assert "val_tier_logits.npy" not in report["written"]


def test_ensemble_rejects_one_run_and_shape_mismatch(tmp_path) -> None:
    _run(tmp_path / "a", 1.0)
    with pytest.raises(ValueError):
        ensemble_runs([tmp_path / "a"], tmp_path / "o1")
    _run(tmp_path / "b", 1.0)
    np.save(tmp_path / "b" / "val_logits.npy", np.zeros((5, 14), dtype=np.float32))
    with pytest.raises(ValueError):
        ensemble_runs([tmp_path / "a", tmp_path / "b"], tmp_path / "o2")