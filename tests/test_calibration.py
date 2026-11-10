"""Unit tests for ``src.calibration`` (Stage 5b)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.calibration import (
    METHODS,
    apply_platt,
    apply_temperature,
    bce_with_logits,
    brier_score,
    calibrate_logits,
    calibrate_run,
    calibration_summary,
    expected_calibration_error,
    fit_platt,
    fit_temperature,
    plot_reliability,
    reliability_bins,
    sigmoid,
)
from src.config import PATHOLOGIES


def test_ece_hand_computed_examples() -> None:
    # 80% confident and right 80% of the time: perfectly calibrated
    labels = np.array([1] * 8 + [0] * 2)
    assert expected_calibration_error(labels, np.full(10, 0.8)) == pytest.approx(0.0)
    # 90% confident but right 50% of the time: |0.5 - 0.9| = 0.4
    labels = np.array([1] * 5 + [0] * 5)
    assert expected_calibration_error(labels, np.full(10, 0.9)) == pytest.approx(0.4)
    assert np.isnan(expected_calibration_error(np.array([]), np.array([])))


def test_ece_handles_probability_one_and_zero() -> None:
    labels = np.array([1, 0])
    assert expected_calibration_error(labels, np.array([1.0, 0.0])) == pytest.approx(0.0)


def test_brier_score_hand_computed() -> None:
    assert brier_score(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(0.25)
    assert brier_score(np.array([0, 1]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_bce_is_stable_for_large_logits() -> None:
    value = bce_with_logits(np.array([800.0, -800.0]), np.array([1, 0]))
    assert np.isfinite(value) and value == pytest.approx(0.0, abs=1e-9)
    # matches the textbook formula on ordinary values
    z, y = np.array([0.3, -1.2]), np.array([1.0, 0.0])
    p = sigmoid(z)
    assert bce_with_logits(z, y) == pytest.approx(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def test_fit_temperature_recovers_known_value() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(0.0, 3.0, size=(20000, 3))
    labels = (rng.random(z.shape) < sigmoid(z / 2.0)).astype(int)   # truth: T = 2
    assert fit_temperature(z, labels) == pytest.approx(2.0, rel=0.1)


def test_temperature_keeps_ranking() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(size=(500, 1))
    y = (rng.random(500) < sigmoid(z[:, 0])).astype(int)
    scaled = apply_temperature(z, 3.7)
    assert roc_auc_score(y, scaled[:, 0]) == pytest.approx(roc_auc_score(y, z[:, 0]))


def test_platt_removes_a_constant_shift() -> None:
    rng = np.random.default_rng(2)
    z = rng.normal(0.0, 2.0, size=(20000, 2))
    labels = (rng.random(z.shape) < sigmoid(z - 2.0)).astype(int)   # model is 2 logits too high
    params = fit_platt(z, labels)
    assert params[:, 1] == pytest.approx([-2.0, -2.0], abs=0.15)
    assert params[:, 0] == pytest.approx([1.0, 1.0], abs=0.1)
    before = expected_calibration_error(labels[:, 0], sigmoid(z[:, 0]))
    after = expected_calibration_error(labels[:, 0], sigmoid(apply_platt(z, params)[:, 0]))
    assert after < before / 3


def test_platt_skips_class_without_both_labels() -> None:
    z = np.random.default_rng(3).normal(size=(50, 2))
    labels = np.stack([np.zeros(50), np.arange(50) % 2], axis=1)
    params = fit_platt(z, labels)
    assert params[0].tolist() == [1.0, 0.0]


def test_calibrate_logits_dispatch_and_error() -> None:
    rng = np.random.default_rng(4)
    z = rng.normal(size=(200, 3))
    y = (rng.random(z.shape) < 0.3).astype(int)
    for method in METHODS:
        cal_val, cal_test, params = calibrate_logits(z, y, z[:50], method)
        assert cal_val.shape == z.shape and cal_test.shape == (50, 3) and params
    with pytest.raises(ValueError):
        calibrate_logits(z, y, z, "isotonic")


def test_reliability_bins_counts_add_up() -> None:
    rng = np.random.default_rng(5)
    p = rng.random((300, 2))
    y = (rng.random((300, 2)) < p).astype(int)
    bins = reliability_bins(y, p, n_bins=10)
    assert sum(bins["count"]) == 600
    assert len(bins["confidence"]) == len(bins["accuracy"]) == len(bins["count"])


def _make_run(tmp_path: Path, shift: float = 2.0) -> tuple[Path, Path]:
    """Synthetic run folder: logits that are too confident and shifted upwards."""
    rng = np.random.default_rng(6)
    processed, run = tmp_path / "processed", tmp_path / "run"
    processed.mkdir(), run.mkdir()
    for split, n in (("val", 1500), ("test", 1500)):
        true_logit = rng.normal(-2.0, 1.5, size=(n, len(PATHOLOGIES)))
        labels = (rng.random(true_logit.shape) < sigmoid(true_logit)).astype(int)
        frame = pd.DataFrame(labels, columns=list(PATHOLOGIES))
        frame.insert(0, "patient_id", np.arange(n) // 2)
        frame.to_csv(processed / f"{split}.csv", index=False)
        np.save(run / f"{split}_logits.npy", (true_logit * 1.8 + shift).astype(np.float32))
    return run, processed


def test_calibrate_run_end_to_end(tmp_path) -> None:
    run, processed = _make_run(tmp_path)
    report = calibrate_run(run, processed, method="auto")
    for name in ("val_logits_calibrated.npy", "test_logits_calibrated.npy", "calibration.json"):
        assert (run / name).exists()
    raw = report["raw"]["test"]["macro_ece"]
    # the shift here cannot be fixed by one temperature, but per-class Platt removes it
    assert report["methods"]["platt"]["test"]["macro_ece"] < raw / 2
    assert report["chosen_method"] == "platt"          # auto picks the lower validation ECE
    assert report["methods"]["temperature"]["parameters"]["temperature"] > 0.0
    assert json.loads((run / "calibration.json").read_text())["fitted_on"] == "validation"
    plot_reliability(report, run / "reliability.png")
    assert (run / "reliability.png").stat().st_size > 0


def test_calibrate_run_rejects_length_mismatch(tmp_path) -> None:
    run, processed = _make_run(tmp_path)
    np.save(run / "val_logits.npy", np.zeros((10, len(PATHOLOGIES)), dtype=np.float32))
    with pytest.raises(ValueError):
        calibrate_run(run, processed)


def test_calibration_summary_shapes() -> None:
    y = np.zeros((20, 14), dtype=int)
    y[:5] = 1
    summary = calibration_summary(y, np.full((20, 14), 0.25))
    assert len(summary["per_class"]) == 14
    assert summary["macro_ece"] == pytest.approx(0.0)


def test_temperature_fixes_pure_overconfidence(tmp_path) -> None:
    run, processed = _make_run(tmp_path, shift=0.0)      # only the scale is wrong
    report = calibrate_run(run, processed, method="temperature")
    assert report["chosen_method"] == "temperature"
    assert report["methods"]["temperature"]["test"]["macro_ece"] < report["raw"]["test"]["macro_ece"]
    assert 1.5 < report["methods"]["temperature"]["parameters"]["temperature"] < 2.1


def test_calibrate_run_rejects_unknown_method(tmp_path) -> None:
    run, processed = _make_run(tmp_path)
    with pytest.raises(ValueError):
        calibrate_run(run, processed, method="isotonic")