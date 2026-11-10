"""Unit tests for ``src.triage`` (Stage 5b)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calibration import sigmoid
from src.config import PATHOLOGIES
from src.metrics import tier_metrics, weighted_kappa
from src.triage import (
    IMMEDIATE,
    URGENT,
    _kappa_and_recall,
    assign_tiers,
    evaluate_tiers,
    tier_confusion,
    tier_scores,
    triage_run,
    tune_tier_thresholds,
)


def _probs(**values: float) -> np.ndarray:
    row = np.full((1, len(PATHOLOGIES)), 0.01)
    for name, value in values.items():
        row[0, PATHOLOGIES.index(name)] = value
    return row


def test_tier_scores_take_the_maximum_within_each_tier() -> None:
    scores = tier_scores(_probs(Pneumothorax=0.2, Edema=0.7, Effusion=0.4, Nodule=0.9))
    assert scores[0, IMMEDIATE] == pytest.approx(0.7)
    assert scores[0, URGENT] == pytest.approx(0.4)
    assert scores[0, 0] == pytest.approx(0.9)


def test_assign_tiers_priority_order() -> None:
    scores = np.array([[0.0, 0.9, 0.9],    # both high -> IMMEDIATE wins
                       [0.0, 0.9, 0.1],    # urgent only
                       [0.0, 0.1, 0.1]])   # neither
    assert assign_tiers(scores, 0.5, 0.5).tolist() == [2, 1, 0]


def test_confusion_and_fast_metrics_match_reference_implementations() -> None:
    rng = np.random.default_rng(0)
    reference = rng.integers(0, 3, 600)
    predicted = np.clip(reference + rng.integers(-1, 2, 600), 0, 2)
    confusion = tier_confusion(reference, predicted)
    assert confusion.sum() == 600 and confusion.shape == (3, 3)
    kappa, recall = _kappa_and_recall(confusion)
    assert kappa == pytest.approx(weighted_kappa(reference, predicted))
    assert recall == pytest.approx(tier_metrics(reference, predicted)["immediate_recall"])


def test_tune_finds_good_thresholds_on_separable_scores() -> None:
    rng = np.random.default_rng(1)
    reference = rng.choice(3, size=3000, p=[0.7, 0.2, 0.1])
    scores = np.zeros((3000, 3))
    scores[:, URGENT] = np.where(reference == URGENT, 0.7, 0.2) + rng.normal(0, 0.05, 3000)
    scores[:, IMMEDIATE] = np.where(reference == IMMEDIATE, 0.8, 0.1) + rng.normal(0, 0.05, 3000)
    tuning = tune_tier_thresholds(reference, scores, recall_floor=0.9)
    assert tuning["feasible"]
    assert tuning["val_immediate_recall"] >= 0.9
    assert tuning["val_weighted_kappa"] > 0.8


def test_tune_reports_infeasible_floor() -> None:
    rng = np.random.default_rng(2)
    reference = rng.integers(0, 3, 400)
    tuning = tune_tier_thresholds(reference, rng.random((400, 3)), recall_floor=1.1)
    assert tuning["feasible"] is False
    assert 0.0 <= tuning["t_immediate"] <= 1.0


def test_evaluate_tiers_intervals_and_trivial_rule() -> None:
    rng = np.random.default_rng(3)
    reference = rng.choice(3, size=1000, p=[0.6, 0.25, 0.15])
    predicted = np.where(rng.random(1000) < 0.8, reference, rng.integers(0, 3, 1000))
    patients = np.arange(1000) // 2
    report = evaluate_tiers(reference, predicted, patients, n_boot=200, seed=0)
    low, high = report["weighted_kappa_ci"]
    assert low <= report["weighted_kappa"] <= high
    assert np.array(report["confusion"]).sum() == 1000
    trivial = evaluate_tiers(reference, np.zeros_like(reference), patients, n_boot=0)
    assert trivial["under_triage_rate"] == pytest.approx(np.mean(reference > 0))
    assert trivial["immediate_flag_rate"] == 0.0
    assert trivial["weighted_kappa"] == pytest.approx(0.0, abs=1e-9)


def _make_run(tmp_path: Path) -> tuple[Path, Path]:
    """Synthetic run: informative but over-confident logits for all 14 findings."""
    rng = np.random.default_rng(4)
    processed, run = tmp_path / "processed", tmp_path / "run"
    processed.mkdir(), run.mkdir()
    for split, n in (("val", 3000), ("test", 3000)):
        labels = (rng.random((n, len(PATHOLOGIES))) < 0.06).astype(int)
        logits = (labels * 2.2 - 1.0) + rng.normal(0, 1.4, labels.shape) - 1.5
        frame = pd.DataFrame(labels, columns=list(PATHOLOGIES))
        frame.insert(0, "patient_id", np.arange(n) // 2)
        frame.to_csv(processed / f"{split}.csv", index=False)
        np.save(run / f"{split}_logits.npy", logits.astype(np.float32))
        np.save(run / f"{split}_logits_calibrated.npy", logits.astype(np.float32))
    return run, processed


def test_triage_run_end_to_end(tmp_path) -> None:
    run, processed = _make_run(tmp_path)
    report = triage_run(run, processed, "logits_calibrated", recall_floor=0.7, n_boot=100)
    assert (run / "triage_metrics.json").exists()
    assert json.loads((run / "triage_metrics.json").read_text())["thresholds_chosen_on"] == "validation"
    tuned, youden = report["tuned"], report["youden_per_class"]
    # the tuned rule must not flag more cases IMMEDIATE than the per-class Youden rule
    assert tuned["immediate_flag_rate"] <= youden["immediate_flag_rate"]
    assert tuned["over_triage_rate"] <= youden["over_triage_rate"]
    assert report["all_routine"]["weighted_kappa"] == pytest.approx(0.0, abs=1e-9)
    assert sum(report["tier_prevalence_test"].values()) == pytest.approx(1.0)
    sweep = report["floor_sweep"]
    assert [row["recall_floor"] for row in sweep] == [0.6, 0.7, 0.8, 0.9]
    # a higher safety floor cannot flag fewer cases IMMEDIATE on the validation-tuned rule
    assert sweep[-1]["immediate_flag_rate"] >= sweep[0]["immediate_flag_rate"]


def test_triage_run_requires_calibrated_logits(tmp_path) -> None:
    run, processed = _make_run(tmp_path)
    (run / "val_logits_calibrated.npy").unlink()
    with pytest.raises(FileNotFoundError):
        triage_run(run, processed, "logits_calibrated", n_boot=0)
    assert triage_run(run, processed, "logits", n_boot=0)["tuned"]["confusion"]