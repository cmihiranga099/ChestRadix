"""Unit tests for ``src.evaluate`` (Stage 5)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.config import PATHOLOGIES
from src.evaluate import (
    batch_threshold_metrics,
    evaluate_predictions,
    evaluate_run,
    fast_auroc,
    report_to_frame,
    sigmoid,
)
from src.metrics import threshold_metrics

C = len(PATHOLOGIES)


def synthetic(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = (rng.random((n, C)) < 0.15).astype(int)
    labels[:, 0] = rng.integers(0, 2, n)
    scores = labels * 1.2 + rng.normal(0, 1.0, (n, C))   # informative logits
    patients = np.arange(n) // 3
    return labels, scores, patients


def test_sigmoid_is_stable_and_bounded() -> None:
    values = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert values.tolist() == [0.0, 0.5, 1.0]


def test_fast_auroc_matches_scikit_learn_with_ties() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.random(300), 1)  # many ties
    assert fast_auroc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert np.isnan(fast_auroc(np.zeros(5, dtype=int), np.arange(5)))


def test_batch_metrics_match_scalar_metrics() -> None:
    labels, scores, _ = synthetic()
    predicted = (scores > 0.5).astype(int)
    batch = batch_threshold_metrics(labels, predicted)
    for c in range(C):
        scalar = threshold_metrics(labels[:, c], predicted[:, c])
        for name in ("sensitivity", "specificity", "f1", "kappa", "mcc"):
            assert batch[name][c] == pytest.approx(scalar[name], nan_ok=True)


def test_evaluate_predictions_structure_and_intervals() -> None:
    labels, scores, patients = synthetic()
    prob = sigmoid(scores)
    report = evaluate_predictions(labels, prob, patients, np.full(C, 0.4), n_boot=100, seed=5)
    assert len(report["per_class"]) == C and report["n_patients"] == 200
    first = report["per_class"][0]
    low, high = first["auroc_ci"]
    assert low <= first["auroc"] <= high
    for name in ("kappa", "mcc", "f1"):
        assert first[f"{name}_ci"][0] <= first[f"{name}_ci"][1]
    assert report["macro"]["auroc_ci"][0] <= report["macro"]["auroc"] <= \
        report["macro"]["auroc_ci"][1]
    assert -1.0 <= report["tier"]["weighted_kappa"] <= 1.0
    assert "weighted_kappa_ci" in report["tier"]


def test_evaluation_is_reproducible() -> None:
    labels, scores, patients = synthetic()
    prob = sigmoid(scores)
    a = evaluate_predictions(labels, prob, patients, np.full(C, 0.5), n_boot=50, seed=9)
    b = evaluate_predictions(labels, prob, patients, np.full(C, 0.5), n_boot=50, seed=9)
    assert a == b


def test_no_bootstrap_gives_no_intervals() -> None:
    labels, scores, patients = synthetic(200)
    report = evaluate_predictions(labels, sigmoid(scores), patients, np.full(C, 0.5), n_boot=0)
    assert "auroc_ci" not in report["per_class"][0]
    assert "[" not in report_to_frame(report).loc[0, "auroc"]


def test_perfect_model_scores_one() -> None:
    labels, _, patients = synthetic(300)
    prob = np.where(labels == 1, 0.95, 0.05)
    report = evaluate_predictions(labels, prob, patients, np.full(C, 0.5), n_boot=20)
    assert report["macro"]["auroc"] == pytest.approx(1.0)
    assert report["macro"]["kappa"] == pytest.approx(1.0)
    assert report["tier"]["weighted_kappa"] == pytest.approx(1.0)
    assert report["tier"]["under_triage_rate"] == 0.0


def _write_split(root: Path, name: str, labels, patients) -> None:
    frame = pd.DataFrame(labels, columns=list(PATHOLOGIES))
    frame.insert(0, "patient_id", patients)
    frame.insert(0, "image", [f"{name}_{i}.png" for i in range(len(labels))])
    frame.to_csv(root / f"{name}.csv", index=False)


def test_evaluate_run_writes_reports(tmp_path: Path) -> None:
    data = tmp_path / "data"
    run = tmp_path / "run"
    data.mkdir(), run.mkdir()
    for name, seed in (("val", 1), ("test", 2)):
        labels, scores, patients = synthetic(240, seed)
        _write_split(data, name, labels, patients)
        np.save(run / f"{name}_logits.npy", scores.astype(np.float32))
    report = evaluate_run(run, data, n_boot=30, seed=3)
    assert report["thresholds_chosen_on"].startswith("validation")
    saved = json.loads((run / "test_metrics.json").read_text())
    assert saved["macro"]["auroc"] == pytest.approx(report["macro"]["auroc"])
    table = pd.read_csv(run / "test_table.csv")
    assert list(table["class"]) == list(PATHOLOGIES)


def test_evaluate_run_rejects_mismatched_lengths(tmp_path: Path) -> None:
    labels, scores, patients = synthetic(50)
    _write_split(tmp_path, "val", labels, patients)
    _write_split(tmp_path, "test", labels, patients)
    np.save(tmp_path / "val_logits.npy", scores[:40])
    np.save(tmp_path / "test_logits.npy", scores)
    with pytest.raises(ValueError):
        evaluate_run(tmp_path, tmp_path, n_boot=5)