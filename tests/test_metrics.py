"""Unit tests for ``src.metrics`` (Stage 5)."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score as sk_f1,
    matthews_corrcoef as sk_mcc,
)

from src.config import PATHOLOGIES
from src.metrics import (
    CLASS_TIERS,
    balanced_accuracy,
    bootstrap_interval,
    choose_thresholds,
    cohen_kappa,
    confusion_counts,
    f1_score,
    matthews_corrcoef,
    patient_bootstrap_indices,
    safe_auprc,
    safe_auroc,
    threshold_metrics,
    tier_metrics,
    tiers_from_binary,
    weighted_kappa,
    youden_threshold,
)


def _vectors(tp: int, fp: int, fn: int, tn: int) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([1] * tp + [0] * fp + [1] * fn + [0] * tn)
    y_pred = np.array([1] * tp + [1] * fp + [0] * fn + [0] * tn)
    return y_true, y_pred


def test_confusion_counts() -> None:
    y_true, y_pred = _vectors(40, 10, 20, 130)
    assert confusion_counts(y_true, y_pred) == (40, 10, 20, 130)


def test_kappa_hand_computed_example() -> None:
    # po = 170/200 = 0.85, pe = (50*60 + 150*140)/200^2 = 0.60 -> kappa = 0.625
    assert cohen_kappa(40, 10, 20, 130) == pytest.approx(0.625)


@pytest.mark.parametrize("counts", [(40, 10, 20, 130), (5, 30, 2, 400), (100, 0, 0, 100),
                                    (0, 20, 30, 50), (12, 7, 9, 3)])
def test_scalar_metrics_match_scikit_learn(counts) -> None:
    y_true, y_pred = _vectors(*counts)
    assert cohen_kappa(*counts) == pytest.approx(cohen_kappa_score(y_true, y_pred))
    assert matthews_corrcoef(*counts) == pytest.approx(sk_mcc(y_true, y_pred))
    assert balanced_accuracy(*counts) == pytest.approx(balanced_accuracy_score(y_true, y_pred))
    assert f1_score(*counts) == pytest.approx(sk_f1(y_true, y_pred))


def test_perfect_and_chance_agreement() -> None:
    assert cohen_kappa(50, 0, 0, 50) == pytest.approx(1.0)
    assert matthews_corrcoef(50, 0, 0, 50) == pytest.approx(1.0)
    assert cohen_kappa(25, 25, 25, 25) == pytest.approx(0.0)
    assert matthews_corrcoef(25, 25, 25, 25) == pytest.approx(0.0)


def test_degenerate_tables_do_not_crash() -> None:
    assert np.isnan(cohen_kappa(0, 0, 0, 0))
    assert matthews_corrcoef(0, 0, 10, 10) == 0.0  # model never predicts positive
    metrics = threshold_metrics(np.zeros(10), np.zeros(10))
    assert np.isnan(metrics["sensitivity"]) and metrics["specificity"] == 1.0


def test_weighted_kappa_matches_scikit_learn() -> None:
    rng = np.random.default_rng(0)
    reference = rng.integers(0, 3, 500)
    predicted = np.clip(reference + rng.integers(-1, 2, 500), 0, 2)
    for weighting in ("quadratic", "linear"):
        assert weighted_kappa(reference, predicted, weighting=weighting) == pytest.approx(
            cohen_kappa_score(reference, predicted, weights=weighting)
        )


def test_weighted_kappa_penalises_large_errors_more() -> None:
    reference = np.array([2] * 20 + [1] * 20 + [0] * 20)
    near = reference.copy()
    near[:5] = 1  # IMMEDIATE read as URGENT
    far = reference.copy()
    far[:5] = 0  # IMMEDIATE read as ROUTINE
    assert weighted_kappa(reference, far) < weighted_kappa(reference, near) < 1.0


def test_weighted_kappa_perfect_and_unknown_weighting() -> None:
    levels = np.array([0, 1, 2, 2, 1, 0])
    assert weighted_kappa(levels, levels) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        weighted_kappa(levels, levels, weighting="cubic")


def test_tiers_from_binary_uses_highest_positive_tier() -> None:
    binary = np.zeros((4, len(PATHOLOGIES)), dtype=int)
    binary[1, PATHOLOGIES.index("Pneumothorax")] = 1   # IMMEDIATE
    binary[2, PATHOLOGIES.index("Effusion")] = 1       # URGENT
    binary[2, PATHOLOGIES.index("Nodule")] = 1         # ROUTINE, lower
    binary[3, PATHOLOGIES.index("Hernia")] = 1         # ROUTINE
    assert tiers_from_binary(binary).tolist() == [0, 2, 1, 0]
    assert set(CLASS_TIERS.tolist()) == {0, 1, 2}


def test_tier_metrics_values() -> None:
    reference = np.array([2, 2, 2, 2, 1, 1, 0, 0])
    predicted = np.array([2, 2, 1, 0, 1, 2, 0, 0])
    result = tier_metrics(reference, predicted)
    assert result["immediate_recall"] == pytest.approx(0.5)
    assert result["under_triage_rate"] == pytest.approx(2 / 8)
    assert result["over_triage_rate"] == pytest.approx(1 / 8)
    assert -1.0 <= result["weighted_kappa"] <= 1.0


def test_ranking_metrics_and_nan_for_single_class() -> None:
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    assert safe_auroc(y, scores) == pytest.approx(0.75)
    assert 0.0 < safe_auprc(y, scores) <= 1.0
    assert np.isnan(safe_auroc(np.zeros(4), scores))
    assert np.isnan(safe_auprc(np.zeros(4), scores))


def test_youden_threshold_separates_clean_data() -> None:
    y = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    threshold = youden_threshold(y, scores)
    assert 0.4 <= threshold <= 0.6
    assert youden_threshold(np.zeros(5), np.linspace(0, 1, 5)) == 0.5
    labels = np.stack([y, y], axis=1)
    probabilities = np.stack([scores, scores], axis=1)
    assert choose_thresholds(labels, probabilities).shape == (2,)


def test_patient_bootstrap_is_deterministic_and_keeps_patients_together() -> None:
    patient_ids = np.array([1, 1, 1, 2, 2, 3, 4, 4, 4, 4])
    first = patient_bootstrap_indices(patient_ids, 20, seed=7)
    second = patient_bootstrap_indices(patient_ids, 20, seed=7)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    sizes = {1: 3, 2: 2, 3: 1, 4: 4}
    for draw in first:
        # each drawn patient contributes all of his or her images
        for pid, size in sizes.items():
            assert np.sum(patient_ids[draw] == pid) % size == 0


def test_bootstrap_interval_contains_point_estimate() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    scores = y * 0.6 + rng.random(400) * 0.7
    patients = np.arange(400) // 2
    draws = patient_bootstrap_indices(patients, 200, seed=3)
    low, high = bootstrap_interval(draws, lambda idx: safe_auroc(y[idx], scores[idx]))
    point = safe_auroc(y, scores)
    assert low <= point <= high
    assert high - low < 0.2