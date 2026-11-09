"""Evaluation metrics for ChestRadix (pure NumPy and scikit-learn, no PyTorch).

Detection metrics, chance-corrected agreement (Cohen's kappa, weighted kappa, MCC),
priority-tier helpers and a patient-level bootstrap for confidence intervals.

Convention: ``y_true`` is a binary vector, ``scores`` are probabilities (or any
monotone score), ``y_pred`` is a binary vector produced with a threshold.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import PATHOLOGIES, PRIORITY_LEVELS, PRIORITY_MAP


# --------------------------------------------------------------------------- #
# Confusion matrix and threshold metrics
# --------------------------------------------------------------------------- #
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(TP, FP, FN, TN)`` for two binary vectors.

    Args:
        y_true: Reference labels (0 or 1).
        y_pred: Predicted labels (0 or 1).

    Returns:
        Tuple of four integers.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    return tp, fp, fn, tn


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide, returning NaN when the denominator is zero."""
    return float(numerator / denominator) if denominator else float("nan")


def sensitivity(tp: int, fp: int, fn: int, tn: int) -> float:
    """Recall of the positive class: TP / (TP + FN)."""
    return _safe_divide(tp, tp + fn)


def specificity(tp: int, fp: int, fn: int, tn: int) -> float:
    """Recall of the negative class: TN / (TN + FP)."""
    return _safe_divide(tn, tn + fp)


def precision(tp: int, fp: int, fn: int, tn: int) -> float:
    """Positive predictive value: TP / (TP + FP)."""
    return _safe_divide(tp, tp + fp)


def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
    """F1-score: 2 TP / (2 TP + FP + FN)."""
    return _safe_divide(2 * tp, 2 * tp + fp + fn)


def balanced_accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    """Mean of sensitivity and specificity."""
    return float(np.nanmean([sensitivity(tp, fp, fn, tn), specificity(tp, fp, fn, tn)]))


def cohen_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    """Cohen's kappa for a 2x2 table: ``(po - pe) / (1 - pe)``.

    ``po`` is the observed agreement and ``pe`` the agreement expected by chance.

    Returns:
        Kappa in [-1, 1]; NaN if the table is empty, 0.0 if chance agreement is 1
        and the raters agree completely (degenerate case).
    """
    total = tp + fp + fn + tn
    if total == 0:
        return float("nan")
    po = (tp + tn) / total
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / total**2
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return float((po - pe) / (1.0 - pe))


def matthews_corrcoef(tp: int, fp: int, fn: int, tn: int) -> float:
    """Matthews correlation coefficient (0.0 when a margin is empty)."""
    denominator = float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    if denominator == 0.0:
        return 0.0
    return float((tp * tn - fp * fn) / np.sqrt(denominator))


def threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute all threshold-based metrics for one class.

    Args:
        y_true: Reference labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with sensitivity, specificity, precision, f1, balanced_accuracy,
        kappa and mcc.
    """
    counts = confusion_counts(y_true, y_pred)
    return {
        "sensitivity": sensitivity(*counts),
        "specificity": specificity(*counts),
        "precision": precision(*counts),
        "f1": f1_score(*counts),
        "balanced_accuracy": balanced_accuracy(*counts),
        "kappa": cohen_kappa(*counts),
        "mcc": matthews_corrcoef(*counts),
    }


# --------------------------------------------------------------------------- #
# Ranking metrics and thresholds
# --------------------------------------------------------------------------- #
def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUROC, NaN when only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def safe_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve), NaN if no positives."""
    if np.sum(y_true) == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Return the threshold that maximises Youden's J = sensitivity + specificity - 1.

    Args:
        y_true: Binary labels of the **validation** split.
        scores: Probabilities for the same images.

    Returns:
        The chosen threshold (0.5 when only one class is present).
    """
    from sklearn.metrics import roc_curve

    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    finite = np.isfinite(thresholds)
    best = np.argmax((tpr - fpr)[finite])
    return float(thresholds[finite][best])


def choose_thresholds(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Choose one Youden threshold per class on the validation data.

    Args:
        labels: ``(N, C)`` binary labels.
        probabilities: ``(N, C)`` probabilities.

    Returns:
        Array of ``C`` thresholds.
    """
    return np.array([youden_threshold(labels[:, c], probabilities[:, c])
                     for c in range(labels.shape[1])])


# --------------------------------------------------------------------------- #
# Priority tiers and weighted kappa
# --------------------------------------------------------------------------- #
TIER_INDEX: dict[str, int] = {name: i for i, name in enumerate(PRIORITY_LEVELS)}
# Tier number of each pathology column (ROUTINE=0, URGENT=1, IMMEDIATE=2).
CLASS_TIERS: np.ndarray = np.array([TIER_INDEX[PRIORITY_MAP[name]] for name in PATHOLOGIES])


def tiers_from_binary(binary: np.ndarray) -> np.ndarray:
    """Case tier = the highest tier among the positive classes (ROUTINE if none).

    Args:
        binary: ``(N, 14)`` binary matrix (labels or thresholded predictions).

    Returns:
        Integer array of shape ``(N,)`` with values 0 (ROUTINE), 1 (URGENT), 2 (IMMEDIATE).
    """
    binary = np.asarray(binary).astype(bool)
    masked = np.where(binary, CLASS_TIERS[None, :], 0)
    return masked.max(axis=1).astype(int)


def weighted_kappa(reference: np.ndarray, predicted: np.ndarray, num_levels: int = 3,
                   weighting: str = "quadratic") -> float:
    """Weighted Cohen's kappa for ordered levels.

    ``kappa_w = 1 - sum(w * O) / sum(w * E)`` with ``w_ij = (i - j)^2 / (k - 1)^2``
    (quadratic) or ``|i - j| / (k - 1)`` (linear). ``O`` holds observed proportions
    and ``E`` the proportions expected by chance from the two margins.

    Args:
        reference: Reference levels (integers 0 .. k-1).
        predicted: Predicted levels.
        num_levels: Number of ordered levels ``k``.
        weighting: ``"quadratic"`` or ``"linear"``.

    Returns:
        Weighted kappa; NaN for empty input, 1.0 when expected disagreement is 0
        and observed disagreement is 0.

    Raises:
        ValueError: If ``weighting`` is unknown.
    """
    reference = np.asarray(reference).astype(int)
    predicted = np.asarray(predicted).astype(int)
    if reference.size == 0:
        return float("nan")
    observed = np.zeros((num_levels, num_levels))
    np.add.at(observed, (reference, predicted), 1.0)
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    distance = np.abs(np.subtract.outer(np.arange(num_levels), np.arange(num_levels)))
    if weighting == "quadratic":
        weights = (distance / (num_levels - 1)) ** 2
    elif weighting == "linear":
        weights = distance / (num_levels - 1)
    else:
        raise ValueError(f"Unknown weighting '{weighting}'")
    denominator = float((weights * expected).sum())
    if denominator == 0.0:
        return 1.0 if float((weights * observed).sum()) == 0.0 else 0.0
    return float(1.0 - (weights * observed).sum() / denominator)


def tier_metrics(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Tier-level metrics: weighted kappa, IMMEDIATE recall, under- and over-triage.

    Args:
        reference: Reference tiers (0, 1, 2) derived from the labels.
        predicted: Predicted tiers.

    Returns:
        Dictionary with ``weighted_kappa``, ``immediate_recall``, ``under_triage_rate``
        and ``over_triage_rate``. Under-triage is the share of cases given a lower tier
        than the reference; over-triage a higher one.
    """
    reference = np.asarray(reference).astype(int)
    predicted = np.asarray(predicted).astype(int)
    immediate = reference == TIER_INDEX["IMMEDIATE"]
    return {
        "weighted_kappa": weighted_kappa(reference, predicted),
        "immediate_recall": _safe_divide(np.sum(immediate & (predicted == reference)),
                                         np.sum(immediate)),
        "under_triage_rate": float(np.mean(predicted < reference)),
        "over_triage_rate": float(np.mean(predicted > reference)),
    }


# --------------------------------------------------------------------------- #
# Patient-level bootstrap
# --------------------------------------------------------------------------- #
def patient_bootstrap_indices(patient_ids: np.ndarray, n_boot: int, seed: int
                              ) -> list[np.ndarray]:
    """Resample **patients** with replacement and return the image indices of each draw.

    Resampling patients (not images) respects that images of one patient are correlated.

    Args:
        patient_ids: Patient id of every image.
        n_boot: Number of bootstrap draws.
        seed: Random seed.

    Returns:
        List of ``n_boot`` index arrays into the image list.
    """
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(patient_ids, return_inverse=True)
    members: list[np.ndarray] = [np.flatnonzero(inverse == k) for k in range(len(unique))]
    draws: list[np.ndarray] = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(unique), size=len(unique))
        draws.append(np.concatenate([members[k] for k in chosen]))
    return draws


def bootstrap_interval(draws: list[np.ndarray], statistic: Callable[[np.ndarray], float],
                       alpha: float = 0.05) -> tuple[float, float]:
    """Percentile confidence interval of ``statistic`` over bootstrap draws.

    Args:
        draws: Index arrays from :func:`patient_bootstrap_indices`.
        statistic: Function mapping an index array to a number.
        alpha: Total tail probability (0.05 gives a 95% interval).

    Returns:
        ``(lower, upper)``; both NaN if no draw gave a finite value.
    """
    values = np.array([statistic(idx) for idx in draws], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return (float(np.percentile(values, 100 * alpha / 2)),
            float(np.percentile(values, 100 * (1 - alpha / 2))))