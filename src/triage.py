"""Priority-tier decision rule for ChestRadix (pure NumPy and pandas, no PyTorch).

The detector gives 14 probabilities. Triage needs one decision per case: IMMEDIATE,
URGENT or ROUTINE. Taking the per-class Youden thresholds and then the highest positive
tier flags far too many cases (every false alarm on any of the 3 IMMEDIATE findings makes
the whole case IMMEDIATE). Here the decision is made at tier level instead:

* ``score_immediate`` = highest probability among the IMMEDIATE findings,
* ``score_urgent``    = highest probability among the URGENT findings,
* a case is IMMEDIATE if ``score_immediate >= t_immediate``, otherwise URGENT if
  ``score_urgent >= t_urgent``, otherwise ROUTINE.

The two thresholds are tuned on the **validation** split only: maximise the quadratic
weighted kappa subject to a minimum IMMEDIATE recall (safety floor). The test split is
used once, with patient-level bootstrap intervals. The reference tier comes from the NIH
labels through the design-assumption mapping, not from a radiologist.

Command line (after ``evaluate``/``calibration`` have produced logits)::

    python -m src.triage --run-dir runs/e1_densenet121_s42 --logits-name logits_calibrated
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.calibration import sigmoid
from src.config import PATHOLOGIES, PRIORITY_LEVELS, locate_processed_dir
from src.metrics import (
    CLASS_TIERS,
    TIER_INDEX,
    choose_thresholds,
    patient_bootstrap_indices,
    tier_metrics,
    tiers_from_binary,
)

LOGGER = logging.getLogger("chestradix.triage")
NUM_TIERS = len(PRIORITY_LEVELS)
URGENT, IMMEDIATE = TIER_INDEX["URGENT"], TIER_INDEX["IMMEDIATE"]
CI_NAMES = ("weighted_kappa", "immediate_recall", "under_triage_rate", "over_triage_rate",
            "immediate_flag_rate")


# --------------------------------------------------------------------------- #
# Scores and decision rule
# --------------------------------------------------------------------------- #
def tier_scores(probabilities: np.ndarray) -> np.ndarray:
    """Highest class probability within each tier.

    Args:
        probabilities: ``(N, 14)`` probabilities in the order of ``PATHOLOGIES``.

    Returns:
        ``(N, 3)`` array; column ``k`` is the tier ``k`` score (ROUTINE=0, URGENT=1,
        IMMEDIATE=2). The ROUTINE column is computed for completeness but not used.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    scores = np.zeros((probabilities.shape[0], NUM_TIERS))
    for tier in range(NUM_TIERS):
        scores[:, tier] = probabilities[:, CLASS_TIERS == tier].max(axis=1)
    return scores


def assign_tiers(scores: np.ndarray, t_urgent: float, t_immediate: float) -> np.ndarray:
    """Apply the two-threshold rule to tier scores.

    Args:
        scores: Output of :func:`tier_scores`.
        t_urgent: Threshold on the URGENT score.
        t_immediate: Threshold on the IMMEDIATE score.

    Returns:
        Integer tiers (0 ROUTINE, 1 URGENT, 2 IMMEDIATE).
    """
    tiers = np.zeros(scores.shape[0], dtype=int)
    tiers[scores[:, URGENT] >= t_urgent] = URGENT
    tiers[scores[:, IMMEDIATE] >= t_immediate] = IMMEDIATE
    return tiers


def tier_confusion(reference: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """3x3 count matrix, rows = reference tier, columns = predicted tier."""
    reference = np.asarray(reference).astype(int)
    predicted = np.asarray(predicted).astype(int)
    flat = np.bincount(reference * NUM_TIERS + predicted, minlength=NUM_TIERS**2)
    return flat.reshape(NUM_TIERS, NUM_TIERS)


def _kappa_and_recall(confusion: np.ndarray) -> tuple[float, float]:
    """Quadratic weighted kappa and IMMEDIATE recall from a confusion matrix."""
    total = confusion.sum()
    if total == 0:
        return float("nan"), float("nan")
    observed = confusion / total
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    distance = np.abs(np.subtract.outer(np.arange(NUM_TIERS), np.arange(NUM_TIERS)))
    weights = (distance / (NUM_TIERS - 1)) ** 2
    denominator = float((weights * expected).sum())
    kappa = (1.0 if float((weights * observed).sum()) == 0.0 else 0.0) if denominator == 0.0 \
        else 1.0 - float((weights * observed).sum()) / denominator
    positives = confusion[IMMEDIATE].sum()
    recall = float(confusion[IMMEDIATE, IMMEDIATE] / positives) if positives else float("nan")
    return float(kappa), recall


# --------------------------------------------------------------------------- #
# Threshold tuning (validation only)
# --------------------------------------------------------------------------- #
def tune_tier_thresholds(reference: np.ndarray, scores: np.ndarray, recall_floor: float = 0.80,
                         n_candidates: int = 60) -> dict[str, Any]:
    """Grid search for ``(t_urgent, t_immediate)`` on validation data.

    Maximises the quadratic weighted kappa among pairs whose IMMEDIATE recall is at least
    ``recall_floor``. If no pair reaches the floor, the pair with the highest IMMEDIATE
    recall is returned and ``feasible`` is False.

    Args:
        reference: Reference tiers (0, 1, 2) of the validation images.
        scores: ``(N, 3)`` tier scores of the validation images.
        recall_floor: Minimum IMMEDIATE recall (safety constraint).
        n_candidates: Number of candidate values per threshold (score quantiles).

    Returns:
        Dictionary with ``t_urgent``, ``t_immediate``, ``feasible``, ``recall_floor`` and
        the validation ``weighted_kappa`` and ``immediate_recall`` of the chosen pair.
    """
    reference = np.asarray(reference).astype(int)
    levels = np.linspace(0.30, 0.995, n_candidates)
    cand_urgent = np.unique(np.quantile(scores[:, URGENT], levels))
    cand_immediate = np.unique(np.quantile(scores[:, IMMEDIATE], levels))

    best: tuple[float, float, float, float] | None = None       # kappa, recall, t_u, t_i
    best_recall: tuple[float, float, float, float] | None = None
    for t_immediate in cand_immediate:
        for t_urgent in cand_urgent:
            predicted = assign_tiers(scores, t_urgent, t_immediate)
            kappa, recall = _kappa_and_recall(tier_confusion(reference, predicted))
            if not np.isfinite(kappa):
                continue
            if recall >= recall_floor and (best is None or kappa > best[0]):
                best = (kappa, recall, float(t_urgent), float(t_immediate))
            if best_recall is None or (recall, kappa) > (best_recall[1], best_recall[0]):
                best_recall = (kappa, recall, float(t_urgent), float(t_immediate))
    chosen = best if best is not None else best_recall
    if chosen is None:
        raise ValueError("Could not tune thresholds (no finite weighted kappa found).")
    return {"t_urgent": chosen[2], "t_immediate": chosen[3], "feasible": best is not None,
            "recall_floor": recall_floor, "val_weighted_kappa": chosen[0],
            "val_immediate_recall": chosen[1]}


# --------------------------------------------------------------------------- #
# Evaluation with patient-level bootstrap
# --------------------------------------------------------------------------- #
def _metrics(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Tier metrics plus the share of cases flagged IMMEDIATE (workload)."""
    values = tier_metrics(reference, predicted)
    values["immediate_flag_rate"] = float(np.mean(predicted == IMMEDIATE))
    return values


def _interval(values: np.ndarray, alpha: float = 0.05) -> list[float]:
    """Percentile interval of the finite values (NaN pair when none)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(finite, 100 * alpha / 2)),
            float(np.percentile(finite, 100 * (1 - alpha / 2)))]


def evaluate_tiers(reference: np.ndarray, predicted: np.ndarray, patient_ids: np.ndarray,
                   n_boot: int = 1000, seed: int = 42) -> dict[str, Any]:
    """Tier metrics, confusion matrix and bootstrap intervals for one decision rule.

    Args:
        reference: Reference tiers.
        predicted: Predicted tiers.
        patient_ids: Patient id per image (the bootstrap resamples patients).
        n_boot: Number of bootstrap draws (0 disables intervals).
        seed: Bootstrap seed.

    Returns:
        Dictionary of metrics, ``<name>_ci`` intervals and the ``confusion`` matrix
        (rows reference, columns predicted, order ROUTINE, URGENT, IMMEDIATE).
    """
    reference = np.asarray(reference).astype(int)
    predicted = np.asarray(predicted).astype(int)
    report: dict[str, Any] = _metrics(reference, predicted)
    report["confusion"] = tier_confusion(reference, predicted).tolist()
    if n_boot > 0:
        draws = patient_bootstrap_indices(np.asarray(patient_ids), n_boot, seed)
        store = {name: np.full(n_boot, np.nan) for name in CI_NAMES}
        for b, idx in enumerate(draws):
            for name, value in _metrics(reference[idx], predicted[idx]).items():
                store[name][b] = value
        for name in CI_NAMES:
            report[f"{name}_ci"] = _interval(store[name])
    return report


# --------------------------------------------------------------------------- #
# Run-level entry point
# --------------------------------------------------------------------------- #
def _floor_sweep(val_ref: np.ndarray, val_scores: np.ndarray, test_ref: np.ndarray,
                 test_scores: np.ndarray, patients: np.ndarray,
                 floors: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)) -> list[dict[str, Any]]:
    """Safety/workload trade-off: tune on validation for several IMMEDIATE-recall floors.

    Returns one row per floor with the test-split point estimates (no intervals).
    """
    rows = []
    for floor in floors:
        tuning = tune_tier_thresholds(val_ref, val_scores, floor)
        predicted = assign_tiers(test_scores, tuning["t_urgent"], tuning["t_immediate"])
        entry = evaluate_tiers(test_ref, predicted, patients, 0)
        rows.append({"recall_floor": floor, "feasible_on_val": tuning["feasible"],
                     "t_urgent": tuning["t_urgent"], "t_immediate": tuning["t_immediate"],
                     **{k: entry[k] for k in ("weighted_kappa", "immediate_recall",
                                              "under_triage_rate", "over_triage_rate",
                                              "immediate_flag_rate")}})
    return rows


def triage_run(run_dir: Path, processed_dir: Path, logits_name: str = "logits_calibrated",
               recall_floor: float = 0.80, n_boot: int = 1000, seed: int = 42) -> dict[str, Any]:
    """Tune the tier rule on validation, evaluate on test and write ``triage_metrics.json``.

    The report compares three decision rules on the same test images: the tuned
    tier-level rule (``tuned``), the per-class Youden rule used by ``evaluate.py``
    (``youden_per_class``) and the trivial rule that calls every case ROUTINE
    (``all_routine``).

    Args:
        run_dir: Run folder containing ``val_<name>.npy`` and ``test_<name>.npy``.
        processed_dir: Folder with ``val.csv`` and ``test.csv``.
        logits_name: ``"logits"`` (raw) or ``"logits_calibrated"``.
        recall_floor: Minimum IMMEDIATE recall on validation.
        n_boot: Bootstrap draws.
        seed: Bootstrap seed.

    Returns:
        The report dictionary.

    Raises:
        FileNotFoundError: If the logits file is missing (run ``calibration`` first).
    """
    for split in ("val", "test"):
        if not (run_dir / f"{split}_{logits_name}.npy").exists():
            raise FileNotFoundError(f"{run_dir}/{split}_{logits_name}.npy not found. "
                                    "Run src.calibration first, or pass --logits-name logits.")
    val = pd.read_csv(processed_dir / "val.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    val_prob = sigmoid(np.load(run_dir / f"val_{logits_name}.npy"))
    test_prob = sigmoid(np.load(run_dir / f"test_{logits_name}.npy"))
    val_labels = val[list(PATHOLOGIES)].to_numpy(dtype=int)
    test_labels = test[list(PATHOLOGIES)].to_numpy(dtype=int)
    if len(val_prob) != len(val) or len(test_prob) != len(test):
        raise ValueError("Saved logits do not match the split CSV files (different lengths).")

    val_ref, test_ref = tiers_from_binary(val_labels), tiers_from_binary(test_labels)
    tuning = tune_tier_thresholds(val_ref, tier_scores(val_prob), recall_floor)
    patients = test["patient_id"].to_numpy()

    tuned_pred = assign_tiers(tier_scores(test_prob), tuning["t_urgent"], tuning["t_immediate"])
    youden = choose_thresholds(val_labels, val_prob)
    youden_pred = tiers_from_binary((test_prob >= youden[None, :]).astype(int))

    report = {
        "logits_name": logits_name,
        "reference": "NIH labels mapped to tiers (design assumption, not a radiologist)",
        "thresholds_chosen_on": "validation",
        "tuning": tuning,
        "tier_prevalence_test": {name: float(np.mean(test_ref == i))
                                 for i, name in enumerate(PRIORITY_LEVELS)},
        "n_images": int(len(test)),
        "n_patients": int(len(np.unique(patients))),
        "tuned": evaluate_tiers(test_ref, tuned_pred, patients, n_boot, seed),
        "youden_per_class": evaluate_tiers(test_ref, youden_pred, patients, n_boot, seed),
        "all_routine": evaluate_tiers(test_ref, np.zeros_like(test_ref), patients, 0, seed),
        "floor_sweep": _floor_sweep(val_ref, tier_scores(val_prob), test_ref,
                                    tier_scores(test_prob), patients),
        "tier_order": list(PRIORITY_LEVELS),
    }
    with (run_dir / "triage_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=None,
                        help="Folder with val.csv and test.csv.")
    parser.add_argument("--logits-name", default="logits_calibrated")
    parser.add_argument("--recall-floor", type=float, default=0.80)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _ci_text(entry: dict[str, Any], name: str) -> str:
    """Format ``value [low, high]`` for the console."""
    low, high = entry.get(f"{name}_ci", (float("nan"), float("nan")))
    return f"{entry[name]:.3f} [{low:.3f}, {high:.3f}]"


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    processed = args.processed_dir or locate_processed_dir()
    report = triage_run(args.run_dir, processed, args.logits_name, args.recall_floor,
                        args.n_boot, args.seed)
    tuning = report["tuning"]
    LOGGER.info("Tuned on validation: t_urgent %.3f, t_immediate %.3f (floor met: %s)",
                tuning["t_urgent"], tuning["t_immediate"], tuning["feasible"])
    for rule in ("tuned", "youden_per_class"):
        entry = report[rule]
        LOGGER.info("[%s] weighted kappa %s | IMMEDIATE recall %s", rule,
                    _ci_text(entry, "weighted_kappa"), _ci_text(entry, "immediate_recall"))
        LOGGER.info("[%s] under-triage %s | over-triage %s | flagged IMMEDIATE %s", rule,
                    _ci_text(entry, "under_triage_rate"), _ci_text(entry, "over_triage_rate"),
                    _ci_text(entry, "immediate_flag_rate"))
    LOGGER.info("Floor sweep (test): floor | IMMEDIATE recall | over-triage | flagged IMMEDIATE")
    for row in report["floor_sweep"]:
        LOGGER.info("  %.2f | %.3f | %.3f | %.3f", row["recall_floor"], row["immediate_recall"],
                    row["over_triage_rate"], row["immediate_flag_rate"])
    LOGGER.info("Wrote %s", args.run_dir / "triage_metrics.json")


if __name__ == "__main__":
    main()