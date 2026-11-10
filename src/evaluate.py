"""Evaluation of a trained model on the validation or test split.

Computes, for each of the 14 findings and as a macro average, AUROC, AUPRC, sensitivity,
specificity, F1-score, Cohen's kappa, MCC and balanced accuracy, plus tier-level metrics
(weighted kappa, IMMEDIATE recall, under-/over-triage). Confidence intervals come from a
patient-level bootstrap. Thresholds are chosen on the **validation** split only.

Command line (after ``train.py`` has written ``val_logits.npy`` and ``test_logits.npy``)::

    python -m src.evaluate --run-dir runs/e1_densenet121_s42
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.config import PATHOLOGIES, locate_processed_dir
from src.metrics import (
    choose_thresholds,
    patient_bootstrap_indices,
    safe_auprc,
    safe_auroc,
    threshold_metrics,
    tier_metrics,
    tiers_from_binary,
)

LOGGER = logging.getLogger("chestradix.evaluate")
CI_METRICS = ("auroc", "sensitivity", "specificity", "f1", "kappa", "mcc")
TIER_CI_METRICS = ("weighted_kappa", "immediate_recall", "under_triage_rate",
                   "over_triage_rate")


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 0.5 * (1.0 + np.tanh(0.5 * logits))


def fast_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUROC from ranks (Mann-Whitney U); NaN when one class is missing.

    Equivalent to ``sklearn.metrics.roc_auc_score`` but faster for the bootstrap.
    """
    positives = int(np.sum(y_true))
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[y_true == 1].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def batch_threshold_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorised threshold metrics for all classes at once.

    Args:
        labels: ``(N, C)`` binary reference labels.
        predicted: ``(N, C)`` binary predictions.

    Returns:
        Dictionary of ``(C,)`` arrays: sensitivity, specificity, f1, kappa, mcc.
    """
    truth = labels.astype(bool)
    pred = predicted.astype(bool)
    tp = (truth & pred).sum(0).astype(float)
    fp = (~truth & pred).sum(0).astype(float)
    fn = (truth & ~pred).sum(0).astype(float)
    tn = (~truth & ~pred).sum(0).astype(float)
    total = tp + fp + fn + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        f1 = 2 * tp / (2 * tp + fp + fn)
        po = (tp + tn) / total
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / total**2
        kappa = np.where(pe == 1.0, np.where(po == 1.0, 1.0, 0.0), (po - pe) / (1.0 - pe))
        denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        mcc = np.where(denom == 0.0, 0.0, (tp * tn - fp * fn) / np.sqrt(denom))
    return {"sensitivity": sens, "specificity": spec, "f1": f1, "kappa": kappa, "mcc": mcc}


def _interval(values: np.ndarray, alpha: float = 0.05) -> list[float]:
    """Percentile interval of the finite values (NaN pair when none)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(finite, 100 * alpha / 2)),
            float(np.percentile(finite, 100 * (1 - alpha / 2)))]


def evaluate_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    patient_ids: np.ndarray,
    thresholds: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate probabilities against labels with patient-level bootstrap intervals.

    Args:
        labels: ``(N, 14)`` binary labels.
        probabilities: ``(N, 14)`` predicted probabilities.
        patient_ids: ``(N,)`` patient id per image.
        thresholds: ``(14,)`` per-class thresholds (chosen on the validation split).
        n_boot: Number of bootstrap draws (0 disables intervals).
        seed: Bootstrap seed.

    Returns:
        Dictionary with ``per_class`` (list of dicts), ``macro``, ``tier`` and settings.
    """
    labels = np.asarray(labels).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    predicted = (probabilities >= thresholds[None, :]).astype(int)
    num_classes = labels.shape[1]

    # ---- point estimates ----
    per_class: list[dict[str, Any]] = []
    for c in range(num_classes):
        entry: dict[str, Any] = {
            "class": PATHOLOGIES[c],
            "positives": int(labels[:, c].sum()),
            "threshold": float(thresholds[c]),
            "auroc": safe_auroc(labels[:, c], probabilities[:, c]),
            "auprc": safe_auprc(labels[:, c], probabilities[:, c]),
        }
        entry.update(threshold_metrics(labels[:, c], predicted[:, c]))
        per_class.append(entry)

    ref_tiers = tiers_from_binary(labels)
    pred_tiers = tiers_from_binary(predicted)
    tier = tier_metrics(ref_tiers, pred_tiers)

    macro = {name: float(np.nanmean([row[name] for row in per_class]))
             for name in ("auroc", "auprc", "sensitivity", "specificity", "f1", "kappa",
                          "mcc", "balanced_accuracy")}

    # ---- bootstrap ----
    if n_boot > 0:
        draws = patient_bootstrap_indices(np.asarray(patient_ids), n_boot, seed)
        store = {name: np.full((n_boot, num_classes), np.nan) for name in CI_METRICS}
        tier_store = {name: np.full(n_boot, np.nan) for name in TIER_CI_METRICS}
        for b, idx in enumerate(draws):
            y, p, pr = labels[idx], probabilities[idx], predicted[idx]
            for c in range(num_classes):
                store["auroc"][b, c] = fast_auroc(y[:, c], p[:, c])
            batch = batch_threshold_metrics(y, pr)
            for name, values in batch.items():
                store[name][b] = values
            for name, value in tier_metrics(ref_tiers[idx], pred_tiers[idx]).items():
                tier_store[name][b] = value
        for c, entry in enumerate(per_class):
            for name in CI_METRICS:
                entry[f"{name}_ci"] = _interval(store[name][:, c])
        for name in CI_METRICS:
            with np.errstate(all="ignore"):
                macro[f"{name}_ci"] = _interval(np.nanmean(store[name], axis=1))
        for name in TIER_CI_METRICS:
            tier[f"{name}_ci"] = _interval(tier_store[name])

    return {
        "n_images": int(labels.shape[0]),
        "n_patients": int(len(np.unique(patient_ids))),
        "n_boot": n_boot,
        "seed": seed,
        "per_class": per_class,
        "macro": macro,
        "tier": tier,
    }


def report_to_frame(report: dict[str, Any]) -> pd.DataFrame:
    """Flatten ``per_class`` into a table with ``value [low, high]`` text for the thesis."""
    rows = []
    for entry in report["per_class"]:
        row: dict[str, Any] = {"class": entry["class"], "positives": entry["positives"],
                               "threshold": round(entry["threshold"], 4)}
        for name in ("auroc", "auprc", "sensitivity", "specificity", "f1", "kappa", "mcc",
                     "balanced_accuracy"):
            value = entry[name]
            low_high = entry.get(f"{name}_ci")
            row[name] = (f"{value:.3f} [{low_high[0]:.3f}, {low_high[1]:.3f}]"
                         if low_high is not None else f"{value:.3f}")
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Model inference and run evaluation
# --------------------------------------------------------------------------- #
def predict_logits(model, loader, device, amp: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader and return logits and labels.

    Args:
        model: Classifier returning ``(B, 14)`` logits.
        loader: DataLoader yielding ``(images, labels)``.
        device: Torch device.
        amp: Use mixed precision (CUDA only).

    Returns:
        Tuple ``(logits, labels)`` as float32 arrays.
    """
    import torch

    model.eval()
    outputs, targets = [], []
    use_amp = amp and torch.device(device).type == "cuda"
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=torch.device(device).type, enabled=use_amp):
                logits = model(images)
            outputs.append(logits.float().cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(outputs).astype(np.float32), np.concatenate(targets).astype(np.float32)


def evaluate_run(run_dir: Path, processed_dir: Path, n_boot: int = 1000, seed: int = 42,
                 logits_name: str = "logits") -> dict[str, Any]:
    """Evaluate a finished run from its saved logits and write the reports.

    Thresholds are chosen on the validation logits, then applied to the test logits.

    Args:
        run_dir: Run folder containing ``val_<logits_name>.npy`` and ``test_<logits_name>.npy``.
        processed_dir: Directory with ``val.csv`` and ``test.csv``.
        n_boot: Bootstrap draws.
        seed: Bootstrap seed.
        logits_name: ``"logits"`` (raw) or ``"logits_calibrated"`` once calibration exists.

    Returns:
        The test report dictionary (also written to ``test_metrics.json`` and ``test_table.csv``).
    """
    val = pd.read_csv(processed_dir / "val.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    val_prob = sigmoid(np.load(run_dir / f"val_{logits_name}.npy"))
    test_prob = sigmoid(np.load(run_dir / f"test_{logits_name}.npy"))
    val_labels = val[list(PATHOLOGIES)].to_numpy(dtype=int)
    test_labels = test[list(PATHOLOGIES)].to_numpy(dtype=int)
    if len(val_prob) != len(val) or len(test_prob) != len(test):
        raise ValueError("Saved logits do not match the split CSV files (different lengths).")

    thresholds = choose_thresholds(val_labels, val_prob)
    report = evaluate_predictions(test_labels, test_prob, test["patient_id"].to_numpy(),
                                  thresholds, n_boot=n_boot, seed=seed)
    report["thresholds_chosen_on"] = "validation (Youden's J)"
    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    report_to_frame(report).to_csv(run_dir / "test_table.csv", index=False)
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logits-name", default="logits")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    processed = args.processed_dir or locate_processed_dir()
    report = evaluate_run(args.run_dir, processed, args.n_boot, args.seed, args.logits_name)
    macro, tier = report["macro"], report["tier"]
    LOGGER.info("Macro AUROC %.4f  kappa %.4f  MCC %.4f", macro["auroc"], macro["kappa"],
                macro["mcc"])
    LOGGER.info("Tier weighted kappa %.4f  IMMEDIATE recall %.4f", tier["weighted_kappa"],
                tier["immediate_recall"])
    LOGGER.info("Wrote %s", args.run_dir / "test_metrics.json")


if __name__ == "__main__":
    main()