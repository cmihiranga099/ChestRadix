"""Probability calibration for ChestRadix (pure NumPy and SciPy, no PyTorch).

The network is trained with a class-weighted loss, so its raw probabilities are usually
too high for rare findings. This module fits a correction on the **validation** logits
only and measures the effect on the test split with the expected calibration error (ECE)
and the Brier score.

Two methods are provided (``auto`` picks the one with the lower validation ECE):

* ``temperature``: one scalar ``T`` for all classes, ``p = sigmoid(z / T)`` (the method
  named in the project plan). It never changes the ranking, so AUROC is unchanged.
* ``platt``: one slope and intercept per class, ``p = sigmoid(a * z + b)``. The intercept
  can undo the shift caused by the positive-class weights, which a single ``T`` cannot.

Command line (after ``train.py`` has written the validation and test logits)::

    python -m src.calibration --run-dir runs/e1_densenet121_s42
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from src.config import PATHOLOGIES, locate_processed_dir
from src.tier_head import TIER_HEAD_NAMES, tier_targets

LOGGER = logging.getLogger("chestradix.calibration")
METHODS = ("temperature", "platt")
CHOICES = METHODS + ("auto",)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(logits, dtype=float)))


def bce_with_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean binary cross-entropy computed from logits (stable for large values)."""
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


# --------------------------------------------------------------------------- #
# Calibration error metrics
# --------------------------------------------------------------------------- #
def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray,
                               n_bins: int = 15) -> float:
    """Expected calibration error of one binary class with equal-width bins.

    ``ECE = sum_b (n_b / N) * |mean(y in bin b) - mean(p in bin b)|``.

    Args:
        labels: Binary labels, shape ``(N,)``.
        probabilities: Predicted probabilities of the positive class, shape ``(N,)``.
        n_bins: Number of bins on [0, 1].

    Returns:
        ECE in [0, 1]; NaN for empty input.
    """
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.size == 0:
        return float("nan")
    bins = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.sum() / labels.size * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(total)


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean squared error between probabilities and binary labels."""
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    return float(np.mean((probabilities - labels) ** 2)) if labels.size else float("nan")


def reliability_bins(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 15
                     ) -> dict[str, list[float]]:
    """Per-bin mean probability, observed frequency and count (pooled over all values).

    Args:
        labels: Array of binary labels (any shape).
        probabilities: Array of probabilities with the same shape.
        n_bins: Number of equal-width bins.

    Returns:
        Dictionary with lists ``confidence``, ``accuracy`` and ``count`` (empty bins skipped).
    """
    labels = np.asarray(labels, dtype=float).ravel()
    probabilities = np.asarray(probabilities, dtype=float).ravel()
    bins = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    out: dict[str, list[float]] = {"confidence": [], "accuracy": [], "count": []}
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            out["confidence"].append(float(probabilities[mask].mean()))
            out["accuracy"].append(float(labels[mask].mean()))
            out["count"].append(float(mask.sum()))
    return out


def calibration_summary(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 15,
                        class_names: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Per-class and macro ECE and Brier score.

    Args:
        labels: ``(N, C)`` binary labels.
        probabilities: ``(N, C)`` probabilities.
        n_bins: Number of ECE bins.
        class_names: Names of the ``C`` columns (defaults to the 14 pathologies).

    Returns:
        Dictionary with ``per_class`` (class, ece, brier) and ``macro_ece``, ``macro_brier``.
    """
    labels = np.asarray(labels)
    names = class_names or PATHOLOGIES
    rows = []
    for c in range(labels.shape[1]):
        name = names[c] if c < len(names) else str(c)
        rows.append({"class": name,
                     "ece": expected_calibration_error(labels[:, c], probabilities[:, c], n_bins),
                     "brier": brier_score(labels[:, c], probabilities[:, c])})
    return {"per_class": rows,
            "macro_ece": float(np.nanmean([r["ece"] for r in rows])),
            "macro_brier": float(np.nanmean([r["brier"] for r in rows]))}


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit one temperature ``T`` by minimising the BCE of ``sigmoid(logits / T)``.

    The search is over ``log T`` so that ``T`` stays positive.

    Args:
        logits: ``(N, C)`` validation logits.
        labels: ``(N, C)`` validation labels.

    Returns:
        The fitted temperature (> 0).
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=float)
    result = minimize_scalar(lambda t: bce_with_logits(logits / np.exp(t), labels),
                             bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(result.x))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide logits by the temperature (calibrated logits)."""
    return np.asarray(logits, dtype=float) / float(temperature)


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fit a slope ``a`` and intercept ``b`` per class on the validation data.

    Args:
        logits: ``(N, C)`` validation logits.
        labels: ``(N, C)`` validation labels.

    Returns:
        Array of shape ``(C, 2)`` with ``(a, b)`` per class; ``(1, 0)`` for a class that
        has only one label value in the validation data.
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=float)
    params = np.tile(np.array([1.0, 0.0]), (logits.shape[1], 1))
    for c in range(logits.shape[1]):
        if len(np.unique(labels[:, c])) < 2:
            continue
        result = minimize(lambda p, c=c: bce_with_logits(p[0] * logits[:, c] + p[1], labels[:, c]),
                          x0=np.array([1.0, 0.0]), method="L-BFGS-B")
        params[c] = result.x
    return params


def apply_platt(logits: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Apply per-class slope and intercept to logits."""
    logits = np.asarray(logits, dtype=float)
    return logits * params[None, :, 0] + params[None, :, 1]


def calibrate_logits(val_logits: np.ndarray, val_labels: np.ndarray, test_logits: np.ndarray,
                     method: str = "temperature") -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a calibration on the validation split and apply it to validation and test logits.

    Args:
        val_logits: ``(N_val, C)`` raw logits.
        val_labels: ``(N_val, C)`` binary labels.
        test_logits: ``(N_test, C)`` raw logits.
        method: ``"temperature"`` or ``"platt"``.

    Returns:
        ``(calibrated_val_logits, calibrated_test_logits, parameters)``.

    Raises:
        ValueError: If ``method`` is unknown.
    """
    if method == "temperature":
        temperature = fit_temperature(val_logits, val_labels)
        return (apply_temperature(val_logits, temperature),
                apply_temperature(test_logits, temperature), {"temperature": temperature})
    if method == "platt":
        params = fit_platt(val_logits, val_labels)
        return (apply_platt(val_logits, params), apply_platt(test_logits, params),
                {"slope_intercept": params.tolist()})
    raise ValueError(f"Unknown method '{method}'. Choose from {METHODS}.")


# --------------------------------------------------------------------------- #
# Run-level entry point
# --------------------------------------------------------------------------- #
def calibrate_run(run_dir: Path, processed_dir: Path, method: str = "auto",
                  n_bins: int = 15) -> dict[str, Any]:
    """Calibrate a finished run and write the calibrated logits and a JSON report.

    Reads ``val_logits.npy`` and ``test_logits.npy``; writes ``val_logits_calibrated.npy``,
    ``test_logits_calibrated.npy`` (for the chosen method) and ``calibration.json`` (ECE and
    Brier before and after for **both** methods, so the choice can be justified).

    Args:
        run_dir: Run folder.
        processed_dir: Folder with ``val.csv`` and ``test.csv`` (the image array is not needed).
        method: ``"temperature"``, ``"platt"`` or ``"auto"`` (the one with the lower macro ECE
            on the validation split). Its logits are saved for later steps.
        n_bins: Number of ECE bins.

    Returns:
        The report dictionary.
    """
    val = pd.read_csv(processed_dir / "val.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    val_logits = np.load(run_dir / "val_logits.npy").astype(float)
    test_logits = np.load(run_dir / "test_logits.npy").astype(float)
    val_labels = val[list(PATHOLOGIES)].to_numpy(dtype=int)
    test_labels = test[list(PATHOLOGIES)].to_numpy(dtype=int)
    if len(val_logits) != len(val) or len(test_logits) != len(test):
        raise ValueError("Saved logits do not match the split CSV files (different lengths).")

    fitted = {name: calibrate_logits(val_logits, val_labels, test_logits, name) for name in METHODS}
    summaries = {name: {"val": calibration_summary(val_labels, sigmoid(fitted[name][0]), n_bins),
                        "test": calibration_summary(test_labels, sigmoid(fitted[name][1]), n_bins)}
                 for name in METHODS}
    if method == "auto":   # decided on validation only
        method = min(METHODS, key=lambda name: summaries[name]["val"]["macro_ece"])
    elif method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {CHOICES}.")

    cal_val, cal_test, _ = fitted[method]
    np.save(run_dir / "val_logits_calibrated.npy", cal_val.astype(np.float32))
    np.save(run_dir / "test_logits_calibrated.npy", cal_test.astype(np.float32))
    report: dict[str, Any] = {
        "fitted_on": "validation", "chosen_method": method, "n_bins": n_bins,
        "raw": {"val": calibration_summary(val_labels, sigmoid(val_logits), n_bins),
                "test": calibration_summary(test_labels, sigmoid(test_logits), n_bins)},
        "methods": {name: {"parameters": fitted[name][2], **summaries[name]} for name in METHODS},
        "reliability_test": {
            "raw": reliability_bins(test_labels, sigmoid(test_logits), n_bins),
            "calibrated": reliability_bins(test_labels, sigmoid(cal_test), n_bins)},
    }
    if (run_dir / "val_tier_logits.npy").exists() and (run_dir / "test_tier_logits.npy").exists():
        report["tier_head"] = _calibrate_tier_head(run_dir, val_labels, test_labels, n_bins)
    with (run_dir / "calibration.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def _calibrate_tier_head(run_dir: Path, val_labels: np.ndarray, test_labels: np.ndarray,
                         n_bins: int) -> dict[str, Any]:
    """Per-output Platt calibration of the two tier-head logits (fitted on validation only).

    Writes ``val_tier_logits_calibrated.npy`` and ``test_tier_logits_calibrated.npy``.

    Returns:
        Report with the fitted ``(slope, intercept)`` per output and ECE/Brier before and after.
    """
    val_logits = np.load(run_dir / "val_tier_logits.npy").astype(float)
    test_logits = np.load(run_dir / "test_tier_logits.npy").astype(float)
    val_targets, test_targets = tier_targets(val_labels), tier_targets(test_labels)
    params = fit_platt(val_logits, val_targets)
    cal_val, cal_test = apply_platt(val_logits, params), apply_platt(test_logits, params)
    np.save(run_dir / "val_tier_logits_calibrated.npy", cal_val.astype(np.float32))
    np.save(run_dir / "test_tier_logits_calibrated.npy", cal_test.astype(np.float32))
    names = TIER_HEAD_NAMES
    return {
        "method": "platt", "outputs": list(names), "slope_intercept": params.tolist(),
        "raw_test": calibration_summary(test_targets, sigmoid(test_logits), n_bins, names),
        "calibrated_val": calibration_summary(val_targets, sigmoid(cal_val), n_bins, names),
        "calibrated_test": calibration_summary(test_targets, sigmoid(cal_test), n_bins, names),
    }


def plot_reliability(report: dict[str, Any], path: Path) -> None:
    """Save a reliability diagram (raw versus calibrated, test split) as a PNG.

    Args:
        report: Report returned by :func:`calibrate_run`.
        path: Output image path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=150)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect")
    for key, label in (("raw", "Raw"), ("calibrated", f"Calibrated ({report['chosen_method']})")):
        bins = report["reliability_test"][key]
        ax.plot(bins["confidence"], bins["accuracy"], marker="o", label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=None,
                        help="Folder with val.csv and test.csv.")
    parser.add_argument("--method", choices=CHOICES, default="auto")
    parser.add_argument("--n-bins", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    processed = args.processed_dir or locate_processed_dir()
    report = calibrate_run(args.run_dir, processed, args.method, args.n_bins)
    raw = report["raw"]["test"]
    LOGGER.info("Test macro ECE / Brier  raw: %.4f / %.4f", raw["macro_ece"], raw["macro_brier"])
    for name, entry in report["methods"].items():
        mark = "  <- saved" if name == report["chosen_method"] else ""
        LOGGER.info("Test macro ECE / Brier  %-11s: %.4f / %.4f  %s%s", name,
                    entry["test"]["macro_ece"], entry["test"]["macro_brier"],
                    entry["parameters"] if name == "temperature" else "", mark)
    if "tier_head" in report:
        tier = report["tier_head"]
        LOGGER.info("Tier head test macro ECE  raw: %.4f  platt: %.4f (saved)",
                    tier["raw_test"]["macro_ece"], tier["calibrated_test"]["macro_ece"])
    plot_reliability(report, args.run_dir / "reliability_test.png")
    LOGGER.info("Wrote %s", args.run_dir / "calibration.json")


if __name__ == "__main__":
    main()