"""Uncertainty flag from the disagreement of several trained models (no PyTorch needed).

Every member of the ensemble gives its own calibrated probabilities. Where the members agree the
prediction is stable; where they disagree it is unreliable. For each case:

* ``u_immediate`` = standard deviation across members of the IMMEDIATE score (the highest
  probability among the IMMEDIATE findings), the number that drives the triage decision;
* ``u_overall``   = mean over the 14 findings of the standard deviation across members.

A case is flagged "low confidence, needs a human look" when its uncertainty is above a cut-off
taken from the **validation** split (the ``1 - flag_rate`` quantile) and applied unchanged to the
test split. The report shows what the flag buys: the ranking quality (AUROC of the IMMEDIATE score
for ``tier == IMMEDIATE``) on the cases that are **not** flagged, versus the flagged ones and all
cases. If the flag works, the retained cases rank better than the flagged ones.

Command line (after every member has been calibrated)::

    python -m src.uncertainty --runs runs/e1_densenet121_s42 runs/e2_densenet121_tier_s42 \
        runs/e4_densenet121_320_s42 --out runs/ens_e1_e2_e4 --processed-dir data/processed
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
from src.config import PATHOLOGIES, locate_processed_dir
from src.evaluate import fast_auroc
from src.metrics import CLASS_TIERS, TIER_INDEX, patient_bootstrap_indices, tiers_from_binary

LOGGER = logging.getLogger("chestradix.uncertainty")
IMMEDIATE = TIER_INDEX["IMMEDIATE"]
FLAG_RATES: tuple[float, ...] = (0.10, 0.20, 0.30)


def member_probabilities(run_dirs: list[Path], split: str, logits_name: str = "logits_calibrated"
                         ) -> np.ndarray:
    """Stack the calibrated probabilities of all members.

    Args:
        run_dirs: Run folders (each with ``<split>_<logits_name>.npy``).
        split: ``"val"`` or ``"test"``.
        logits_name: Logit file suffix.

    Returns:
        Array of shape ``(M, N, 14)``.

    Raises:
        ValueError: If fewer than two members are given or their shapes differ.
    """
    if len(run_dirs) < 2:
        raise ValueError("Uncertainty from disagreement needs at least two members.")
    probs = [sigmoid(np.load(d / f"{split}_{logits_name}.npy")) for d in run_dirs]
    if len({p.shape for p in probs}) != 1:
        raise ValueError(f"Member outputs differ in shape: {[p.shape for p in probs]}")
    return np.stack(probs)


def immediate_scores(probabilities: np.ndarray) -> np.ndarray:
    """Highest probability among the IMMEDIATE findings, for any leading shape ``(..., 14)``."""
    return probabilities[..., CLASS_TIERS == IMMEDIATE].max(axis=-1)


def uncertainty_scores(members: np.ndarray) -> dict[str, np.ndarray]:
    """Disagreement between members.

    Args:
        members: ``(M, N, 14)`` probabilities.

    Returns:
        Dictionary with ``score`` (ensemble IMMEDIATE score from the mean probabilities),
        ``u_immediate`` and ``u_overall``, each of shape ``(N,)``.
    """
    return {
        "score": immediate_scores(members.mean(axis=0)),
        "u_immediate": immediate_scores(members).std(axis=0),
        "u_overall": members.std(axis=0).mean(axis=1),
    }


def selective_report(reference: np.ndarray, score: np.ndarray, uncertainty: np.ndarray,
                     cutoff: float, draws: list[np.ndarray] | None = None) -> dict[str, Any]:
    """What a flag with a given cut-off does to the IMMEDIATE ranking.

    Args:
        reference: Reference tiers.
        score: Ensemble IMMEDIATE score.
        uncertainty: Uncertainty per case.
        cutoff: Cases with ``uncertainty >= cutoff`` are flagged.
        draws: Patient-level bootstrap index arrays (optional, for intervals).

    Returns:
        Flag rate, IMMEDIATE prevalence and AUROC for flagged, retained and all cases.
    """
    truth = (np.asarray(reference).astype(int) == IMMEDIATE).astype(int)
    flagged = uncertainty >= cutoff

    def measures(idx: np.ndarray | slice) -> tuple[float, float, float]:
        t, s, f = truth[idx], score[idx], flagged[idx]
        keep = ~f
        auc_keep = fast_auroc(t[keep], s[keep]) if keep.any() else float("nan")
        auc_flag = fast_auroc(t[f], s[f]) if f.any() else float("nan")
        return float(f.mean()), auc_keep, auc_flag

    flag_rate, auc_keep, auc_flag = measures(slice(None))
    report: dict[str, Any] = {
        "cutoff": float(cutoff), "flag_rate": flag_rate,
        "immediate_prevalence_flagged": float(truth[flagged].mean()) if flagged.any() else float("nan"),
        "immediate_prevalence_retained": float(truth[~flagged].mean()) if (~flagged).any() else float("nan"),
        "auroc_all": fast_auroc(truth, score), "auroc_retained": auc_keep, "auroc_flagged": auc_flag,
    }
    if draws:
        samples = np.array([measures(idx) for idx in draws])
        for k, name in ((1, "auroc_retained"), (2, "auroc_flagged")):
            finite = samples[np.isfinite(samples[:, k]), k]
            report[f"{name}_ci"] = [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]
    return report


def uncertainty_run(run_dirs: list[Path], out_dir: Path, processed_dir: Path,
                    logits_name: str = "logits_calibrated", n_boot: int = 500, seed: int = 42,
                    flag_rates: tuple[float, ...] = FLAG_RATES) -> dict[str, Any]:
    """Compute the flag on validation/test, report what it buys and write ``uncertainty.json``.

    Args:
        run_dirs: Calibrated member runs.
        out_dir: Folder for ``uncertainty.json`` and the per-case uncertainty arrays.
        processed_dir: Folder with ``val.csv`` and ``test.csv``.
        logits_name: Logit file suffix of the members.
        n_boot: Patient-level bootstrap draws for intervals (0 disables).
        seed: Bootstrap seed.
        flag_rates: Target shares of cases to flag.

    Returns:
        The report dictionary.
    """
    val = pd.read_csv(processed_dir / "val.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    val_members = member_probabilities(run_dirs, "val", logits_name)
    test_members = member_probabilities(run_dirs, "test", logits_name)
    if val_members.shape[1] != len(val) or test_members.shape[1] != len(test):
        raise ValueError("Member outputs do not match the split CSV files (different lengths).")
    val_ref = tiers_from_binary(val[list(PATHOLOGIES)].to_numpy(dtype=int))
    test_ref = tiers_from_binary(test[list(PATHOLOGIES)].to_numpy(dtype=int))
    val_u, test_u = uncertainty_scores(val_members), uncertainty_scores(test_members)
    draws = (patient_bootstrap_indices(test["patient_id"].to_numpy(), n_boot, seed)
             if n_boot > 0 else None)

    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "members": [str(d) for d in run_dirs], "cutoffs_chosen_on": "validation",
        "reference": "NIH labels mapped to tiers (design assumption, not a radiologist)",
        "measures": {},
    }
    for name in ("u_immediate", "u_overall"):
        np.save(out_dir / f"val_{name}.npy", val_u[name].astype(np.float32))
        np.save(out_dir / f"test_{name}.npy", test_u[name].astype(np.float32))
        rows = []
        for rate in flag_rates:
            cutoff = float(np.quantile(val_u[name], 1.0 - rate))
            row = selective_report(test_ref, test_u["score"], test_u[name], cutoff, draws)
            row["target_flag_rate"] = rate
            rows.append(row)
        report["measures"][name] = rows
    with (out_dir / "uncertainty.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--logits-name", default="logits_calibrated")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    processed = args.processed_dir or locate_processed_dir()
    report = uncertainty_run(args.runs, args.out, processed, args.logits_name, args.n_boot, args.seed)
    for name, rows in report["measures"].items():
        LOGGER.info("=== %s: flag the most uncertain X%% of cases ===", name)
        LOGGER.info("  X%% | realised | IMMEDIATE AUROC retained [CI] | flagged | all | "
                    "IMMEDIATE share flagged vs retained")
        for row in rows:
            low, high = row.get("auroc_retained_ci", (float("nan"), float("nan")))
            LOGGER.info("  %3.0f | %.3f | %.3f [%.3f, %.3f] | %.3f | %.3f | %.3f vs %.3f",
                        100 * row["target_flag_rate"], row["flag_rate"], row["auroc_retained"],
                        low, high, row["auroc_flagged"], row["auroc_all"],
                        row["immediate_prevalence_flagged"], row["immediate_prevalence_retained"])
    LOGGER.info("Wrote %s", args.out / "uncertainty.json")


if __name__ == "__main__":
    main()