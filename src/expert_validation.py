"""Validation against radiologist labels (Google expert labels for NIH ChestX-ray14, all 14 findings).

The 810 PA test images were read independently by five board-certified radiologists. The released
``test_labels.csv`` is the majority vote of three of them (the *ground-truth readers*); the other two
(*held-out readers*) serve as human comparison points. This module answers three questions on the
images that are also in our test split, using the model's saved predictions (no GPU, no retraining):

1. **Label noise.** How well do the original NIH labels agree with the radiologist labels
   (precision and recall of NIH labels per finding, and tier agreement)?
2. **Model quality against experts.** Per-finding AUROC and the triage metrics (weighted kappa,
   IMMEDIATE recall, under/over-triage, ranking AUROC) with the radiologist tier as reference,
   next to the same numbers with the NIH tier as reference on the same images. The tier thresholds
   are the ones tuned on the validation split by ``src.triage``; nothing is tuned on these images.
3. **Human comparison.** The two held-out radiologists and the model are each compared with the
   ground-truth tier, so the model's agreement can be read against the human level.

The tier of a case is the highest tier among its positive findings, exactly as for the NIH labels.
The mapping of findings to tiers is still our design assumption; the radiologists only supply
better findings, not the mapping.

Command line::

    python -m src.expert_validation --run-dir runs/ens_e1_e2_e4 --processed-dir data/processed \
        --expert-labels expert/test_labels.csv --readers expert/test_individual_readers.csv
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
from src.metrics import (
    patient_bootstrap_indices,
    safe_auroc,
    tier_metrics,
    tiers_from_binary,
    weighted_kappa,
)
from src.triage import _interval, _load_scores, assign_tiers, evaluate_tiers, ranking_metrics

LOGGER = logging.getLogger("chestradix.expert_validation")
GROUND_TRUTH_READERS: tuple[str, ...] = ("4343882785", "4343883593", "4343883996")
MIN_POSITIVES = 10


def _binary(frame: pd.DataFrame) -> np.ndarray:
    """``(N, 14)`` 0/1 matrix from the YES/NO finding columns (``Pleural Thickening`` spelling handled)."""
    columns = {name.replace(" ", "_"): name for name in frame.columns}
    missing = [name for name in PATHOLOGIES if name not in columns]
    if missing:
        raise ValueError(f"Expert label file is missing finding columns: {missing}")
    return np.stack([(frame[columns[name]].astype(str).str.upper() == "YES").to_numpy()
                     for name in PATHOLOGIES], axis=1).astype(int)


def _model_tiers(run_dir: Path, logits_name: str, rows: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Predicted tiers and scores of the matched images with the validation-tuned thresholds."""
    report_path = run_dir / "triage_metrics.json"
    if not report_path.exists():
        raise FileNotFoundError(f"{report_path} not found. Run src.triage for this run first.")
    triage = json.loads(report_path.read_text(encoding="utf-8"))
    primary = triage["primary_source"]
    tuning = triage["sources"][primary]["tuning"]
    scores = _load_scores(run_dir, logits_name)[primary][1][rows]
    predicted = assign_tiers(scores, tuning["t_urgent"], tuning["t_immediate"])
    probabilities = sigmoid(np.load(run_dir / f"test_{logits_name}.npy"))[rows]
    return predicted, scores, probabilities, {"source": primary, **{k: tuning[k] for k in
                                                                   ("t_urgent", "t_immediate")}}


def _per_finding(probabilities: np.ndarray, expert: np.ndarray, nih: np.ndarray,
                 draws: list[np.ndarray] | None) -> dict[str, Any]:
    """AUROC against expert and NIH labels, and how well the NIH labels agree with the experts."""
    rows: dict[str, Any] = {}
    eligible: list[int] = []
    for k, name in enumerate(PATHOLOGIES):
        positives = int(expert[:, k].sum())
        entry: dict[str, Any] = {"expert_positives": positives, "nih_positives": int(nih[:, k].sum())}
        tp = int(((nih[:, k] == 1) & (expert[:, k] == 1)).sum())
        entry["nih_precision_vs_expert"] = tp / nih[:, k].sum() if nih[:, k].sum() else float("nan")
        entry["nih_recall_vs_expert"] = tp / positives if positives else float("nan")
        if positives >= MIN_POSITIVES and (len(expert) - positives) >= MIN_POSITIVES:
            entry["auroc_expert"] = safe_auroc(expert[:, k], probabilities[:, k])
            entry["auroc_nih"] = safe_auroc(nih[:, k], probabilities[:, k])
            eligible.append(k)
        else:
            entry["note"] = f"fewer than {MIN_POSITIVES} expert positives, AUROC not reported"
        rows[name] = entry

    def macro(labels: np.ndarray, idx: np.ndarray) -> float:
        values = [safe_auroc(labels[idx, k], probabilities[idx, k]) for k in eligible]
        return float(np.nanmean(values)) if values else float("nan")

    everything = np.arange(len(expert))
    summary: dict[str, Any] = {"eligible_findings": [PATHOLOGIES[k] for k in eligible]}
    for label, labels in (("expert", expert), ("nih", nih)):
        summary[f"macro_auroc_{label}"] = macro(labels, everything)
        if draws:
            summary[f"macro_auroc_{label}_ci"] = _interval(
                np.array([macro(labels, idx) for idx in draws]))
    if draws:
        difference = np.array([macro(expert, idx) - macro(nih, idx) for idx in draws])
        summary["macro_auroc_difference_ci"] = _interval(difference)
    summary["macro_auroc_difference"] = summary["macro_auroc_expert"] - summary["macro_auroc_nih"]
    return {"per_finding": rows, "summary": summary}


def _reader_comparison(readers_path: Path, images: np.ndarray, reference: np.ndarray,
                       model_pred: np.ndarray) -> dict[str, Any]:
    """Each radiologist and the model against the ground-truth tier."""
    frame = pd.read_csv(readers_path, dtype={"Reader ID": str})
    frame = frame[frame["Image ID"].isin(set(images))]
    out: dict[str, Any] = {"model": tier_metrics(reference, model_pred)}
    for reader, part in frame.groupby("Reader ID"):
        part = part.drop_duplicates("Image ID").set_index("Image ID").reindex(images)
        if part["Atelectasis"].isna().any():
            continue
        tiers = tiers_from_binary(_binary(part))
        entry = tier_metrics(reference, tiers)
        entry["role"] = "ground_truth_reader" if reader in GROUND_TRUTH_READERS else "held_out_reader"
        entry["kappa_with_model"] = weighted_kappa(tiers, model_pred)
        out[reader] = entry
    return out


def expert_validation_run(run_dir: Path, processed_dir: Path, expert_labels: Path,
                          readers_path: Path | None = None, logits_name: str = "logits_calibrated",
                          n_boot: int = 1000, seed: int = 42) -> dict[str, Any]:
    """Run the validation against radiologist labels and write ``expert_validation.json``.

    Args:
        run_dir: Run (or ensemble) folder with ``test_<logits_name>.npy`` and ``triage_metrics.json``.
        processed_dir: Folder with ``test.csv`` (``image``, ``patient_id`` and the 14 labels).
        expert_labels: Google ``all_findings_expert_labels/test_labels.csv``.
        readers_path: Optional ``test_individual_readers.csv`` for the human comparison.
        logits_name: Logit file suffix.
        n_boot: Patient-level bootstrap draws (0 disables intervals).
        seed: Bootstrap seed.

    Returns:
        The report dictionary.

    Raises:
        ValueError: If no expert image is in the test split.
    """
    test = pd.read_csv(processed_dir / "test.csv")
    expert_frame = pd.read_csv(expert_labels).drop_duplicates("Image ID").set_index("Image ID")
    rows = np.flatnonzero(test["image"].isin(expert_frame.index).to_numpy())
    if len(rows) == 0:
        raise ValueError("None of the expert-labelled images is in the test split.")
    images = test["image"].to_numpy()[rows]
    expert = _binary(expert_frame.loc[images])
    nih = test[list(PATHOLOGIES)].to_numpy(dtype=int)[rows]
    patients = test["patient_id"].to_numpy()[rows]
    expert_ref, nih_ref = tiers_from_binary(expert), tiers_from_binary(nih)
    predicted, scores, probabilities, thresholds = _model_tiers(run_dir, logits_name, rows)
    draws = patient_bootstrap_indices(patients, n_boot, seed) if n_boot > 0 else None

    report: dict[str, Any] = {
        "run_dir": str(run_dir), "n_expert_images": int(len(expert_frame)),
        "n_matched_in_our_test_split": int(len(rows)),
        "n_patients": int(len(np.unique(patients))),
        "thresholds_from_validation": thresholds,
        "tier_prevalence_expert": {n: float(np.mean(expert_ref == i))
                                   for i, n in enumerate(("ROUTINE", "URGENT", "IMMEDIATE"))},
        "tier_prevalence_nih": {n: float(np.mean(nih_ref == i))
                                for i, n in enumerate(("ROUTINE", "URGENT", "IMMEDIATE"))},
        "nih_tier_vs_expert_tier": {
            "weighted_kappa": weighted_kappa(expert_ref, nih_ref),
            **{k: v for k, v in tier_metrics(expert_ref, nih_ref).items() if k != "weighted_kappa"}},
        "findings": _per_finding(probabilities, expert, nih, draws),
        "triage_vs_expert": evaluate_tiers(expert_ref, predicted, patients, n_boot, seed, draws=draws),
        "triage_vs_nih_same_images": evaluate_tiers(nih_ref, predicted, patients, n_boot, seed, draws=draws),
        "ranking_vs_expert": ranking_metrics(expert_ref, scores, draws),
        "ranking_vs_nih_same_images": ranking_metrics(nih_ref, scores, draws),
        "note": ("Findings are radiologist-labelled; the finding-to-tier mapping is still a design "
                 "assumption. 'nih_tier_vs_expert_tier' treats the NIH labels as a predictor of the "
                 "radiologist tier."),
    }
    if readers_path is not None:
        report["readers_vs_ground_truth"] = _reader_comparison(readers_path, images, expert_ref, predicted)
    with (run_dir / "expert_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expert-labels", type=Path, required=True)
    parser.add_argument("--readers", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--logits-name", default="logits_calibrated")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _ci(entry: dict[str, Any], name: str) -> str:
    low, high = entry[f"{name}_ci"]
    return f"{entry[name]:.3f} [{low:.3f}, {high:.3f}]"


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    processed = args.processed_dir or locate_processed_dir()
    report = expert_validation_run(args.run_dir, processed, args.expert_labels, args.readers,
                                   args.logits_name, args.n_boot, args.seed)
    LOGGER.info("Expert images matched in our test split: %d of %d (%d patients)",
                report["n_matched_in_our_test_split"], report["n_expert_images"], report["n_patients"])
    prevalence = report["tier_prevalence_expert"]
    LOGGER.info("Tier prevalence (expert): ROUTINE %.2f | URGENT %.2f | IMMEDIATE %.2f",
                prevalence["ROUTINE"], prevalence["URGENT"], prevalence["IMMEDIATE"])
    agreement = report["nih_tier_vs_expert_tier"]
    LOGGER.info("NIH-label tier vs radiologist tier: weighted kappa %.3f | IMMEDIATE recall %.3f",
                agreement["weighted_kappa"], agreement["immediate_recall"])
    summary = report["findings"]["summary"]
    low, high = summary.get("macro_auroc_expert_ci", (float("nan"),) * 2)
    LOGGER.info("Macro AUROC over %d findings: vs expert %.3f [%.3f, %.3f] | vs NIH labels %.3f",
                len(summary["eligible_findings"]), summary["macro_auroc_expert"], low, high,
                summary["macro_auroc_nih"])
    LOGGER.info("Per finding (AUROC expert | NIH | expert positives | NIH-label precision, recall vs expert):")
    for name, row in report["findings"]["per_finding"].items():
        if "auroc_expert" in row:
            LOGGER.info("  %-19s %.3f | %.3f | %3d | %.2f, %.2f", name, row["auroc_expert"],
                        row["auroc_nih"], row["expert_positives"], row["nih_precision_vs_expert"],
                        row["nih_recall_vs_expert"])
        else:
            LOGGER.info("  %-19s (too few expert positives: %d)", name, row["expert_positives"])
    for label, key in (("vs radiologists", "triage_vs_expert"), ("vs NIH labels", "triage_vs_nih_same_images")):
        entry = report[key]
        LOGGER.info("Triage %s: weighted kappa %s | IMMEDIATE recall %s | under-triage %.3f | over-triage %.3f",
                    label, _ci(entry, "weighted_kappa"), _ci(entry, "immediate_recall"),
                    entry["under_triage_rate"], entry["over_triage_rate"])
    for label, key in (("radiologists", "ranking_vs_expert"), ("NIH labels", "ranking_vs_nih_same_images")):
        entry = report[key]
        LOGGER.info("IMMEDIATE ranking AUROC vs %s: %s", label, _ci(entry, "immediate_auroc"))
    if "readers_vs_ground_truth" in report:
        LOGGER.info("Against the ground-truth tier (weighted kappa | IMMEDIATE recall | under | over):")
        for who, row in report["readers_vs_ground_truth"].items():
            role = row.get("role", "model")
            LOGGER.info("  %-12s %-19s %.3f | %.3f | %.3f | %.3f", who[-6:] if who != "model" else who,
                        role, row["weighted_kappa"], row["immediate_recall"], row["under_triage_rate"],
                        row["over_triage_rate"])
    LOGGER.info("Wrote %s", args.run_dir / "expert_validation.json")


if __name__ == "__main__":
    main()