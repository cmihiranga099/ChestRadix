"""Tests for the validation against radiologist labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import NUM_CLASSES, PATHOLOGIES
from src.expert_validation import GROUND_TRUTH_READERS, expert_validation_run
from src.triage import triage_run


def _setup(tmp_path: Path, flip: float = 0.0):
    """Synthetic run: logits informative about the expert labels; NIH labels = expert + noise."""
    rng = np.random.default_rng(3)
    processed, run = tmp_path / "processed", tmp_path / "run"
    processed.mkdir(), run.mkdir()
    sizes = {"val": 3000, "test": 1200}
    for split, n in sizes.items():
        truth = (rng.random((n, NUM_CLASSES)) < 0.12).astype(int)
        nih = np.where(rng.random(truth.shape) < flip, 1 - truth, truth) if split == "test" else truth
        frame = pd.DataFrame(nih, columns=list(PATHOLOGIES))
        frame.insert(0, "patient_id", np.arange(n) // 2)
        frame.insert(0, "image", [f"{split}_{i:05d}.png" for i in range(n)])
        frame.to_csv(processed / f"{split}.csv", index=False)
        logits = truth * 2.2 - 1.1 + rng.normal(0, 1.0, truth.shape)
        np.save(run / f"{split}_logits_calibrated.npy", logits.astype(np.float32))
        if split == "test":
            test_truth = truth
    triage_run(run, processed, "logits_calibrated", n_boot=0)
    # expert file: first 300 test images (+ 20 images that are not in our split)
    names = [f"test_{i:05d}.png" for i in range(300)] + [f"other_{i}.png" for i in range(20)]
    values = np.vstack([test_truth[:300], np.zeros((20, NUM_CLASSES), dtype=int)])
    expert = pd.DataFrame(np.where(values == 1, "YES", "NO"),
                          columns=[c.replace("_", " ") for c in PATHOLOGIES])
    expert.insert(0, "Image ID", names)
    expert.insert(1, "Patient ID", np.arange(len(names)))
    expert.insert(2, "Abnormal", "NO")
    expert["Other"] = "NO"
    expert.to_csv(tmp_path / "test_labels.csv", index=False)
    return run, processed, tmp_path / "test_labels.csv", expert, test_truth


def _readers(tmp_path: Path, expert: pd.DataFrame) -> Path:
    parts = []
    for reader in (*GROUND_TRUTH_READERS, "111", "222"):
        part = expert.copy()
        part.insert(2, "Reader ID", reader)
        parts.append(part)
    path = tmp_path / "readers.csv"
    pd.concat(parts).to_csv(path, index=False)
    return path


def test_matching_and_agreement_when_nih_equals_expert(tmp_path) -> None:
    run, processed, labels, expert, _ = _setup(tmp_path, flip=0.0)
    report = expert_validation_run(run, processed, labels, n_boot=50)
    assert report["n_expert_images"] == 320 and report["n_matched_in_our_test_split"] == 300
    assert report["nih_tier_vs_expert_tier"]["weighted_kappa"] == pytest.approx(1.0)
    summary = report["findings"]["summary"]
    assert summary["macro_auroc_expert"] == pytest.approx(summary["macro_auroc_nih"])
    assert summary["macro_auroc_expert"] > 0.8
    assert (run / "expert_validation.json").exists()


def test_noisy_nih_labels_look_worse_than_expert_labels(tmp_path) -> None:
    run, processed, labels, _, _ = _setup(tmp_path, flip=0.15)
    report = expert_validation_run(run, processed, labels, n_boot=50)
    summary = report["findings"]["summary"]
    assert summary["macro_auroc_expert"] > summary["macro_auroc_nih"] + 0.05
    assert report["nih_tier_vs_expert_tier"]["weighted_kappa"] < 0.9
    assert (report["ranking_vs_expert"]["immediate_auroc"]
            > report["ranking_vs_nih_same_images"]["immediate_auroc"])


def test_reader_comparison_and_rare_findings_are_skipped(tmp_path) -> None:
    run, processed, labels, expert, _ = _setup(tmp_path)
    readers = _readers(tmp_path, expert)
    report = expert_validation_run(run, processed, labels, readers, n_boot=0)
    comparison = report["readers_vs_ground_truth"]
    assert comparison["111"]["role"] == "held_out_reader"
    assert comparison["111"]["weighted_kappa"] == pytest.approx(1.0)      # identical to ground truth
    assert comparison[GROUND_TRUTH_READERS[0]]["role"] == "ground_truth_reader"
    assert 0.0 < comparison["model"]["weighted_kappa"] < 1.0
    # a finding with fewer than 10 expert positives has no AUROC
    rare = json.loads(json.dumps(report["findings"]["per_finding"]))
    assert all(("auroc_expert" in row) == (row["expert_positives"] >= 10) for row in rare.values())


def test_errors_are_clear(tmp_path) -> None:
    run, processed, labels, expert, _ = _setup(tmp_path)
    (run / "triage_metrics.json").unlink()
    with pytest.raises(FileNotFoundError):
        expert_validation_run(run, processed, labels, n_boot=0)
    other = expert.copy()
    other["Image ID"] = [f"zzz_{i}.png" for i in range(len(other))]
    other.to_csv(tmp_path / "none.csv", index=False)
    triage_run(run, processed, "logits_calibrated", n_boot=0)
    with pytest.raises(ValueError):
        expert_validation_run(run, processed, tmp_path / "none.csv", n_boot=0)
        