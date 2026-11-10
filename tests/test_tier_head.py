"""Tests for Stage 5c: tier head, tier calibration, ranking and workload analysis."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calibration import calibrate_run, sigmoid
from src.config import NUM_CLASSES, PATHOLOGIES, TrainConfig
from src.tier_head import (
    NUM_TIER_OUTPUTS,
    split_outputs,
    tier_targets,
)
from src.triage import (
    IMMEDIATE,
    URGENT,
    head_scores,
    ranking_metrics,
    triage_run,
    workload_sweep,
)


def _labels(*positives: str) -> np.ndarray:
    row = np.zeros((1, NUM_CLASSES), dtype=int)
    for name in positives:
        row[0, PATHOLOGIES.index(name)] = 1
    return row


def test_tier_targets_follow_the_highest_positive_tier() -> None:
    binary = np.vstack([_labels(), _labels("Nodule"), _labels("Effusion", "Nodule"),
                        _labels("Pneumothorax"), _labels("Edema", "Effusion")])
    assert tier_targets(binary).tolist() == [[0, 0], [0, 0], [1, 0], [1, 1], [1, 1]]


def test_tier_targets_torch_matches_numpy() -> None:
    torch = pytest.importorskip("torch")
    from src.tier_head import tier_targets_torch

    rng = np.random.default_rng(0)
    binary = (rng.random((200, NUM_CLASSES)) < 0.1).astype(np.float32)
    assert np.array_equal(tier_targets_torch(torch.from_numpy(binary)).numpy(),
                          tier_targets(binary))


def test_split_outputs_handles_both_layouts() -> None:
    fourteen, sixteen = np.zeros((5, 14)), np.arange(5 * 16).reshape(5, 16)
    assert split_outputs(fourteen)[1] is None
    findings, tier = split_outputs(sixteen)
    assert findings.shape == (5, 14) and tier.shape == (5, NUM_TIER_OUTPUTS)
    assert np.array_equal(tier, sixteen[:, 14:])


def test_head_scores_layout() -> None:
    scores = head_scores(np.array([[0.7, 0.2], [0.1, 0.05]]))
    assert scores[:, URGENT].tolist() == [0.7, 0.1]
    assert scores[:, IMMEDIATE].tolist() == [0.2, 0.05]


# ------------------------------------------------------------------ training smoke
def test_tier_head_training_writes_tier_logits_and_loads(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from src.model import load_checkpoint
    from src.tier_head import load_checkpoint_any
    from src.train import train_one
    from tests.test_train import SIZE, make_tiny_dir

    processed = tmp_path / "data"
    processed.mkdir()
    make_tiny_dir(processed)
    config = replace(TrainConfig(), image_size=SIZE, epochs=1, batch_size=4, num_workers=0,
                     amp=False, backbone="resnet50")
    run = tmp_path / "run"
    summary = train_one(config, processed, run, pretrained=False, max_batches=2, tier_head=True)

    assert summary["tier_head"] is True and "val_immediate_auroc" in summary
    assert np.load(run / "val_logits.npy").shape == (8, NUM_CLASSES)        # unchanged layout
    assert np.load(run / "test_logits.npy").shape == (8, NUM_CLASSES)
    assert np.load(run / "val_tier_logits.npy").shape == (8, NUM_TIER_OUTPUTS)
    assert json.loads((run / "config.json").read_text())["tier_head"] is True

    model, meta = load_checkpoint_any(run / "best.pt")
    assert meta["extra"]["num_outputs"] == NUM_CLASSES + NUM_TIER_OUTPUTS
    with pytest.raises(RuntimeError):                      # the old loader expects 14 outputs
        load_checkpoint(run / "best.pt")


def test_plain_training_is_unchanged_without_tier_head(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from src.tier_head import load_checkpoint_any
    from src.train import train_one
    from tests.test_train import SIZE, make_tiny_dir

    processed = tmp_path / "data"
    processed.mkdir()
    make_tiny_dir(processed)
    config = replace(TrainConfig(), image_size=SIZE, epochs=1, batch_size=4, num_workers=0,
                     amp=False, backbone="resnet50")
    run = tmp_path / "run"
    train_one(config, processed, run, pretrained=False, max_batches=2)
    assert not (run / "val_tier_logits.npy").exists()
    model, _ = load_checkpoint_any(run / "best.pt")
    assert model.fc[-1].out_features == NUM_CLASSES


# ------------------------------------------------------------------ synthetic run folder
def _make_run(tmp_path: Path, head_quality: float = 2.5) -> tuple[Path, Path]:
    """Run folder with 14 weak finding logits and 2 stronger tier-head logits."""
    rng = np.random.default_rng(7)
    processed, run = tmp_path / "processed", tmp_path / "run"
    processed.mkdir(), run.mkdir()
    for split, n in (("val", 4000), ("test", 4000)):
        labels = (rng.random((n, NUM_CLASSES)) < 0.06).astype(int)
        frame = pd.DataFrame(labels, columns=list(PATHOLOGIES))
        frame.insert(0, "patient_id", np.arange(n) // 2)
        frame.to_csv(processed / f"{split}.csv", index=False)
        findings = labels * 1.2 - 0.6 + rng.normal(0, 1.4, labels.shape) + 1.0   # shifted, weak
        tier = (tier_targets(labels) * head_quality - head_quality / 2
                + rng.normal(0, 1.0, (n, 2)) + 1.0)
        np.save(run / f"{split}_logits.npy", findings.astype(np.float32))
        np.save(run / f"{split}_tier_logits.npy", tier.astype(np.float32))
    return run, processed


def test_calibration_also_calibrates_the_tier_head(tmp_path) -> None:
    run, processed = _make_run(tmp_path)
    report = calibrate_run(run, processed)
    tier = report["tier_head"]
    assert tier["outputs"] == ["urgent_or_higher", "immediate"]
    assert tier["calibrated_test"]["macro_ece"] < tier["raw_test"]["macro_ece"] / 2
    for name in ("val_tier_logits_calibrated.npy", "test_tier_logits_calibrated.npy"):
        assert (run / name).exists()


def test_triage_uses_the_head_and_beats_weak_findings(tmp_path) -> None:
    run, processed = _make_run(tmp_path)
    calibrate_run(run, processed)
    report = triage_run(run, processed, "logits_calibrated", recall_floor=0.7, n_boot=60)
    assert set(report["sources"]) == {"classes", "head"} and report["primary_source"] == "head"
    head, classes = report["sources"]["head"], report["sources"]["classes"]
    # the head sees a stronger signal, so it must rank IMMEDIATE cases better
    assert head["ranking"]["immediate_auroc"] > classes["ranking"]["immediate_auroc"]
    low, high = head["ranking"]["immediate_auroc_ci"]
    assert low <= head["ranking"]["immediate_auroc"] <= high
    assert head["tuned"]["immediate_flag_rate"] <= classes["tuned"]["immediate_flag_rate"] + 1e-9
    assert report["tuned"] == head["tuned"]                      # top-level alias = primary source


def test_triage_without_a_head_falls_back_to_classes(tmp_path) -> None:
    run, processed = _make_run(tmp_path)
    for split in ("val", "test"):
        (run / f"{split}_tier_logits.npy").unlink()
    report = triage_run(run, processed, "logits", n_boot=0)
    assert list(report["sources"]) == ["classes"] and report["primary_source"] == "classes"


def test_workload_sweep_is_monotone_and_hand_checkable() -> None:
    reference = np.array([2] * 10 + [0] * 90)
    scores = np.zeros((100, 3))
    scores[:10, IMMEDIATE] = np.linspace(0.9, 0.5, 10)                  # all IMMEDIATE above 0.5
    scores[10:, IMMEDIATE] = np.linspace(0.45, 0.0, 90)                  # all others below
    rows = workload_sweep(reference, scores, reference, scores, caps=(0.10, 0.20, 0.50))
    assert [round(r["flag_rate"], 2) for r in rows] == [0.10, 0.20, 0.50]
    assert rows[0]["immediate_recall"] == pytest.approx(1.0)             # top 10% = the 10 cases
    assert rows[0]["precision"] == pytest.approx(1.0)
    assert rows[1]["precision"] == pytest.approx(0.5, abs=0.05)          # 10 true in 20 flagged
    assert rows[1]["lift"] == pytest.approx(5.0, abs=0.5)                # 0.5 / 0.10 prevalence
    recalls = [r["immediate_recall"] for r in rows]
    assert recalls == sorted(recalls)                                    # more workload, more recall


def test_ranking_metrics_perfect_and_random_scores() -> None:
    rng = np.random.default_rng(1)
    reference = rng.choice(3, size=2000, p=[0.7, 0.2, 0.1])
    perfect = np.zeros((2000, 3))
    perfect[:, IMMEDIATE] = (reference == IMMEDIATE) + rng.random(2000) * 0.01
    perfect[:, URGENT] = (reference >= URGENT) + rng.random(2000) * 0.01
    assert ranking_metrics(reference, perfect)["immediate_auroc"] == pytest.approx(1.0)
    noise = rng.random((2000, 3))
    assert ranking_metrics(reference, noise)["immediate_auroc"] == pytest.approx(0.5, abs=0.06)