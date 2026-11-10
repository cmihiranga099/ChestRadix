"""Smoke and unit tests for ``src.train`` (Stage 5). Uses tiny synthetic data."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.config import PATHOLOGIES, TrainConfig, array_name
from src.evaluate import evaluate_run
from src.model import load_checkpoint
from src.train import macro_auroc, make_scheduler, train_one

SIZE = 64


def make_tiny_dir(root: Path, per_split: int = 8) -> None:
    rng = np.random.default_rng(0)
    names: list[str] = []
    for split in ("train", "val", "test"):
        split_names = [f"{split}_{i}.png" for i in range(per_split)]
        labels = (rng.random((per_split, len(PATHOLOGIES))) < 0.4).astype(int)
        labels[:2] = 1   # guarantee positives and negatives in every column
        labels[2:4] = 0
        frame = pd.DataFrame(labels, columns=list(PATHOLOGIES))
        frame.insert(0, "patient_id", np.arange(per_split) + {"train": 0, "val": 100, "test": 200}[split])
        frame.insert(0, "image", split_names)
        frame.to_csv(root / f"{split}.csv", index=False)
        names += split_names
    np.save(root / array_name(SIZE), rng.integers(0, 256, (len(names), SIZE, SIZE), dtype=np.uint8))
    pd.DataFrame({"image": names, "row": np.arange(len(names))}).to_csv(
        root / "image_index.csv", index=False)


def test_macro_auroc_ignores_single_class_columns() -> None:
    labels = np.array([[0, 1], [1, 1], [0, 1], [1, 1]])
    logits = np.array([[0.1, 0.5], [0.9, 0.2], [0.2, 0.9], [0.8, 0.1]])
    assert macro_auroc(labels, logits) == pytest.approx(1.0)  # second column skipped


def test_scheduler_warms_up_then_decays() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = make_scheduler(optimizer, total_steps=100, warmup_steps=10)
    rates = []
    for _ in range(100):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[0] < rates[9] <= 1.0
    assert rates[10] == pytest.approx(1.0, abs=0.01)
    assert rates[-1] < 0.01


def test_smoke_training_writes_artefacts_and_evaluates(tmp_path: Path) -> None:
    data, run = tmp_path / "data", tmp_path / "run"
    data.mkdir()
    make_tiny_dir(data)
    config = replace(TrainConfig(), image_size=SIZE, batch_size=4, epochs=2, patience=2,
                     num_workers=0, amp=False, warmup_epochs=1)
    summary = train_one(config, data, run, pretrained=False, max_batches=2)

    assert summary["epochs_run"] >= 1 and (run / "best.pt").exists()
    for name in ("config.json", "history.json", "summary.json", "val_logits.npy",
                 "test_logits.npy"):
        assert (run / name).exists()
    assert np.load(run / "test_logits.npy").shape == (8, 14)
    assert len(json.loads((run / "history.json").read_text())) == summary["epochs_run"]

    model, meta = load_checkpoint(run / "best.pt")
    assert meta["backbone"] == "densenet121" and meta["extra"]["epoch"] == summary["best_epoch"]

    report = evaluate_run(run, data, n_boot=10, seed=1)
    assert report["n_images"] == 8 and "kappa" in report["macro"]


def test_training_is_reproducible(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    make_tiny_dir(data)
    config = replace(TrainConfig(), image_size=SIZE, batch_size=4, epochs=1, patience=2,
                     num_workers=0, amp=False)
    train_one(config, data, tmp_path / "a", pretrained=False, max_batches=2)
    train_one(config, data, tmp_path / "b", pretrained=False, max_batches=2)
    first = np.load(tmp_path / "a" / "val_logits.npy")
    second = np.load(tmp_path / "b" / "val_logits.npy")
    np.testing.assert_allclose(first, second, atol=1e-5)