"""Tests for the heat-map artefact check."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.artifact_check import CATEGORIES, artifact_check, classify_peak, format_report
from src.config import TrainConfig
from src.model import build_model, save_checkpoint


def _cam(y: int, x: int, size: int = 100) -> np.ndarray:
    cam = np.zeros((size, size), dtype=np.float32)
    cam[y, x] = 1.0
    return cam


def test_classify_peak_categories() -> None:
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[10:90, 10:90] = 1
    assert classify_peak(_cam(50, 3), mask) == "background"
    assert classify_peak(_cam(50, 50), mask) == "body"
    mask[:] = 1
    assert classify_peak(_cam(50, 3), mask) == "border"
    assert classify_peak(_cam(12, 12), mask) == "corner"
    assert classify_peak(_cam(50, 50), mask) == "body"


def test_artifact_check_end_to_end(tmp_path) -> None:
    torch.manual_seed(0)
    processed = tmp_path / "subset"
    processed.mkdir()
    n = 6
    images = np.zeros((n, 64, 64), dtype=np.uint8)
    images[:, 10:60, 10:54] = 120
    np.save(processed / "images_64.npy", images)
    names = [f"{i}.png" for i in range(n)]
    pd.DataFrame({"image": names, "row": range(n)}).to_csv(processed / "image_index.csv", index=False)
    pd.DataFrame({"image": names, "finding": ["Mass"] * n, "x": 300, "y": 300, "w": 300, "h": 300}).to_csv(
        processed / "bbox.csv", index=False)
    model = build_model("densenet121", 14, 0.0, pretrained=False)
    save_checkpoint(tmp_path / "best.pt", model, TrainConfig(image_size=64, backbone="densenet121"))
    out = tmp_path / "out" / "artifact_check.json"
    report = artifact_check([tmp_path / "best.pt"], processed, out)
    assert out.exists() and report["n_pairs"] == n
    for block in ("without_mask", "with_mask"):
        assert set(report[block]) == set(CATEGORIES) and abs(sum(report[block].values()) - 1) < 1e-9
    assert report["with_mask"]["background"] == 0.0   # the mask removes every background peak
    assert "no mask" in format_report(report)