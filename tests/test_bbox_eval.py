"""Tests for the bounding-box localisation check."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bbox_eval import bbox_run, box_mask, centre_prior, load_bboxes, map_scores
from src.model import build_model, save_checkpoint
from src.config import TrainConfig


def test_box_mask_scales_1024_boxes() -> None:
    mask = box_mask(np.array([[512, 512, 256, 256]]), 224)
    assert mask.shape == (224, 224)
    assert mask[128:160, 128:160].all() and not mask[:100, :100].any()
    assert mask.mean() == pytest.approx(1 / 16, abs=0.01)


def test_map_scores_perfect_and_wrong_maps() -> None:
    mask = box_mask(np.array([[512, 512, 256, 256]]), 64)
    good = np.zeros((64, 64), dtype=np.float32)
    good[mask] = 1.0
    bad = np.zeros((64, 64), dtype=np.float32)
    bad[2, 2] = 1.0
    perfect = map_scores(good, mask)
    assert perfect["hit"] == 1 and perfect["iou"] == pytest.approx(1.0) and perfect["energy"] == pytest.approx(1.0)
    missed = map_scores(bad, mask)
    assert missed["hit"] == 0 and missed["iou"] == 0 and missed["energy"] == 0
    assert map_scores(np.zeros((64, 64), dtype=np.float32), mask)["energy"] == 0


def test_centre_prior_peaks_in_the_middle() -> None:
    prior = centre_prior(64)
    assert prior.max() == pytest.approx(1.0, abs=0.01)
    assert prior[32, 32] > prior[2, 2]


def test_load_bboxes_reads_raw_nih_layout_and_renames_infiltrate(tmp_path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text("Image Index,Finding Label,Bbox [x,y,w,h],,,,\n"
                   "a.png,Infiltrate,10.0,20.0,30.0,40.0,,,\n", encoding="utf-8")
    frame = load_bboxes(raw)
    assert list(frame.columns) == ["image", "finding", "x", "y", "w", "h"]
    assert frame.loc[0, "finding"] == "Infiltration" and frame.loc[0, "w"] == 30.0
    bad = tmp_path / "bad.csv"
    bad.write_text("image,finding,x,y,w,h\na.png,Nonsense,1,2,3,4\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_bboxes(bad)


def test_bbox_run_end_to_end_with_tiny_model(tmp_path) -> None:
    import torch

    torch.manual_seed(0)
    processed = tmp_path / "subset"
    processed.mkdir()
    n = 12
    images = (np.random.default_rng(0).random((n, 64, 64)) * 255).astype(np.uint8)
    np.save(processed / "images_64.npy", images)
    pd.DataFrame({"image": [f"{i}.png" for i in range(n)], "row": range(n)}).to_csv(
        processed / "image_index.csv", index=False)
    pd.DataFrame({"image": [f"{i}.png" for i in range(n)], "finding": ["Mass"] * n,
                  "x": 300, "y": 300, "w": 300, "h": 300}).to_csv(processed / "bbox.csv", index=False)
    model = build_model("densenet121", 14, 0.0, pretrained=False)
    save_checkpoint(tmp_path / "best.pt", model, TrainConfig(image_size=64, backbone="densenet121"))
    report = bbox_run([tmp_path / "best.pt"], processed, tmp_path / "out")
    assert report["n_pairs"] == n and (tmp_path / "out" / "bbox_localisation.json").exists()
    overall = report["overall"]
    assert 0 <= overall["cam"]["pointing_hit"] <= 1 and overall["centre"]["pointing_hit"] == 1.0
    assert "Mass" in report["per_finding"]