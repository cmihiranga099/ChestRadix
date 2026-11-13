"""Tests for the heat-map methods and the comparison on boxes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.cam_compare import compare_run, format_table, parse_config
from src.cam_methods import METHODS, cam_maps, focused_overlay
from src.config import TrainConfig
from src.model import build_model, save_checkpoint


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return build_model("densenet121", 14, 0.0, pretrained=False).eval()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("layer", [None, "features.denseblock3"])
def test_cam_maps_shape_range_and_layer(tiny_model, method, layer) -> None:
    batch = torch.randn(2, 3, 64, 64)
    cams = cam_maps(tiny_model, batch, torch.tensor([3, 5]), "densenet121", method, layer)
    assert cams.shape == (2, 64, 64) and cams.dtype == np.float32
    assert cams.min() >= 0.0 and cams.max() <= 1.0 and cams.max() > 0.0


def test_unknown_method_and_config_are_rejected(tiny_model) -> None:
    with pytest.raises(ValueError):
        cam_maps(tiny_model, torch.randn(1, 3, 64, 64), 0, "densenet121", "nope")
    with pytest.raises(ValueError):
        parse_config("nope:features.denseblock3")
    assert parse_config("layercam:features.denseblock3") == ("layercam", "features.denseblock3")
    assert parse_config("gradcam") == ("gradcam", None)


def test_focused_overlay_hides_weak_regions() -> None:
    gray = np.full((8, 8), 100, dtype=np.uint8)
    cam = np.zeros((8, 8), dtype=np.float32)
    cam[:, :4], cam[0, 0] = 0.3, 1.0
    whole, focused = focused_overlay(gray, cam, level=0.0), focused_overlay(gray, cam, level=0.5)
    assert focused.shape == (8, 8, 3) and focused.dtype == np.uint8
    assert (focused[4, 2] == focused[4, 6]).all()            # weak area (0.3) is untouched
    assert not (whole[4, 2] == whole[4, 6]).all()            # but visible without the level
    with pytest.raises(ValueError):
        focused_overlay(gray, cam, level=1.0)


def test_compare_run_end_to_end(tmp_path) -> None:
    torch.manual_seed(0)
    processed = tmp_path / "subset"
    processed.mkdir()
    n = 10
    images = (np.random.default_rng(0).random((n, 64, 64)) * 255).astype(np.uint8)
    np.save(processed / "images_64.npy", images)
    pd.DataFrame({"image": [f"{i}.png" for i in range(n)], "row": range(n)}).to_csv(
        processed / "image_index.csv", index=False)
    pd.DataFrame({"image": [f"{i}.png" for i in range(n)], "finding": ["Mass"] * n,
                  "x": 300, "y": 300, "w": 300, "h": 300}).to_csv(processed / "bbox.csv", index=False)
    model = build_model("densenet121", 14, 0.0, pretrained=False)
    save_checkpoint(tmp_path / "best.pt", model, TrainConfig(image_size=64, backbone="densenet121"))
    configs = ["gradcam", "layercam:features.denseblock3"]
    report = compare_run([tmp_path / "best.pt"], processed, tmp_path / "out", configs)
    assert set(report) == set(configs) and (tmp_path / "out" / "cam_compare.json").exists()
    for summary in report.values():
        assert summary["n_pairs"] == n and 0 <= summary["overall"]["cam"]["pointing_hit"] <= 1
    assert "layercam:features.denseblock3" in format_table(report)