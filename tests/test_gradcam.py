"""Tests for Grad-CAM maps, overlay and the faithfulness check."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from src.config import NUM_CLASSES, TrainConfig
from src.gradcam import deletion_scores, faithfulness_run, gradcam_maps, overlay


class Toy(nn.Module):
    """Channel 0 copies input channel 0; class 3 reads the average of that channel."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 2, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(2, NUM_CLASSES)
        with torch.no_grad():
            self.features[0].weight.zero_()
            self.features[0].bias.zero_()
            self.features[0].weight[0, 0, 1, 1] = 1.0
            self.head.weight.zero_()
            self.head.bias.fill_(-1.0)
            self.head.weight[3, 0] = 6.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).mean(dim=(2, 3)))


def _blob_batch() -> torch.Tensor:
    images = torch.zeros(2, 3, 32, 32)
    images[:, 0, 2:10, 2:10] = 1.0                    # bright blob, top left
    return images


def test_cam_is_scaled_and_peaks_on_the_evidence() -> None:
    model = Toy().eval()
    cams = gradcam_maps(model, _blob_batch(), 3, layer_path="features.1")
    assert cams.shape == (2, 32, 32) and cams.min() >= 0 and cams.max() <= 1 + 1e-6
    assert not np.isnan(cams).any()
    inside, outside = cams[:, 2:10, 2:10].mean(), cams[:, 16:, 16:].mean()
    assert inside > 5 * max(outside, 1e-3)


def test_cam_of_a_class_without_evidence_is_all_zero() -> None:
    model = Toy().eval()
    cams = gradcam_maps(model, _blob_batch(), 0, layer_path="features.1")   # class 0 has zero weights
    assert np.allclose(cams, 0)


def test_deleting_the_important_pixels_hurts_more_than_random() -> None:
    model = Toy().eval()
    images = _blob_batch()
    target = torch.tensor([3, 3])
    cams = gradcam_maps(model, images, target, layer_path="features.1")
    out = deletion_scores(model, images, target, cams, fraction=0.06)
    assert (out["drop_cam"] > out["drop_random"]).all()
    assert (out["drop_cam"] > 0).all()


def test_overlay_returns_a_uint8_rgb_image() -> None:
    gray = np.full((32, 32), 128, dtype=np.uint8)
    cam = np.zeros((32, 32), dtype=np.float32)
    cam[:8, :8] = 1.0
    image = overlay(gray, cam)
    assert image.shape == (32, 32, 3) and image.dtype == np.uint8
    assert tuple(image[20, 20]) == (128, 128, 128)            # untouched where the map is 0
    assert tuple(image[2, 2]) != (128, 128, 128)


def test_faithfulness_run_writes_report_and_figure(tmp_path: Path) -> None:
    from src.model import build_model, save_checkpoint
    from tests.test_train import SIZE, make_tiny_dir

    processed = tmp_path / "data"
    processed.mkdir()
    make_tiny_dir(processed)
    config = replace(TrainConfig(), image_size=SIZE, backbone="resnet50")
    torch.manual_seed(0)
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(checkpoint, build_model("resnet50", pretrained=False), config)

    report = faithfulness_run(checkpoint, processed, tmp_path / "out", n=8, examples=3,
                              batch_size=4)
    assert report["n_images"] == 6                     # 2 of the 8 tiny images have no finding
    low, high = report["mean_difference_ci"]
    assert low <= report["mean_difference"] <= high
    assert (tmp_path / "out" / "gradcam_examples.png").exists()
    assert json.loads((tmp_path / "out" / "gradcam_faithfulness.json").read_text())["split"] == "test"