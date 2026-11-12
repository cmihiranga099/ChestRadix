"""Tests for single-image inference and the API (tiny random models, no real data)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api import create_app
from src.config import PATHOLOGIES
from src.inference import TriagePredictor
from src.model import build_model, save_checkpoint
from src.train import TrainConfig


def _run(path: Path, seed: int) -> Path:
    import torch

    path.mkdir(parents=True)
    torch.manual_seed(seed)
    model = build_model("densenet121", 14, 0.0, pretrained=False)
    save_checkpoint(path / "best.pt", model, TrainConfig(image_size=64, backbone="densenet121"))
    (path / "calibration.json").write_text(json.dumps({
        "chosen_method": "temperature", "methods": {"temperature": {"parameters": {"temperature": 1.5}}}}))
    return path


@pytest.fixture(scope="module")
def predictor(tmp_path_factory) -> TriagePredictor:
    root = tmp_path_factory.mktemp("runs")
    members = [_run(root / "a", 0), _run(root / "b", 1)]
    ensemble = root / "ens"
    ensemble.mkdir()
    (ensemble / "calibration.json").write_text(json.dumps({
        "chosen_method": "platt", "methods": {"platt": {"parameters": {
            "slope_intercept": [[1.0, 0.0]] * 14}}}}))
    (ensemble / "triage_metrics.json").write_text(json.dumps({
        "primary_source": "classes",
        "sources": {"classes": {"tuning": {"t_urgent": 0.5, "t_immediate": 0.5}}}}))
    np.save(ensemble / "val_u_overall.npy", np.linspace(0, 0.1, 100))
    return TriagePredictor(members, ensemble, flag_rate=0.2)


def _png(size: int = 90) -> bytes:
    array = (np.random.default_rng(0).random((size, size)) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def test_predict_shapes_and_fields(predictor) -> None:
    result = predictor.predict(Image.open(io.BytesIO(_png())).convert("RGB"))
    assert result.tier in ("ROUTINE", "URGENT", "IMMEDIATE")
    assert set(result.probabilities) == set(PATHOLOGIES)
    assert result.cam.shape == (64, 64) and result.overlay.shape == (64, 64, 3)
    assert 0 <= result.cam.min() and result.cam.max() <= 1
    assert result.flagged == (result.uncertainty >= result.uncertainty_cutoff)


def test_explains_requested_finding_and_rejects_unknown(predictor) -> None:
    image = Image.open(io.BytesIO(_png()))
    assert predictor.predict(image, "Mass").explained_finding == "Mass"
    with pytest.raises(ValueError):
        predictor.predict(image, "Not_a_finding")


def test_needs_two_members(tmp_path) -> None:
    with pytest.raises(ValueError):
        TriagePredictor([tmp_path], tmp_path)


def test_api(predictor) -> None:
    client = TestClient(create_app(predictor))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/findings").json()["findings"] == list(PATHOLOGIES)
    reply = client.post("/predict", files={"file": ("x.png", _png(), "image/png")})
    assert reply.status_code == 200
    body = reply.json()
    assert body["tier"] in ("ROUTINE", "URGENT", "IMMEDIATE") and "images" in body
    assert "gradcam_overlay_png_base64" in body["images"]
    slim = client.post("/predict?include_images=false", files={"file": ("x.png", _png(), "image/png")})
    assert "images" not in slim.json()
    assert client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 400
    assert client.post("/predict?finding=Nope", files={"file": ("x.png", _png(), "image/png")}).status_code == 422