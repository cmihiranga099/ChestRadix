"""FastAPI service: ``GET /health``, ``GET /findings`` and ``POST /predict`` (multipart image).

Start (from the project root)::

    python -m uvicorn src.api:app --port 8000

Settings come from environment variables: ``TRIAGE_MEMBERS`` (comma-separated run folders,
default ``runs/e1_densenet121_s42,runs/e2_densenet121_tier_s42``), ``TRIAGE_ENSEMBLE``
(default ``runs/ens_e1_e2``), ``TRIAGE_FLAG_RATE`` (default 0.2) and ``TRIAGE_DEVICE``.
Uploads are processed in memory and never written to disk.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from src.config import PATHOLOGIES
from src.inference import DISCLAIMER, TriagePredictor

LOGGER = logging.getLogger("chestradix.api")
MAX_BYTES = 25 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 100_000_000


def _png_base64(array: np.ndarray) -> str:
    """Encode an image array as a base64 PNG string."""
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_app(predictor: TriagePredictor | None = None) -> FastAPI:
    """Build the application.

    Args:
        predictor: A ready predictor (used by tests). When ``None`` it is created on first use
            from the environment settings, so importing this module stays cheap.
    """
    app = FastAPI(title="Chest X-ray triage (research prototype)", version="1.0")
    state: dict[str, Any] = {"predictor": predictor}
    lock = threading.Lock()

    def get_predictor() -> TriagePredictor:
        with lock:
            if state["predictor"] is None:
                members = [Path(p) for p in os.environ.get(
                    "TRIAGE_MEMBERS", "runs/e1_densenet121_s42,runs/e2_densenet121_tier_s42").split(",")]
                state["predictor"] = TriagePredictor(
                    members, Path(os.environ.get("TRIAGE_ENSEMBLE", "runs/ens_e1_e2")),
                    float(os.environ.get("TRIAGE_FLAG_RATE", "0.2")), os.environ.get("TRIAGE_DEVICE", "cpu"))
            return state["predictor"]

    @app.get("/health")
    def health() -> dict[str, Any]:
        loaded = state["predictor"] is not None
        return {"status": "ok", "model_loaded": loaded, "disclaimer": DISCLAIMER}

    @app.get("/findings")
    def findings() -> dict[str, list[str]]:
        return {"findings": list(PATHOLOGIES)}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...),
                      finding: str | None = Query(None, description="Finding to explain with Grad-CAM"),
                      include_images: bool = Query(True)) -> dict[str, Any]:
        data = await file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "File is larger than 25 MB.")
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise HTTPException(400, "The file is not a readable image (PNG or JPEG expected).") from error
        if finding is not None and finding not in PATHOLOGIES:
            raise HTTPException(422, f"Unknown finding '{finding}'. Choose from {list(PATHOLOGIES)}.")
        try:
            predictor_ = get_predictor()
        except FileNotFoundError as error:
            raise HTTPException(503, f"Model files missing: {error}") from error
        with lock:
            result = predictor_.predict(image, finding)
        body = result.to_dict()
        if include_images:
            body["images"] = {"original_png_base64": _png_base64(result.gray),
                              "gradcam_overlay_png_base64": _png_base64(result.overlay)}
        return body

    return app


app = create_app()