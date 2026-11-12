"""Single-image inference: the full triage pipeline as one object (used by the API and the app).

For one radiograph the predictor

1. resizes it exactly as the training data was made (grey-scale, Lanczos, square),
2. runs every member model and averages the **raw logits** (the ensemble of Stage 6),
3. calibrates the averaged logits with the per-class calibration fitted on validation data,
4. assigns the tier (IMMEDIATE / URGENT / ROUTINE) with the thresholds tuned on validation,
5. measures **disagreement between the members** (each calibrated on its own) and flags the case
   "low confidence, needs a human look" when it is above the validation cut-off,
6. explains the finding that decided the tier with Grad-CAM (average over the members).

Everything it needs is read from run folders that the earlier stages already wrote:

* ``<member>/best.pt`` and ``<member>/calibration.json`` for each member,
* ``<ensemble>/calibration.json`` and ``<ensemble>/triage_metrics.json`` for the ensemble of the
  same members (``python -m src.ensemble``, ``src.calibration``, ``src.triage``),
* ``<ensemble>/val_u_overall.npy`` from ``python -m src.uncertainty`` run on the same members.

This is a research prototype. It does not check that an upload is a chest radiograph.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from src.calibration import sigmoid
from src.config import NUM_CLASSES, PATHOLOGIES
from src.dataset import build_transforms
from src.gradcam import gradcam_maps, overlay
from src.metrics import CLASS_TIERS
from src.tier_head import load_checkpoint_any
from src.triage import assign_tiers, tier_scores
from src.uncertainty import uncertainty_scores

LOGGER = logging.getLogger("chestradix.inference")
TIER_NAMES: tuple[str, ...] = ("ROUTINE", "URGENT", "IMMEDIATE")
DISCLAIMER = ("Research prototype for decision support only. Not a medical device and not validated "
              "for clinical use. A qualified clinician must review every case.")


@dataclass
class Prediction:
    """Result for one radiograph."""

    tier: str
    urgent_score: float
    immediate_score: float
    probabilities: dict[str, float]
    deciding_finding: str
    explained_finding: str
    uncertainty: float
    uncertainty_cutoff: float
    flagged: bool
    flag_rate: float
    member_immediate_scores: list[float]
    gray: np.ndarray = field(repr=False)
    cam: np.ndarray = field(repr=False)
    overlay: np.ndarray = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary (without the images)."""
        return {
            "tier": self.tier,
            "tier_scores": {"urgent": self.urgent_score, "immediate": self.immediate_score},
            "deciding_finding": self.deciding_finding,
            "explained_finding": self.explained_finding,
            "probabilities": self.probabilities,
            "uncertainty": {"value": self.uncertainty, "cutoff": self.uncertainty_cutoff,
                            "flagged": self.flagged, "target_flag_rate": self.flag_rate,
                            "member_immediate_scores": self.member_immediate_scores},
            "disclaimer": DISCLAIMER,
        }


def _calibration(run_dir: Path) -> Callable[[np.ndarray], np.ndarray]:
    """Calibration function of a run from its ``calibration.json`` (temperature or Platt)."""
    report = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    method = report["chosen_method"]
    parameters = report["methods"][method]["parameters"]
    if method == "platt":
        slope_intercept = np.asarray(parameters["slope_intercept"], dtype=float)
        return lambda logits: logits * slope_intercept[None, :, 0] + slope_intercept[None, :, 1]
    temperature = float(parameters["temperature"])
    return lambda logits: logits / temperature


class TriagePredictor:
    """Ensemble triage, uncertainty flag and Grad-CAM for single images."""

    def __init__(self, member_runs: list[Path], ensemble_run: Path, flag_rate: float = 0.2,
                 device: str = "cpu") -> None:
        """Load the members and the ensemble settings.

        Args:
            member_runs: Run folders of the members (each with ``best.pt`` and ``calibration.json``).
            ensemble_run: Run folder of their ensemble (calibration, triage thresholds and the
                validation uncertainty values).
            flag_rate: Share of cases to flag as low confidence (the cut-off is the ``1 - flag_rate``
                quantile of the validation uncertainty).
            device: ``"cpu"`` or ``"cuda"``.

        Raises:
            ValueError: If fewer than two members are given, their image sizes differ, or the
                ensemble was not triaged with the class-score rule.
        """
        if len(member_runs) < 2:
            raise ValueError("The uncertainty flag needs at least two members.")
        self.device = device
        self.models: list[torch.nn.Module] = []
        self.backbones: list[str] = []
        self.member_calibrations: list[Callable[[np.ndarray], np.ndarray]] = []
        sizes: set[int] = set()
        for run in map(Path, member_runs):
            model, meta = load_checkpoint_any(run / "best.pt", map_location=device)
            self.models.append(model.to(device).eval())
            self.backbones.append(meta["backbone"])
            self.member_calibrations.append(_calibration(run))
            sizes.add(int(meta["image_size"]))
        if len(sizes) != 1:
            raise ValueError(f"All members must use the same image size, got {sorted(sizes)}.")
        self.image_size = sizes.pop()
        self.transform = build_transforms(False, self.image_size)

        ensemble = Path(ensemble_run)
        self.ensemble_calibration = _calibration(ensemble)
        triage = json.loads((ensemble / "triage_metrics.json").read_text(encoding="utf-8"))
        if triage["primary_source"] != "classes":
            raise ValueError("The app supports the class-score triage rule (no tier head in the "
                             "ensemble). Use an ensemble of runs without a common tier head.")
        tuning = triage["sources"]["classes"]["tuning"]
        self.t_urgent, self.t_immediate = float(tuning["t_urgent"]), float(tuning["t_immediate"])
        reference = np.load(ensemble / "val_u_overall.npy")
        self.flag_rate = float(flag_rate)
        self.cutoff = float(np.quantile(reference, 1.0 - self.flag_rate))
        LOGGER.info("Loaded %d members at %d px; thresholds urgent %.3f immediate %.3f; "
                    "uncertainty cut-off %.4f (flag %.0f%%)", len(self.models), self.image_size,
                    self.t_urgent, self.t_immediate, self.cutoff, 100 * self.flag_rate)

    # ------------------------------------------------------------------ preprocessing
    def preprocess(self, image: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
        """Grey-scale, Lanczos resize to the training size, then the evaluation transform.

        Returns:
            The uint8 grey image ``(S, S)`` and the normalised tensor ``(1, 3, S, S)``.
        """
        gray = image.convert("L").resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        array = np.asarray(gray, dtype=np.uint8)
        tensor = self.transform(torch.from_numpy(array.copy()).unsqueeze(0))
        return array, tensor.unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------ prediction
    def predict(self, image: Image.Image, finding: str | None = None) -> Prediction:
        """Triage one radiograph.

        Args:
            image: The radiograph (any mode; converted to grey-scale).
            finding: Finding to explain with Grad-CAM. By default the finding that decided the tier.

        Returns:
            The :class:`Prediction`.

        Raises:
            ValueError: If ``finding`` is not one of the 14 findings.
        """
        if finding is not None and finding not in PATHOLOGIES:
            raise ValueError(f"Unknown finding '{finding}'. Choose from {list(PATHOLOGIES)}.")
        gray, batch = self.preprocess(image)
        with torch.no_grad():
            member_logits = np.stack([model(batch)[0, :NUM_CLASSES].cpu().numpy().astype(float)
                                      for model in self.models])
        probabilities = sigmoid(self.ensemble_calibration(member_logits.mean(axis=0)[None]))[0]
        scores = tier_scores(probabilities[None])
        tier = int(assign_tiers(scores, self.t_urgent, self.t_immediate)[0])

        member_probabilities = np.stack([sigmoid(calibrate(logits[None]))
                                         for calibrate, logits in zip(self.member_calibrations, member_logits)])
        spread = uncertainty_scores(member_probabilities)
        uncertainty = float(spread["u_overall"][0])
        member_immediate = member_probabilities[:, 0, CLASS_TIERS == 2].max(axis=1)

        candidates = np.where(CLASS_TIERS == tier, probabilities, -1.0) if tier > 0 else probabilities
        deciding = int(np.argmax(candidates))
        explained = PATHOLOGIES.index(finding) if finding else deciding
        cams = np.stack([gradcam_maps(model, batch, explained, backbone)[0]
                         for model, backbone in zip(self.models, self.backbones)]).mean(axis=0)
        span = cams.max() - cams.min()
        cam = ((cams - cams.min()) / span if span > 1e-12 else np.zeros_like(cams)).astype(np.float32)

        return Prediction(
            tier=TIER_NAMES[tier], urgent_score=float(scores[0, 1]), immediate_score=float(scores[0, 2]),
            probabilities={name: float(p) for name, p in zip(PATHOLOGIES, probabilities)},
            deciding_finding=PATHOLOGIES[deciding], explained_finding=PATHOLOGIES[explained],
            uncertainty=uncertainty, uncertainty_cutoff=self.cutoff, flagged=bool(uncertainty >= self.cutoff),
            flag_rate=self.flag_rate, member_immediate_scores=[float(v) for v in member_immediate],
            gray=gray, cam=cam, overlay=overlay(gray, cam))