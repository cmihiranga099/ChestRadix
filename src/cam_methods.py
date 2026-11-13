"""Sharper heat maps: more Grad-CAM variants and a layer choice, plus a focused overlay.

The default Grad-CAM (:func:`src.gradcam.gradcam_maps`) reads the last DenseNet layer
(``features.norm5``), which is only 7 x 7 at 224 px, so a map is a few blurry blobs. This module
adds, without changing the existing code:

* ``gradcam``   the original method, on any layer,
* ``gradcampp`` Grad-CAM++ (Chattopadhay et al.): better weights when several regions are involved,
* ``layercam``  LayerCAM (Jiang et al.): weights every location by its own gradient, which works
  well on earlier, higher-resolution layers (``features.denseblock3`` is 14 x 14,
  ``features.denseblock2`` is 28 x 28),

and :func:`focused_overlay`, which draws only the strongest part of a map so that a weak, wide
red area is not shown.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.gradcam import gradcam_maps, overlay
from src.model import activation_and_gradient

METHODS: tuple[str, ...] = ("gradcam", "gradcampp", "layercam")


def _finish(cam: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """Upsample, min-max scale each map to [0, 1] (a flat map gives zeros) and return NumPy."""
    cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)[:, 0]
    low = cam.amin(dim=(1, 2), keepdim=True)
    span = cam.amax(dim=(1, 2), keepdim=True) - low
    cam = torch.where(span > 1e-12, (cam - low) / span.clamp_min(1e-12), torch.zeros_like(cam))
    return cam.cpu().numpy().astype(np.float32)


def cam_maps(model: nn.Module, images: torch.Tensor, class_index: int | torch.Tensor,
             backbone: str = "densenet121", method: str = "gradcam",
             layer_path: str | None = None) -> np.ndarray:
    """Heat maps for one finding per image with a chosen method and layer.

    Args:
        model: Model in evaluation mode.
        images: Normalised batch ``(B, 3, H, W)``.
        class_index: Finding index (one for all images, or one per image) whose logit is explained.
        backbone: Backbone name (selects the default layer).
        method: ``"gradcam"``, ``"gradcampp"`` or ``"layercam"``.
        layer_path: Target layer, e.g. ``features.denseblock3`` (default: the last layer).

    Returns:
        Array ``(B, H, W)`` in [0, 1].

    Raises:
        ValueError: If ``method`` is unknown.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(METHODS)}.")
    if method == "gradcam":
        return gradcam_maps(model, images, class_index, backbone, layer_path)
    _, activation, gradient = activation_and_gradient(model, images, class_index, backbone, layer_path)
    if method == "layercam":
        cam = F.relu((F.relu(gradient) * activation).sum(dim=1, keepdim=True))
    else:
        squared, cubed = gradient ** 2, gradient ** 3
        denominator = 2 * squared + activation.sum(dim=(2, 3), keepdim=True) * cubed
        alpha = squared / torch.where(denominator != 0, denominator, torch.ones_like(denominator))
        weights = (alpha * F.relu(gradient)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
    return _finish(cam, tuple(images.shape[-2:]))


def focused_overlay(gray: np.ndarray, cam: np.ndarray, level: float = 0.5, alpha: float = 0.5) -> np.ndarray:
    """Overlay that shows only the part of the map above ``level`` (rescaled to 0..1).

    Args:
        gray: ``(H, W)`` uint8 radiograph.
        cam: ``(H, W)`` map in [0, 1].
        level: Share of the map maximum below which nothing is drawn (0 = whole map).
        alpha: Opacity of the strongest region.

    Returns:
        ``(H, W, 3)`` uint8 RGB image.

    Raises:
        ValueError: If ``level`` is not in [0, 1).
    """
    if not 0.0 <= level < 1.0:
        raise ValueError("level must be in [0, 1).")
    focused = np.clip((np.asarray(cam, dtype=np.float32) - level) / (1.0 - level), 0.0, 1.0)
    return overlay(gray, focused, alpha=alpha)