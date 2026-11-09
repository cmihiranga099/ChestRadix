"""Model definitions for ChestRadix.

Provides the DenseNet-121 (main) and ResNet-50 (baseline) classifiers with a
shared 14-class head, an activation/gradient capture helper for Grad-CAM, and
checkpoint save/load utilities.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import TracebackType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from src.config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    PATHOLOGIES,
    TrainConfig,
)

# Layer whose activations Grad-CAM explains, per backbone.
DEFAULT_TARGET_LAYERS: dict[str, str] = {
    "densenet121": "features.norm5",
    "resnet50": "layer4",
}
# DenseNet applies ReLU right after ``features.norm5`` inside ``forward``; the
# captured activation therefore has that ReLU applied (see ActivationCapture).
RELU_AFTER_TARGET: dict[str, bool] = {"densenet121": True, "resnet50": False}


def build_model(
    backbone: str = "densenet121",
    num_classes: int = NUM_CLASSES,
    dropout: float = 0.2,
    pretrained: bool = True,
) -> nn.Module:
    """Create a classifier with a ``Dropout`` + ``Linear`` head.

    Args:
        backbone: ``"densenet121"`` or ``"resnet50"``.
        num_classes: Number of output logits.
        dropout: Dropout probability before the final linear layer.
        pretrained: Load ImageNet weights (downloaded on first use).

    Returns:
        The model (outputs raw logits, no sigmoid).

    Raises:
        ValueError: If the backbone is unknown.
    """
    if backbone == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(model.classifier.in_features, num_classes)
        )
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    else:
        raise ValueError(f"Unknown backbone '{backbone}'. Use 'densenet121' or 'resnet50'.")
    return model


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters.

    Args:
        model: Any module.

    Returns:
        Count of parameters with ``requires_grad``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Grad-CAM support
# --------------------------------------------------------------------------- #
class ActivationCapture:
    """Context manager that records the activation of one layer.

    torchvision's DenseNet applies an **in-place** ReLU to the output of
    ``features.norm5``, which breaks ordinary backward hooks. This helper avoids
    the problem: the forward hook replaces the layer output with a clone and
    keeps the (optionally ReLU-ed) original in ``activation``, which stays part
    of the autograd graph so ``torch.autograd.grad`` can differentiate with
    respect to it. The hook is always removed on exit, so parallel requests
    cannot share state.
    """

    def __init__(self, model: nn.Module, layer_path: str, relu_after: bool = False) -> None:
        """Prepare the capture.

        Args:
            model: The model to inspect.
            layer_path: Dotted path of the target module, e.g. ``features.norm5``.
            relu_after: Apply ReLU to the layer output before storing it.
        """
        self.layer = model.get_submodule(layer_path)
        self.relu_after = relu_after
        self.activation: torch.Tensor | None = None
        self._handle: Any = None

    def _hook(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        stored = F.relu(output) if self.relu_after else output
        self.activation = stored
        return stored.clone()

    def __enter__(self) -> "ActivationCapture":
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def activation_and_gradient(
    model: nn.Module,
    images: torch.Tensor,
    class_index: int | torch.Tensor,
    backbone: str = "densenet121",
    layer_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return logits, target-layer activation and its gradient for a class logit.

    Args:
        model: Model in evaluation mode.
        images: Input batch of shape ``(B, 3, H, W)``.
        class_index: Class (int, or one index per image) whose **logit** is explained.
        backbone: Backbone name, used to choose the default layer and ReLU handling.
        layer_path: Override of the target layer path.

    Returns:
        Tuple ``(logits, activation, gradient)``. ``activation`` and ``gradient``
        are detached tensors of shape ``(B, C, h, w)``.
    """
    layer = layer_path or DEFAULT_TARGET_LAYERS[backbone]
    relu_after = RELU_AFTER_TARGET[backbone] if layer_path is None else False
    with torch.enable_grad(), ActivationCapture(model, layer, relu_after) as capture:
        images = images.detach().clone().requires_grad_(False)
        logits = model(images)
        activation = capture.activation
        assert activation is not None
        if isinstance(class_index, int):
            score = logits[:, class_index].sum()
        else:
            score = logits.gather(1, class_index.view(-1, 1)).sum()
        (gradient,) = torch.autograd.grad(score, activation, retain_graph=False)
    return logits.detach(), activation.detach(), gradient.detach()


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    """Return the current git commit hash, or ``unknown`` if unavailable."""
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True, timeout=5)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def save_checkpoint(path: Path, model: nn.Module, config: TrainConfig,
                    extra: dict[str, Any] | None = None) -> None:
    """Save weights together with everything needed to rebuild and use the model.

    Args:
        path: Destination file (parent folders are created).
        model: Trained model.
        config: Training configuration of the run.
        extra: Additional metadata (for example the best epoch or temperature).
    """
    payload = {
        "state_dict": model.state_dict(),
        "backbone": config.backbone,
        "dropout": config.dropout,
        "classes": list(PATHOLOGIES),
        "image_size": config.image_size,
        "normalisation": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "config": config.to_dict(),
        "git_commit": _git_commit(),
        "extra": extra or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Args:
        path: Checkpoint file.
        map_location: Device to load the tensors onto.

    Returns:
        Tuple ``(model in eval mode, metadata dictionary without the weights)``.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(payload["backbone"], len(payload["classes"]), payload["dropout"],
                        pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    return model, metadata