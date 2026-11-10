"""Tier head helpers for ChestRadix (Stage 5c).

Besides the 14 finding logits the network can learn two extra ordinal outputs that answer
the triage question directly:

* ``urgent_or_higher``: is any URGENT or IMMEDIATE finding present (tier >= 1)?
* ``immediate``: is any IMMEDIATE finding present (tier >= 2)?

The targets are derived from the 14 labels, so no new annotation is needed. This module is
free of any PyTorch import at load time (torch is imported inside the functions that need
it), so calibration and triage can use :func:`tier_targets` without installing PyTorch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.config import NUM_CLASSES
from src.metrics import CLASS_TIERS, tiers_from_binary

NUM_TIER_OUTPUTS = 2
TIER_HEAD_NAMES: tuple[str, str] = ("urgent_or_higher", "immediate")


def tier_targets(binary: np.ndarray) -> np.ndarray:
    """Binary tier-head targets from the 14 binary labels.

    Args:
        binary: ``(N, 14)`` binary labels.

    Returns:
        ``(N, 2)`` float array: column 0 is ``tier >= URGENT``, column 1 is ``tier == IMMEDIATE``.
    """
    tiers = tiers_from_binary(binary)
    return np.stack([tiers >= 1, tiers >= 2], axis=1).astype(np.float32)


def tier_targets_torch(labels):
    """Same as :func:`tier_targets` for a ``(B, 14)`` torch tensor (stays on its device).

    Args:
        labels: Float tensor of shape ``(B, 14)`` with values 0 or 1.

    Returns:
        Float tensor of shape ``(B, 2)``.
    """
    import torch

    tiers = torch.as_tensor(CLASS_TIERS, device=labels.device)
    highest = torch.where(labels > 0.5, tiers, torch.zeros_like(tiers)).max(dim=1).values
    return torch.stack([highest >= 1, highest >= 2], dim=1).float()


def load_checkpoint_any(path: Path, map_location: Any = "cpu"):
    """Load a checkpoint with 14 outputs (E1) or 16 outputs (tier head, E2+).

    ``src.model.load_checkpoint`` always rebuilds a 14-output model, so it cannot load a
    tier-head checkpoint. This function reads the output size from the checkpoint.

    Args:
        path: Checkpoint written by ``save_checkpoint``.
        map_location: Device to load the tensors onto.

    Returns:
        Tuple ``(model in eval mode, metadata dictionary without the weights)``.
    """
    import torch

    from src.model import build_model

    payload = torch.load(path, map_location=map_location, weights_only=False)
    outputs = int(payload.get("extra", {}).get("num_outputs", len(payload["classes"])))
    model = build_model(payload["backbone"], outputs, payload["dropout"], pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, {key: value for key, value in payload.items() if key != "state_dict"}


def split_outputs(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Split model outputs into the 14 finding logits and the optional tier-head logits.

    Args:
        logits: ``(N, 14)`` or ``(N, 16)`` array.

    Returns:
        ``(finding_logits, tier_logits or None)``.
    """
    logits = np.asarray(logits)
    if logits.shape[1] == NUM_CLASSES:
        return logits, None
    return logits[:, :NUM_CLASSES], logits[:, NUM_CLASSES:NUM_CLASSES + NUM_TIER_OUTPUTS]