"""Central configuration for ChestRadix.

Everything that more than one module needs lives here: the 14 pathology names,
the disease-to-priority mapping, image and normalisation constants, seeds, path
discovery (laptop, Kaggle, Colab) and the training hyperparameters.
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #
PATHOLOGIES: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)
NUM_CLASSES: int = len(PATHOLOGIES)
NO_FINDING_COLUMN: str = "no_finding"

# --------------------------------------------------------------------------- #
# Disease-to-priority mapping (design assumption, not clinically validated)
# --------------------------------------------------------------------------- #
PRIORITY_LEVELS: tuple[str, ...] = ("ROUTINE", "URGENT", "IMMEDIATE")  # low to high
PRIORITY_COLORS: dict[str, str] = {
    "IMMEDIATE": "red",
    "URGENT": "yellow",
    "ROUTINE": "green",
}
PRIORITY_MAP: dict[str, str] = {
    "Pneumothorax": "IMMEDIATE",
    "Edema": "IMMEDIATE",
    "Consolidation": "IMMEDIATE",
    "Pneumonia": "URGENT",
    "Effusion": "URGENT",
    "Mass": "URGENT",
    "Atelectasis": "URGENT",
    "Infiltration": "URGENT",
    "Cardiomegaly": "URGENT",
    "Nodule": "ROUTINE",
    "Hernia": "ROUTINE",
    "Emphysema": "ROUTINE",
    "Fibrosis": "ROUTINE",
    "Pleural_Thickening": "ROUTINE",
}

# --------------------------------------------------------------------------- #
# Image and normalisation constants
# --------------------------------------------------------------------------- #
IMAGE_SIZE: int = 224
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
SEEDS: tuple[int, int, int] = (42, 43, 44)
SPLITS: tuple[str, str, str] = ("train", "val", "test")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
ENV_DATA_DIR: str = "CHESTRADIX_DATA_DIR"
ENV_RUNS_DIR: str = "CHESTRADIX_RUNS_DIR"


def on_kaggle() -> bool:
    """Return True when running inside a Kaggle notebook."""
    return Path("/kaggle/working").exists()


def on_colab() -> bool:
    """Return True when running inside Google Colab."""
    return "google.colab" in sys.modules


def array_name(image_size: int = IMAGE_SIZE) -> str:
    """Return the file name of the packed image array.

    Args:
        image_size: Edge length of the stored images.

    Returns:
        File name such as ``images_224.npy``.
    """
    return f"images_{image_size}.npy"


def locate_processed_dir(image_size: int = IMAGE_SIZE) -> Path:
    """Find the directory that holds the packed images and split CSV files.

    Search order: the ``CHESTRADIX_DATA_DIR`` environment variable, the
    repository's ``data/processed`` folder, then dataset folders mounted under
    ``/kaggle/input`` (up to three levels deep).

    Args:
        image_size: Edge length of the stored images.

    Returns:
        Directory containing ``images_<size>.npy``.

    Raises:
        FileNotFoundError: If the data cannot be found anywhere.
    """
    marker = array_name(image_size)
    env_value = os.environ.get(ENV_DATA_DIR)
    if env_value:
        candidate = Path(env_value)
        if (candidate / marker).exists():
            return candidate
        raise FileNotFoundError(f"{ENV_DATA_DIR}={env_value} does not contain {marker}")

    local = REPO_ROOT / "data" / "processed"
    if (local / marker).exists():
        return local

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for pattern in (marker, f"*/{marker}", f"*/*/{marker}", f"*/*/*/{marker}"):
            for hit in kaggle_input.glob(pattern):
                return hit.parent

    raise FileNotFoundError(
        f"Could not find {marker}. Set {ENV_DATA_DIR}, copy the processed data to "
        f"{local}, or attach the Kaggle dataset to the notebook."
    )


def runs_dir() -> Path:
    """Return the directory where experiment run folders are written.

    Returns:
        ``CHESTRADIX_RUNS_DIR`` if set, ``/kaggle/working/runs`` on Kaggle,
        otherwise ``<repo>/runs``.
    """
    env_value = os.environ.get(ENV_RUNS_DIR)
    if env_value:
        return Path(env_value)
    if on_kaggle():
        return Path("/kaggle/working/runs")
    return REPO_ROOT / "runs"


# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AugmentConfig:
    """Training-time augmentation settings (validation and test use none)."""

    crop_scale: tuple[float, float] = (0.85, 1.0)
    crop_ratio: tuple[float, float] = (0.95, 1.05)
    rotation_degrees: float = 10.0
    brightness: float = 0.15
    contrast: float = 0.15
    noise_std: float = 0.02
    horizontal_flip: bool = False  # off by default: flipping swaps left/right anatomy


@dataclass(frozen=True)
class TrainConfig:
    """Training settings for one experiment."""

    seed: int = 42
    backbone: str = "densenet121"
    image_size: int = IMAGE_SIZE
    batch_size: int = 64
    epochs: int = 15
    patience: int = 4
    warmup_epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    dropout: float = 0.2
    amp: bool = True
    num_workers: int = 2
    pos_weight_mode: str = "full"  # "full", "sqrt", "capped" or "none"
    pos_weight_cap: float = 20.0
    augment: AugmentConfig = field(default_factory=AugmentConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a JSON-serialisable dictionary."""
        return asdict(self)