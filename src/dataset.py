"""PyTorch dataset, augmentations and data loaders for ChestRadix.

Images come from one packed ``images_<size>.npy`` array (written by
``pack_images.py``) that is memory-mapped lazily, so the dataset object stays
cheap to pickle for DataLoader worker processes (including on Windows).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from src.config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    PATHOLOGIES,
    SPLITS,
    AugmentConfig,
    TrainConfig,
    array_name,
)

Transform = Callable[[torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------- #
# Small transform building blocks (nn.Module so they pickle cleanly)
# --------------------------------------------------------------------------- #
class RepeatChannels(nn.Module):
    """Repeat a single-channel image to ``channels`` channels."""

    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Expand ``(1, H, W)`` to ``(channels, H, W)``."""
        return image.expand(self.channels, -1, -1).contiguous()


class AddGaussianNoise(nn.Module):
    """Add Gaussian noise to an image in the range [0, 1] and clamp the result."""

    def __init__(self, std: float) -> None:
        super().__init__()
        self.std = std

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return the noisy image."""
        if self.std <= 0:
            return image
        return (image + torch.randn_like(image) * self.std).clamp_(0.0, 1.0)


def build_transforms(
    train: bool, image_size: int, augment: AugmentConfig | None = None
) -> Transform:
    """Build the transform pipeline for one split.

    The input is a ``uint8`` tensor of shape ``(1, H, W)``; the output is a
    normalised ``float32`` tensor of shape ``(3, image_size, image_size)``.

    Args:
        train: Whether to include random augmentation.
        image_size: Output edge length.
        augment: Augmentation settings (defaults are used when ``None``).

    Returns:
        A callable transform.
    """
    augment = augment or AugmentConfig()
    steps: list[nn.Module] = []
    if train:
        steps.append(
            v2.RandomResizedCrop(
                image_size, scale=augment.crop_scale, ratio=augment.crop_ratio,
                antialias=True,
            )
        )
        if augment.horizontal_flip:
            steps.append(v2.RandomHorizontalFlip(p=0.5))
        steps.append(v2.RandomRotation(degrees=augment.rotation_degrees))
        steps.append(
            v2.ColorJitter(brightness=augment.brightness, contrast=augment.contrast)
        )
    steps.append(v2.ToDtype(torch.float32, scale=True))
    if train:
        steps.append(AddGaussianNoise(augment.noise_std))
    steps.append(RepeatChannels(3))
    steps.append(v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)))
    return v2.Compose(steps)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ChestXrayDataset(Dataset):
    """Chest X-ray dataset backed by a packed uint8 array.

    Each item is ``(image, labels)`` with ``image`` of shape ``(3, S, S)`` and
    ``labels`` a float vector of length 14.
    """

    def __init__(
        self,
        processed_dir: Path,
        split: str,
        transform: Transform | None = None,
        image_size: int = 224,
    ) -> None:
        """Create the dataset.

        Args:
            processed_dir: Directory with ``<split>.csv``, ``image_index.csv`` and
                the packed array.
            split: ``"train"``, ``"val"`` or ``"test"``.
            transform: Transform applied to each ``(1, H, W)`` uint8 image.
            image_size: Edge length of the stored images.

        Raises:
            ValueError: If the split is unknown or images are missing from the index.
        """
        if split not in SPLITS:
            raise ValueError(f"Unknown split '{split}', expected one of {SPLITS}")
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self._array_path = self.processed_dir / array_name(image_size)
        self._array: np.ndarray | None = None

        frame = pd.read_csv(self.processed_dir / f"{split}.csv")
        index = pd.read_csv(self.processed_dir / "image_index.csv")
        row_of = dict(zip(index["image"], index["row"]))
        missing = [name for name in frame["image"] if name not in row_of]
        if missing:
            raise ValueError(
                f"{len(missing)} images of the {split} split are not in image_index.csv "
                f"(first: {missing[0]})"
            )
        self.images: list[str] = frame["image"].tolist()
        self.patient_ids: np.ndarray = frame["patient_id"].to_numpy()
        self.rows: np.ndarray = np.array([row_of[name] for name in self.images],
                                         dtype=np.int64)
        self.labels: np.ndarray = frame[list(PATHOLOGIES)].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        """Return the number of images in the split."""
        return len(self.images)

    def _images(self) -> np.ndarray:
        """Open the memory-mapped array lazily (once per process)."""
        if self._array is None:
            self._array = np.load(self._array_path, mmap_mode="r")
        return self._array

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the transformed image and its label vector."""
        raw = np.array(self._images()[self.rows[item]])  # copy out of the memmap
        image = torch.from_numpy(raw).unsqueeze(0)  # (1, H, W) uint8
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.from_numpy(self.labels[item])

    def __getstate__(self) -> dict:
        """Drop the memory map before pickling (needed for worker processes)."""
        state = self.__dict__.copy()
        state["_array"] = None
        return state


# --------------------------------------------------------------------------- #
# Class weights, seeding, loaders
# --------------------------------------------------------------------------- #
def compute_pos_weight(labels: np.ndarray, mode: str, cap: float = 20.0) -> torch.Tensor:
    """Compute per-class ``pos_weight`` for ``BCEWithLogitsLoss``.

    Args:
        labels: Binary label matrix of the **training** split, shape ``(N, 14)``.
        mode: ``"none"`` (all ones), ``"full"`` (negatives / positives),
            ``"sqrt"`` (square root of that ratio) or ``"capped"`` (ratio, at most ``cap``).
        cap: Upper bound used in ``"capped"`` mode.

    Returns:
        Float tensor of shape ``(14,)``.

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    ratio = negatives / np.maximum(positives, 1.0)
    if mode == "none":
        weights = np.ones(labels.shape[1])
    elif mode == "full":
        weights = ratio
    elif mode == "sqrt":
        weights = np.sqrt(ratio)
    elif mode == "capped":
        weights = np.minimum(ratio, cap)
    else:
        raise ValueError(f"Unknown pos_weight mode '{mode}'")
    return torch.tensor(weights, dtype=torch.float32)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    """Seed NumPy and ``random`` inside a DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int,
    drop_last: bool = False,
) -> DataLoader:
    """Create a reproducible DataLoader.

    Args:
        dataset: Dataset to wrap.
        batch_size: Batch size.
        shuffle: Whether to shuffle (training only).
        num_workers: Number of worker processes.
        seed: Seed for the shuffling generator.
        drop_last: Drop the last incomplete batch.

    Returns:
        A configured DataLoader.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=drop_last,
    )


def build_dataloaders(processed_dir: Path, config: TrainConfig) -> dict[str, DataLoader]:
    """Create train, validation and test loaders.

    Args:
        processed_dir: Directory with the packed array and split CSV files.
        config: Training configuration.

    Returns:
        Dictionary with the keys ``train``, ``val`` and ``test``.
    """
    loaders: dict[str, DataLoader] = {}
    for split in SPLITS:
        is_train = split == "train"
        transform = build_transforms(is_train, config.image_size, config.augment)
        dataset = ChestXrayDataset(processed_dir, split, transform, config.image_size)
        loaders[split] = make_loader(
            dataset, config.batch_size, shuffle=is_train, num_workers=config.num_workers,
            seed=config.seed, drop_last=is_train,
        )
    return loaders


def save_augmentation_preview(
    processed_dir: Path, output_path: Path, config: TrainConfig, count: int = 6
) -> Path:
    """Save a figure comparing original and augmented training images.

    Args:
        processed_dir: Directory with the packed array and split CSV files.
        output_path: Where to write the PNG figure.
        config: Training configuration (augmentation settings are used).
        count: Number of images to show.

    Returns:
        The path of the written figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plain = ChestXrayDataset(processed_dir, "train", build_transforms(False, config.image_size),
                             config.image_size)
    augmented = ChestXrayDataset(
        processed_dir, "train", build_transforms(True, config.image_size, config.augment),
        config.image_size,
    )
    count = min(count, len(plain))
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    figure, axes = plt.subplots(2, count, figsize=(2.2 * count, 4.6), squeeze=False)
    for column in range(count):
        for row, dataset in enumerate((plain, augmented)):
            image, labels = dataset[column]
            shown = (image * std + mean)[0].clamp(0, 1).numpy()
            axes[row][column].imshow(shown, cmap="gray", vmin=0, vmax=1)
            axes[row][column].axis("off")
            if row == 0:
                names = [PATHOLOGIES[i] for i in range(NUM_CLASSES) if labels[i] > 0]
                axes[row][column].set_title(", ".join(names[:2]) or "No finding", fontsize=7)
    axes[0][0].text(-0.05, 0.5, "original", transform=axes[0][0].transAxes,
                    rotation=90, va="center", ha="right")
    axes[1][0].text(-0.05, 0.5, "augmented", transform=axes[1][0].transAxes,
                    rotation=90, va="center", ha="right")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120)
    plt.close(figure)
    return output_path