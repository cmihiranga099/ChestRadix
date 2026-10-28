"""Unit tests for ``src.dataset``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import IMAGENET_MEAN, IMAGENET_STD, PATHOLOGIES, TrainConfig, array_name
from src.dataset import (
    ChestXrayDataset,
    build_dataloaders,
    build_transforms,
    compute_pos_weight,
    save_augmentation_preview,
)

SIZE = 224


def make_processed_dir(root: Path, per_split: int = 8, seed: int = 0) -> dict[str, np.ndarray]:
    """Write a fake processed directory in the format produced by pack_images.py."""
    rng = np.random.default_rng(seed)
    all_names: list[str] = []
    for split in ("train", "val", "test"):
        names = [f"{split}_{i:03d}.png" for i in range(per_split)]
        labels = (rng.random((per_split, len(PATHOLOGIES))) < 0.3).astype(int)
        frame = pd.DataFrame(labels, columns=list(PATHOLOGIES))
        frame.insert(0, "patient_id", np.arange(per_split) + (1000 if split == "val" else 0))
        frame.insert(0, "image", names)
        frame.to_csv(root / f"{split}.csv", index=False)
        all_names.extend(names)
    array = rng.integers(0, 256, size=(len(all_names), SIZE, SIZE), dtype=np.uint8)
    np.save(root / array_name(SIZE), array)
    pd.DataFrame({"image": all_names, "row": np.arange(len(all_names))}).to_csv(
        root / "image_index.csv", index=False
    )
    return dict(zip(all_names, array))


@pytest.fixture()
def processed(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    return tmp_path, make_processed_dir(tmp_path)


def test_item_shapes_and_dtypes(processed) -> None:
    root, _ = processed
    dataset = ChestXrayDataset(root, "train", build_transforms(False, SIZE), SIZE)
    image, labels = dataset[0]
    assert len(dataset) == 8
    assert image.shape == (3, SIZE, SIZE) and image.dtype == torch.float32
    assert labels.shape == (14,) and labels.dtype == torch.float32
    assert set(labels.unique().tolist()) <= {0.0, 1.0}


def test_eval_transform_returns_normalised_original_pixels(processed) -> None:
    root, originals = processed
    dataset = ChestXrayDataset(root, "val", build_transforms(False, SIZE), SIZE)
    image, _ = dataset[2]
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    restored = ((image * std + mean) * 255).round().to(torch.uint8)
    expected = torch.from_numpy(originals[dataset.images[2]])
    for channel in range(3):
        assert torch.equal(restored[channel], expected)


def test_eval_transform_is_deterministic(processed) -> None:
    root, _ = processed
    dataset = ChestXrayDataset(root, "test", build_transforms(False, SIZE), SIZE)
    assert torch.equal(dataset[1][0], dataset[1][0])


def test_train_augmentation_is_reproducible_under_seed(processed) -> None:
    root, _ = processed
    dataset = ChestXrayDataset(root, "train", build_transforms(True, SIZE), SIZE)
    torch.manual_seed(5)
    first = dataset[3][0]
    torch.manual_seed(5)
    second = dataset[3][0]
    torch.manual_seed(6)
    other = dataset[3][0]
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert first.shape == (3, SIZE, SIZE)


def test_dataloaders_produce_expected_batches(processed) -> None:
    root, _ = processed
    config = TrainConfig(batch_size=4, num_workers=0)
    loaders = build_dataloaders(root, config)
    images, labels = next(iter(loaders["train"]))
    assert images.shape == (4, 3, SIZE, SIZE) and labels.shape == (4, 14)
    assert sum(len(batch[0]) for batch in loaders["val"]) == 8


def test_dataloader_works_with_worker_processes(processed) -> None:
    root, _ = processed
    config = TrainConfig(batch_size=4, num_workers=2)
    images, _ = next(iter(build_dataloaders(root, config)["val"]))
    assert images.shape == (4, 3, SIZE, SIZE)


def test_unknown_split_and_missing_index_entries_are_rejected(processed) -> None:
    root, _ = processed
    with pytest.raises(ValueError, match="Unknown split"):
        ChestXrayDataset(root, "holdout", None, SIZE)
    index = pd.read_csv(root / "image_index.csv")
    index[index["image"] != "train_000.png"].to_csv(root / "image_index.csv", index=False)
    with pytest.raises(ValueError, match="not in image_index.csv"):
        ChestXrayDataset(root, "train", None, SIZE)


def test_compute_pos_weight_modes() -> None:
    labels = np.zeros((100, 14), dtype=np.float32)
    labels[:50, 0] = 1  # class 0: 50 positives, ratio 1
    labels[:2, 1] = 1  # class 1: 2 positives, ratio 49
    labels[:, 2] = 0   # class 2: no positives, ratio 100
    assert torch.allclose(compute_pos_weight(labels, "none"), torch.ones(14))
    full = compute_pos_weight(labels, "full")
    assert full[0] == pytest.approx(1.0) and full[1] == pytest.approx(49.0)
    assert full[2] == pytest.approx(100.0)
    assert compute_pos_weight(labels, "sqrt")[1] == pytest.approx(7.0)
    assert compute_pos_weight(labels, "capped", cap=20.0)[1] == pytest.approx(20.0)
    with pytest.raises(ValueError):
        compute_pos_weight(labels, "other")


def test_augmentation_preview_is_saved(processed, tmp_path: Path) -> None:
    root, _ = processed
    output = save_augmentation_preview(root, tmp_path / "figs" / "preview.png",
                                       TrainConfig(), count=4)
    assert output.exists() and output.stat().st_size > 1000


def test_dataset_and_transform_are_picklable_for_windows_workers(processed) -> None:
    import pickle

    root, _ = processed
    dataset = ChestXrayDataset(root, "train", build_transforms(True, SIZE), SIZE)
    _ = dataset[0]  # opens the memory map
    clone = pickle.loads(pickle.dumps(dataset))
    assert clone._array is None and len(clone) == len(dataset)
    assert clone[1][0].shape == (3, SIZE, SIZE)