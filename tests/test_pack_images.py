"""Unit tests for ``src.pack_images``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.pack_images import collect_image_names, pack_images

SIZE = 16


def build_processed_dir(root: Path, names_per_split: dict[str, list[str]],
                        size: int = SIZE) -> dict[str, np.ndarray]:
    """Create a fake processed directory with PNGs and split CSVs."""
    image_dir = root / f"images_{size}"
    image_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    originals: dict[str, np.ndarray] = {}
    for split, names in names_per_split.items():
        pd.DataFrame({"image": names, "patient_id": range(len(names))}).to_csv(
            root / f"{split}.csv", index=False
        )
        for name in names:
            array = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
            Image.fromarray(array, "L").save(image_dir / name)
            originals[name] = array
    return originals


def test_collect_image_names_keeps_order_and_uniqueness(tmp_path: Path) -> None:
    build_processed_dir(tmp_path, {"train": ["a.png", "b.png"], "val": ["c.png"],
                                   "test": ["d.png"]})
    assert collect_image_names(tmp_path) == ["a.png", "b.png", "c.png", "d.png"]


def test_pack_images_roundtrip(tmp_path: Path) -> None:
    originals = build_processed_dir(
        tmp_path, {"train": ["a.png", "b.png", "c.png"], "val": ["d.png"], "test": ["e.png"]}
    )
    path = pack_images(tmp_path, SIZE, workers=2, delete_images=False)
    packed = np.load(path)
    index = pd.read_csv(tmp_path / "image_index.csv")
    assert packed.shape == (5, SIZE, SIZE) and packed.dtype == np.uint8
    assert len(index) == 5
    for name, row in zip(index["image"], index["row"]):
        assert np.array_equal(packed[row], originals[name])
    assert (tmp_path / f"images_{SIZE}").exists()


def test_pack_images_can_delete_pngs(tmp_path: Path) -> None:
    build_processed_dir(tmp_path, {"train": ["a.png"], "val": ["b.png"], "test": ["c.png"]})
    pack_images(tmp_path, SIZE, workers=1, delete_images=True)
    assert not (tmp_path / f"images_{SIZE}").exists()
    assert (tmp_path / f"images_{SIZE}.npy").exists()


def test_pack_images_rejects_wrong_size(tmp_path: Path) -> None:
    build_processed_dir(tmp_path, {"train": ["a.png"], "val": ["b.png"], "test": ["c.png"]},
                        size=8)
    (tmp_path / "images_8").rename(tmp_path / f"images_{SIZE}")  # 8 px images, 16 expected
    with pytest.raises(ValueError, match="expected"):
        pack_images(tmp_path, SIZE, workers=1, delete_images=False)