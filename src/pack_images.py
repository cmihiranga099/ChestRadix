"""Pack resized radiographs into one uint8 NumPy array.

Reads the split CSVs written by ``data_prep.py``, decodes every PNG exactly once
and stores all images in a single ``images_<size>.npy`` file with shape
``(N, size, size)`` and dtype ``uint8``. A companion ``image_index.csv`` maps each
image file name to its row in the array.

Why: Kaggle limits the number of files that a notebook can keep as output, and
decoding 100k PNG files in every training epoch is a data-loading bottleneck.
One array can be memory-mapped, so training reads raw bytes instead.

Example (Kaggle, after data_prep.py):
    python pack_images.py --processed-dir /kaggle/working/processed --delete-images
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from PIL import Image
from tqdm import tqdm

LOGGER = logging.getLogger("pack_images")
SPLIT_FILES: tuple[str, ...] = ("train.csv", "val.csv", "test.csv")


def _load_one(task: tuple[str, int]) -> np.ndarray:
    """Decode one grayscale PNG and check its shape.

    Args:
        task: Tuple ``(path, expected size)``.

    Returns:
        ``uint8`` array of shape ``(size, size)``.

    Raises:
        ValueError: If the image does not have the expected shape.
    """
    path, size = task
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    if array.shape != (size, size):
        raise ValueError(f"{path} has shape {array.shape}, expected {(size, size)}")
    return array


def collect_image_names(processed_dir: Path) -> list[str]:
    """Return the unique image names of all splits, in train, val, test order.

    Args:
        processed_dir: Directory containing ``train.csv``, ``val.csv``, ``test.csv``.

    Returns:
        List of unique image file names.
    """
    names: list[str] = []
    for csv_name in SPLIT_FILES:
        names.extend(pd.read_csv(processed_dir / csv_name, usecols=["image"])["image"])
    return list(dict.fromkeys(names))


def pack_images(
    processed_dir: Path, image_size: int, workers: int, delete_images: bool
) -> Path:
    """Pack all images referenced by the split CSVs into one ``.npy`` file.

    Args:
        processed_dir: Output directory of ``data_prep.py``.
        image_size: Edge length of the resized images.
        workers: Number of worker processes used for decoding.
        delete_images: Remove the PNG folder after a successful, verified pack.

    Returns:
        Path of the written ``.npy`` file.

    Raises:
        RuntimeError: If the verification of the packed array fails.
    """
    image_dir = processed_dir / f"images_{image_size}"
    names = collect_image_names(processed_dir)
    LOGGER.info("Packing %d images from %s", len(names), image_dir)

    array_path = processed_dir / f"images_{image_size}.npy"
    packed = open_memmap(array_path, mode="w+", dtype=np.uint8,
                         shape=(len(names), image_size, image_size))
    tasks = [(str(image_dir / name), image_size) for name in names]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(_load_one, tasks, chunksize=256)
        for row, array in enumerate(tqdm(results, total=len(tasks), desc="Packing")):
            packed[row] = array
    packed.flush()

    pd.DataFrame({"image": names, "row": np.arange(len(names))}).to_csv(
        processed_dir / "image_index.csv", index=False
    )

    # Verification: re-decode a random sample and compare with the packed rows.
    rng = np.random.default_rng(0)
    sample = rng.choice(len(names), size=min(100, len(names)), replace=False)
    reread = np.load(array_path, mmap_mode="r")
    for row in sample:
        if not np.array_equal(reread[row], _load_one(tasks[int(row)])):
            raise RuntimeError(f"Verification failed for row {row} ({names[int(row)]})")
    LOGGER.info("Verified %d random rows. Array shape: %s", len(sample), reread.shape)
    del reread, packed

    if delete_images:
        shutil.rmtree(image_dir)
        LOGGER.info("Deleted %s", image_dir)
    return array_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Pack resized images into one .npy")
    parser.add_argument("--processed-dir", type=Path, required=True,
                        help="Output directory of data_prep.py.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--delete-images", action="store_true",
                        help="Delete the PNG folder after a verified pack.")
    return parser.parse_args()


def main() -> None:
    """Run the packing step."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    path = pack_images(args.processed_dir, args.image_size, args.workers, args.delete_images)
    LOGGER.info("Done. Wrote %s (%.2f GB)", path, path.stat().st_size / 1e9)


if __name__ == "__main__":
    main()