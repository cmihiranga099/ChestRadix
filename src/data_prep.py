
"""NIH ChestX-ray14 preprocessing: Lanczos resize and patient-level split.

Steps performed:
    1. Locate the metadata CSV and all raw PNG radiographs under ``--raw-dir``.
    2. Resize every radiograph to ``--image-size`` x ``--image-size`` grayscale
       PNG using Lanczos resampling (resumable, multi-process).
    3. Build a patient-level, multi-label stratified train/val/test split so that
       no patient ever appears in more than one split.
    4. Write ``train.csv``, ``val.csv``, ``test.csv`` and ``split_stats.json``.

Example (Kaggle, CPU session):
    python data_prep.py --raw-dir /kaggle/input/<dataset-folder> \
        --out-dir /kaggle/working/processed --split-mode official

Smoke test on a small random subset of patients:
    python data_prep.py --raw-dir /kaggle/input/<dataset-folder> \
        --out-dir /kaggle/working/processed_test --limit 300
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

LOGGER = logging.getLogger("data_prep")

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
NO_FINDING_LABEL: str = "No Finding"
NO_FINDING_COLUMN: str = "no_finding"
DEFAULT_IMAGE_SIZE: int = 224
SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")
OPTIONAL_COLUMNS: dict[str, str] = {
    "Patient Age": "patient_age",
    "Patient Gender": "patient_gender",
    "View Position": "view_position",
}


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def find_file(root: Path, name: str) -> Path:
    """Recursively find a file by exact name under ``root``.

    Args:
        root: Directory to search.
        name: Exact file name to look for.

    Returns:
        Path of the first match.

    Raises:
        FileNotFoundError: If no file with that name exists under ``root``.
    """
    for path in root.rglob(name):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find '{name}' under {root}")


def index_images(root: Path) -> dict[str, Path]:
    """Map every PNG file name under ``root`` to its full path.

    Args:
        root: Directory containing the raw NIH image folders.

    Returns:
        Dictionary of ``file name -> full path``.
    """
    LOGGER.info("Indexing raw images under %s ...", root)
    index = {path.name: path for path in root.rglob("*.png")}
    LOGGER.info("Found %d raw PNG files.", len(index))
    return index


def read_name_list(path: Path) -> set[str]:
    """Read a text file with one image file name per line.

    Args:
        path: Path to e.g. ``train_val_list.txt``.

    Returns:
        Set of image file names.
    """
    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def load_metadata(csv_path: Path) -> pd.DataFrame:
    """Load NIH metadata and expand pathology strings into binary columns.

    Args:
        csv_path: Path to ``Data_Entry_2017.csv`` (or a compatible CSV).

    Returns:
        DataFrame with columns ``image``, ``patient_id``, one 0/1 column per
        pathology, ``no_finding`` and any available optional columns.

    Raises:
        ValueError: If required columns are missing or unknown labels appear.
    """
    raw = pd.read_csv(csv_path)
    raw.columns = [str(col).strip() for col in raw.columns]
    required = {"Image Index", "Finding Labels", "Patient ID"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Metadata CSV is missing columns: {sorted(missing)}")

    label_sets = [
        {label.strip() for label in str(value).split("|")}
        for value in raw["Finding Labels"]
    ]
    known = set(PATHOLOGIES) | {NO_FINDING_LABEL}
    unknown = set().union(*label_sets) - known
    if unknown:
        raise ValueError(f"Unknown finding labels in metadata: {sorted(unknown)}")

    meta = pd.DataFrame(
        {
            "image": raw["Image Index"].astype(str).str.strip(),
            "patient_id": raw["Patient ID"].astype(int),
        }
    )
    for pathology in PATHOLOGIES:
        meta[pathology] = np.fromiter(
            (int(pathology in labels) for labels in label_sets),
            dtype=np.int8,
            count=len(label_sets),
        )
    meta[NO_FINDING_COLUMN] = (meta[list(PATHOLOGIES)].sum(axis=1) == 0).astype(
        np.int8
    )
    for source, target in OPTIONAL_COLUMNS.items():
        if source in raw.columns:
            meta[target] = raw[source].to_numpy()
    return meta


# --------------------------------------------------------------------------- #
# Resizing
# --------------------------------------------------------------------------- #
def _resize_one(task: tuple[str, str, int]) -> tuple[str, bool, str]:
    """Resize a single radiograph to a square grayscale PNG with Lanczos.

    Args:
        task: Tuple ``(source path, destination path, image size)``.

    Returns:
        Tuple ``(image file name, success flag, error message)``.
    """
    source, destination, size = task
    name = Path(destination).name
    try:
        if os.path.exists(destination):
            return name, True, ""
        temporary = destination + ".part"
        with Image.open(source) as image:
            resized = image.convert("L").resize(
                (size, size), Image.Resampling.LANCZOS
            )
            resized.save(temporary, format="PNG", compress_level=3)
        os.replace(temporary, destination)
        return name, True, ""
    except Exception as exc:  # noqa: BLE001 - report any decode/IO failure
        return name, False, f"{type(exc).__name__}: {exc}"


def resize_images(
    tasks: list[tuple[str, str, int]], workers: int
) -> tuple[set[str], list[str]]:
    """Resize many images in parallel.

    Args:
        tasks: List of ``(source, destination, size)`` tuples.
        workers: Number of worker processes.

    Returns:
        Tuple ``(set of successfully processed names, list of error messages)``.
    """
    succeeded: set[str] = set()
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(_resize_one, tasks, chunksize=64)
        for name, ok, message in tqdm(results, total=len(tasks), desc="Resizing"):
            if ok:
                succeeded.add(name)
            else:
                errors.append(f"{name}: {message}")
    return succeeded, errors


# --------------------------------------------------------------------------- #
# Patient-level multi-label stratified splitting
# --------------------------------------------------------------------------- #
def _choose_split(
    primary: np.ndarray, secondary: np.ndarray, rng: np.random.Generator
) -> int:
    """Pick a split index by maximal primary score, breaking ties by secondary.

    Args:
        primary: Score per split (e.g. remaining desired positives of a label).
        secondary: Tie-break score per split (remaining desired patients).
        rng: Random generator for final random tie-breaking.

    Returns:
        Index of the chosen split.
    """
    candidates = np.flatnonzero(np.isclose(primary, primary.max()))
    if candidates.size > 1:
        tie = secondary[candidates]
        candidates = candidates[np.isclose(tie, tie.max())]
    return int(rng.choice(candidates))


def iterative_patient_split(
    patient_labels: np.ndarray, ratios: Sequence[float], rng: np.random.Generator
) -> np.ndarray:
    """Assign patients to splits with iterative multi-label stratification.

    Implements the iterative stratification algorithm of Sechidis et al. (2011):
    labels are processed from rarest to most frequent, and each patient carrying
    the current label goes to the split that still needs the most positives of
    that label.

    Args:
        patient_labels: Binary matrix of shape ``(n_patients, n_labels)``.
        ratios: Desired split proportions (normalised internally).
        rng: Random generator.

    Returns:
        Integer array of shape ``(n_patients,)`` with the split index per patient.
    """
    weights = np.asarray(ratios, dtype=np.float64)
    weights = weights / weights.sum()
    n_patients = patient_labels.shape[0]

    desired_total = weights * n_patients
    desired_label = weights[:, None] * patient_labels.sum(axis=0)[None, :]
    assignment = np.full(n_patients, -1, dtype=np.int64)

    while True:
        unassigned = assignment < 0
        if not unassigned.any():
            break
        remaining = patient_labels[unassigned].sum(axis=0)
        candidates = np.flatnonzero(remaining > 0)

        if candidates.size == 0:
            # Only patients without any positive label are left.
            for index in rng.permutation(np.flatnonzero(unassigned)):
                split = _choose_split(desired_total, desired_total, rng)
                assignment[index] = split
                desired_total[split] -= 1
            break

        label = int(candidates[np.argmin(remaining[candidates])])
        members = np.flatnonzero(unassigned & (patient_labels[:, label] > 0))
        for index in rng.permutation(members):
            split = _choose_split(desired_label[:, label], desired_total, rng)
            assignment[index] = split
            desired_total[split] -= 1
            desired_label[split] -= patient_labels[index]
    return assignment


def split_by_patient(
    frame: pd.DataFrame, ratios: Sequence[float], seed: int
) -> list[pd.DataFrame]:
    """Split an image-level frame into patient-disjoint parts.

    Args:
        frame: Image-level metadata containing ``patient_id`` and pathology columns.
        ratios: Desired proportions of patients per part.
        seed: Random seed.

    Returns:
        List of DataFrames, one per ratio entry.
    """
    patient_labels = frame.groupby("patient_id")[list(PATHOLOGIES)].max()
    rng = np.random.default_rng(seed)
    assignment = iterative_patient_split(
        patient_labels.to_numpy(dtype=np.int64), ratios, rng
    )
    mapping = pd.Series(assignment, index=patient_labels.index)
    split_index = frame["patient_id"].map(mapping).to_numpy()
    return [
        frame[split_index == part].reset_index(drop=True)
        for part in range(len(ratios))
    ]


def make_splits(
    meta: pd.DataFrame,
    mode: str,
    ratios: Sequence[float],
    val_fraction: float,
    seed: int,
    raw_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Create train/val/test DataFrames according to the chosen split mode.

    Args:
        meta: Image-level metadata of all usable images.
        mode: ``"patient"`` (custom ratios) or ``"official"`` (NIH test list).
        ratios: Train/val/test ratios used in ``"patient"`` mode.
        val_fraction: Validation share of the train_val pool in ``"official"`` mode.
        seed: Random seed.
        raw_dir: Directory in which the official list files are searched.

    Returns:
        Dictionary mapping ``train``/``val``/``test`` to DataFrames.

    Raises:
        ValueError: If ``mode`` is unknown.
        RuntimeError: If the official lists share patients (should never happen).
    """
    if mode == "patient":
        train, val, test = split_by_patient(meta, ratios, seed)
        return {"train": train, "val": val, "test": test}

    if mode == "official":
        test_names = read_name_list(find_file(raw_dir, "test_list.txt"))
        trainval_names = read_name_list(find_file(raw_dir, "train_val_list.txt"))
        test = meta[meta["image"].isin(test_names)].reset_index(drop=True)
        trainval = meta[meta["image"].isin(trainval_names)].reset_index(drop=True)
        if set(test["patient_id"]) & set(trainval["patient_id"]):
            raise RuntimeError("Official train_val and test lists share patients.")
        train, val = split_by_patient(trainval, [1.0 - val_fraction, val_fraction], seed)
        return {"train": train, "val": val, "test": test}

    raise ValueError(f"Unknown split mode: {mode}")


def assert_no_patient_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Verify that no patient appears in more than one split.

    Args:
        splits: Dictionary of split DataFrames.

    Raises:
        RuntimeError: If any patient occurs in two or more splits.
    """
    names = list(splits)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            overlap = set(splits[first]["patient_id"]) & set(splits[second]["patient_id"])
            if overlap:
                raise RuntimeError(
                    f"Patient leakage between {first} and {second}: "
                    f"{len(overlap)} shared patients."
                )
    LOGGER.info("Leakage check passed: all splits are patient-disjoint.")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def summarize(splits: dict[str, pd.DataFrame]) -> dict:
    """Compute per-split counts, prevalences and training ``pos_weight`` values.

    Args:
        splits: Dictionary of split DataFrames.

    Returns:
        JSON-serialisable statistics dictionary.
    """
    stats: dict = {}
    for name, frame in splits.items():
        count = len(frame)
        stats[name] = {
            "images": count,
            "patients": int(frame["patient_id"].nunique()),
            "no_finding_images": int(frame[NO_FINDING_COLUMN].sum()),
            "positives": {p: int(frame[p].sum()) for p in PATHOLOGIES},
            "prevalence": {
                p: round(float(frame[p].mean()), 5) if count else 0.0
                for p in PATHOLOGIES
            },
        }
    train = splits["train"]
    positives = train[list(PATHOLOGIES)].sum()
    negatives = len(train) - positives
    stats["suggested_pos_weight"] = {
        p: round(float(negatives[p] / max(int(positives[p]), 1)), 4)
        for p in PATHOLOGIES
    }
    return stats


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="NIH ChestX-ray14 preprocessing")
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="Directory containing the NIH CSV and image folders.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"),
                        help="Output directory (images_224/ and CSV files).")
    parser.add_argument("--metadata-csv", type=Path, default=None,
                        help="Metadata CSV (default: search Data_Entry_2017.csv).")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--split-mode", choices=("patient", "official"),
                        default="patient")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15),
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Ratios for --split-mode patient.")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="Validation share of train_val for --split-mode official.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only N random patients (smoke test).")
    parser.add_argument("--skip-resize", action="store_true",
                        help="Reuse existing resized images; only rebuild splits.")
    return parser.parse_args()


def main() -> None:
    """Run the full preprocessing pipeline."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    out_dir: Path = args.out_dir
    image_dir = out_dir / f"images_{args.image_size}"
    image_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.metadata_csv or find_file(args.raw_dir, "Data_Entry_2017.csv")
    LOGGER.info("Loading metadata from %s", csv_path)
    meta = load_metadata(csv_path)
    LOGGER.info("Metadata rows: %d | patients: %d", len(meta), meta["patient_id"].nunique())

    # Keep only rows whose image file actually exists.
    raw_index: dict[str, Path] = {}
    if args.skip_resize:
        available = {path.name for path in image_dir.glob("*.png")}
    else:
        raw_index = index_images(args.raw_dir)
        available = set(raw_index)
    missing_count = int((~meta["image"].isin(available)).sum())
    if missing_count:
        LOGGER.warning("Dropping %d metadata rows without an image file.", missing_count)
    meta = meta[meta["image"].isin(available)].reset_index(drop=True)

    if args.limit is not None:
        rng = np.random.default_rng(args.seed)
        patients = meta["patient_id"].unique()
        chosen = rng.choice(patients, size=min(args.limit, len(patients)), replace=False)
        meta = meta[meta["patient_id"].isin(chosen)].reset_index(drop=True)
        LOGGER.info("Smoke-test subset: %d patients, %d images.", len(chosen), len(meta))

    if not args.skip_resize:
        tasks = [
            (str(raw_index[name]), str(image_dir / name), args.image_size)
            for name in meta["image"]
        ]
        succeeded, errors = resize_images(tasks, args.workers)
        if errors:
            LOGGER.warning("%d images failed and will be dropped. First errors: %s",
                           len(errors), errors[:5])
        meta = meta[meta["image"].isin(succeeded)].reset_index(drop=True)

    splits = make_splits(meta, args.split_mode, args.ratios, args.val_fraction,
                         args.seed, args.raw_dir)
    assert_no_patient_leakage(splits)

    for name in SPLIT_NAMES:
        splits[name].to_csv(out_dir / f"{name}.csv", index=False)
        LOGGER.info("%-5s | images: %6d | patients: %6d", name,
                    len(splits[name]), splits[name]["patient_id"].nunique())

    stats = summarize(splits)
    stats["config"] = {
        "split_mode": args.split_mode,
        "ratios": list(args.ratios),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "image_size": args.image_size,
        "limit": args.limit,
    }
    with (out_dir / "split_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    LOGGER.info("Done. Outputs written to %s", out_dir.resolve())


if __name__ == "__main__":
    main()