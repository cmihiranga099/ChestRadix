"""How often does the heat-map peak land on an artefact instead of the body?

For every image/finding pair of the NIH box subset the heat map is computed (LayerCAM, averaged
over the models) and the location of its strongest point is classified:

* ``background``  outside the body (black background, floating markers such as "R"),
* ``border``      inside the outer ``border`` share of the picture (edges, collimation, clothing),
* ``corner``      in the top-left or top-right corner square (where side markers sit),
* ``body``        everything else.

It is done twice, without and with the body mask used in :mod:`src.inference`, so the effect of
the mask can be reported as a number (a limitation of heat maps, not a clinical result).

Two sets of images can be checked: the box subset (images with a labelled finding, the map is for
that finding) and, with ``--no-finding``, a random sample of test images that have **no finding**
(the map is for the finding the models score highest, as the app shows it for ROUTINE images).

Command line::

    python -m src.artifact_check --checkpoints runs/e1_densenet121_s42/best.pt runs/e2_densenet121_tier_s42/best.pt \
        --processed-dir data/bbox_subset --out runs/ens_e1_e2/artifact_check.json
    python -m src.artifact_check --checkpoints runs/e1_densenet121_s42/best.pt runs/e2_densenet121_tier_s42/best.pt \
        --processed-dir data/processed --no-finding --n 500 --out runs/ens_e1_e2/artifact_check_nofinding.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.bbox_eval import load_bboxes
from src.cam_methods import cam_maps
from src.config import NUM_CLASSES, PATHOLOGIES
from src.dataset import build_transforms
from src.inference import CAM_LAYERS, body_mask
from src.tier_head import load_checkpoint_any

LOGGER = logging.getLogger("chestradix.artifact_check")
CATEGORIES: tuple[str, ...] = ("background", "border", "corner", "body")


def classify_peak(cam: np.ndarray, mask: np.ndarray, border: float = 0.08, corner: float = 0.18) -> str:
    """Category of the strongest point of one map (see the module text); ties take the first point."""
    height, width = cam.shape
    y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    if mask[y, x] == 0:
        return "background"
    by, bx = int(border * height), int(border * width)
    if y < by or y >= height - by or x < bx or x >= width - bx:
        return "border"
    cy, cx = int(corner * height), int(corner * width)
    if y < cy and (x < cx or x >= width - cx):
        return "corner"
    return "body"


def _load(checkpoints: list[Path], device: str) -> tuple[list[torch.nn.Module], list[str], int]:
    """Load the members; all must share one image size."""
    models, backbones, sizes = [], [], set()
    for path in checkpoints:
        model, meta = load_checkpoint_any(path, map_location=device)
        models.append(model.to(device).eval())
        backbones.append(meta["backbone"])
        sizes.add(int(meta["image_size"]))
    if len(sizes) != 1:
        raise ValueError(f"All checkpoints must share one image size, got {sorted(sizes)}.")
    return models, backbones, sizes.pop()


def _count_peaks(models: list[torch.nn.Module], backbones: list[str], size: int, array: np.ndarray,
                 rows: list[int], targets: list[int] | None, batch_size: int, device: str) -> dict[str, Any]:
    """Classify the map peak of every image; ``targets=None`` explains the highest-scoring finding."""
    transform = build_transforms(False, size)
    counts = {"without_mask": dict.fromkeys(CATEGORIES, 0), "with_mask": dict.fromkeys(CATEGORIES, 0)}
    for start in range(0, len(rows), batch_size):
        grays = [np.array(array[row]) for row in rows[start:start + batch_size]]
        batch = torch.stack([transform(torch.from_numpy(g).unsqueeze(0)) for g in grays]).to(device)
        if targets is None:
            with torch.no_grad():
                logits = torch.stack([model(batch)[:, :NUM_CLASSES] for model in models]).mean(dim=0)
            target = logits.argmax(dim=1)
        else:
            target = torch.tensor(targets[start:start + batch_size], device=device)
        maps = np.mean([cam_maps(model, batch, target, backbone, "layercam", CAM_LAYERS[backbone])
                        for model, backbone in zip(models, backbones)], axis=0)
        for gray, cam in zip(grays, maps):
            mask = body_mask(gray)
            counts["without_mask"][classify_peak(cam, np.ones_like(mask))] += 1
            counts["with_mask"][classify_peak(cam * mask, mask)] += 1
        LOGGER.info("%d / %d images", min(start + batch_size, len(rows)), len(rows))
    report: dict[str, Any] = {"n_pairs": len(rows)}
    for name, block in counts.items():
        report[name] = {key: value / len(rows) for key, value in block.items()}
    return report


def _write(report: dict[str, Any], out_path: Path | None) -> dict[str, Any]:
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def artifact_check(checkpoints: list[Path], processed_dir: Path, out_path: Path | None = None,
                   bbox_csv: Path | None = None, batch_size: int = 16, max_pairs: int | None = None,
                   device: str = "cpu") -> dict[str, Any]:
    """Share of heat-map peaks per category on the box subset, without and with the body mask.

    Returns:
        ``{"n_pairs": n, "without_mask": {category: share}, "with_mask": {category: share}}``
        (also written to ``out_path`` when given).

    Raises:
        ValueError: If image sizes differ or no box belongs to an image in the index.
    """
    models, backbones, size = _load(checkpoints, device)
    index = pd.read_csv(processed_dir / "image_index.csv")
    row_of = dict(zip(index["image"], index["row"]))
    boxes = load_bboxes(bbox_csv or processed_dir / "bbox.csv")
    boxes = boxes[boxes["image"].isin(row_of)]
    if boxes.empty:
        raise ValueError("No box belongs to an image in image_index.csv.")
    pairs = sorted({(image, finding) for image, finding in zip(boxes["image"], boxes["finding"])})
    if max_pairs:
        pairs = pairs[:max_pairs]
    array = np.load(processed_dir / f"images_{size}.npy", mmap_mode="r")
    report = _count_peaks(models, backbones, size, array, [row_of[image] for image, _ in pairs],
                          [PATHOLOGIES.index(finding) for _, finding in pairs], batch_size, device)
    return _write(report, out_path)


def no_finding_check(checkpoints: list[Path], processed_dir: Path, out_path: Path | None = None,
                     split: str = "test", n: int = 500, seed: int = 42, batch_size: int = 16,
                     device: str = "cpu") -> dict[str, Any]:
    """Same check on a random sample of images of ``split`` with no finding at all.

    The map explains the finding the models score highest (what the app shows for ROUTINE images).

    Raises:
        ValueError: If the split has no image without a finding.
    """
    models, backbones, size = _load(checkpoints, device)
    frame = pd.read_csv(processed_dir / f"{split}.csv")
    normal = frame[frame[list(PATHOLOGIES)].sum(axis=1) == 0]
    if normal.empty:
        raise ValueError(f"The {split} split has no image without a finding.")
    sample = normal.sample(n=min(n, len(normal)), random_state=seed)
    index = pd.read_csv(processed_dir / "image_index.csv")
    row_of = dict(zip(index["image"], index["row"]))
    array = np.load(processed_dir / f"images_{size}.npy", mmap_mode="r")
    report = _count_peaks(models, backbones, size, array, [row_of[name] for name in sample["image"]],
                          None, batch_size, device)
    return _write(report, out_path)


def format_report(report: dict[str, Any]) -> str:
    """Small table: share of peaks per category, without and with the body mask."""
    lines = [f"{report['n_pairs']} images / image-finding pairs", f"{'where the peak is':<18}{'no mask':>10}{'mask':>10}"]
    for name in CATEGORIES:
        lines.append(f"{name:<18}{report['without_mask'][name]:>10.1%}{report['with_mask'][name]:>10.1%}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bbox-csv", type=Path, default=None)
    parser.add_argument("--no-finding", action="store_true",
                        help="check a sample of no-finding images of --split instead of the box subset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args = parse_args()
    if args.no_finding:
        report = no_finding_check(args.checkpoints, args.processed_dir, args.out, args.split, args.n,
                                  args.seed, args.batch_size, args.device)
    else:
        report = artifact_check(args.checkpoints, args.processed_dir, args.out, args.bbox_csv,
                                args.batch_size, args.max_pairs, args.device)
    print(format_report(report))


if __name__ == "__main__":
    main()