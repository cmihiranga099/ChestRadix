"""Compare heat-map methods and layers on the NIH radiologist boxes (pointing game).

For each configuration the same image/finding pairs as :mod:`src.bbox_eval` are scored, so the
numbers are comparable with the Stage 11 result. A configuration is written ``method`` or
``method:layer``, for example ``gradcam``, ``gradcampp``, ``layercam:features.denseblock3``.

Command line::

    python -m src.cam_compare --checkpoints runs/e1_densenet121_s42/best.pt --processed-dir data/bbox_subset \
        --out runs/e1_densenet121_s42/cam_compare \
        --configs gradcam gradcampp layercam:features.denseblock3 layercam:features.denseblock2
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

from src.bbox_eval import box_mask, centre_prior, load_bboxes, map_scores, summarise
from src.cam_methods import METHODS, cam_maps
from src.config import NUM_CLASSES, PATHOLOGIES
from src.dataset import build_transforms
from src.tier_head import load_checkpoint_any

LOGGER = logging.getLogger("chestradix.cam_compare")


def parse_config(text: str) -> tuple[str, str | None]:
    """Split ``method[:layer]`` into its parts.

    Raises:
        ValueError: If the method is unknown.
    """
    method, _, layer = text.partition(":")
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}' in '{text}'. Choose from {list(METHODS)}.")
    return method, (layer or None)


def compare_run(checkpoints: list[Path], processed_dir: Path, out_dir: Path, configs: list[str],
                bbox_csv: Path | None = None, threshold: float = 0.5, iou_min: float = 0.1,
                seed: int = 42, batch_size: int = 16, max_pairs: int | None = None,
                device: str = "cpu") -> dict[str, Any]:
    """Score several heat-map configurations against the boxes (maps averaged over the models).

    Args:
        checkpoints: ``best.pt`` files (same image size).
        processed_dir: Folder with the packed images, ``image_index.csv`` and ``bbox.csv``.
        out_dir: Output folder for ``cam_compare.json``.
        configs: Configurations ``method`` or ``method:layer``.
        bbox_csv: Box file (default ``<processed_dir>/bbox.csv``).
        threshold: Share of the map maximum used to binarise a map for IoU.
        iou_min: IoU needed for an "IoU hit".
        seed: Seed for the shuffled baseline and the bootstrap.
        batch_size: Images per forward pass.
        max_pairs: Use only the first N pairs (quick check).
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        ``{config: summary}`` in the format of :func:`src.bbox_eval.summarise` (also written to disk).

    Raises:
        ValueError: If a configuration is unknown, no box matches an image or the image sizes differ.
    """
    parsed = {text: parse_config(text) for text in configs}
    models, backbones, sizes = [], [], set()
    for path in checkpoints:
        model, meta = load_checkpoint_any(path, map_location=device)
        models.append(model.to(device).eval())
        backbones.append(meta["backbone"])
        sizes.add(int(meta["image_size"]))
    if len(sizes) != 1:
        raise ValueError(f"All checkpoints must share one image size, got {sorted(sizes)}.")
    size = sizes.pop()
    transform = build_transforms(False, size)

    index = pd.read_csv(processed_dir / "image_index.csv")
    row_of = dict(zip(index["image"], index["row"]))
    boxes = load_bboxes(bbox_csv or processed_dir / "bbox.csv")
    boxes = boxes[boxes["image"].isin(row_of)]
    if boxes.empty:
        raise ValueError("No box belongs to an image in image_index.csv.")
    pairs = [(image, finding, group[["x", "y", "w", "h"]].to_numpy())
             for (image, finding), group in boxes.groupby(["image", "finding"], sort=True)]
    if max_pairs:
        pairs = pairs[:max_pairs]
    array = np.load(processed_dir / f"images_{size}.npy", mmap_mode="r")

    cams: dict[str, list[np.ndarray]] = {text: [] for text in configs}
    probabilities = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        batch = torch.stack([transform(torch.from_numpy(np.array(array[row_of[image]])).unsqueeze(0))
                             for image, _, _ in chunk]).to(device)
        target = torch.tensor([PATHOLOGIES.index(finding) for _, finding, _ in chunk], device=device)
        probs = []
        for model in models:
            with torch.no_grad():
                logit = model(batch)[:, :NUM_CLASSES].gather(1, target[:, None])[:, 0]
            probs.append(torch.sigmoid(logit).cpu().numpy())
        probabilities.append(np.mean(probs, axis=0))
        for text, (method, layer) in parsed.items():
            cams[text].append(np.mean([cam_maps(model, batch, target, backbone, method, layer)
                                       for model, backbone in zip(models, backbones)], axis=0))
        LOGGER.info("%d / %d pairs", min(start + batch_size, len(pairs)), len(pairs))
    probabilities = np.concatenate(probabilities)

    rng = np.random.default_rng(seed)
    shuffle = rng.permutation(len(pairs))
    if len(pairs) > 1:
        for i in np.where(shuffle == np.arange(len(pairs)))[0]:
            j = (i + 1) % len(pairs)
            shuffle[i], shuffle[j] = shuffle[j], shuffle[i]
    prior = centre_prior(size)

    report: dict[str, Any] = {}
    for text in configs:
        maps = np.concatenate(cams[text])
        rows = []
        for k, (image, finding, box) in enumerate(pairs):
            mask = box_mask(box, size)
            row: dict[str, Any] = {"image": image, "finding": finding, "probability": float(probabilities[k])}
            for source, cam in (("cam", maps[k]), ("centre", prior), ("shuffled", maps[shuffle[k]])):
                for key, value in map_scores(cam, mask, threshold).items():
                    row[f"{source}_{key}"] = value
            rows.append(row)
        report[text] = summarise(rows, iou_min, seed)
        report[text]["n_pairs"] = len(pairs)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cam_compare.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def format_table(report: dict[str, Any], iou_min: float = 0.1) -> str:
    """Overall table (one row per configuration) and a per-finding pointing-hit table."""
    key = f"iou_ge_{iou_min}"
    lines = [f"{'config':<36} | pointing hit | minus centre [95% CI]   | IoU>={iou_min} | energy | hit if p>=0.5"]
    for text, summary in report.items():
        overall = summary["overall"]
        low, high = overall["pointing_hit_minus_centre_ci"]
        detected = summary.get("model_detected_p_ge_0.5")
        det = f"{detected['cam']['pointing_hit']:.2f}" if detected else "-"
        lines.append(f"{text:<36} | {overall['cam']['pointing_hit']:>12.2f} | "
                     f"{overall['pointing_hit_minus_centre']:+.3f} [{low:+.3f}, {high:+.3f}] | "
                     f"{overall['cam'][key]:>8.2f} | {overall['cam']['mean_energy']:.2f}   | {det}")
    findings = sorted({name for summary in report.values() for name in summary["per_finding"]})
    lines.append("")
    lines.append(f"{'pointing hit by finding':<24}" + "".join(f"{text[:18]:>20}" for text in report))
    for name in findings:
        cells = []
        for summary in report.values():
            block = summary["per_finding"].get(name)
            cells.append(f"{block['cam']['pointing_hit']:.2f}" if block else "-")
        lines.append(f"{name:<24}" + "".join(f"{cell:>20}" for cell in cells))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=["gradcam", "gradcampp", "layercam:features.denseblock3"])
    parser.add_argument("--bbox-csv", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--iou-min", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args = parse_args()
    report = compare_run(args.checkpoints, args.processed_dir, args.out, args.configs, args.bbox_csv,
                         args.threshold, args.iou_min, args.seed, args.batch_size, args.max_pairs, args.device)
    print(format_table(report, args.iou_min))
    print(f"\nWrote {args.out / 'cam_compare.json'}")


if __name__ == "__main__":
    main()