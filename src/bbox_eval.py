"""Localisation check of Grad-CAM against radiologist bounding boxes (NIH ``BBox_List_2017.csv``).

NIH provides boxes drawn by radiologists for about 880 test-style images and 8 findings
(Atelectasis, Cardiomegaly, Effusion, Infiltrate, Mass, Nodule, Pneumonia, Pneumothorax). For each
(image, finding) pair with a box this module asks whether the Grad-CAM map **for that finding**
lands on the box:

* **pointing hit**: the hottest pixel of the map lies inside a box (any box of that finding),
* **IoU hit**: the map thresholded at ``threshold`` of its maximum overlaps the boxes with IoU at
  least ``iou_min``,
* **energy**: share of the map's total heat that falls inside the boxes.

Two baselines show how much of that is real: a **centre prior** (a fixed Gaussian blob in the middle
of the image, which already hits many boxes because lesions and hearts are central) and a
**shuffled** map (the Grad-CAM of another image). Only a clear margin over both is evidence that the
map follows the image. Boxes are in 1024-px image coordinates and are rescaled to the packed size.

Input folder (``--processed-dir``): ``images_224.npy`` and ``image_index.csv`` (columns ``image``,
``row``) plus ``bbox.csv`` (columns ``image, finding, x, y, w, h``; the raw NIH file is also read).

CLI::

    python -m src.bbox_eval --checkpoints runs/e1_densenet121_s42/best.pt --processed-dir data/bbox_subset --out runs/e1_densenet121_s42/bbox
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

from src.config import NUM_CLASSES, PATHOLOGIES
from src.dataset import build_transforms
from src.gradcam import gradcam_maps
from src.tier_head import load_checkpoint_any

LOGGER = logging.getLogger("chestradix.bbox_eval")
BOX_SIZE = 1024.0                       # NIH box coordinates refer to 1024 x 1024 images
FINDING_ALIAS = {"Infiltrate": "Infiltration"}


def load_bboxes(path: Path) -> pd.DataFrame:
    """Read boxes as columns ``image, finding, x, y, w, h`` (clean or raw NIH file).

    Raises:
        ValueError: If the file has fewer than six columns or a finding is not one of the 14.
    """
    frame = pd.read_csv(path)
    if frame.shape[1] < 6:
        raise ValueError(f"{path} needs at least 6 columns (image, finding, x, y, w, h).")
    frame = frame.iloc[:, :6].copy()
    frame.columns = ["image", "finding", "x", "y", "w", "h"]
    frame["finding"] = frame["finding"].replace(FINDING_ALIAS)
    unknown = sorted(set(frame["finding"]) - set(PATHOLOGIES))
    if unknown:
        raise ValueError(f"Unknown findings in the box file: {unknown}")
    return frame.astype({"x": float, "y": float, "w": float, "h": float})


def box_mask(boxes: np.ndarray, size: int) -> np.ndarray:
    """Boolean ``(size, size)`` mask of the union of boxes ``(x, y, w, h)`` given in 1024-px units."""
    mask = np.zeros((size, size), dtype=bool)
    scale = size / BOX_SIZE
    for x, y, w, h in np.atleast_2d(boxes):
        x0, y0 = int(np.floor(x * scale)), int(np.floor(y * scale))
        x1, y1 = int(np.ceil((x + w) * scale)), int(np.ceil((y + h) * scale))
        mask[max(y0, 0):min(y1, size), max(x0, 0):min(x1, size)] = True
    return mask


def map_scores(cam: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Pointing hit, IoU and heat share of one map against one box mask."""
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    total = float(cam.sum())
    binary = cam >= threshold * float(cam.max()) if cam.max() > 0 else np.zeros_like(mask)
    union = float((binary | mask).sum())
    return {"hit": float(mask[peak]),
            "iou": float((binary & mask).sum() / union) if union else 0.0,
            "energy": float(cam[mask].sum() / total) if total > 0 else 0.0}


def centre_prior(size: int, sigma: float = 0.25) -> np.ndarray:
    """Fixed Gaussian blob in the image centre (baseline map)."""
    axis = (np.arange(size) + 0.5) / size - 0.5
    grid = np.exp(-(axis[None, :] ** 2 + axis[:, None] ** 2) / (2 * sigma ** 2))
    return (grid / grid.max()).astype(np.float32)


def _bootstrap(values: np.ndarray, n_boot: int = 1000, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarise(rows: list[dict[str, Any]], iou_min: float, seed: int = 42) -> dict[str, Any]:
    """Aggregate per-pair results (mean with 95% bootstrap interval) overall and per finding."""
    frame = pd.DataFrame(rows)

    def block(part: pd.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {"n": int(len(part))}
        for source in ("cam", "centre", "shuffled"):
            hit = part[f"{source}_hit"].to_numpy(float)
            iou_hit = (part[f"{source}_iou"].to_numpy(float) >= iou_min).astype(float)
            out[source] = {
                "pointing_hit": float(hit.mean()), "pointing_hit_ci": _bootstrap(hit, seed=seed),
                f"iou_ge_{iou_min}": float(iou_hit.mean()),
                "mean_iou": float(part[f"{source}_iou"].mean()),
                "mean_energy": float(part[f"{source}_energy"].mean())}
        diff = part["cam_hit"].to_numpy(float) - part["centre_hit"].to_numpy(float)
        out["pointing_hit_minus_centre"] = float(diff.mean())
        out["pointing_hit_minus_centre_ci"] = _bootstrap(diff, seed=seed)
        return out

    report = {"overall": block(frame), "per_finding": {}}
    for name, part in frame.groupby("finding"):
        if len(part) >= 5:
            report["per_finding"][name] = block(part)
    detected = frame[frame["probability"] >= 0.5]
    report["model_detected_p_ge_0.5"] = block(detected) if len(detected) >= 5 else None
    return report


def bbox_run(checkpoints: list[Path], processed_dir: Path, out_dir: Path, bbox_csv: Path | None = None,
             threshold: float = 0.5, iou_min: float = 0.1, seed: int = 42, batch_size: int = 16,
             device: str = "cpu") -> dict[str, Any]:
    """Run the pointing-game and IoU check for one or more models (maps averaged over models).

    Args:
        checkpoints: ``best.pt`` files (14 or 16 outputs, same image size).
        processed_dir: Folder with ``images_224.npy``, ``image_index.csv`` and ``bbox.csv``.
        out_dir: Output folder for ``bbox_localisation.json``.
        bbox_csv: Box file (default ``<processed_dir>/bbox.csv``).
        threshold: Share of the map maximum used to binarise the map for IoU.
        iou_min: IoU needed for an "IoU hit".
        seed: Seed for the shuffled baseline and the bootstrap.
        batch_size: Images per forward pass.
        device: ``"cpu"`` or ``"cuda"``.

    Raises:
        ValueError: If no box matches an image in the index or the image sizes differ.
    """
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
    array = np.load(processed_dir / f"images_{size}.npy", mmap_mode="r")

    cams, probabilities = [], []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        batch = torch.stack([transform(torch.from_numpy(np.array(array[row_of[image]])).unsqueeze(0))
                             for image, _, _ in chunk]).to(device)
        target = torch.tensor([PATHOLOGIES.index(finding) for _, finding, _ in chunk], device=device)
        maps, probs = [], []
        for model, backbone in zip(models, backbones):
            maps.append(gradcam_maps(model, batch, target, backbone))
            with torch.no_grad():
                logit = model(batch)[:, :NUM_CLASSES].gather(1, target[:, None])[:, 0]
            probs.append(torch.sigmoid(logit).cpu().numpy())
        cams.append(np.mean(maps, axis=0))
        probabilities.append(np.mean(probs, axis=0))
    cams, probabilities = np.concatenate(cams), np.concatenate(probabilities)

    rng = np.random.default_rng(seed)
    shuffle = rng.permutation(len(pairs))
    if len(pairs) > 1:                                   # never pair a map with itself
        for i in np.where(shuffle == np.arange(len(pairs)))[0]:
            j = (i + 1) % len(pairs)
            shuffle[i], shuffle[j] = shuffle[j], shuffle[i]
    prior = centre_prior(size)

    rows = []
    for k, (image, finding, box) in enumerate(pairs):
        mask = box_mask(box, size)
        row: dict[str, Any] = {"image": image, "finding": finding, "probability": float(probabilities[k]),
                               "box_area_share": float(mask.mean())}
        for source, cam in (("cam", cams[k]), ("centre", prior), ("shuffled", cams[shuffle[k]])):
            for key, value in map_scores(cam, mask, threshold).items():
                row[f"{source}_{key}"] = value
        rows.append(row)

    report = summarise(rows, iou_min, seed)
    report.update({"checkpoints": [str(p) for p in checkpoints], "n_pairs": len(pairs),
                   "n_images": len({p[0] for p in pairs}), "threshold": threshold, "iou_min": iou_min,
                   "note": ("Boxes are radiologist-drawn NIH annotations. Compare the Grad-CAM with the "
                            "centre-prior and shuffled baselines, not with zero.")})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bbox_localisation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "bbox_localisation_pairs.csv", index=False)
    return report


def _print(report: dict[str, Any], iou_min: float) -> None:
    def line(label: str, block: dict[str, Any]) -> str:
        c, m, s = block["cam"], block["centre"], block["shuffled"]
        key = f"iou_ge_{iou_min}"
        return (f"{label:<20} n={block['n']:>4} | pointing hit: CAM {c['pointing_hit']:.2f} "
                f"centre {m['pointing_hit']:.2f} shuffled {s['pointing_hit']:.2f} | IoU>={iou_min}: "
                f"CAM {c[key]:.2f} centre {m[key]:.2f} shuffled {s[key]:.2f} | energy CAM {c['mean_energy']:.2f} "
                f"centre {m['mean_energy']:.2f}")

    print(f"{report['n_pairs']} (image, finding) pairs from {report['n_images']} images")
    print(line("ALL", report["overall"]))
    low, high = report["overall"]["pointing_hit_minus_centre_ci"]
    print(f"Pointing hit, CAM minus centre prior: {report['overall']['pointing_hit_minus_centre']:+.3f} "
          f"[{low:+.3f}, {high:+.3f}]")
    if report["model_detected_p_ge_0.5"]:
        print(line("model p>=0.5", report["model_detected_p_ge_0.5"]))
    for name, block in report["per_finding"].items():
        print(line(name, block))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bbox-csv", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--iou-min", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    report = bbox_run(args.checkpoints, args.processed_dir, args.out, args.bbox_csv, args.threshold,
                      args.iou_min, args.seed, args.batch_size, args.device)
    _print(report, args.iou_min)
    print(f"Wrote {args.out / 'bbox_localisation.json'}")


if __name__ == "__main__":
    main()