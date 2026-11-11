"""Grad-CAM explanations and a faithfulness check for the chest X-ray classifier.

Grad-CAM (Selvaraju et al.) weights the channels of the last convolutional layer by the average
gradient of the explained logit, sums them, keeps the positive part and upsamples it to the image.
The hooks live in :mod:`src.model` (``activation_and_gradient``); this module adds:

* :func:`gradcam_maps`      heat maps in [0, 1] for a batch,
* :func:`overlay`           a heat map drawn over the radiograph (RGB, uint8),
* :func:`deletion_scores`   a faithfulness test that needs no extra annotations: blank out the
  fraction of pixels the map marks as most important and see how much the probability of the
  explained finding falls, compared with blanking the same number of random pixels,
* :func:`faithfulness_run`  that test on a sample of test images, plus a figure of examples.

A map that is faithful lowers the probability more than random blanking does. This shows that the
highlighted region matters **to the model**; it does not show that the region is the true lesion
(that would need bounding boxes, see the NIH BBox file).

Command line::

    python -m src.gradcam --checkpoint runs/e1_densenet121_s42/best.pt --out runs/e1_densenet121_s42/gradcam \
        --processed-dir data/processed --n 200 --examples 6
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.config import NUM_CLASSES, PATHOLOGIES, locate_processed_dir
from src.dataset import ChestXrayDataset, build_transforms
from src.model import activation_and_gradient
from src.tier_head import load_checkpoint_any

LOGGER = logging.getLogger("chestradix.gradcam")


def gradcam_maps(model: nn.Module, images: torch.Tensor, class_index: int | torch.Tensor,
                 backbone: str = "densenet121", layer_path: str | None = None) -> np.ndarray:
    """Grad-CAM heat maps for one finding per image.

    Args:
        model: Model in evaluation mode.
        images: Normalised batch ``(B, 3, H, W)``.
        class_index: Finding index (one for all images, or one per image) whose logit is explained.
        backbone: Backbone name (selects the default target layer).
        layer_path: Optional override of the target layer.

    Returns:
        Array ``(B, H, W)`` with values in [0, 1]; each map is min-max scaled, and an image whose
        map is flat (no positive evidence) gives all zeros.
    """
    _, activation, gradient = activation_and_gradient(model, images, class_index, backbone, layer_path)
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    low = cam.amin(dim=(1, 2), keepdim=True)
    span = cam.amax(dim=(1, 2), keepdim=True) - low
    cam = torch.where(span > 1e-12, (cam - low) / span.clamp_min(1e-12), torch.zeros_like(cam))
    return cam.cpu().numpy().astype(np.float32)


def overlay(gray: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Draw a heat map over a grey-scale radiograph.

    Args:
        gray: ``(H, W)`` uint8 radiograph.
        cam: ``(H, W)`` map in [0, 1] (same size as ``gray``).
        alpha: Opacity of the heat map where the map is 1.

    Returns:
        ``(H, W, 3)`` uint8 RGB image.
    """
    from matplotlib import colormaps

    base = np.repeat(gray[..., None].astype(np.float32) / 255.0, 3, axis=2)
    heat = colormaps["jet"](np.clip(cam, 0.0, 1.0))[..., :3]
    weight = (alpha * np.clip(cam, 0.0, 1.0))[..., None]
    blended = np.clip((1.0 - weight) * base + weight * heat, 0.0, 1.0)
    return (blended * 255).astype(np.uint8)


def deletion_scores(model: nn.Module, images: torch.Tensor, class_index: torch.Tensor,
                    cams: np.ndarray, fraction: float = 0.2, seed: int = 0) -> dict[str, np.ndarray]:
    """Probability drop when the most important pixels are blanked, versus random pixels.

    Blanking sets the normalised input to 0 (the ImageNet mean grey).

    Args:
        model: Model in evaluation mode.
        images: Normalised batch ``(B, 3, H, W)``.
        class_index: Explained finding per image, shape ``(B,)``.
        cams: Heat maps ``(B, H, W)``.
        fraction: Share of pixels to blank.
        seed: Seed of the random baseline.

    Returns:
        Dictionary with per-image arrays ``base`` (probability), ``drop_cam`` and ``drop_random``
        (probability lost by each kind of blanking).
    """
    rng = np.random.default_rng(seed)
    flat = cams.reshape(len(cams), -1)
    # Rank pixels with a tiny random tie-break so a flat map does not blank an arbitrary block.
    order = np.argsort(-(flat + rng.random(flat.shape) * 1e-6), axis=1)
    keep = max(1, int(round(fraction * flat.shape[1])))
    cam_mask = np.zeros_like(flat, dtype=bool)
    rand_mask = np.zeros_like(flat, dtype=bool)
    rows = np.arange(len(flat))[:, None]
    cam_mask[rows, order[:, :keep]] = True
    rand_mask[rows, np.argsort(rng.random(flat.shape), axis=1)[:, :keep]] = True

    def probability(batch: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = model(batch)[:, :NUM_CLASSES]
        return torch.sigmoid(logits).gather(1, class_index.view(-1, 1))[:, 0].cpu().numpy()

    def blank(mask: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(mask.reshape(cams.shape)).to(images.device)
        return images * (~tensor).unsqueeze(1).to(images.dtype)

    base = probability(images)
    return {"base": base, "drop_cam": base - probability(blank(cam_mask)),
            "drop_random": base - probability(blank(rand_mask))}


def _bootstrap_ci(values: np.ndarray, draws: int = 2000, seed: int = 0) -> list[float]:
    """95% bootstrap interval of the mean."""
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _select_cases(dataset: ChestXrayDataset, n: int, seed: int) -> np.ndarray:
    """Random sample of test images that have at least one positive finding."""
    candidates = np.flatnonzero(dataset.labels.sum(axis=1) > 0)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(candidates, size=min(n, len(candidates)), replace=False))


def save_examples(path: Path, grays: list[np.ndarray], cams: np.ndarray, titles: list[str]) -> Path:
    """Figure with the radiograph and its overlay for each example (two rows)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(grays)
    figure, axes = plt.subplots(2, count, figsize=(2.4 * count, 5.2), squeeze=False)
    for column in range(count):
        axes[0][column].imshow(grays[column], cmap="gray", vmin=0, vmax=255)
        axes[1][column].imshow(overlay(grays[column], cams[column]))
        axes[0][column].set_title(titles[column], fontsize=7)
        axes[0][column].axis("off")
        axes[1][column].axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def faithfulness_run(checkpoint: Path, processed_dir: Path, out_dir: Path, split: str = "test",
                     n: int = 200, fraction: float = 0.2, examples: int = 6, seed: int = 42,
                     batch_size: int = 16, device: str = "cpu") -> dict[str, Any]:
    """Faithfulness test and example figure on a random sample of images with a true finding.

    For each image the explained finding is the **true positive finding with the highest model
    logit**, so the map answers "where did the model look to support a finding that is really there".

    Args:
        checkpoint: ``best.pt`` of a run (14 or 16 outputs).
        processed_dir: Folder with the packed arrays and split CSV files.
        out_dir: Output folder for ``gradcam_faithfulness.json`` and ``gradcam_examples.png``.
        split: Split to sample.
        n: Number of images.
        fraction: Share of pixels blanked.
        examples: Number of images in the example figure (0 disables).
        seed: Sampling and baseline seed.
        batch_size: Images per forward pass.
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        The report dictionary (also written to disk).
    """
    model, meta = load_checkpoint_any(checkpoint, map_location=device)
    model.to(device).eval()
    size, backbone = int(meta["image_size"]), meta["backbone"]
    dataset = ChestXrayDataset(processed_dir, split, build_transforms(False, size), size)
    chosen = _select_cases(dataset, n, seed)
    grays = [np.array(dataset._images()[dataset.rows[i]]) for i in chosen]

    drops_cam, drops_random, bases, targets, cams_all = [], [], [], [], []
    for start in range(0, len(chosen), batch_size):
        ids = chosen[start:start + batch_size]
        images = torch.stack([dataset[int(i)][0] for i in ids]).to(device)
        labels = torch.from_numpy(dataset.labels[ids]).to(device)
        with torch.no_grad():
            logits = model(images)[:, :NUM_CLASSES]
        target = torch.where(labels > 0.5, logits, torch.full_like(logits, -1e9)).argmax(dim=1)
        cams = gradcam_maps(model, images, target, backbone)
        scores = deletion_scores(model, images, target, cams, fraction, seed + start)
        drops_cam.append(scores["drop_cam"])
        drops_random.append(scores["drop_random"])
        bases.append(scores["base"])
        targets.append(target.cpu().numpy())
        cams_all.append(cams)

    drop_cam, drop_random = np.concatenate(drops_cam), np.concatenate(drops_random)
    base, target = np.concatenate(bases), np.concatenate(targets)
    difference = drop_cam - drop_random
    per_class = {}
    for index in np.unique(target):
        mask = target == index
        if mask.sum() >= 5:
            per_class[PATHOLOGIES[int(index)]] = {
                "n": int(mask.sum()), "mean_drop_cam": float(drop_cam[mask].mean()),
                "mean_drop_random": float(drop_random[mask].mean())}
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint), "split": split, "n_images": int(len(chosen)),
        "blanked_fraction": fraction, "mean_probability": float(base.mean()),
        "mean_drop_cam": float(drop_cam.mean()), "mean_drop_random": float(drop_random.mean()),
        "mean_difference": float(difference.mean()),
        "mean_difference_ci": _bootstrap_ci(difference, seed=seed),
        "share_cam_beats_random": float((difference > 0).mean()),
        "per_class": per_class,
        "note": "Faithfulness to the model, not proof of lesion localisation.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "gradcam_faithfulness.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    if examples > 0:
        cams = np.concatenate(cams_all)[:examples]
        titles = [f"{PATHOLOGIES[int(t)]}\np={b:.2f}" for t, b in zip(target[:examples], base[:examples])]
        save_examples(out_dir / "gradcam_examples.png", grays[:examples], cams, titles)
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    processed = args.processed_dir or locate_processed_dir()
    report = faithfulness_run(args.checkpoint, processed, args.out, args.split, args.n,
                              args.fraction, args.examples, args.seed, device=args.device)
    low, high = report["mean_difference_ci"]
    LOGGER.info("Images: %d | blanked %.0f%% of pixels", report["n_images"], 100 * report["blanked_fraction"])
    LOGGER.info("Mean probability drop  Grad-CAM %.3f | random %.3f", report["mean_drop_cam"],
                report["mean_drop_random"])
    LOGGER.info("Difference %.3f [%.3f, %.3f] | Grad-CAM beats random on %.0f%% of images",
                report["mean_difference"], low, high, 100 * report["share_cam_beats_random"])
    LOGGER.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()