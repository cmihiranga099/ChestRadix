"""Training script for ChestRadix.

Trains DenseNet-121 (or the ResNet-50 baseline) on the packed NIH data with a
patient-level split, then saves the best checkpoint (by validation macro-AUROC), the
training history and the raw validation and test logits (needed later for calibration
and for ``evaluate.py``).

With ``--tier-head`` the network gets two extra outputs (``tier >= URGENT`` and
``tier == IMMEDIATE``, derived from the 14 labels). The 14 finding logits are still saved
as ``val_logits.npy`` / ``test_logits.npy`` (so every later step works unchanged) and the
two tier logits as ``val_tier_logits.npy`` / ``test_tier_logits.npy``.

Example (Kaggle or laptop)::

    python -m src.train --backbone densenet121 --seed 42 --run-name e1_densenet121_s42
    python -m src.train --tier-head --seed 42 --run-name e2_densenet121_tier_s42
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.config import NUM_CLASSES, TrainConfig, locate_processed_dir, runs_dir
from src.dataset import build_dataloaders, compute_pos_weight, seed_everything
from src.evaluate import fast_auroc, predict_logits
from src.model import build_model, count_parameters, save_checkpoint
from src.tier_head import (
    NUM_TIER_OUTPUTS,
    load_checkpoint_any,
    split_outputs,
    tier_targets,
    tier_targets_torch,
)

LOGGER = logging.getLogger("chestradix.train")


def pick_device() -> torch.device:
    """Return CUDA if available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def macro_auroc(labels: np.ndarray, logits: np.ndarray) -> float:
    """Mean AUROC over the classes that have both positives and negatives.

    Args:
        labels: ``(N, C)`` binary labels.
        logits: ``(N, C)`` raw model outputs.

    Returns:
        Macro AUROC (NaN if no class is evaluable).
    """
    values = [fast_auroc(labels[:, c].astype(int), logits[:, c]) for c in range(labels.shape[1])]
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int
                   ) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up followed by cosine decay to zero (stepped once per batch).

    Args:
        optimizer: Optimizer to schedule.
        total_steps: Total number of optimisation steps.
        warmup_steps: Number of warm-up steps.

    Returns:
        A ``LambdaLR`` scheduler.
    """
    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one(
    config: TrainConfig,
    processed_dir: Path,
    run_dir: Path,
    pretrained: bool = True,
    max_batches: int | None = None,
    time_budget_minutes: float | None = None,
    tier_head: bool = False,
    tier_loss_weight: float = 1.0,
) -> dict[str, Any]:
    """Train one model and write all artefacts into ``run_dir``.

    Args:
        config: Training configuration.
        processed_dir: Directory with the packed array and split CSV files.
        run_dir: Output folder for this run.
        pretrained: Start from ImageNet weights.
        max_batches: Limit the batches per epoch (for quick smoke tests only).
        time_budget_minutes: Stop after the epoch in which this wall-clock budget is used up.
        tier_head: Add two tier outputs (``tier >= URGENT``, ``tier == IMMEDIATE``) trained
            together with the 14 findings. Model selection still uses the 14-class macro-AUROC.
        tier_loss_weight: Weight of the tier-head loss relative to the 14-class loss.

    Returns:
        Summary dictionary (also written to ``summary.json``).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    device = pick_device()
    use_amp = config.amp and device.type == "cuda"
    LOGGER.info("Device: %s | AMP: %s | run: %s", device, use_amp, run_dir.name)

    loaders = build_dataloaders(processed_dir, config)
    train_ds = loaders["train"].dataset
    pos_weight = compute_pos_weight(train_ds.labels, config.pos_weight_mode,
                                    config.pos_weight_cap).to(device)
    num_outputs = NUM_CLASSES + (NUM_TIER_OUTPUTS if tier_head else 0)
    model = build_model(config.backbone, num_classes=num_outputs, dropout=config.dropout,
                        pretrained=pretrained).to(device)
    LOGGER.info("%s | %d outputs | %.1fM trainable parameters", config.backbone, num_outputs,
                count_parameters(model) / 1e6)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    tier_pos_weight = compute_pos_weight(tier_targets(train_ds.labels), config.pos_weight_mode,
                                         config.pos_weight_cap).to(device)
    tier_criterion = nn.BCEWithLogitsLoss(pos_weight=tier_pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    steps_per_epoch = len(loaders["train"]) if max_batches is None else min(
        max_batches, len(loaders["train"]))
    total_steps = max(1, steps_per_epoch * config.epochs)
    scheduler = make_scheduler(optimizer, total_steps, steps_per_epoch * config.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({**config.to_dict(), "pretrained": pretrained, "max_batches": max_batches,
                   "pos_weight": pos_weight.cpu().tolist(), "tier_head": tier_head,
                   "tier_loss_weight": tier_loss_weight,
                   "tier_pos_weight": tier_pos_weight.cpu().tolist()}, handle, indent=2)

    best_auroc, best_epoch, stale = -1.0, -1, 0
    history: list[dict[str, float]] = []
    started = time.time()
    checkpoint_path = run_dir / "best.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        running, batches, epoch_start = 0.0, 0, time.time()
        for step, (images, labels) in enumerate(loaders["train"]):
            if max_batches is not None and step >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images).float()
            loss = criterion(outputs[:, :NUM_CLASSES], labels)
            if tier_head:
                loss = loss + tier_loss_weight * tier_criterion(
                    outputs[:, NUM_CLASSES:], tier_targets_torch(labels))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss.item())
            batches += 1

        val_outputs, val_labels = predict_logits(model, loaders["val"], device, use_amp)
        val_logits, val_tier = split_outputs(val_outputs)
        val_auroc = macro_auroc(val_labels, val_logits)
        record = {"epoch": epoch, "train_loss": running / max(1, batches),
                  "val_macro_auroc": val_auroc, "lr": optimizer.param_groups[0]["lr"],
                  "seconds": time.time() - epoch_start}
        tier_note = ""
        if val_tier is not None:
            targets = tier_targets(val_labels)
            record["val_urgent_plus_auroc"] = fast_auroc(targets[:, 0].astype(int), val_tier[:, 0])
            record["val_immediate_auroc"] = fast_auroc(targets[:, 1].astype(int), val_tier[:, 1])
            tier_note = " | tier AUROC urgent+ %.4f immediate %.4f" % (
                record["val_urgent_plus_auroc"], record["val_immediate_auroc"])
        history.append(record)
        LOGGER.info("epoch %2d | loss %.4f | val macro-AUROC %.4f%s | %.0fs", epoch,
                    record["train_loss"], val_auroc, tier_note, record["seconds"])

        if val_auroc > best_auroc:
            best_auroc, best_epoch, stale = val_auroc, epoch, 0
            save_checkpoint(checkpoint_path, model, config,
                            extra={"epoch": epoch, "val_macro_auroc": val_auroc,
                                   "num_outputs": num_outputs, "tier_head": tier_head})
        else:
            stale += 1
        with (run_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        if stale >= config.patience:
            LOGGER.info("Early stopping (no improvement for %d epochs)", config.patience)
            break
        if time_budget_minutes and (time.time() - started) / 60 > time_budget_minutes:
            LOGGER.info("Time budget of %.0f minutes used up; stopping", time_budget_minutes)
            break

    # Reload the best weights and save raw logits for calibration and evaluation.
    best_model, _ = load_checkpoint_any(checkpoint_path, map_location=device)
    best_model.to(device)
    summary: dict[str, Any] = {"run": run_dir.name, "best_epoch": best_epoch,
                               "best_val_macro_auroc": best_auroc,
                               "epochs_run": len(history), "tier_head": tier_head,
                               "minutes": (time.time() - started) / 60}
    for split in ("val", "test"):
        outputs, labels = predict_logits(best_model, loaders[split], device, use_amp)
        logits, tier_logits = split_outputs(outputs)
        np.save(run_dir / f"{split}_logits.npy", logits)
        summary[f"{split}_macro_auroc"] = macro_auroc(labels, logits)
        if tier_logits is not None:
            np.save(run_dir / f"{split}_tier_logits.npy", tier_logits)
            targets = tier_targets(labels)
            summary[f"{split}_urgent_plus_auroc"] = fast_auroc(targets[:, 0].astype(int),
                                                               tier_logits[:, 0])
            summary[f"{split}_immediate_auroc"] = fast_auroc(targets[:, 1].astype(int),
                                                             tier_logits[:, 1])
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    LOGGER.info("Done: best epoch %d, val %.4f, test %.4f", best_epoch,
                summary["val_macro_auroc"], summary["test_macro_auroc"])
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default=defaults.backbone,
                        choices=["densenet121", "resnet50"])
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--pos-weight-mode", default=defaults.pos_weight_mode,
                        choices=["full", "sqrt", "capped", "none"])
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--tier-head", action="store_true",
                        help="Add the two tier outputs (urgent_or_higher, immediate).")
    parser.add_argument("--tier-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit batches per epoch (smoke tests only).")
    parser.add_argument("--time-budget-min", type=float, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    config = replace(
        TrainConfig(), backbone=args.backbone, seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, dropout=args.dropout, patience=args.patience,
        num_workers=args.num_workers, pos_weight_mode=args.pos_weight_mode,
        image_size=args.image_size, amp=not args.no_amp,
    )
    processed = args.processed_dir or locate_processed_dir(config.image_size)
    name = args.run_name or f"{config.backbone}_s{config.seed}"
    train_one(config, processed, runs_dir() / name, pretrained=not args.no_pretrained,
              max_batches=args.max_batches, time_budget_minutes=args.time_budget_min,
              tier_head=args.tier_head, tier_loss_weight=args.tier_loss_weight)


if __name__ == "__main__":
    main()