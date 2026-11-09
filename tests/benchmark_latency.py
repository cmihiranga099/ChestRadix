"""Measure CPU latency of preprocessing and model inference.

Usage (from the repository root, virtual environment active):
    python tests/benchmark_latency.py
    python tests/benchmark_latency.py --threads 4 --runs 100 --channels-last

Grad-CAM and API timings are added in Stage 8.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import IMAGE_SIZE  # noqa: E402
from src.dataset import build_transforms  # noqa: E402
from src.model import build_model  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    """Return the q-th percentile (0-100) of a list of numbers."""
    return float(np.percentile(np.asarray(values), q))


def summarize(name: str, values_ms: list[float]) -> str:
    """Format mean, p50 and p95 for one timing series."""
    return (f"{name:<12} mean {statistics.fmean(values_ms):7.1f} ms | "
            f"p50 {percentile(values_ms, 50):7.1f} ms | p95 {percentile(values_ms, 95):7.1f} ms")


def main() -> None:
    """Run the benchmark and print a summary."""
    parser = argparse.ArgumentParser(description="ChestRadix CPU latency benchmark")
    parser.add_argument("--backbone", default="densenet121", choices=("densenet121", "resnet50"))
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=0, help="0 keeps the PyTorch default")
    parser.add_argument("--channels-last", action="store_true")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    model = build_model(args.backbone, pretrained=False).eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    transform = build_transforms(train=False, image_size=IMAGE_SIZE)
    rng = np.random.default_rng(0)
    raw = [torch.from_numpy(rng.integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))
           .unsqueeze(0) for _ in range(args.runs + args.warmup)]

    pre_ms: list[float] = []
    fwd_ms: list[float] = []
    with torch.inference_mode():
        for step, image in enumerate(raw):
            start = time.perf_counter()
            batch = transform(image).unsqueeze(0)
            if args.channels_last:
                batch = batch.contiguous(memory_format=torch.channels_last)
            middle = time.perf_counter()
            torch.sigmoid(model(batch))
            end = time.perf_counter()
            if step >= args.warmup:
                pre_ms.append((middle - start) * 1000)
                fwd_ms.append((end - middle) * 1000)

    total_ms = [a + b for a, b in zip(pre_ms, fwd_ms)]
    print(f"backbone={args.backbone} threads={torch.get_num_threads()} "
          f"channels_last={args.channels_last} runs={args.runs}")
    print(summarize("preprocess", pre_ms))
    print(summarize("forward", fwd_ms))
    print(summarize("total", total_ms))
    print("Target: /predict p95 below 350 ms")


if __name__ == "__main__":
    main()