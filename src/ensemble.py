"""Average the raw logits of several finished runs into a new run folder (no PyTorch needed).

The runs must have been evaluated on the same validation and test images in the same order
(true for every run made with ``train.py`` from the same packed data). The averaged raw logits
are written as ``val_logits.npy`` / ``test_logits.npy`` (and the tier-head logits when **every**
run has them), so ``calibration``, ``evaluate`` and ``triage`` work on the result unchanged.

Example::

    python -m src.ensemble --runs runs/e1_densenet121_s42 runs/e2_densenet121_tier_s42 \
        --out runs/ens_e1_e2
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("chestradix.ensemble")


def ensemble_runs(run_dirs: list[Path], out_dir: Path) -> dict[str, object]:
    """Average raw logits over ``run_dirs`` and write them into ``out_dir``.

    Args:
        run_dirs: Two or more run folders containing ``val_logits.npy`` and ``test_logits.npy``.
        out_dir: Output folder (created if needed).

    Returns:
        Small report (members and which arrays were written).

    Raises:
        ValueError: If fewer than two runs are given or the array shapes differ.
    """
    if len(run_dirs) < 2:
        raise ValueError("Give at least two runs to ensemble.")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for prefix in ("", "tier_"):
        for split in ("val", "test"):
            name = f"{split}_{prefix}logits.npy"
            if not all((d / name).exists() for d in run_dirs):
                continue
            arrays = [np.load(d / name).astype(np.float64) for d in run_dirs]
            if len({a.shape for a in arrays}) != 1:
                raise ValueError(f"{name}: shapes differ between runs {[a.shape for a in arrays]}")
            np.save(out_dir / name, np.mean(arrays, axis=0).astype(np.float32))
            written.append(name)
    if not {"val_logits.npy", "test_logits.npy"} <= set(written):
        raise ValueError("val_logits.npy and test_logits.npy are required in every run.")
    report = {"members": [str(d) for d in run_dirs], "written": written, "average": "raw logits"}
    (out_dir / "ensemble.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = ensemble_runs(args.runs, args.out)
    LOGGER.info("Wrote %s from %d runs: %s", args.out, len(args.runs), report["written"])


if __name__ == "__main__":
    main()