"""Queue simulation: does triage shorten the wait of the cases that cannot wait?

Radiograph reports are produced by ``c`` radiologists working in parallel. Cases arrive at random
(Poisson) and each takes a random reading time (exponential). The policies differ only in **who is
read next** when a radiologist becomes free (reading is never interrupted):

* ``fifo``              first come, first served (the baseline of most departments),
* ``model_tier``        by the tier the model predicts (IMMEDIATE, then URGENT, then ROUTINE),
* ``model_tier_aging``  like ``model_tier`` but a case that has waited longer than ``aging_minutes``
                        is read next, so no ROUTINE case waits indefinitely,
* ``model_score``       by the model's continuous IMMEDIATE score,
* ``oracle``            by the true tier (an upper bound: perfect triage).

Cases are drawn from the real test predictions, so the model's mistakes (an IMMEDIATE case that is
called ROUTINE) cost real waiting time. Within one (seed, load) every policy sees the **same**
arrivals, reading times and cases, so differences between policies are paired.

The headline number is the wait of the cases that truly are IMMEDIATE. Arrival rate, reading time and
number of radiologists are **assumptions** (no hospital data): read the results as the relative
benefit of a policy under a given load, not as absolute minutes in a real department.

Command line::

    python -m src.queue_sim --run-dir runs/ens_e1_e2_e4 --processed-dir data/processed
"""
from __future__ import annotations

import argparse
import heapq
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PATHOLOGIES, locate_processed_dir
from src.metrics import tiers_from_binary
from src.triage import IMMEDIATE, _load_scores, assign_tiers

LOGGER = logging.getLogger("chestradix.queue_sim")
POLICIES: tuple[str, ...] = ("fifo", "model_tier", "model_tier_aging", "model_score", "oracle")
TIER_NAMES: tuple[str, ...] = ("ROUTINE", "URGENT", "IMMEDIATE")


# --------------------------------------------------------------------------- #
# Discrete-event simulation
# --------------------------------------------------------------------------- #
def simulate(arrivals: np.ndarray, services: np.ndarray, keys: np.ndarray, servers: int = 2,
             aging: float | None = None) -> np.ndarray:
    """Start time of every case under a non-preemptive priority rule.

    Args:
        arrivals: Arrival times in minutes, sorted ascending.
        services: Reading time of each case in minutes.
        keys: Priority key of each case; the **smallest** key is read first, ties by arrival.
        servers: Number of radiologists.
        aging: If given, a case that has waited at least this many minutes is read before any
            other (oldest first).

    Returns:
        Array of start times (the wait of case ``i`` is ``start[i] - arrivals[i]``).
    """
    n = len(arrivals)
    start = np.empty(n)
    free = [0.0] * servers
    heapq.heapify(free)
    by_key: list[tuple[float, float, int]] = []
    by_age: list[tuple[float, int]] = []
    served = np.zeros(n, dtype=bool)
    next_case, done = 0, 0
    while done < n:
        now = heapq.heappop(free)
        if next_case < n and arrivals[next_case] > now and not _any_waiting(by_key, served):
            now = float(arrivals[next_case])
        while next_case < n and arrivals[next_case] <= now:
            heapq.heappush(by_key, (float(keys[next_case]), float(arrivals[next_case]), next_case))
            if aging is not None:
                heapq.heappush(by_age, (float(arrivals[next_case]), next_case))
            next_case += 1
        chosen = -1
        if aging is not None:
            while by_age and served[by_age[0][1]]:
                heapq.heappop(by_age)
            if by_age and now - by_age[0][0] >= aging:
                chosen = heapq.heappop(by_age)[1]
        if chosen < 0:
            while True:
                _, _, candidate = heapq.heappop(by_key)
                if not served[candidate]:
                    chosen = candidate
                    break
        served[chosen] = True
        start[chosen] = now
        heapq.heappush(free, now + float(services[chosen]))
        done += 1
    return start


def _any_waiting(by_key: list[tuple[float, float, int]], served: np.ndarray) -> bool:
    """Whether an unserved case is waiting (lazy deletion of served entries)."""
    while by_key and served[by_key[0][2]]:
        heapq.heappop(by_key)
    return bool(by_key)


# --------------------------------------------------------------------------- #
# Case pool from the real predictions
# --------------------------------------------------------------------------- #
def build_pool(run_dir: Path, processed_dir: Path, expert_labels: Path | None = None,
               logits_name: str = "logits_calibrated") -> dict[str, np.ndarray]:
    """True tier, predicted tier and IMMEDIATE score of every case that can arrive.

    Args:
        run_dir: Run folder with test logits and ``triage_metrics.json`` (thresholds tuned on
            validation).
        processed_dir: Folder with ``test.csv``.
        expert_labels: If given, only the radiologist-labelled images are used and the true tier
            comes from the radiologists; otherwise all test images with the NIH tier.
        logits_name: Logit file suffix.

    Returns:
        Dictionary with ``true``, ``predicted`` and ``score`` arrays.
    """
    test = pd.read_csv(processed_dir / "test.csv")
    triage = json.loads((run_dir / "triage_metrics.json").read_text(encoding="utf-8"))
    primary = triage["primary_source"]
    tuning = triage["sources"][primary]["tuning"]
    scores = _load_scores(run_dir, logits_name)[primary][1]
    rows = np.arange(len(test))
    labels = test[list(PATHOLOGIES)].to_numpy(dtype=int)
    if expert_labels is not None:
        from src.expert_validation import _binary

        expert = pd.read_csv(expert_labels).drop_duplicates("Image ID").set_index("Image ID")
        rows = np.flatnonzero(test["image"].isin(expert.index).to_numpy())
        labels = _binary(expert.loc[test["image"].to_numpy()[rows]])
    else:
        labels = labels[rows]
    scores = scores[rows]
    return {"true": tiers_from_binary(labels),
            "predicted": assign_tiers(scores, tuning["t_urgent"], tuning["t_immediate"]),
            "score": scores[:, IMMEDIATE]}


def _keys(policy: str, pool: dict[str, np.ndarray], cases: np.ndarray) -> np.ndarray:
    """Priority keys (smaller is read first) of the drawn cases for a policy."""
    if policy == "fifo":
        return np.zeros(len(cases))
    if policy in ("model_tier", "model_tier_aging"):
        return -pool["predicted"][cases].astype(float)
    if policy == "model_score":
        return -pool["score"][cases]
    if policy == "oracle":
        return -pool["true"][cases].astype(float)
    raise ValueError(f"Unknown policy '{policy}'")


def _wait_stats(wait: np.ndarray, true: np.ndarray, target: float) -> dict[str, float]:
    """Wait statistics per true tier for one simulation run."""
    out: dict[str, float] = {}
    for tier, name in enumerate(TIER_NAMES):
        w = wait[true == tier]
        if len(w) == 0:
            continue
        out[f"{name}_mean"] = float(w.mean())
        out[f"{name}_p90"] = float(np.percentile(w, 90))
        out[f"{name}_p95"] = float(np.percentile(w, 95))
        out[f"{name}_within_target"] = float(np.mean(w <= target))
    return out


def run_sweep(pool: dict[str, np.ndarray], loads: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9, 0.95),
              servers: int = 2, service_minutes: float = 8.0, cases: int = 2000, seeds: int = 50,
              target_minutes: float = 30.0, aging_minutes: float = 120.0, warmup: float = 0.1,
              policies: tuple[str, ...] = POLICIES, seed: int = 42) -> dict[str, Any]:
    """Simulate every policy at every load over many seeds.

    Args:
        pool: Output of :func:`build_pool`.
        loads: Utilisation levels (arrival rate x reading time / servers); must be below 1.
        servers: Number of radiologists.
        service_minutes: Mean reading time.
        cases: Cases per simulated run.
        seeds: Independent runs per load.
        target_minutes: "Seen in time" target used for the share of cases read within it.
        aging_minutes: Waiting time after which ``model_tier_aging`` promotes a case.
        warmup: Share of the first cases left out of the statistics (the queue starts empty).
        policies: Policies to simulate.
        seed: Base seed.

    Returns:
        Dictionary with per-load, per-policy statistics averaged over seeds with 95% intervals
        over seeds, and the paired reduction of the IMMEDIATE wait relative to FIFO.

    Raises:
        ValueError: If a load is not below 1.
    """
    if any(load >= 1.0 or load <= 0.0 for load in loads):
        raise ValueError("Loads must be between 0 and 1 (the queue is unstable at or above 1).")
    skip = int(warmup * cases)
    report: dict[str, Any] = {
        "assumptions": {"servers": servers, "service_minutes": service_minutes,
                        "arrivals": "Poisson", "reading_time": "exponential",
                        "target_minutes": target_minutes, "aging_minutes": aging_minutes,
                        "cases_per_run": cases, "runs_per_load": seeds, "warmup_share": warmup},
        "pool": {"n": int(len(pool["true"])),
                 "tier_prevalence": {n: float(np.mean(pool["true"] == i)) for i, n in enumerate(TIER_NAMES)}},
        "loads": {},
    }
    for load_index, load in enumerate(loads):
        rate = load * servers / service_minutes
        runs: dict[str, list[dict[str, float]]] = {p: [] for p in policies}
        for run in range(seeds):
            rng = np.random.default_rng([seed, load_index, run])
            arrivals = np.cumsum(rng.exponential(1.0 / rate, cases))
            services = rng.exponential(service_minutes, cases)
            drawn = rng.integers(0, len(pool["true"]), cases)
            true = pool["true"][drawn]
            for policy in policies:
                aging = aging_minutes if policy == "model_tier_aging" else None
                start = simulate(arrivals, services, _keys(policy, pool, drawn), servers, aging)
                wait = (start - arrivals)[skip:]
                runs[policy].append(_wait_stats(wait, true[skip:], target_minutes))
        entry: dict[str, Any] = {"policies": {}, "immediate_mean_wait_reduction_vs_fifo": {}}
        for policy in policies:
            keys = sorted({k for r in runs[policy] for k in r})
            stats = {}
            for key in keys:
                values = np.array([r.get(key, np.nan) for r in runs[policy]])
                stats[key] = {"mean": float(np.nanmean(values)),
                              "ci": [float(np.nanpercentile(values, 2.5)),
                                     float(np.nanpercentile(values, 97.5))]}
            entry["policies"][policy] = stats
        if "fifo" in runs:
            base = np.array([r["IMMEDIATE_mean"] for r in runs["fifo"]])
            for policy in policies:
                if policy == "fifo":
                    continue
                other = np.array([r["IMMEDIATE_mean"] for r in runs[policy]])
                minutes, share = base - other, (base - other) / base
                entry["immediate_mean_wait_reduction_vs_fifo"][policy] = {
                    "minutes": float(minutes.mean()),
                    "minutes_ci": [float(np.percentile(minutes, 2.5)), float(np.percentile(minutes, 97.5))],
                    "share": float(share.mean()),
                    "share_ci": [float(np.percentile(share, 2.5)), float(np.percentile(share, 97.5))]}
        report["loads"][f"{load:.2f}"] = entry
    return report


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--expert-labels", type=Path, default=None,
                        help="Use the radiologist-labelled images and tiers instead of NIH labels.")
    parser.add_argument("--loads", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9, 0.95])
    parser.add_argument("--servers", type=int, default=2)
    parser.add_argument("--service-minutes", type=float, default=8.0)
    parser.add_argument("--cases", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--target-minutes", type=float, default=30.0)
    parser.add_argument("--aging-minutes", type=float, default=120.0)
    parser.add_argument("--logits-name", default="logits_calibrated")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON (default: in the run folder).")
    return parser.parse_args()


def _fmt(stats: dict[str, Any], key: str) -> str:
    entry = stats.get(key)
    return "   -   " if entry is None else f"{entry['mean']:6.1f}"


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    processed = args.processed_dir or locate_processed_dir()
    pool = build_pool(args.run_dir, processed, args.expert_labels, args.logits_name)
    report = run_sweep(pool, tuple(args.loads), args.servers, args.service_minutes, args.cases,
                       args.seeds, args.target_minutes, args.aging_minutes)
    source = "radiologist" if args.expert_labels else "NIH"
    out = args.out or args.run_dir / ("queue_simulation_expert.json" if args.expert_labels
                                      else "queue_simulation.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    prevalence = report["pool"]["tier_prevalence"]
    LOGGER.info("Cases from %d test images (%s tiers): ROUTINE %.2f | URGENT %.2f | IMMEDIATE %.2f",
                report["pool"]["n"], source, prevalence["ROUTINE"], prevalence["URGENT"],
                prevalence["IMMEDIATE"])
    LOGGER.info("%d radiologists, mean reading time %.0f min, %d runs x %d cases per load",
                args.servers, args.service_minutes, args.seeds, args.cases)
    for load, entry in report["loads"].items():
        LOGGER.info("--- load %s ---", load)
        LOGGER.info("%-17s | IMMEDIATE mean | p90 | in %.0f min | URGENT mean | ROUTINE mean | p95",
                    "policy", args.target_minutes)
        for policy, stats in entry["policies"].items():
            within = stats["IMMEDIATE_within_target"]["mean"] * 100
            LOGGER.info("%-17s | %s | %s | %5.1f%% | %s | %s | %s", policy, _fmt(stats, "IMMEDIATE_mean"),
                        _fmt(stats, "IMMEDIATE_p90"), within, _fmt(stats, "URGENT_mean"),
                        _fmt(stats, "ROUTINE_mean"), _fmt(stats, "ROUTINE_p95"))
        for policy, cut in entry["immediate_mean_wait_reduction_vs_fifo"].items():
            low, high = cut["share_ci"]
            LOGGER.info("  %s cuts the mean IMMEDIATE wait by %.1f min (%.0f%% [%.0f%%, %.0f%%]) vs FIFO",
                        policy, cut["minutes"], 100 * cut["share"], 100 * low, 100 * high)
    LOGGER.info("Wrote %s", out)


if __name__ == "__main__":
    main()