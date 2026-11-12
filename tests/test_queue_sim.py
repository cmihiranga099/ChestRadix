"""Tests for the queue simulation."""
from __future__ import annotations

import numpy as np
import pytest

from src.queue_sim import POLICIES, build_pool, run_sweep, simulate


def test_fifo_and_priority_order_on_a_hand_checked_example() -> None:
    arrivals, services = np.array([0.0, 1.0, 2.0]), np.array([5.0, 5.0, 5.0])
    fifo = simulate(arrivals, services, np.zeros(3), servers=1)
    assert fifo.tolist() == [0.0, 5.0, 10.0]
    priority = simulate(arrivals, services, np.array([0.0, 1.0, 0.0]), servers=1)
    assert priority.tolist() == [0.0, 10.0, 5.0]          # case 2 (key 0) jumps ahead of case 1


def test_aging_serves_a_case_that_waited_too_long() -> None:
    arrivals, services = np.array([0.0, 1.0, 2.0]), np.array([5.0, 5.0, 5.0])
    keys = np.array([0.0, 1.0, 0.0])
    aged = simulate(arrivals, services, keys, servers=1, aging=3.0)
    assert aged.tolist() == [0.0, 5.0, 10.0]               # case 1 had waited 4 >= 3 minutes at t=5
    not_yet = simulate(arrivals, services, keys, servers=1, aging=100.0)
    assert not_yet.tolist() == [0.0, 10.0, 5.0]


def test_two_servers_and_idle_gaps() -> None:
    arrivals, services = np.array([0.0, 0.0, 0.0, 50.0]), np.array([10.0, 10.0, 10.0, 1.0])
    start = simulate(arrivals, services, np.zeros(4), servers=2)
    assert start.tolist() == [0.0, 0.0, 10.0, 50.0]        # the late case finds an idle radiologist


def test_simulation_matches_the_mm1_formula() -> None:
    rng = np.random.default_rng(0)
    n, mean, load = 60000, 8.0, 0.5
    arrivals = np.cumsum(rng.exponential(mean / load, n))
    services = rng.exponential(mean, n)
    start = simulate(arrivals, services, np.zeros(n), servers=1)
    wait = (start - arrivals)[n // 10:]
    expected = mean * load / (1 - load)                    # M/M/1: Wq = s * rho / (1 - rho) = 8 min
    assert wait.mean() == pytest.approx(expected, rel=0.12)
    assert (start >= arrivals).all()


def _pool(n: int = 4000, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    true = rng.choice(3, size=n, p=[0.5, 0.3, 0.2])
    score = np.clip(0.2 + 0.5 * (true == 2) + rng.normal(0, 0.2, n), 0, 1)
    predicted = np.where(score > 0.5, 2, np.where(true >= 1, 1, 0))
    return {"true": true, "predicted": predicted, "score": score}


def test_triage_shortens_the_immediate_wait_and_oracle_is_best() -> None:
    report = run_sweep(_pool(), loads=(0.9,), servers=2, cases=1500, seeds=12)
    policies = report["loads"]["0.90"]["policies"]
    wait = {p: policies[p]["IMMEDIATE_mean"]["mean"] for p in POLICIES}
    assert wait["model_tier"] < 0.6 * wait["fifo"]
    assert wait["oracle"] <= wait["model_tier"] + 1e-9
    cut = report["loads"]["0.90"]["immediate_mean_wait_reduction_vs_fifo"]["model_tier"]
    assert cut["share_ci"][0] > 0.3
    # priority is paid for by the ROUTINE cases, aging limits that cost
    assert policies["model_tier"]["ROUTINE_mean"]["mean"] > policies["fifo"]["ROUTINE_mean"]["mean"]
    assert (policies["model_tier_aging"]["ROUTINE_p95"]["mean"]
            < policies["model_tier"]["ROUTINE_p95"]["mean"])


def test_policies_converge_when_the_department_is_nearly_idle() -> None:
    report = run_sweep(_pool(), loads=(0.2,), servers=4, cases=1500, seeds=8)
    policies = report["loads"]["0.20"]["policies"]
    assert policies["fifo"]["IMMEDIATE_mean"]["mean"] < 3.0
    assert (policies["fifo"]["IMMEDIATE_mean"]["mean"]
            - policies["oracle"]["IMMEDIATE_mean"]["mean"]) < 1.0


def test_loads_must_be_stable() -> None:
    with pytest.raises(ValueError):
        run_sweep(_pool(200), loads=(1.0,), cases=100, seeds=2)


def test_pool_from_a_run_uses_expert_tiers_when_given(tmp_path) -> None:
    from tests.test_expert_validation import _setup

    run, processed, labels, _, _ = _setup(tmp_path)
    nih = build_pool(run, processed)
    expert = build_pool(run, processed, labels)
    assert len(nih["true"]) == 1200 and len(expert["true"]) == 300
    assert set(np.unique(expert["predicted"])) <= {0, 1, 2}
    assert expert["score"].shape == (300,)