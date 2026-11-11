"""Tests for the ensemble-disagreement uncertainty flag."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.calibration import sigmoid
from src.config import NUM_CLASSES, PATHOLOGIES
from src.metrics import tiers_from_binary
from src.uncertainty import immediate_scores, selective_report, uncertainty_run, uncertainty_scores


def _members(n: int = 6000, seed: int = 0):
    """Three members; the first 30% of cases are hard (members disagree and are uninformative)."""
    rng = np.random.default_rng(seed)
    labels = (rng.random((n, NUM_CLASSES)) < 0.08).astype(int)
    hard = np.arange(n) < int(0.3 * n)
    logits = []
    for _ in range(3):
        informative = labels * 2.0 - 1.0 + rng.normal(0, 0.3, labels.shape)
        noise = rng.normal(0, 2.0, labels.shape)
        logits.append(np.where(hard[:, None], noise, informative))
    return labels, hard, np.stack([sigmoid(z) for z in logits])


def test_scores_have_expected_shapes_and_agreeing_members_give_zero() -> None:
    same = np.stack([np.full((5, NUM_CLASSES), 0.3)] * 3)
    out = uncertainty_scores(same)
    assert out["score"].shape == (5,) and np.allclose(out["u_immediate"], 0)
    assert np.allclose(out["u_overall"], 0)
    assert np.allclose(immediate_scores(same), 0.3)


def test_flag_finds_the_hard_cases_and_retained_cases_rank_better() -> None:
    labels, hard, members = _members()
    out = uncertainty_scores(members)
    reference = tiers_from_binary(labels)
    cutoff = float(np.quantile(out["u_immediate"], 0.7))
    report = selective_report(reference, out["score"], out["u_immediate"], cutoff)
    flagged = out["u_immediate"] >= cutoff
    assert hard[flagged].mean() > 0.7                         # the flag picks the hard cases
    assert report["auroc_retained"] > report["auroc_all"] > report["auroc_flagged"]
    assert report["flag_rate"] == pytest.approx(0.3, abs=0.02)


def test_uncertainty_run_writes_report(tmp_path) -> None:
    labels, _, members = _members(n=3000)
    processed = tmp_path / "processed"
    processed.mkdir()
    for split in ("val", "test"):
        frame = pd.DataFrame(labels, columns=list(PATHOLOGIES))
        frame.insert(0, "patient_id", np.arange(len(frame)) // 2)
        frame.to_csv(processed / f"{split}.csv", index=False)
    runs = []
    for k in range(3):
        run = tmp_path / f"run{k}"
        run.mkdir()
        logit = np.log(members[k] / (1 - members[k]))
        for split in ("val", "test"):
            np.save(run / f"{split}_logits_calibrated.npy", logit.astype(np.float32))
        runs.append(run)
    report = uncertainty_run(runs, tmp_path / "out", processed, n_boot=30)
    rows = report["measures"]["u_immediate"]
    assert [r["target_flag_rate"] for r in rows] == [0.1, 0.2, 0.3]
    assert all(r["auroc_retained"] >= r["auroc_all"] - 1e-9 for r in rows)
    assert (tmp_path / "out" / "uncertainty.json").exists()
    assert (tmp_path / "out" / "test_u_immediate.npy").exists()


def test_needs_at_least_two_members(tmp_path) -> None:
    from src.uncertainty import member_probabilities

    with pytest.raises(ValueError):
        member_probabilities([tmp_path], "val")