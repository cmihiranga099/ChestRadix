"""Unit tests for ``src.data_prep`` (Stage 2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_prep import (
    NO_FINDING_COLUMN,
    PATHOLOGIES,
    assert_no_patient_leakage,
    iterative_patient_split,
    load_metadata,
    split_by_patient,
)


def make_frame(n_patients: int = 400, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic image-level frame with 1 to 4 images per patient.

    Label prevalences are varied on purpose: some classes are common, one is
    rare (about 8% of patients) so stratification is actually tested.
    """
    rng = np.random.default_rng(seed)
    prevalence = np.linspace(0.30, 0.02, len(PATHOLOGIES))
    prevalence[-1] = 0.08  # one deliberately rare class
    rows: list[dict] = []
    for patient_id in range(n_patients):
        patient_labels = (rng.random(len(PATHOLOGIES)) < prevalence).astype(int)
        for image_number in range(int(rng.integers(1, 5))):
            row = {"image": f"{patient_id:08d}_{image_number:03d}.png",
                   "patient_id": patient_id}
            row.update({name: int(value) for name, value in zip(PATHOLOGIES, patient_labels)})
            row[NO_FINDING_COLUMN] = int(patient_labels.sum() == 0)
            rows.append(row)
    return pd.DataFrame(rows)


def write_csv(path: Path, finding_labels: list[str]) -> Path:
    """Write a minimal NIH-style metadata CSV."""
    frame = pd.DataFrame(
        {
            "Image Index": [f"{i:08d}_000.png" for i in range(len(finding_labels))],
            "Finding Labels": finding_labels,
            "Patient ID": list(range(len(finding_labels))),
            "Patient Age": [40] * len(finding_labels),
            "Patient Gender": ["M"] * len(finding_labels),
            "View Position": ["PA"] * len(finding_labels),
        }
    )
    frame.to_csv(path, index=False)
    return path


def test_load_metadata_parses_multi_label_strings(tmp_path: Path) -> None:
    csv_path = write_csv(tmp_path / "meta.csv",
                         ["Atelectasis|Effusion", "No Finding", "Hernia"])
    meta = load_metadata(csv_path)
    assert list(meta["image"]) == ["00000000_000.png", "00000001_000.png", "00000002_000.png"]
    assert meta.loc[0, "Atelectasis"] == 1 and meta.loc[0, "Effusion"] == 1
    assert meta.loc[0, NO_FINDING_COLUMN] == 0
    assert meta.loc[1, list(PATHOLOGIES)].sum() == 0
    assert meta.loc[1, NO_FINDING_COLUMN] == 1
    assert meta.loc[2, "Hernia"] == 1
    assert {"patient_age", "patient_gender", "view_position"} <= set(meta.columns)


def test_load_metadata_rejects_unknown_label(tmp_path: Path) -> None:
    csv_path = write_csv(tmp_path / "meta.csv", ["Atelectasis", "NotADisease"])
    with pytest.raises(ValueError, match="Unknown finding labels"):
        load_metadata(csv_path)


def test_load_metadata_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "meta.csv"
    pd.DataFrame({"Image Index": ["a.png"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_metadata(path)


def test_split_is_patient_disjoint_and_complete() -> None:
    frame = make_frame()
    train, val, test = split_by_patient(frame, (0.70, 0.15, 0.15), seed=1)
    assert len(train) + len(val) + len(test) == len(frame)
    assert_no_patient_leakage({"train": train, "val": val, "test": test})
    shares = np.array([train.patient_id.nunique(), val.patient_id.nunique(),
                       test.patient_id.nunique()]) / frame.patient_id.nunique()
    assert np.allclose(shares, [0.70, 0.15, 0.15], atol=0.03)


def test_split_keeps_rare_label_in_every_part() -> None:
    frame = make_frame(n_patients=600, seed=3)
    rare = PATHOLOGIES[-1]
    parts = split_by_patient(frame, (0.70, 0.15, 0.15), seed=2)
    positive_patients = [part.loc[part[rare] == 1, "patient_id"].nunique() for part in parts]
    total = sum(positive_patients)
    assert total > 20, "synthetic data should contain enough rare positives"
    assert min(positive_patients) >= 3
    assert 0.60 <= positive_patients[0] / total <= 0.80


def test_split_is_reproducible_and_seed_sensitive() -> None:
    frame = make_frame()
    first = split_by_patient(frame, (0.70, 0.15, 0.15), seed=7)
    second = split_by_patient(frame, (0.70, 0.15, 0.15), seed=7)
    other = split_by_patient(frame, (0.70, 0.15, 0.15), seed=8)
    assert first[2].image.tolist() == second[2].image.tolist()
    assert first[2].image.tolist() != other[2].image.tolist()


def test_iterative_split_assigns_every_patient_including_all_negative() -> None:
    rng = np.random.default_rng(0)
    labels = (rng.random((200, 5)) < 0.1).astype(int)
    labels[:50] = 0  # patients without any finding
    assignment = iterative_patient_split(labels, (0.7, 0.15, 0.15), rng)
    assert assignment.shape == (200,)
    assert set(np.unique(assignment)) <= {0, 1, 2}
    assert (assignment >= 0).all()


def test_leakage_assertion_raises_on_shared_patient() -> None:
    first = pd.DataFrame({"patient_id": [1, 2]})
    second = pd.DataFrame({"patient_id": [2, 3]})
    with pytest.raises(RuntimeError, match="Patient leakage"):
        assert_no_patient_leakage({"train": first, "test": second})