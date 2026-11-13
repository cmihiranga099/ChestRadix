"""Tests for the ablation table."""
from __future__ import annotations

import json

import pytest

from src.ablation import ablation_table, collect_row, markdown_table


def _write_run(path, auroc=0.78, with_triage=True) -> None:
    path.mkdir(parents=True)
    (path / "test_metrics.json").write_text(json.dumps(
        {"macro": {"auroc": auroc, "auroc_ci": [auroc - 0.01, auroc + 0.01]}}), encoding="utf-8")
    if with_triage:
        classes = {"tuned": {"weighted_kappa": 0.28, "weighted_kappa_ci": [0.26, 0.30],
                             "immediate_recall": 0.92, "under_triage_rate": 0.07, "over_triage_rate": 0.52},
                   "ranking": {"immediate_auroc": 0.72, "immediate_auroc_ci": [0.71, 0.73]}}
        (path / "triage_metrics.json").write_text(json.dumps(
            {"primary_source": "classes", "sources": {"classes": classes}}), encoding="utf-8")


def test_collect_row_reads_numbers(tmp_path) -> None:
    _write_run(tmp_path / "a")
    row = collect_row(tmp_path / "a")
    assert row["run"] == "a" and row["macro_auroc"] == "0.780" and row["macro_auroc_ci"] == "[0.770, 0.790]"
    assert row["immediate_recall"] == "0.920" and row["immediate_auroc"] == "0.720"


def test_missing_files_become_dashes(tmp_path) -> None:
    (tmp_path / "empty").mkdir()
    _write_run(tmp_path / "no_triage", with_triage=False)
    assert collect_row(tmp_path / "empty")["macro_auroc"] == "-"
    row = collect_row(tmp_path / "no_triage")
    assert row["macro_auroc"] == "0.780" and row["weighted_kappa"] == "-" and row["immediate_auroc_ci"] == "-"


def test_table_csv_and_unknown_run(tmp_path) -> None:
    _write_run(tmp_path / "a")
    _write_run(tmp_path / "b", auroc=0.77)
    rows = ablation_table([tmp_path / "a", tmp_path / "b"], tmp_path / "out" / "ablation.csv")
    assert [r["run"] for r in rows] == ["a", "b"]
    assert (tmp_path / "out" / "ablation.csv").read_text(encoding="utf-8").startswith("run,macro_auroc")
    assert markdown_table(rows).count("\n") == 3
    with pytest.raises(FileNotFoundError):
        ablation_table([tmp_path / "missing"])