"""Collect the headline numbers of several runs into one ablation table.

Reads ``test_metrics.json``, ``triage_metrics.json`` and ``summary.json`` (when present) from each
run folder, including ensemble folders, and prints a Markdown table plus a CSV file. Triage numbers
always come from the class-score rule (``sources.classes``) so that every row is comparable; a run
without that source shows ``-``.

Usage::

    python -m src.ablation --runs runs/e1_densenet121_s42 runs/e0_resnet50_s42 --out runs/ablation.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("chestradix.ablation")
COLUMNS: tuple[str, ...] = ("run", "macro_auroc", "macro_auroc_ci", "weighted_kappa", "weighted_kappa_ci",
                            "immediate_recall", "under_triage", "over_triage", "immediate_auroc",
                            "immediate_auroc_ci")


def _load(path: Path) -> dict[str, Any] | None:
    """Return the parsed JSON file or ``None`` if it does not exist."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _fmt(value: float | None, digits: int = 3) -> str:
    """Format a number, or ``-`` when it is missing."""
    return "-" if value is None else f"{value:.{digits}f}"


def _ci(interval: list[float] | None) -> str:
    """Format a confidence interval as ``[low, high]`` or ``-``."""
    return "-" if not interval else f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def collect_row(run_dir: Path) -> dict[str, str]:
    """Headline numbers of one run folder.

    Args:
        run_dir: Run or ensemble folder.

    Returns:
        One table row keyed by :data:`COLUMNS`; missing values are ``-``.
    """
    run_dir = Path(run_dir)
    test = _load(run_dir / "test_metrics.json")
    triage = _load(run_dir / "triage_metrics.json")
    macro = test["macro"] if test else {}
    classes = triage["sources"].get("classes") if triage else None
    tuned = classes["tuned"] if classes else {}
    ranking = classes["ranking"] if classes else {}
    return {
        "run": run_dir.name,
        "macro_auroc": _fmt(macro.get("auroc")),
        "macro_auroc_ci": _ci(macro.get("auroc_ci")),
        "weighted_kappa": _fmt(tuned.get("weighted_kappa")),
        "weighted_kappa_ci": _ci(tuned.get("weighted_kappa_ci")),
        "immediate_recall": _fmt(tuned.get("immediate_recall")),
        "under_triage": _fmt(tuned.get("under_triage_rate")),
        "over_triage": _fmt(tuned.get("over_triage_rate")),
        "immediate_auroc": _fmt(ranking.get("immediate_auroc")),
        "immediate_auroc_ci": _ci(ranking.get("immediate_auroc_ci")),
    }


def markdown_table(rows: list[dict[str, str]]) -> str:
    """Render the rows as a Markdown table."""
    header = ["Run", "Macro AUROC [95% CI]", "Weighted kappa [95% CI]", "IMMEDIATE recall",
              "Under-triage", "Over-triage", "IMMEDIATE AUROC [95% CI]"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join([
            row["run"], f"{row['macro_auroc']} {row['macro_auroc_ci']}",
            f"{row['weighted_kappa']} {row['weighted_kappa_ci']}", row["immediate_recall"],
            row["under_triage"], row["over_triage"],
            f"{row['immediate_auroc']} {row['immediate_auroc_ci']}"]) + " |")
    return "\n".join(lines)


def ablation_table(runs: list[Path], out_csv: Path | None = None) -> list[dict[str, str]]:
    """Build the table for several runs and optionally write it as CSV.

    Args:
        runs: Run or ensemble folders, in the order they should appear.
        out_csv: Optional CSV path.

    Returns:
        The rows.

    Raises:
        FileNotFoundError: If a run folder does not exist.
    """
    rows = []
    for run in map(Path, runs):
        if not run.is_dir():
            raise FileNotFoundError(f"Run folder not found: {run}")
        rows.append(collect_row(run))
    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV file.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = ablation_table(args.runs, args.out)
    print(markdown_table(rows))
    if args.out:
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()