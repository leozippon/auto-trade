#!/usr/bin/env python3
"""Compare experiments by their walk-forward record, as one markdown table.

Read-only over each experiment's ledger: the same ``walk_forward_report`` and
``experiment_verdict`` the per-experiment report uses, so a row here can never
disagree with that report. One row per Epoch, because the graduation term reads
the last Epoch's transitions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

add_repo_src(__file__)

from autotrade.pipelines.ledger import (
    ExperimentLedger,
    experiment_verdict,
    latest_fold_records,
    latest_heldout_records,
)
from autotrade.pipelines.reporting import walk_forward_report

_COLUMNS = (
    "experiment",
    "epoch",
    "transitions",
    "positive",
    "chain return",
    "benchmark",
    "chain excess",
    "sharpe/step",
    "excess@2x slip",
    "null pctile",
    "held-out",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument(
        "experiment_ids",
        nargs="*",
        help="Defaults to every experiment directory carrying a ledger.",
    )
    args = parser.parse_args()
    rows: list[list[str]] = []
    for ledger_path in _ledgers(args.experiments_root, args.experiment_ids):
        rows.extend(_rows(ledger_path))
    print(_markdown(rows))
    return 0


def _ledgers(root: Path, experiment_ids: list[str]) -> list[Path]:
    names = experiment_ids or sorted(
        item.name for item in root.iterdir() if item.is_dir()
    )
    paths = []
    for name in names:
        path = root / name / "ledgers" / "experiment_ledger.jsonl"
        if path.is_file():
            paths.append(path)
        elif experiment_ids:
            raise SystemExit(f"no ledger for experiment {name}: {path}")
    return paths


def _rows(ledger_path: Path) -> list[list[str]]:
    ledger = ExperimentLedger(ledger_path)
    experiment_id = ledger_path.parents[1].name
    folds = list(latest_fold_records(ledger.read("fold")).values())
    # strict=False: an experiment still running has periods without a verdict,
    # which is a state to report, not a failure.
    verdict = experiment_verdict(latest_heldout_records(ledger.read("heldout")), strict=False)
    heldout = str((verdict or {}).get("status") or "not_reached")
    rows = []
    for epoch in walk_forward_report(folds):
        chain = epoch["chain"]
        rows.append(
            [
                experiment_id,
                str(epoch["epoch_id"]),
                str(chain["transitions"]),
                str(chain["positive_transitions"]),
                _pct(chain["return"]),
                _pct(chain["benchmark_return"]),
                _pct(chain["excess_return"]),
                _ratio(chain["sharpe_per_transition"]),
                _pct(chain["excess_at_2x_slippage_sum"]),
                _share(chain["mean_excess_percentile"]),
                heldout,
            ]
        )
    if not rows:
        rows.append([experiment_id, "-", "0", "0", *["-"] * 6, heldout])
    return rows


def _pct(value: object) -> str:
    return "-" if value is None else f"{float(value) * 100:+.2f}%"


def _ratio(value: object) -> str:
    return "-" if value is None else f"{float(value):+.2f}"


def _share(value: object) -> str:
    return "-" if value is None else f"{float(value):.2f}"


def _markdown(rows: list[list[str]]) -> str:
    widths = [
        max(len(column), *(len(row[index]) for row in rows)) if rows else len(column)
        for index, column in enumerate(_COLUMNS)
    ]
    lines = [
        "| " + " | ".join(column.ljust(width) for column, width in zip(_COLUMNS, widths)) + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines += [
        "| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |"
        for row in rows
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
