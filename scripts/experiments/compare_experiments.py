#!/usr/bin/env python3
"""Compare experiments by their walk-forward record, as one markdown table.

Read-only over each experiment's ledger: the same ``walk_forward_report`` and
``experiment_verdict`` the per-experiment report uses, so a row here can never
disagree with that report. One row per Epoch, because the graduation term reads
the last Epoch's transitions. Every chain figure is a forward figure: each
transition is the inherited strategy on ground it was not developed on, and
``scored on`` says whether that ground was the Fold's new quarter alone (a
trailing window) or the whole Validation window. ``DSR (last)`` is the
deflated-Sharpe probability of the Epoch's latest frozen node, the one a
Held-out would replay.
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
    "scored on",
    "transitions",
    "positive",
    "chain return",
    "benchmark",
    "chain excess",
    "sharpe/step",
    "excess@2x slip",
    "null pctile",
    "DSR (last)",
    "held-out",
)
_SCORED_ON = {"step_result": "step quarter", "validation_result": "full window"}


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
                _scored_on(epoch["transitions"]),
                str(chain["transitions"]),
                str(chain["positive_transitions"]),
                _pct(chain["return"]),
                _pct(chain["benchmark_return"]),
                _pct(chain["excess_return"]),
                _ratio(chain["sharpe_per_transition"]),
                _pct(chain["excess_at_2x_slippage_sum"]),
                _share(chain["mean_excess_percentile"]),
                _share(_last_deflated_sharpe(folds, str(epoch["epoch_id"]))),
                heldout,
            ]
        )
    if not rows:
        rows.append([experiment_id, "-", "-", "0", "0", *["-"] * 7, heldout])
    return rows


def _scored_on(transitions: list[dict[str, object]]) -> str:
    sources = {str(row.get("source")) for row in transitions}
    if not sources:
        return "-"
    if len(sources) > 1:
        return "mixed"
    return _SCORED_ON.get(sources.pop(), "-")


def _last_deflated_sharpe(folds: list[dict[str, object]], epoch_id: str) -> float | None:
    """Deflated-Sharpe probability of the Epoch's latest frozen node, or None."""
    frozen = sorted(
        (
            record
            for record in folds
            if str(record.get("epoch_id")) == epoch_id
            and record.get("frozen_strategy_artifact_id")
            and isinstance(record.get("selection_statistics"), dict)
        ),
        key=lambda record: str(record.get("validation_period") or ""),
    )
    if not frozen:
        return None
    value = frozen[-1]["selection_statistics"].get("deflated_sharpe_probability")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


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
