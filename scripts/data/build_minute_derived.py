#!/usr/bin/env python3
"""Build the minute-derived raw datasets (docs/data-documentation.md §1.7).

Reduces each ``stk_mins_1min_by_date`` partition to one PIT-stamped row per
stock-day for every dataset in ``DERIVED`` -- ``intraday_flow`` (signed order
flow) and ``intraday_stats`` (realized moments and the intraday profile) -- and
writes ``data/raw/<dataset>/trade_date=<D>.parquet`` with the lake's usual
sidecar. One streamed pass, one minute partition in memory at a time and read
once for all datasets still missing that day; existing output partitions are
skipped unless ``--rebuild``, so the nightly run costs one trade date.

Run it after the minute download has landed:

    ~/miniconda3/envs/quant/bin/python scripts/data/build_minute_derived.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

add_repo_src(__file__)

from autotrade.data_sources.tushare.io import write_parquet
from autotrade.environment.data.intraday_flow import (
    INTRADAY_FLOW_DATASET,
    MINUTE_COLUMNS,
    MINUTE_DATASET,
    aggregate_intraday_flow,
)
from autotrade.environment.data.intraday_stats import (
    INTRADAY_STATS_DATASET,
    INTRADAY_STATS_MINUTE_COLUMNS,
    aggregate_intraday_stats,
)

# dataset -> (stock-day reducer, minute columns it reads)
DERIVED = {
    INTRADAY_FLOW_DATASET: (aggregate_intraday_flow, MINUTE_COLUMNS),
    INTRADAY_STATS_DATASET: (aggregate_intraday_stats, INTRADAY_STATS_MINUTE_COLUMNS),
}
_PARTITION = re.compile(r"trade_date=(\d{8})\.parquet$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--start-date", default="", help="YYYYMMDD; default: earliest minute partition.")
    parser.add_argument("--end-date", default="", help="YYYYMMDD; default: latest minute partition.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DERIVED),
        help="Build only this dataset (repeatable); default: every derived dataset.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recompute partitions that already exist instead of skipping them.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw_dir = Path(args.raw_dir)
    minute_dir = raw_dir / MINUTE_DATASET
    if not minute_dir.is_dir():
        raise SystemExit(f"missing minute dataset: {minute_dir}")
    datasets = sorted(set(args.dataset or DERIVED))

    sources: list[tuple[str, Path]] = []
    for path in sorted(minute_dir.glob("trade_date=*.parquet")):
        match = _PARTITION.search(path.name)
        if match is None:
            continue
        day = match.group(1)
        if args.start_date and day < args.start_date:
            continue
        if args.end_date and day > args.end_date:
            continue
        sources.append((day, path))
    if not sources:
        raise SystemExit(f"no minute partitions in {minute_dir} for the requested window")

    started = time.perf_counter()
    written = {dataset: [] for dataset in datasets}
    rows = dict.fromkeys(datasets, 0)
    days_done = 0
    for day, path in sources:
        todo = [
            dataset
            for dataset in datasets
            if args.rebuild or not (raw_dir / dataset / f"trade_date={day}.parquet").exists()
        ]
        if not todo:
            continue
        columns = sorted({column for dataset in todo for column in DERIVED[dataset][1]})
        minutes = pd.read_parquet(path, columns=columns)
        for dataset in todo:
            aggregate = DERIVED[dataset][0]
            frame = aggregate(minutes)
            if frame.empty:
                # A minute partition that yields no tradable stock-day is a
                # source defect, not an empty session: refuse rather than
                # publish a zero-row partition that later reads as "no data".
                raise SystemExit(f"{path} produced no {dataset} rows")
            write_parquet(
                raw_dir / dataset / f"trade_date={day}.parquet",
                frame,
                api_name=f"derived:{dataset}",
                params={
                    "dataset": dataset,
                    "source_dataset": MINUTE_DATASET,
                    "source_partition": path.name,
                    "trade_date": day,
                },
                fields=list(frame.columns),
            )
            written[dataset].append(day)
            rows[dataset] += len(frame)
        del minutes
        days_done += 1
        if days_done % 100 == 0:
            print(
                json.dumps(
                    {
                        "note": "minute_derived_progress",
                        "days_done": days_done,
                        "last_trade_date": day,
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
    print(
        json.dumps(
            {
                "status": "ok",
                "minute_partitions": len(sources),
                "datasets": {
                    dataset: {
                        "output_dir": str(raw_dir / dataset),
                        "partitions_written": len(written[dataset]),
                        "partitions_skipped": len(sources) - len(written[dataset]),
                        "rows": rows[dataset],
                        "first_trade_date": written[dataset][0] if written[dataset] else None,
                        "last_trade_date": written[dataset][-1] if written[dataset] else None,
                    }
                    for dataset in datasets
                },
                "elapsed_seconds": round(time.perf_counter() - started, 1),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
