#!/usr/bin/env python3
"""Build the derived ``intraday_flow`` raw dataset (docs/data-documentation.md §3.1).

Reduces each ``stk_mins_1min_by_date`` partition to one PIT-stamped order-flow
row per stock-day and writes ``data/raw/intraday_flow/trade_date=<D>.parquet``
with the lake's usual sidecar. One streamed pass, one minute partition in
memory at a time; existing output partitions are skipped unless ``--rebuild``,
so the nightly run costs one trade date.

Run it after the evening minute download has landed:

    ~/miniconda3/envs/quant/bin/python scripts/data/build_intraday_flow.py
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

_PARTITION = re.compile(r"trade_date=(\d{8})\.parquet$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--start-date", default="", help="YYYYMMDD; default: earliest minute partition.")
    parser.add_argument("--end-date", default="", help="YYYYMMDD; default: latest minute partition.")
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
    output_dir = raw_dir / INTRADAY_FLOW_DATASET

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
    written: list[str] = []
    skipped = 0
    rows_total = 0
    for day, path in sources:
        target = output_dir / f"trade_date={day}.parquet"
        if target.exists() and not args.rebuild:
            skipped += 1
            continue
        minutes = pd.read_parquet(path, columns=list(MINUTE_COLUMNS))
        flow = aggregate_intraday_flow(minutes)
        del minutes
        if flow.empty:
            # A minute partition that yields no tradable stock-day is a source
            # defect, not an empty trading session: refuse rather than publish
            # a zero-row partition that later reads as "no order flow that day".
            raise SystemExit(f"{path} produced no intraday_flow rows")
        write_parquet(
            target,
            flow,
            api_name=f"derived:{INTRADAY_FLOW_DATASET}",
            params={
                "dataset": INTRADAY_FLOW_DATASET,
                "source_dataset": MINUTE_DATASET,
                "source_partition": path.name,
                "trade_date": day,
            },
            fields=list(flow.columns),
        )
        written.append(day)
        rows_total += len(flow)
        del flow
        if len(written) % 100 == 0:
            print(
                json.dumps(
                    {
                        "note": "intraday_flow_progress",
                        "partitions_done": len(written),
                        "partitions_total": len(sources) - skipped,
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
                "dataset": INTRADAY_FLOW_DATASET,
                "output_dir": str(output_dir),
                "partitions_written": len(written),
                "partitions_skipped": skipped,
                "rows": rows_total,
                "first_trade_date": written[0] if written else None,
                "last_trade_date": written[-1] if written else None,
                "elapsed_seconds": round(time.perf_counter() - started, 1),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
