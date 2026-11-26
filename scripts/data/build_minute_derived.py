#!/usr/bin/env python3
"""Build the minute-derived raw datasets (docs/DATA_DOCUMENTATION.md §1.7).

Reduces each ``stk_mins_1min_by_date`` partition to the PIT-stamped rows of
every dataset in ``DERIVED`` -- one row per stock-day for ``intraday_flow``
(signed order flow) and ``intraday_stats`` (realized moments and the intraday
profile), one per bar for the ``intraday_bars_<N>min`` datasets -- and writes
``data/raw/<dataset>/trade_date=<D>.parquet`` with the lake's usual sidecar.
One streamed pass, one minute partition in memory at a time and read once for
all datasets still missing that day. Each sidecar names the write of the minute
partition it was reduced from (``source_write_id``); an output partition that
is one committed write of the minute partition's present write is skipped
unless ``--rebuild``, so the nightly run costs one trade date, while a
partition a killed run left half-committed, and every partition of a day whose
minutes the lake has since re-written, is built again.

Run it after the minute download has landed:

    ~/miniconda3/envs/quant/bin/python scripts/data/build_minute_derived.py
"""
from __future__ import annotations

import argparse
import functools
import json
import re
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

add_repo_src(__file__)

from autotrade.data_sources.tushare.io import committed_partition_intact, write_parquet
from autotrade.environment.data.contracts import INTRADAY_BARS_DATASETS
from autotrade.environment.data.intraday_bars import (
    INTRADAY_BARS_MINUTE_COLUMNS,
    aggregate_intraday_bars,
)
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
from autotrade.environment.data.pit import WRITE_ID_METADATA_KEY, parquet_meta

# dataset -> (reducer of one day's minute bars, minute columns it reads)
DERIVED = {
    INTRADAY_FLOW_DATASET: (aggregate_intraday_flow, MINUTE_COLUMNS),
    INTRADAY_STATS_DATASET: (aggregate_intraday_stats, INTRADAY_STATS_MINUTE_COLUMNS),
    **{
        dataset: (functools.partial(aggregate_intraday_bars, dataset=dataset), INTRADAY_BARS_MINUTE_COLUMNS)
        for dataset in INTRADAY_BARS_DATASETS
    },
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
        help="Recompute partitions that are already committed instead of skipping them.",
    )
    return parser


def _reduced_from(output: Path, source_write_id: str) -> bool:
    """Whether ``output`` is one committed write reduced from that write of its minute partition."""
    if not committed_partition_intact(output):
        return False
    return (parquet_meta(output).get("params") or {}).get("source_write_id") == source_write_id


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
        # One open file gives both the identity and the rows, so the write a
        # sidecar names is the write that was reduced even if the minute job
        # replaces the partition meanwhile.
        source = pq.ParquetFile(path)
        source_write_id = (source.schema_arrow.metadata or {}).get(WRITE_ID_METADATA_KEY, b"").decode("ascii")
        if not source_write_id:
            raise SystemExit(f"{path} carries no write id, so what is derived from it could not follow a re-write")
        todo = [
            dataset
            for dataset in datasets
            if args.rebuild or not _reduced_from(raw_dir / dataset / f"trade_date={day}.parquet", source_write_id)
        ]
        if not todo:
            continue
        columns = sorted({column for dataset in todo for column in DERIVED[dataset][1]})
        minutes = source.read(columns=columns, use_pandas_metadata=True).to_pandas()
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
                    "source_write_id": source_write_id,
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
