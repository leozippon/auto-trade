"""A repository data lake with research releases published the production way.

Each release is committed in the live lake (the updater lock plus a committed
generation marker) and published by ``pin_research_release`` into a scratch
pin, exactly as the seed prebuild pins its workspace, so the tree on disk is
the one a real deployment holds. Paths are the console's own data roots.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from autotrade.environment.data.contracts import BENCHMARK_INDEXES
from autotrade.environment.data.research_release import (
    ResearchRelease,
    pin_research_release,
)

RAW_DIR = "data/raw"
FUNDAMENTAL_EVENTS_ROOT = "data/pit/fundamental_events"
FUNDAMENTAL_EVENTS_STATUS = "results/data_quality/fundamental_events_status.json"
# Two SSE trading days inside the default Held-out quarter, the least a
# release must reach for an experiment on that geometry to start.
HELDOUT_REACHING_DAYS = ("20250630", "20260630", "20260701", "20260702")
# How far back benchmark history reaches in the backfilled lake (the 20140701
# `download --history-floor` backfill), which an eight-year research period
# needs; pass it as ``history_start``. Moving the floor is this one edit.
BACKFILLED_HISTORY_START = "20140701"
# Benchmark partitions whose vendor history starts after that floor, as the
# backfilled release holds them: CSI 1000's first constituent section (listed
# 2014-10-17), STAR 50's base-date bar and its first section.
LATER_VENDOR_STARTS = {
    ("index_weight", "000852.SH"): "20141031",
    ("index_daily", "000688.SH"): "20191231",
    ("index_weight", "000688.SH"): "20200731",
}


def publish_release(
    repo_root: Path,
    generation_id: str,
    *,
    datasets: Iterable[str],
    trading_days: Sequence[str] = HELDOUT_REACHING_DAYS,
    benchmark_indexes: Sequence[str] = tuple(BENCHMARK_INDEXES),
    history_start: str = "20200102",
) -> ResearchRelease:
    """Commit ``generation_id`` in the live lake and publish its release.

    ``datasets`` get one parquet partition each and ``trading_days`` are both
    the SSE calendar's open days and the daily partitions. ``index_daily`` and
    ``index_weight`` always get their per-index partitions, because every real
    release carries them and an arm's ``benchmark_index`` is checked against
    them at create time; ``benchmark_indexes`` narrows that set to model a
    release published before an index existed. Each partition's first row is
    dated ``history_start`` -- by default the first trading day of the 2020
    research-history floor, the lake before its backfill -- or the index's
    later vendor start. A later call commits a newer generation over the same
    lake, as the nightly chain does.
    """

    raw = repo_root / RAW_DIR
    events_root = repo_root / FUNDAMENTAL_EVENTS_ROOT
    events_root.mkdir(parents=True, exist_ok=True)
    (repo_root / FUNDAMENTAL_EVENTS_STATUS).parent.mkdir(parents=True, exist_ok=True)
    lock = repo_root / ".runtime/tushare/locks/tushare_update.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.touch()
    for name in datasets:
        _write_pair(raw / name / "part.parquet")
    for code in benchmark_indexes:
        # Real footers: the create-time check reads each benchmark partition's
        # first trade_date against research_start.
        for dataset, key in (("index_daily", "ts_code"), ("index_weight", "index_code")):
            first = max(history_start, LATER_VENDOR_STARTS.get((dataset, code), history_start))
            _write_pair(
                raw / dataset / f"{key}={code}" / f"year={first[:4]}.parquet",
                pd.DataFrame({"trade_date": [first]}),
            )
    for day in trading_days:
        _write_pair(raw / "daily" / f"trade_date={day}.parquet")
    for year in sorted({day[:4] for day in trading_days}):
        calendar = raw / "trade_cal" / "exchange=SSE" / f"year={year}.parquet"
        calendar.parent.mkdir(parents=True, exist_ok=True)
        # Replaced, never rewritten in place: a published release hardlinks
        # the previous file, as it does the updater's atomic writes.
        staging = calendar.with_name(f"{calendar.name}.tmp")
        pd.DataFrame(
            {"cal_date": [day for day in trading_days if day[:4] == year], "is_open": 1}
        ).to_parquet(staging, index=False)
        staging.replace(calendar)
        calendar.with_suffix(".parquet.meta.json").write_text("{}\n", encoding="utf-8")
    (raw / ".raw_generation.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "state": "committed",
                "generation_id": generation_id,
                "completed_at": "2026-09-14T00:00:00+00:00",
                "transaction": {"job": "fixture"},
            }
        ),
        encoding="utf-8",
    )
    return pin_research_release(
        experiment_dir=repo_root / "release_pins" / generation_id,
        raw_dir=raw,
        fundamental_events_root=events_root,
        fundamental_events_status=repo_root / FUNDAMENTAL_EVENTS_STATUS,
    )


def _write_pair(parquet: Path, frame: pd.DataFrame | None = None) -> None:
    if parquet.exists():
        return
    parquet.parent.mkdir(parents=True, exist_ok=True)
    if frame is None:
        parquet.write_bytes(b"fixture")
    else:
        frame.to_parquet(parquet, index=False)
    parquet.with_suffix(".parquet.meta.json").write_text("{}\n", encoding="utf-8")
