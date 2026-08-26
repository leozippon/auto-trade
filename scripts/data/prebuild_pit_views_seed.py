#!/usr/bin/env python3
"""Prebuild the exploration PIT view seed (docs/environment-design.md).

Pins the current research generation, walks the console explore calendar, and
writes completed decision/replay views into ``data/pit_views_seed/explore/``.
That tree is not a live worker cache_root. New experiments hardlink it when
``provider.json`` matches.

Reuses ``ResearchPITSnapshotProvider`` / ``SnapshotBuilder``; does not fork a
second builder. Fail-fast if the 2022Q1 first fold cannot see 2021 lookback.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

REPO_ROOT = add_repo_src(__file__)

from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS
from autotrade.pipelines.pit_backend import ResearchPITSnapshotProvider
from autotrade.pipelines.pit_views_seed import (
    DEFAULT_PIT_VIEWS_SEED,
    DEFAULT_PIT_VIEWS_SEED_WORKSPACE,
    iter_plan_pit_jobs,
)
from autotrade.pipelines.worker import _snapshot_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--seed",
        type=Path,
        default=None,
        help="Seed cache_root. Default: <repo>/data/pit_views_seed/explore",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Scratch experiment dir for the research-release pin. "
        "Default: <repo>/data/pit_views_seed/explore_workspace",
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--fundamental-events-root", type=Path, default=None)
    parser.add_argument("--fundamental-events-status", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    seed = (args.seed or repo_root / DEFAULT_PIT_VIEWS_SEED).resolve()
    workspace = (args.workspace or repo_root / DEFAULT_PIT_VIEWS_SEED_WORKSPACE).resolve()
    raw_dir = (args.raw_dir or repo_root / "data/raw").resolve()
    events_root = (
        args.fundamental_events_root or repo_root / "data/pit/fundamental_events"
    ).resolve()
    events_status = (
        args.fundamental_events_status
        or repo_root / "results/data_quality/fundamental_events_status.json"
    ).resolve()
    config = _snapshot_config(dict(WEB_CREATE_DEFAULTS))
    workspace.mkdir(parents=True, exist_ok=True)
    seed.mkdir(parents=True, exist_ok=True)
    provider = ResearchPITSnapshotProvider(
        experiment_dir=workspace,
        raw_dir=raw_dir,
        fundamental_events_root=events_root,
        fundamental_events_status=events_status,
        config=config,
        cache_root=seed,
    )
    window_months = WEB_CREATE_DEFAULTS["window_months"]
    min_region_trade_days = WEB_CREATE_DEFAULTS["min_region_trade_days"]
    if type(window_months) is not int or type(min_region_trade_days) is not int:
        raise TypeError("explore calendar window_months/min_region_trade_days must be int")
    jobs = iter_plan_pit_jobs(
        provider.trading_days,
        first_test_period=str(WEB_CREATE_DEFAULTS["first_test_period"]),
        last_test_period=str(WEB_CREATE_DEFAULTS["last_test_period"]),
        heldout_first_period=str(WEB_CREATE_DEFAULTS["heldout_first_period"]),
        heldout_last_period=str(WEB_CREATE_DEFAULTS["heldout_last_period"]),
        fold_period=str(WEB_CREATE_DEFAULTS["fold_period"]),
        window_months=window_months,
        min_region_trade_days=min_region_trade_days,
    )
    print(
        json.dumps(
            {
                "seed": str(seed),
                "workspace": str(workspace),
                "generation_id": provider.release.generation_id,
                "release_raw_dir": str(provider.release.raw_dir),
                "jobs": len(jobs),
                "first_test_period": WEB_CREATE_DEFAULTS["first_test_period"],
                "last_test_period": WEB_CREATE_DEFAULTS["last_test_period"],
                "heldout_first_period": WEB_CREATE_DEFAULTS["heldout_first_period"],
                "heldout_last_period": WEB_CREATE_DEFAULTS["heldout_last_period"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    for index, (phase, start, end, decision) in enumerate(jobs, start=1):
        print(
            f"[{index}/{len(jobs)}] {phase} {start}..{end} decision={decision.isoformat()}",
            flush=True,
        )
        provider.prepare(
            fold=None,
            phase=phase,
            start=start,
            end=end,
            decision_time=decision,
        )
    print(
        json.dumps({"status": "ok", "seed": str(seed), "jobs": len(jobs)}, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
