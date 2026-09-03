#!/usr/bin/env python3
"""Prebuild the exploration PIT view seed (docs/environment-design.md).

Pins the current research generation, walks the console research calendar, and
writes completed decision/replay views into ``data/pit_views_seed/explore/``.
That tree is not a live worker cache_root. New experiments hardlink it when
``provider.json`` matches.

The calendar comes from the console creation defaults (one source); every field
can be overridden so a seed can be prebuilt for a plan before it becomes the
default. ``--dry-run`` prints the plan and exits without building anything,
using a scratch cache root so it never touches the seed.

Reuses ``ResearchPITSnapshotProvider`` / ``SnapshotBuilder``; does not fork a
second builder.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path
from time import perf_counter

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

REPO_ROOT = add_repo_src(__file__)

from autotrade.environment.strategy import StrategySchedule
from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS
from autotrade.pipelines.pit_backend import (
    ResearchPITSnapshotProvider,
    prebuild_asof_stash,
)
from autotrade.pipelines.pit_views_seed import (
    DEFAULT_PIT_VIEWS_SEED,
    DEFAULT_PIT_VIEWS_SEED_WORKSPACE,
    PLAN_PARAMETERS,
    iter_plan_pit_jobs,
    plan_parameters,
)
from autotrade.pipelines.worker import _snapshot_config

# Scratch cache_root for --dry-run: planning must not bind or create views in
# the real seed, and a stale seed contract must not block printing the plan.
DRY_RUN_CACHE_NAME = "dry_run_cache"


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
    calendar = parser.add_argument_group(
        "calendar overrides", "default: the console creation defaults"
    )
    calendar.add_argument("--fold-period", default=None)
    calendar.add_argument("--development-first-period", default=None)
    calendar.add_argument("--development-last-period", default=None)
    calendar.add_argument("--heldout-first-period", default=None)
    calendar.add_argument("--heldout-last-period", default=None)
    calendar.add_argument("--window-months", type=int, default=None)
    calendar.add_argument(
        "--validation-periods",
        type=int,
        default=None,
        help="periods in each Fold's trailing validation window (1 = the Fold's own period)",
    )
    calendar.add_argument("--min-region-trade-days", type=int, default=None)
    calendar.add_argument(
        "--test-stage",
        dest="test_stage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="cut the development window into rolling Folds with a test region",
    )
    parser.add_argument(
        "--asof-stash",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also encode each region's rolling as-of parts, so the first "
        "backtest of an experiment hardlinks them instead of building them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned jobs and exit without building any view",
    )
    return parser


def create_parameters(args: argparse.Namespace) -> dict[str, object]:
    """Console creation defaults with the calendar overrides applied.

    One parameter set feeds both the snapshot configuration and the calendar,
    exactly as creating an experiment does: ``window_months``, for instance, is
    the data window AND the Fold input window.
    """

    params = dict(WEB_CREATE_DEFAULTS)
    for name in PLAN_PARAMETERS:
        override = getattr(args, name, None)
        if override is not None:
            params[name] = override
    return params


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
    params = create_parameters(args)
    config = _snapshot_config(params)
    # The stash is keyed by the schedule an experiment will actually replay on.
    schedule = StrategySchedule(
        str(params["strategy_period"]),  # type: ignore[arg-type]
        str(params["inference_time"]),
    )
    plan = plan_parameters(params)
    workspace.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        cache_root = workspace / DRY_RUN_CACHE_NAME
        shutil.rmtree(cache_root, ignore_errors=True)
    else:
        cache_root = seed
        seed.mkdir(parents=True, exist_ok=True)
    provider = ResearchPITSnapshotProvider(
        experiment_dir=workspace,
        raw_dir=raw_dir,
        fundamental_events_root=events_root,
        fundamental_events_status=events_status,
        config=config,
        cache_root=cache_root,
    )
    jobs = iter_plan_pit_jobs(provider.trading_days, **plan)  # type: ignore[arg-type]
    windows = {(start, end, decision) for _phase, start, end, decision in jobs}
    print(
        json.dumps(
            {
                "seed": str(seed),
                "workspace": str(workspace),
                "generation_id": provider.release.generation_id,
                "release_raw_dir": str(provider.release.raw_dir),
                "dry_run": bool(args.dry_run),
                "jobs": len(jobs),
                "decision_snapshots": len({decision for *_rest, decision in jobs}),
                "replay_sources": len(windows),
                "asof_stash": bool(args.asof_stash),
                "schedule": schedule.to_record(),
                "calendar": {
                    name: str(value) for name, value in sorted(plan.items())
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    stashed: set[tuple[str, str]] = set()
    for index, (phase, start, end, decision) in enumerate(jobs, start=1):
        print(
            f"[{index}/{len(jobs)}] {phase} {start}..{end} decision={decision.isoformat()}",
            flush=True,
        )
        if args.dry_run:
            continue
        started = perf_counter()
        bundle = provider.prepare(
            fold=None,
            phase=phase,
            start=start,
            end=end,
            decision_time=decision,
        )
        print(f"    prepared in {perf_counter() - started:.1f}s", flush=True)
        if not args.asof_stash:
            continue
        # One stash per region and schedule: the phases of a region share it.
        key = (Path(bundle.decision_ref).name, Path(bundle.replay_ref).name)
        if key in stashed:
            continue
        stashed.add(key)
        report = prebuild_asof_stash(
            snapshot_dir=bundle.decision_ref,
            replay_dir=bundle.replay_ref,
            schedule=schedule,
            phase=phase,
            generation_id=bundle.generation_id,
            start=start,
            end=end,
            host_dir=workspace / "asof_stash_build" / uuid.uuid4().hex,
        )
        status = "reused" if report.pop("reused") else "built"
        print(
            f"    asof stash {status} {json.dumps(report, sort_keys=True)}",
            flush=True,
        )
    print(
        json.dumps(
            {
                "status": "planned" if args.dry_run else "ok",
                "seed": str(seed),
                "jobs": len(jobs),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
