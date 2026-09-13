#!/usr/bin/env python3
"""Prebuild the exploration PIT view seed (docs/data-documentation.md §3.4).

Pins the current research generation, plans the seed of one research geometry
(``pit_views_seed.plan_seed``) and writes the completed decision views and
replay slots into ``data/pit_views_seed/explore/``. That tree is not a
live worker cache_root. New experiments hardlink it when ``provider.json``
matches.

The geometry defaults to ``DEFAULT_RESEARCH_GEOMETRY`` and the snapshot
configuration to the console creation defaults; every field of either can be
overridden. The geometry decides which views a seed carries; the snapshot
configuration decides what is inside them, and therefore which experiments may
reuse the seed at all — an experiment selecting datasets the seed was not built
for finds no matching contract and cold-builds every view. A seed for another
snapshot configuration therefore needs its own ``--seed``/``--workspace``
directory: one tree binds one contract. ``--dry-run`` prints the plan and exits
without building anything, using a scratch cache root so it never touches the
seed; it still pins the release into the workspace, which the build then
reuses.

Each slot is prepared at its own anchor, which also builds the decision view at
that anchor, so the Held-out slot leaves one decision view (at forward end)
that no stage reads. The as-of stash is encoded for the first slot of each
chain; a later slot of a chain continues the as-of tree of the slots before
it, which this backend cannot encode offline, so those parts are built on
first use.

Reuses ``ResearchPITSnapshotProvider`` / ``SnapshotBuilder`` and the worker's
own ``_snapshot_config``; does not fork a second builder or a second reading of
the experiment parameters.
"""
from __future__ import annotations

import argparse
import dataclasses
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

from autotrade.environment.data.snapshot import DEFAULT_DATASETS
from autotrade.environment.strategy import StrategySchedule
from autotrade.pipelines.calendar import GEOMETRY_PARAMETERS
from autotrade.pipelines.config import (
    DEFAULT_PIT_VIEWS_SEED,
    DEFAULT_PIT_VIEWS_SEED_WORKSPACE,
    DEFAULT_RESEARCH_GEOMETRY,
    SnapshotBundle,
)
from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS
from autotrade.pipelines.pit_backend import (
    ResearchPITSnapshotProvider,
    prebuild_asof_stash,
)
from autotrade.pipelines.pit_views_seed import plan_seed
from autotrade.pipelines.worker import _snapshot_config

# Scratch cache_root for --dry-run: planning must not bind or create views in
# the real seed, and a stale seed contract must not block printing the plan.
DRY_RUN_CACHE_NAME = "dry_run_cache"

# The snapshot-identity parameters a seed can be prebuilt for, by the shape
# their override takes on the command line. Names, types and semantics are the
# experiment parameters' own: `_snapshot_config` reads both, so a seed built
# with these overrides carries exactly the contract that experiment writes.
DATASET_DOMAINS = {
    "fundamental_datasets": "fundamentals",
    "macro_datasets": "macro",
    "events_datasets": "events",
    "text_datasets": "text",
}
DOMAIN_TOGGLES = (
    "include_fundamentals",
    "include_macro",
    "include_events",
    "include_text",
    "include_intraday",
)
WINDOW_PARAMETERS = (
    "window_months",
    "daily_window_months",
    "fundamentals_window_months",
    "events_window_months",
    "macro_window_months",
    "text_window_months",
    "intraday_trade_days",
)
SNAPSHOT_PARAMETERS = (*DATASET_DOMAINS, *DOMAIN_TOGGLES, *WINDOW_PARAMETERS)


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _dataset_list(text: str) -> tuple[str, ...]:
    """A comma-separated dataset selection; membership is checked downstream.

    `_snapshot_config` rejects empty names, duplicates and names outside the
    domain's SELECTABLE_DATASETS, so this parses and never validates twice.
    """

    return tuple(item.strip() for item in text.split(","))


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
    geometry = parser.add_argument_group(
        "research geometry overrides",
        "YYYYMMDD; default: "
        + ", ".join(
            f"{name}={value}" for name, value in DEFAULT_RESEARCH_GEOMETRY.to_record().items()
        ),
    )
    for name in GEOMETRY_PARAMETERS:
        geometry.add_argument(_flag(name), default=None)
    snapshot = parser.add_argument_group(
        "snapshot overrides",
        "default: the console creation defaults. These decide the seed's "
        "provider contract, so a seed built with any of them is reusable only "
        "by an experiment created with the same values, and needs its own "
        "--seed/--workspace directory.",
    )
    for name, domain in DATASET_DOMAINS.items():
        snapshot.add_argument(
            _flag(name),
            type=_dataset_list,
            default=None,
            metavar="NAME,NAME",
            help=f"comma-separated {domain} datasets. REPLACES this domain's "
            "default set, exactly as the experiment parameter of the same name "
            "does, so to ADD a dataset pass the whole default set plus the "
            f"extras. Unknown names are rejected. Default set: "
            + ",".join(DEFAULT_DATASETS[domain]),
        )
    for name in DOMAIN_TOGGLES:
        snapshot.add_argument(
            _flag(name),
            action=argparse.BooleanOptionalAction,
            default=None,
            help="load this domain at all; off drops it from both the decision "
            "snapshot and the replay slots",
        )
    for name in WINDOW_PARAMETERS:
        snapshot.add_argument(_flag(name), type=int, default=None)
    parser.add_argument(
        "--asof-stash",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also encode the rolling as-of parts of each chain's first slot, so "
        "the first replay of an experiment hardlinks them instead of building them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned jobs and exit without building any view",
    )
    return parser


def create_parameters(args: argparse.Namespace) -> dict[str, object]:
    """Console creation defaults with the snapshot overrides applied.

    The result is a creation parameter mapping, not a second dialect, so the
    seed identity it produces is byte-for-byte the one an experiment created
    with these values writes.
    """

    params = dict(WEB_CREATE_DEFAULTS)
    for name in SNAPSHOT_PARAMETERS:
        override = getattr(args, name, None)
        if override is not None:
            params[name] = override
    return params


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Refused before anything is pinned: a malformed geometry has no plan.
    geometry = dataclasses.replace(
        DEFAULT_RESEARCH_GEOMETRY,
        **{
            name: getattr(args, name)
            for name in GEOMETRY_PARAMETERS
            if getattr(args, name) is not None
        },
    )
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
    plan = plan_seed(geometry, provider.trading_days)
    jobs = plan.jobs
    print(
        json.dumps(
            {
                "seed": str(seed),
                "workspace": str(workspace),
                "generation_id": provider.release.generation_id,
                "release_raw_dir": str(provider.release.raw_dir),
                "dry_run": bool(args.dry_run),
                "jobs": len(jobs),
                "asof_stash": bool(args.asof_stash),
                "schedule": schedule.to_record(),
                "plan": plan.to_record(),
                # The contract an experiment has to match to reuse this tree;
                # logging it makes the run self-describing next to the
                # provider.json the build writes.
                "snapshot_config": config.to_record(),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    prepared: dict[str, tuple[str, SnapshotBundle]] = {}
    for index, (phase, slot) in enumerate(jobs, start=1):
        print(
            f"[{index}/{len(jobs)}] {slot.label} {phase} {slot.start}..{slot.end} "
            f"decision={slot.anchor.isoformat()}",
            flush=True,
        )
        if args.dry_run:
            continue
        started = perf_counter()
        prepared[slot.label] = (
            phase,
            provider.prepare(
                phase=phase,
                start=slot.start,
                end=slot.end,
                decision_time=slot.anchor,
            ),
        )
        print(f"    prepared in {perf_counter() - started:.1f}s", flush=True)
    if args.asof_stash:
        for decision, (head, *rest) in plan.stash_chains:
            print(
                f"asof stash decision={decision.isoformat()} {head.label}"
                + (f"; on first use: {','.join(slot.label for slot in rest)}" if rest else ""),
                flush=True,
            )
            if args.dry_run:
                continue
            phase, bundle = prepared[head.label]
            report = prebuild_asof_stash(
                snapshot_dir=bundle.decision_ref,
                replay_dir=bundle.replay_ref,
                schedule=schedule,
                phase=phase,
                generation_id=bundle.generation_id,
                start=head.start,
                end=head.end,
                host_dir=workspace / "asof_stash_build" / uuid.uuid4().hex,
            )
            status = "reused" if report.pop("reused") else "built"
            print(f"    {status} {json.dumps(report, sort_keys=True)}", flush=True)
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
