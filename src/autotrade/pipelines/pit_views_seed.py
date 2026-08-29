"""One-shot hardlink of a matching exploration PIT view seed into an experiment.

The seed tree is a repo-adjacent, gitignored snapshot of completed ``decision/``,
``replay/``, and tiny ``bundles/`` views. It is not a live worker ``cache_root``.
New experiments hardlink those views into ``experiments/<id>/pit_views/`` only
when ``provider.json`` matches the contract this experiment would write.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import stat
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.runtime import chmod_tree
from autotrade.pipelines.config import SNAPSHOT_CACHE_FORMAT_VERSION
from autotrade.pipelines.folds import build_fold_schedule, heldout_periods
from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS

DEFAULT_PIT_VIEWS_SEED = Path("data/pit_views_seed/explore")
DEFAULT_PIT_VIEWS_SEED_WORKSPACE = Path("data/pit_views_seed/explore_workspace")

# The calendar a seed is planned over. One source: the console creation
# defaults an experiment is actually created with, so a seed prebuilt without
# overrides matches what the next experiment asks the provider to build.
PLAN_PARAMETERS: tuple[str, ...] = (
    "fold_period",
    "development_first_period",
    "development_last_period",
    "test_stage",
    "heldout_first_period",
    "heldout_last_period",
    "window_months",
    "min_region_trade_days",
)
_INT_PLAN_PARAMETERS = frozenset({"window_months", "min_region_trade_days"})
_BOOL_PLAN_PARAMETERS = frozenset({"test_stage"})


def plan_parameters(params: Mapping[str, object] | None = None) -> dict[str, object]:
    """Calendar keywords for ``iter_plan_pit_jobs`` from creation parameters.

    Defaults to the console creation defaults, so a seed prebuilt without
    overrides plans exactly the calendar the next experiment is created with.
    """

    source = WEB_CREATE_DEFAULTS if params is None else params
    plan: dict[str, object] = {}
    for name in PLAN_PARAMETERS:
        value = source[name]
        if name in _INT_PLAN_PARAMETERS:
            if type(value) is not int:
                raise TypeError(f"calendar default {name} must be an int")
            plan[name] = value
        elif name in _BOOL_PLAN_PARAMETERS:
            if type(value) is not bool:
                raise TypeError(f"calendar default {name} must be a bool")
            plan[name] = value
        else:
            plan[name] = str(value)
    return plan


def pit_cache_provider_record(
    *,
    generation_id: str,
    release_raw_dir: str | Path,
    snapshot_config: SnapshotConfig,
) -> dict[str, object]:
    """The on-disk PIT cache contract written to ``provider.json``."""

    return {
        "schema_version": SNAPSHOT_CACHE_FORMAT_VERSION,
        "generation_id": generation_id,
        "release_raw_dir": str(release_raw_dir),
        "snapshot_config": snapshot_config.to_record(),
    }


def seed_pit_views(
    experiment_pit_views: Path,
    seed: Path,
    *,
    expected_provider: Mapping[str, object],
    required: bool = False,
) -> bool:
    """Hardlink completed seed views into an experiment PIT cache.

    Returns True when the seed contract matched and views were applied (or
    already present). Returns False when the default seed is missing or its
    contract does not match — the experiment then cold-builds. An explicit
    seed (``required=True``) fails fast on a missing tree or a mismatch.
    Never writes outside ``experiment_pit_views``.

    Prebuilt ``asof_stash`` parts come across too, so the first backtest over a
    slot hardlinks the day-by-day as-of parts instead of encoding them. Their
    stash contract names only what determines the parts, which is why parts
    encoded offline are valid here. Unlike a view, a stash keeps growing (a
    replay can reach a day the prebuild did not cover), so its directories are
    published writable while the parts themselves stay read-only.
    """

    dest = Path(experiment_pit_views)
    seed = Path(seed)
    if not seed.exists():
        if required:
            raise FileNotFoundError(f"PIT view seed does not exist: {seed}")
        return False
    if not seed.is_dir() or seed.is_symlink():
        raise RuntimeError(f"PIT view seed must be a real directory: {seed}")
    provider_path = seed / "provider.json"
    if not provider_path.is_file() or provider_path.is_symlink():
        raise RuntimeError(f"PIT view seed is missing provider.json: {provider_path}")
    seed_record = _load_json(provider_path)
    if seed_record != dict(expected_provider):
        if required:
            raise RuntimeError(
                f"PIT view seed {seed} does not match this experiment's provider "
                "contract; refusing to mix views"
            )
        return False
    dest_root = dest.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    for source_view in _completed_seed_views(seed):
        _publish_seed_entry(source_view, seed, dest_root, dir_mode=0o555)
    for source_stash in _completed_seed_stashes(seed):
        _publish_seed_entry(source_stash, seed, dest_root, dir_mode=0o755)
    return True


def _publish_seed_entry(
    source: Path, seed: Path, dest_root: Path, *, dir_mode: int
) -> None:
    """Hardlink one seed entry into place, or leave an existing one alone."""

    target = dest_root / source.relative_to(seed)
    _assert_inside(target, dest_root)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    _assert_inside(target.parent, dest_root)
    staging = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    _assert_inside(staging, dest_root)
    try:
        _hardlink_tree(source, staging, dest_root=dest_root)
        chmod_tree(staging, file_mode=0o444, dir_mode=dir_mode)
        try:
            staging.replace(target)
        except OSError:
            if not target.exists():
                raise
    finally:
        if staging.exists():
            chmod_tree(staging, file_mode=0o644, dir_mode=0o755)
            shutil.rmtree(staging)


def iter_plan_pit_jobs(
    trading_days: list[str],
    *,
    development_first_period: str,
    development_last_period: str,
    heldout_first_period: str,
    heldout_last_period: str,
    fold_period: str,
    window_months: int,
    min_region_trade_days: int,
    test_stage: bool,
) -> tuple[tuple[str, str, str, datetime], ...]:
    """Unique Meta/Fold/frozen_test/held-out prepare jobs for one fold plan.

    The plan comes from the schedule API, never from a second calendar: the
    regions and decision anchors are exactly the ``FoldSpec`` and held-out
    periods the pipeline will ask the provider to prepare. A fold without a
    test region (the default single-window development Fold) contributes no
    frozen_test job.

    Epoch count does not multiply the set: later epochs reuse the same decision
    times and replay windows. Several phases routinely share one region — meta
    and valid always do, and on a contiguous calendar the previous fold's test
    does too — so the returned tuples repeat a region once per phase while the
    provider builds it once. Jobs are ordered by decision time so later
    decision snapshots can reuse the previous one's events.
    """

    folds = build_fold_schedule(
        development_first_period,
        development_last_period,
        trading_days,
        window_months=window_months,
        period=fold_period,
        min_region_trade_days=min_region_trade_days,
        test_stage=test_stage,
    )
    jobs: list[tuple[str, str, str, datetime]] = []
    for fold in folds:
        jobs.append(
            ("meta", fold.validation_start, fold.validation_end, fold.valid_decision_time)
        )
        jobs.append(
            ("valid", fold.validation_start, fold.validation_end, fold.valid_decision_time)
        )
        if fold.has_test:
            assert fold.test_start is not None and fold.test_end is not None
            assert fold.test_decision_time is not None
            jobs.append(
                ("frozen_test", fold.test_start, fold.test_end, fold.test_decision_time)
            )
    for period in heldout_periods(
        heldout_first_period,
        heldout_last_period,
        trading_days,
        period=fold_period,
        min_region_trade_days=min_region_trade_days,
    ):
        jobs.append(
            (
                "heldout",
                str(period["start"]),
                str(period["end"]),
                period["decision_time"],  # type: ignore[arg-type]
            )
        )
    jobs.sort(key=lambda job: (job[3], job[1], job[0]))
    return tuple(jobs)


def _completed_seed_views(seed: Path) -> list[Path]:
    views: list[Path] = []
    for name in ("decision", "replay"):
        root = seed / name
        if not root.is_dir() or root.is_symlink():
            continue
        views.extend(
            path
            for path in root.iterdir()
            if path.is_dir() and not path.is_symlink() and not _skip_seed_name(path.name)
        )
    bundles = seed / "bundles"
    if bundles.is_dir() and not bundles.is_symlink():
        for phase in bundles.iterdir():
            if not phase.is_dir() or phase.is_symlink() or _skip_seed_name(phase.name):
                continue
            views.extend(
                path
                for path in phase.iterdir()
                if path.is_dir()
                and not path.is_symlink()
                and not _skip_seed_name(path.name)
            )
    return views


def _completed_seed_stashes(seed: Path) -> list[Path]:
    """Every stash directory the seed prebuild finished, by its contract file."""

    root = seed / "asof_stash"
    if not root.is_dir() or root.is_symlink():
        return []
    return sorted(
        contract.parent
        for contract in root.rglob("contract.json")
        if contract.is_file() and not contract.is_symlink()
    )


def _hardlink_tree(source: Path, dest: Path, *, dest_root: Path) -> None:
    _assert_inside(dest, dest_root)
    source_mode = source.lstat().st_mode
    if stat.S_ISLNK(source_mode):
        raise RuntimeError(f"symbolic link is forbidden in a PIT view seed: {source}")
    if stat.S_ISDIR(source_mode):
        if dest.exists() and not dest.is_dir():
            raise RuntimeError(f"PIT view seed destination is not a directory: {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        _assert_inside(dest, dest_root)
        for child in source.iterdir():
            if _skip_seed_name(child.name):
                continue
            _hardlink_tree(child, dest / child.name, dest_root=dest_root)
        return
    if not stat.S_ISREG(source_mode):
        raise RuntimeError(f"unsupported PIT view seed entry: {source}")
    if dest.exists():
        return
    try:
        os.link(source, dest)
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            raise RuntimeError(
                f"PIT view seed is on a different filesystem than {dest}; "
                "hardlink is required and copy is refused"
            ) from exc
        raise


def _assert_inside(path: Path, root: Path) -> None:
    resolved = path.resolve()
    root = root.resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise RuntimeError(f"PIT view seed refused to write outside {root}: {resolved}")


def _skip_seed_name(name: str) -> bool:
    lowered = name.lower()
    return name.startswith(".") or lowered.endswith(".lock") or ".tmp" in lowered


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid PIT cache record {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"PIT cache record is not an object: {path}")
    return value


__all__ = [
    "DEFAULT_PIT_VIEWS_SEED",
    "DEFAULT_PIT_VIEWS_SEED_WORKSPACE",
    "PLAN_PARAMETERS",
    "iter_plan_pit_jobs",
    "pit_cache_provider_record",
    "plan_parameters",
    "seed_pit_views",
]
