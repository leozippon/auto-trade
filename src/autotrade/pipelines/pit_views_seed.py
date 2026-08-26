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

DEFAULT_PIT_VIEWS_SEED = Path("data/pit_views_seed/explore")
DEFAULT_PIT_VIEWS_SEED_WORKSPACE = Path("data/pit_views_seed/explore_workspace")


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
    Never copies ``asof_stash`` or writes outside ``experiment_pit_views``.
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
        target = dest_root / source_view.relative_to(seed)
        _assert_inside(target, dest_root)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        _assert_inside(target.parent, dest_root)
        staging = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
        _assert_inside(staging, dest_root)
        try:
            _hardlink_tree(source_view, staging, dest_root=dest_root)
            chmod_tree(staging, file_mode=0o444, dir_mode=0o555)
            try:
                staging.replace(target)
            except OSError:
                if not target.exists():
                    raise
        finally:
            if staging.exists():
                chmod_tree(staging, file_mode=0o644, dir_mode=0o755)
                shutil.rmtree(staging)
    return True


def iter_plan_pit_jobs(
    trading_days: list[str],
    *,
    first_test_period: str,
    last_test_period: str,
    heldout_first_period: str,
    heldout_last_period: str,
    fold_period: str = "quarter",
    window_months: int = 21,
    min_region_trade_days: int = 2,
) -> tuple[tuple[str, str, str, datetime], ...]:
    """Unique Meta/Fold/frozen_test/held-out prepare jobs for one fold plan.

    Epoch count does not multiply the set: later epochs reuse the same
    decision times and replay windows. Jobs are ordered by decision time so
    later decision snapshots can reuse prior events.
    """

    folds = build_fold_schedule(
        first_test_period,
        last_test_period,
        trading_days,
        window_months=window_months,
        period=fold_period,
        min_region_trade_days=min_region_trade_days,
    )
    jobs: list[tuple[str, str, str, datetime]] = []
    for fold in folds:
        jobs.append(
            ("meta", fold.validation_start, fold.validation_end, fold.valid_decision_time)
        )
        jobs.append(
            ("valid", fold.validation_start, fold.validation_end, fold.valid_decision_time)
        )
        jobs.append(
            (
                "frozen_test",
                fold.test_start,
                fold.test_end,
                fold.test_decision_time,
            )
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
    "iter_plan_pit_jobs",
    "pit_cache_provider_record",
    "seed_pit_views",
]
