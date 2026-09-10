"""One-shot hardlink of a matching exploration PIT view seed into an experiment.

The seed tree is a repo-adjacent, gitignored snapshot of completed ``decision/``,
``replay/``, and tiny ``bundles/`` views. It is not a live worker ``cache_root``.
New experiments hardlink those views into ``experiments/<id>/pit_views/`` only
when ``provider.json`` matches the contract this experiment would write.

Which tree an experiment reads is its ``pit_views_seed`` parameter: the default
one carries the default dataset selection, and an arm that selects other
datasets points at a tree prebuilt for exactly its own selection
(``scripts/data/prebuild_pit_views_seed.py``).
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
from autotrade.pipelines.folds import (
    build_fold_schedule,
    deployment_fold,
    heldout_periods,
)
from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS

# What a finished view carries: a snapshot restates its manifest and a bundle
# its data summary. Any other directory in the layout is a level the provider
# still writes into, so it is descended, never published as a view.
_VIEW_MARKERS = ("manifest.json", "data_summary.json")

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
    "validation_periods",
    "min_region_trade_days",
    "deployment_adjustment_start",
)
_INT_PLAN_PARAMETERS = frozenset(
    {"window_months", "validation_periods", "min_region_trade_days"}
)
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


def assert_seed_snapshot_config(seed: Path, snapshot_config: SnapshotConfig) -> None:
    """Refuse a seed prebuilt for a different snapshot configuration or cache format.

    The create-time half of the contract check. Two of the four fields of
    ``pit_cache_provider_record`` — the pinned generation and its release
    path — exist only once the experiment runs, so what a create request can be
    judged against is the part the seed was prebuilt for: the cache format and
    the snapshot configuration, which is exactly what decides whether a dataset
    selection has views here at all. ``seed_pit_views`` still compares the
    whole record before it links anything.
    """

    provider_path = Path(seed) / "provider.json"
    if not provider_path.is_file() or provider_path.is_symlink():
        raise ValueError(f"PIT view seed is missing provider.json: {provider_path}")
    record = _load_json(provider_path)
    # The on-disk contract version is also known at create time: a seed built
    # under an older one holds views this code would never link, so naming it
    # would only fail at worker start after a misleading acceptance.
    version = record.get("schema_version")
    if version != SNAPSHOT_CACHE_FORMAT_VERSION:
        raise ValueError(
            f"PIT view seed {seed} was prebuilt under snapshot cache format {version!r}; "
            f"this code writes {SNAPSHOT_CACHE_FORMAT_VERSION}, so rebuild the seed under a new directory"
        )
    recorded = record.get("snapshot_config")
    expected = snapshot_config.to_record()
    if recorded != expected:
        raise ValueError(
            f"PIT view seed {seed} was prebuilt for a different snapshot "
            f"configuration: seed has {json.dumps(recorded, ensure_ascii=False, sort_keys=True)}, "
            f"this experiment needs {json.dumps(expected, ensure_ascii=False, sort_keys=True)}"
        )
    # ``provider.json`` is written when the build binds its cache root, not when
    # it finishes, so the contract above says nothing about how much of the tree
    # exists yet. A staged slot does: it is the view the provider is writing
    # right now, and it is renamed into place only once that view is complete.
    staged = _staged_seed_slots(Path(seed))
    if staged:
        raise ValueError(
            f"PIT view seed {seed} has an unfinished build: "
            f"{len(staged)} staged slot(s) still present, first "
            f"{staged[0].relative_to(seed)}. Wait for "
            "scripts/data/prebuild_pit_views_seed.py to report status ok for "
            "this seed, or remove the staged slots a killed build left behind"
        )


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
    Never writes outside ``experiment_pit_views``, and re-seeding an already
    seeded experiment is a no-op.

    The result is indistinguishable from a cold build: each view is published
    read-only, exactly as the provider publishes its own, while the layout
    directories around it stay writable, so the provider can still take the
    lock beside a seeded slot and build the slots the seed does not carry.

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


def seed_pit_view_slots(
    experiment_pit_views: Path,
    seed: Path,
    *,
    expected_provider: Mapping[str, object],
    decision_key: str,
    phase: str,
    replay_slot: str,
) -> None:
    """Hardlink exactly one decision view and one phase replay slot from a seed.

    The deployment adjustment's two views (docs/pipeline-design.md §3.4): an
    experiment whose own views were built under an older cache format cannot
    extend them with this code, so its deployment session takes just these
    two slots -- plus the bundle and the prebuilt as-of parts of that pair,
    when the seed carries them -- into its own cache root. The contract check
    is the whole-record comparison ``seed_pit_views`` applies, and it fails
    explicitly on a missing tree, a mismatch, or a seed that lacks either
    slot: cold-building here would cost hours and look like a slow session.
    """

    seed = Path(seed)
    if not seed.is_dir() or seed.is_symlink():
        raise RuntimeError(f"PIT view seed does not exist: {seed}")
    provider_path = seed / "provider.json"
    if not provider_path.is_file() or provider_path.is_symlink():
        raise RuntimeError(f"PIT view seed is missing provider.json: {provider_path}")
    if _load_json(provider_path) != dict(expected_provider):
        raise RuntimeError(
            f"PIT view seed {seed} does not match this experiment's provider "
            "contract; refusing to mix views"
        )
    required = (seed / "decision" / decision_key, seed / "replay" / phase / replay_slot)
    missing = [
        view
        for view in required
        if not any((view / marker).is_file() for marker in _VIEW_MARKERS)
    ]
    if missing:
        raise RuntimeError(
            f"PIT view seed {seed} lacks the deployment slot(s): "
            + ", ".join(str(view.relative_to(seed)) for view in missing)
        )
    dest_root = Path(experiment_pit_views).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    for view in required:
        _publish_seed_entry(view, seed, dest_root, dir_mode=0o555)
    bundle = seed / "bundles" / phase / replay_slot
    if any((bundle / marker).is_file() for marker in _VIEW_MARKERS):
        _publish_seed_entry(bundle, seed, dest_root, dir_mode=0o555)
    stash_root = seed / "asof_stash" / "decision" / decision_key / "replay" / replay_slot
    for contract in sorted(stash_root.rglob("contract.json")):
        if contract.is_file() and not contract.is_symlink():
            _publish_seed_entry(contract.parent, seed, dest_root, dir_mode=0o755)


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
    validation_periods: int = 1,
    deployment_adjustment_start: str = "",
) -> tuple[tuple[str, str, str, datetime], ...]:
    """Unique Meta/Fold/frozen_test/held-out prepare jobs for one fold plan.

    The plan comes from the schedule API, never from a second calendar: the
    regions and decision anchors are exactly the ``FoldSpec`` and held-out
    periods the pipeline will ask the provider to prepare. A fold without a
    test region (the default regular Fold) contributes no frozen_test job.

    Epoch count does not multiply the set: later epochs reuse the same decision
    times and replay windows. Several phases routinely share one region — meta
    and valid always do, and on a contiguous calendar the previous fold's test
    does too — so the returned tuples repeat a region once per phase while the
    provider builds it once. Jobs are sorted by decision time, then region and
    phase, so the plan and a prebuild's progress log are the same on every run.

    With ``deployment_adjustment_start`` set, the deployment adjustment's
    Validation slot (that start through the release's last trading day,
    ``folds.deployment_fold``) is planned too.
    """

    folds = build_fold_schedule(
        development_first_period,
        development_last_period,
        trading_days,
        window_months=window_months,
        period=fold_period,
        min_region_trade_days=min_region_trade_days,
        test_stage=test_stage,
        validation_periods=validation_periods,
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
    if deployment_adjustment_start:
        deployment = deployment_fold(
            deployment_adjustment_start,
            trading_days,
            window_months=window_months,
            min_region_trade_days=min_region_trade_days,
        )
        jobs.append(
            (
                "valid",
                deployment.validation_start,
                deployment.validation_end,
                deployment.valid_decision_time,
            )
        )
    jobs.sort(key=lambda job: (job[3], job[1], job[0]))
    return tuple(jobs)


def _completed_seed_views(seed: Path) -> list[Path]:
    """Every completed view in the seed, wherever the layout puts it.

    Views sit at different depths: ``decision/<slot>``, the unphased
    ``replay/<slot>`` source, the ``replay/<phase>/<slot>`` views hardlinked
    from it, and ``bundles/<phase>/<slot>``. A view is therefore recognised by
    the marker the provider writes when it publishes one, never by its depth.
    Publishing a layout level instead of a view would freeze that level
    read-only in the experiment, and the provider could then neither take the
    lock beside a seeded slot nor stage a new slot next to it.
    """

    views: list[Path] = []
    for name in ("decision", "replay", "bundles"):
        views.extend(_marked_seed_views(seed / name))
    return views


def _staged_seed_slots(seed: Path, root: Path | None = None, *, depth: int = 4) -> list[Path]:
    """Slots a build is still writing into this seed, if any.

    The provider stages a slot as ``.<slot>.<uuid>.tmp`` beside its destination
    and renames it in only when the view is complete, so a staged directory is
    on-disk evidence that a prebuild is running here or was killed part way.

    This proves an unfinished build; it cannot prove a finished one. The
    prebuild reports its own completion only on stdout (``{"status": "ok"}``),
    leaving nothing in the tree for a later reader, so a build stopped between
    two jobs still looks like a complete seed here. Walks exactly as far as
    ``_marked_seed_views``: down the layout levels, stopping at any published
    view.
    """

    if root is None:
        return [
            path
            for name in ("decision", "replay", "bundles", "asof_stash")
            for path in _staged_seed_slots(seed, seed / name, depth=depth)
        ]
    if depth <= 0 or not root.is_dir() or root.is_symlink():
        return []
    staged: list[Path] = []
    for path in sorted(root.iterdir()):
        if path.is_symlink() or not path.is_dir():
            continue
        if ".tmp" in path.name.lower():
            staged.append(path)
        elif not any((path / marker).is_file() for marker in _VIEW_MARKERS):
            staged.extend(_staged_seed_slots(seed, path, depth=depth - 1))
    return staged


def _marked_seed_views(root: Path) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        return []
    views: list[Path] = []
    for path in sorted(root.iterdir()):
        if _skip_seed_name(path.name) or path.is_symlink() or not path.is_dir():
            continue
        if any((path / marker).is_file() for marker in _VIEW_MARKERS):
            views.append(path)
        else:
            views.extend(_marked_seed_views(path))
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
    "PLAN_PARAMETERS",
    "assert_seed_snapshot_config",
    "iter_plan_pit_jobs",
    "pit_cache_provider_record",
    "plan_parameters",
    "seed_pit_view_slots",
    "seed_pit_views",
]
