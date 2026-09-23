"""Immutable, on-demand research inputs over the mutable live data lake.

The cron writer keeps ``data/raw`` as its live tree.  An experiment pins a
committed generation into a shared hardlink checkpoint, then stores a durable
experiment-local manifest.  Resume always reuses that manifest; a new
experiment started during ``updating``/``dirty`` uses the last complete release.
An experiment that names the release it must read (the one its PIT view seed
was built from) pins exactly that already-published release instead.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow.parquet as pq

from autotrade.data_quality import read_quality_report
from autotrade.environment.data.contracts import (
    DOMAIN_REPORT_TYPES,
    DOMAIN_STATUS_FILES,
    LEGACY_STATUS_REPORT_TYPES,
    RAW_GENERATION_FILENAME,
    benchmark_index_label,
    require_committed_generation,
)

_SCHEMA_VERSION = 2
_PIN_DIR_NAME = "research_release"
_MANIFEST_NAME = "manifest.json"
_SAFE_GENERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class ResearchRelease:
    raw_dir: Path
    fundamental_events_root: Path
    fundamental_events_status: Path
    generation_id: str


@dataclass(frozen=True)
class _GlobalRelease:
    root: Path
    raw_dir: Path
    fundamental_events_root: Path
    baseline_quality_dir: Path
    generation_id: str
    fundamental_status_name: str
    source_raw_dir: Path
    source_fundamental_events_root: Path
    source_quality_dir: Path


def _dataset_dir_populated(path: Path) -> bool:
    # ≥1 parquet anywhere below (layouts nest: exchange=/ts_code=/trade_date=);
    # rglob short-circuits at the first hit, so populated dirs cost ~one stat.
    return path.is_dir() and next(path.rglob("*.parquet"), None) is not None


def _require_raw_datasets(raw_dir: Path, required: tuple[str, ...], *, context: str) -> None:
    # A release materialized before a dataset was added to the lake silently
    # lacks its directory (or carries an empty one from an interrupted
    # backfill); the snapshot would only fail deep inside the first build
    # (observed: lzp-test22 pinned a pre-derivatives release during a nightly
    # update). Fail at pin time with the remedy instead.
    missing = [name for name in required if not _dataset_dir_populated(raw_dir / name)]
    if missing:
        raise RuntimeError(
            f"{context} lacks configured raw datasets {missing}: the release predates these datasets; "
            "recreate the experiment while the updater is idle (a fresh release will include them), "
            "or exclude the datasets from the experiment config"
        )


_BENCHMARK_INDEX_PARTITION_KEYS = {"index_daily": "ts_code", "index_weight": "index_code"}


def _first_trade_date(partition_dir: Path) -> str | None:
    """Earliest ``trade_date`` held under one benchmark partition directory.

    Its files are ``year=YYYY.parquet``: the earliest year with any row decides,
    and zero-row years (an index before it launched) are skipped by their
    footer alone.
    """

    for path in sorted(partition_dir.glob("year=*.parquet")):
        if pq.ParquetFile(path).metadata.num_rows:
            dates = pd.read_parquet(path, columns=["trade_date"])["trade_date"]
            return str(dates.astype(str).str.replace("-", "").str[:8].min())
    return None


def require_benchmark_index(
    raw_dir: str | Path,
    benchmark_index: str,
    *,
    datasets: tuple[str, ...],
    research_start: str,
) -> None:
    """Refuse a ``benchmark_index`` the pinned release cannot key a replay on.

    The parameter selects one partition of the benchmark's daily bars and one
    of its constituent table, and both are read only at replay end, where a
    missing one degrades to an unmeasured benchmark and an unmeasurable
    verdict. A release that predates an index therefore has to be refused where
    the arm names it -- and so does one whose history of the index starts
    inside the research period: without a bar before ``research_start`` the
    first research year has no benchmark return to neutralize against, and
    without a constituent section dated before it the zero-skill panel has no
    membership to match the first entries on. Only the datasets the arm
    actually consumes are checked (``datasets`` is its
    ``required_release_raw_datasets``): an arm that mounts no ``index_weight``
    draws its zero-skill panel on the float-cap decile, and one that mounts no
    macro domain has no benchmark series by its own configuration -- neither is
    this parameter's doing.
    """

    label = benchmark_index_label(benchmark_index)
    root = Path(raw_dir)
    mounted = [
        (dataset, root / dataset / f"{key}={benchmark_index}")
        for dataset, key in _BENCHMARK_INDEX_PARTITION_KEYS.items()
        if dataset in datasets
    ]
    missing = [
        f"{partition.parent.name}/{partition.name}"
        for _dataset, partition in mounted
        if not _dataset_dir_populated(partition)
    ]
    if missing:
        raise ValueError(
            f"benchmark_index {benchmark_index} ({label}) is absent from research release "
            f"{root}: {missing}; pick an index the release carries or recreate the "
            "experiment on a release that includes it"
        )
    late = {
        f"{dataset}/{partition.name}": first
        for dataset, partition in mounted
        if (first := _first_trade_date(partition)) is None or first >= research_start
    }
    if late:
        raise ValueError(
            f"benchmark_index {benchmark_index} ({label}) has no history before research_start "
            f"{research_start} in research release {root}: earliest trade_date {late}; the first "
            "research year could not be graded against it. Start research later, or recreate "
            "the experiment on a release whose history of this index reaches back before it"
        )


def pin_research_release(
    *,
    experiment_dir: str | Path,
    raw_dir: str | Path,
    fundamental_events_root: str | Path,
    fundamental_events_status: str | Path,
    required_raw_datasets: tuple[str, ...] = (),
    generation_id: str | None = None,
) -> ResearchRelease:
    """Pin immutable data inputs for one experiment.

    Production mode requires both the cron flock and generation marker.  Data
    roots without that pair retain live-path behavior for local/synthetic tests.
    ``required_raw_datasets`` are dataset directory names the experiment's
    snapshot config will read: every returned release must contain them.

    ``generation_id`` names the release to pin instead of the newest committed
    one. That release must already be published under ``research_releases/``
    (a release directory is immutable, so neither the updater lock nor the live
    generation is consulted), and an experiment already pinned to another
    generation is refused.
    """

    experiment_dir = Path(experiment_dir).resolve()
    raw_dir = Path(raw_dir).resolve()
    fundamental_events_root = Path(fundamental_events_root).resolve()
    fundamental_events_status = Path(fundamental_events_status).resolve()
    pin_dir = experiment_dir / _PIN_DIR_NAME
    pin_manifest = pin_dir / _MANIFEST_NAME
    experiment_dir.mkdir(parents=True, exist_ok=True)

    with _exclusive_flock(experiment_dir / f".{_PIN_DIR_NAME}.lock"):
        if pin_manifest.exists():
            release = _load_experiment_pin(
                pin_manifest,
                raw_dir.parent / "research_releases",
                raw_dir=raw_dir,
                fundamental_events_root=fundamental_events_root,
                fundamental_events_status=fundamental_events_status,
            )
            if generation_id is not None and release.generation_id != generation_id:
                raise RuntimeError(
                    f"experiment is pinned to research release {release.generation_id or release.raw_dir}, "
                    f"not the requested release {generation_id}"
                )
            _require_raw_datasets(
                release.raw_dir,
                required_raw_datasets,
                context=f"pinned research release {release.generation_id or release.raw_dir}",
            )
            return release

        if generation_id is not None:
            _reject_unpinned_legacy_experiment(experiment_dir)
            shared = _named_release(
                generation_id,
                raw_dir=raw_dir,
                fundamental_events_root=fundamental_events_root,
                fundamental_events_status=fundamental_events_status,
                required_raw_datasets=required_raw_datasets,
            )
            # The live quality files describe the newest generation, so the
            # named release's own baseline copy is the one that belongs to it.
            return _publish_experiment_pin(
                experiment_dir, shared, quality_source=shared.baseline_quality_dir
            )

        update_lock = raw_dir.parent.parent / ".runtime" / "tushare" / "locks" / "tushare_update.lock"
        generation_path = raw_dir / RAW_GENERATION_FILENAME
        if not update_lock.exists() and not generation_path.exists():
            _require_raw_datasets(raw_dir, required_raw_datasets, context=f"live raw dir {raw_dir}")
            return ResearchRelease(
                raw_dir=raw_dir,
                fundamental_events_root=fundamental_events_root,
                fundamental_events_status=fundamental_events_status,
                generation_id="",
            )
        if not update_lock.exists() or not generation_path.exists():
            raise RuntimeError(
                "production research inputs require both the updater lock and raw generation marker: "
                f"lock={update_lock.exists()} marker={generation_path.exists()}"
            )

        _reject_unpinned_legacy_experiment(experiment_dir)
        release_root = raw_dir.parent / "research_releases"
        release_root.mkdir(parents=True, exist_ok=True)

        try:
            observed = _read_generation(generation_path)
        except RuntimeError:
            observed = {}

        # Cron holds LOCK_EX before changing raw or PIT. Never queue behind a
        # long update: publish current committed data only when LOCK_SH wins.
        with _try_shared_flock(update_lock) as lock_acquired:
            if lock_acquired:
                generation = _read_generation(generation_path)
                try:
                    generation_id = _committed_generation_id(generation)
                except RuntimeError:
                    pass
                else:
                    with _exclusive_flock(release_root / ".registry.lock"):
                        shared = _load_global_release(release_root / generation_id)
                        if shared is None:
                            shared = _create_global_release(
                                release_root=release_root,
                                generation=generation,
                                raw_dir=raw_dir,
                                fundamental_events_root=fundamental_events_root,
                                quality_dir=fundamental_events_status.parent,
                                fundamental_status_name=fundamental_events_status.name,
                            )
                        else:
                            _assert_source_contract(
                                shared,
                                raw_dir,
                                fundamental_events_root,
                                fundamental_events_status,
                            )
                    _require_raw_datasets(
                        shared.raw_dir,
                        required_raw_datasets,
                        context=f"research release {shared.generation_id}",
                    )
                    return _publish_experiment_pin(
                        experiment_dir, shared, quality_source=fundamental_events_status.parent
                    )

        shared = _select_existing_release(
            release_root,
            preferred_generation=str(observed.get("generation_id") or ""),
            raw_dir=raw_dir,
            fundamental_events_root=fundamental_events_root,
            fundamental_events_status=fundamental_events_status,
            required_raw_datasets=required_raw_datasets,
        )
        if shared is None:
            raise RuntimeError(
                "live research data is unavailable and no immutable release covers the configured "
                f"raw datasets {sorted(required_raw_datasets)}; retry when the updater is idle so a "
                "fresh release can be pinned"
            )
        return _publish_experiment_pin(
            experiment_dir, shared, quality_source=shared.baseline_quality_dir
        )


def published_research_release(
    *,
    generation_id: str,
    raw_dir: str | Path,
    fundamental_events_root: str | Path,
    fundamental_events_status: str | Path,
    required_raw_datasets: tuple[str, ...] = (),
) -> ResearchRelease:
    """Read one already-published release without pinning it.

    Verifies exactly what ``pin_research_release(generation_id=...)`` verifies
    before it pins, and writes nothing, so a create request can be judged
    before its experiment directory exists. The status path is the release's
    baseline quality copy, which is what that pin copies in.
    """

    shared = _named_release(
        generation_id,
        raw_dir=Path(raw_dir).resolve(),
        fundamental_events_root=Path(fundamental_events_root).resolve(),
        fundamental_events_status=Path(fundamental_events_status).resolve(),
        required_raw_datasets=required_raw_datasets,
    )
    return ResearchRelease(
        raw_dir=shared.raw_dir,
        fundamental_events_root=shared.fundamental_events_root,
        fundamental_events_status=shared.baseline_quality_dir / shared.fundamental_status_name,
        generation_id=shared.generation_id,
    )


def _named_release(
    generation_id: str,
    *,
    raw_dir: Path,
    fundamental_events_root: Path,
    fundamental_events_status: Path,
    required_raw_datasets: tuple[str, ...],
) -> _GlobalRelease:
    shared = _published_release(
        raw_dir.parent / "research_releases",
        generation_id,
        raw_dir=raw_dir,
        fundamental_events_root=fundamental_events_root,
        fundamental_events_status=fundamental_events_status,
        description=f"research release {generation_id}",
    )
    _require_raw_datasets(
        shared.raw_dir, required_raw_datasets, context=f"research release {generation_id}"
    )
    return shared


def _published_release(
    release_root: Path,
    generation_id: str,
    *,
    raw_dir: Path,
    fundamental_events_root: Path,
    fundamental_events_status: Path,
    description: str,
) -> _GlobalRelease:
    """The complete global release ``generation_id`` for this source contract."""

    _validate_generation_text(generation_id)
    expected_global = (release_root / generation_id / _MANIFEST_NAME).resolve()
    try:
        shared = _load_global_release(expected_global.parent, strict=True)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise RuntimeError(f"{description} is corrupt: {expected_global.parent}: {exc}") from exc
    if shared is None:
        raise RuntimeError(f"{description} is missing or incomplete: {expected_global.parent}")
    _assert_source_contract(
        shared,
        raw_dir,
        fundamental_events_root,
        fundamental_events_status,
    )
    return shared


def _reject_unpinned_legacy_experiment(experiment_dir: Path) -> None:
    ledger = experiment_dir / "ledgers" / "experiment_ledger.jsonl"
    if not ledger.exists():
        return
    try:
        populated = bool(ledger.read_text(encoding="utf-8").strip())
    except OSError as exc:
        raise RuntimeError(f"cannot inspect legacy experiment ledger: {ledger}: {exc}") from exc
    if populated:
        raise RuntimeError(
            "experiment has ledger records but no research-release pin; refusing to resume with a different "
            f"data generation: {ledger}"
        )


def _create_global_release(
    *,
    release_root: Path,
    generation: dict[str, object],
    raw_dir: Path,
    fundamental_events_root: Path,
    quality_dir: Path,
    fundamental_status_name: str,
) -> _GlobalRelease:
    generation_id = _committed_generation_id(generation)
    target = release_root / generation_id
    staging = release_root / f".{generation_id}.{uuid.uuid4().hex[:10]}.tmp"
    if target.exists():
        loaded = _load_global_release(target)
        if loaded is None:
            raise RuntimeError(f"research release target exists but is incomplete: {target}")
        return loaded
    try:
        staging.mkdir()
        _clone_tree(raw_dir, staging / "raw", kind="raw")
        _clone_tree(fundamental_events_root, staging / "fundamental_events", kind="pit")
        quality_files = _copy_quality_files(
            quality_dir,
            staging / "baseline_quality",
            fundamental_status_name=fundamental_status_name,
        )
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "kind": "research_release",
            "generation_id": generation_id,
            "created_at": _utc_now(),
            "raw_generation": generation,
            "fundamental_status_name": fundamental_status_name,
            "quality_files": quality_files,
            "source": {
                "raw_dir": str(raw_dir),
                "fundamental_events_root": str(fundamental_events_root),
                "quality_dir": str(quality_dir),
            },
        }
        _write_json(staging / _MANIFEST_NAME, manifest)
        staging.rename(target)
        published = _load_global_release(target)
        if published is None:  # Defensive: staging was fully validated before rename.
            raise RuntimeError(f"published research release failed validation: {target}")
        return published
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _publish_experiment_pin(
    experiment_dir: Path,
    shared: _GlobalRelease,
    *,
    quality_source: Path,
) -> ResearchRelease:
    target = experiment_dir / _PIN_DIR_NAME
    if target.exists():
        raise RuntimeError(f"research-release pin directory already exists unexpectedly: {target}")

    staging = experiment_dir / f".{_PIN_DIR_NAME}.{uuid.uuid4().hex[:10]}.tmp"
    try:
        staging.mkdir()
        quality_files = _copy_quality_files(
            quality_source,
            staging / "quality",
            fundamental_status_name=shared.fundamental_status_name,
        )
        _write_json(
            staging / _MANIFEST_NAME,
            {
                "schema_version": _SCHEMA_VERSION,
                "kind": "experiment_research_release",
                "generation_id": shared.generation_id,
                "created_at": _utc_now(),
                "quality_files": quality_files,
            },
        )
        staging.rename(target)
        return _load_experiment_pin(
            target / _MANIFEST_NAME,
            shared.root.parent,
            raw_dir=shared.source_raw_dir,
            fundamental_events_root=shared.source_fundamental_events_root,
            fundamental_events_status=shared.source_quality_dir / shared.fundamental_status_name,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_experiment_pin(
    manifest_path: Path,
    release_root: Path,
    *,
    raw_dir: Path,
    fundamental_events_root: Path,
    fundamental_events_status: Path,
) -> ResearchRelease:
    if manifest_path.is_symlink() or manifest_path.parent.is_symlink():
        raise RuntimeError(f"research-release pin must not use symlinks: {manifest_path}")
    payload = _read_json(manifest_path, description="experiment research-release manifest")
    if (
        payload.get("kind") != "experiment_research_release"
        or payload.get("schema_version") != _SCHEMA_VERSION
    ):
        raise RuntimeError(f"invalid research-release pin kind: {manifest_path}")
    generation_id = str(payload.get("generation_id") or "")
    shared = _published_release(
        release_root,
        generation_id,
        raw_dir=raw_dir,
        fundamental_events_root=fundamental_events_root,
        fundamental_events_status=fundamental_events_status,
        description="pinned global research release",
    )

    quality_dir = manifest_path.parent / "quality"
    if quality_dir.is_symlink() or not quality_dir.is_dir():
        raise RuntimeError(f"pinned quality directory is missing or invalid: {quality_dir}")
    quality_files = payload.get("quality_files")
    if not isinstance(quality_files, dict):
        raise RuntimeError(f"research-release pin has no quality file inventory: {manifest_path}")
    _verify_quality_files(quality_dir, quality_files, shared.fundamental_status_name)
    status_path = quality_dir / shared.fundamental_status_name
    return ResearchRelease(
        raw_dir=shared.raw_dir,
        fundamental_events_root=shared.fundamental_events_root,
        fundamental_events_status=status_path,
        generation_id=generation_id,
    )


def _select_existing_release(
    release_root: Path,
    *,
    preferred_generation: str,
    raw_dir: Path,
    fundamental_events_root: Path,
    fundamental_events_status: Path,
    required_raw_datasets: tuple[str, ...] = (),
) -> _GlobalRelease | None:
    def _covers_required(release: _GlobalRelease) -> bool:
        return all(_dataset_dir_populated(release.raw_dir / name) for name in required_raw_datasets)

    if preferred_generation and _SAFE_GENERATION_ID.fullmatch(preferred_generation):
        preferred = _load_global_release(release_root / preferred_generation)
        if preferred is not None and _covers_required(preferred):
            _assert_source_contract(
                preferred,
                raw_dir,
                fundamental_events_root,
                fundamental_events_status,
            )
            return preferred
    candidates: list[tuple[str, Path]] = []
    for path in release_root.iterdir() if release_root.exists() else ():
        if not path.is_dir() or path.name.startswith("."):
            continue
        try:
            payload = _read_json(path / _MANIFEST_NAME, description="research-release manifest")
        except RuntimeError:
            continue
        if payload.get("kind") == "research_release":
            candidates.append((str(payload.get("created_at") or ""), path))
    for _, path in sorted(candidates, reverse=True):
        loaded = _load_global_release(path)
        if loaded is not None and _covers_required(loaded) and _source_contract_matches(
            loaded,
            raw_dir,
            fundamental_events_root,
            fundamental_events_status,
        ):
            return loaded
    return None


def _load_global_release(path: Path, *, strict: bool = False) -> _GlobalRelease | None:
    """Load a global release directory, or None when it is structurally absent
    or incomplete. ``strict=True`` (the pin path) propagates read/validation
    exceptions so corruption keeps its precise diagnostic; ``strict=False``
    (candidate scans) treats such releases as unusable and returns None."""
    manifest_path = path / _MANIFEST_NAME
    if (
        path.is_symlink()
        or not path.is_dir()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
    ):
        return None
    try:
        payload = _read_json(manifest_path, description="research-release manifest")
        if (
            payload.get("kind") != "research_release"
            or payload.get("schema_version") != _SCHEMA_VERSION
        ):
            return None
        generation_id = str(payload.get("generation_id") or "")
        _validate_generation_text(generation_id)
        if path.name != generation_id:
            return None
        raw_dir = path / "raw"
        fundamental_root = path / "fundamental_events"
        baseline_quality = path / "baseline_quality"
        roots = (raw_dir, fundamental_root, baseline_quality)
        if any(root.is_symlink() or not root.is_dir() for root in roots):
            return None
        generation = _read_generation(raw_dir / RAW_GENERATION_FILENAME)
        if _committed_generation_id(generation) != generation_id:
            return None
        fundamental_status_name = str(
            payload.get("fundamental_status_name") or DOMAIN_STATUS_FILES["fundamentals"]
        )
        if Path(fundamental_status_name).name != fundamental_status_name:
            return None
        quality_files = payload.get("quality_files")
        if not isinstance(quality_files, dict):
            return None
        _verify_quality_files(baseline_quality, quality_files, fundamental_status_name)
        source = payload.get("source")
        if not isinstance(source, dict):
            return None
        source_paths = tuple(
            Path(str(source.get(key) or ""))
            for key in ("raw_dir", "fundamental_events_root", "quality_dir")
        )
        if any(not source_path.is_absolute() for source_path in source_paths):
            return None
        return _GlobalRelease(
            root=path,
            raw_dir=raw_dir,
            fundamental_events_root=fundamental_root,
            baseline_quality_dir=baseline_quality,
            generation_id=generation_id,
            fundamental_status_name=fundamental_status_name,
            source_raw_dir=source_paths[0],
            source_fundamental_events_root=source_paths[1],
            source_quality_dir=source_paths[2],
        )
    except (OSError, RuntimeError, ValueError, TypeError):
        if strict:
            raise
        return None


def _source_contract_matches(
    shared: _GlobalRelease,
    raw_dir: Path,
    fundamental_events_root: Path,
    fundamental_events_status: Path,
) -> bool:
    return (
        shared.source_raw_dir == raw_dir.resolve()
        and shared.source_fundamental_events_root == fundamental_events_root.resolve()
        and shared.source_quality_dir == fundamental_events_status.resolve().parent
        and shared.fundamental_status_name == fundamental_events_status.name
    )


def _assert_source_contract(
    shared: _GlobalRelease,
    raw_dir: Path,
    fundamental_events_root: Path,
    fundamental_events_status: Path,
) -> None:
    if not _source_contract_matches(
        shared,
        raw_dir,
        fundamental_events_root,
        fundamental_events_status,
    ):
        raise RuntimeError(
            f"research release {shared.generation_id} was published for a different raw/PIT/status contract"
        )


def _clone_tree(source: Path, destination: Path, *, kind: str) -> None:
    if not source.is_dir() or source.is_symlink():
        raise RuntimeError(f"research-release source must be a real directory: {source}")
    destination.mkdir(parents=True)
    _clone_directory(source, destination, kind=kind)


def _clone_directory(
    source: Path,
    destination: Path,
    *,
    kind: str,
) -> None:
    parquets: set[str] = set()
    metadata_targets: set[str] = set()
    with os.scandir(source) as entries:
        for entry in entries:
            if _is_temporary_name(entry.name):
                raise RuntimeError(
                    f"temporary file/directory found while creating research release: {source / entry.name}"
                )
            mode = entry.stat(follow_symlinks=False).st_mode
            src = source / entry.name
            dst = destination / entry.name
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"symbolic link is forbidden in a research release: {src}")
            if stat.S_ISDIR(mode):
                dst.mkdir()
                _clone_directory(src, dst, kind=kind)
            elif stat.S_ISREG(mode):
                if kind == "raw" and entry.name.endswith(".parquet.meta.json"):
                    metadata_targets.add(entry.name[: -len(".meta.json")])
                elif kind == "raw" and entry.name.endswith(".parquet"):
                    parquets.add(entry.name)
                if _should_hardlink(entry.name, kind=kind):
                    os.link(src, dst)
                else:
                    shutil.copy2(src, dst, follow_symlinks=False)
            else:
                raise RuntimeError(f"special file is forbidden in a research release: {src}")
    if kind == "raw":
        _validate_raw_pairs(source, parquets, metadata_targets)


def _should_hardlink(name: str, *, kind: str) -> bool:
    if kind == "raw":
        return name.endswith(".parquet") or name.endswith(".parquet.meta.json")
    if kind == "pit":
        return name.endswith(".parquet")
    raise ValueError(f"unknown research-release tree kind: {kind}")


def _is_temporary_name(name: str) -> bool:
    return ".tmp" in name.lower() or name.lower() == "tmp"


def _validate_raw_pairs(source: Path, parquets: set[str], metadata_targets: set[str]) -> None:
    missing_meta = sorted(parquets - metadata_targets, key=str)
    orphan_meta = sorted(metadata_targets - parquets, key=str)
    if missing_meta or orphan_meta:
        raise RuntimeError(
            f"raw parquet/metadata pairing failed under {source}: "
            f"missing_meta={missing_meta[:10]} orphan_meta={orphan_meta[:10]}"
        )


def _copy_quality_files(
    source: Path,
    destination: Path,
    *,
    fundamental_status_name: str,
) -> dict[str, bool]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, bool] = {}
    for name in _quality_names(fundamental_status_name):
        src = source / name
        dst = destination / name
        if src.is_symlink():
            raise RuntimeError(f"quality status must not be a symlink: {src}")
        if not src.exists():
            copied[name] = False
            continue
        mode = src.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise RuntimeError(f"quality status must be a regular non-symlink file: {src}")
        shutil.copy2(src, dst, follow_symlinks=False)
        try:
            read_quality_report(
                dst,
                expected_report_type=_quality_report_type(name, fundamental_status_name),
            )
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid quality status copied from {src}: {exc}") from exc
        copied[name] = True
    return copied


def _verify_quality_files(
    directory: Path,
    inventory: dict[object, object],
    fundamental_status_name: str,
) -> None:
    # A frozen release is self-describing: verify it against its OWN recorded
    # file set, not the live taxonomy (publishers always record the current
    # full set, so demanding equality here made every later audit-taxonomy
    # change retroactively corrupt all pinned releases). The declared
    # fundamentals status must at least be present in the record.
    if fundamental_status_name not in {str(name) for name in inventory}:
        raise RuntimeError(f"research-release quality file set is invalid: {directory}")
    for raw_name, raw_expected in inventory.items():
        name = str(raw_name)
        if Path(name).name != name:
            raise RuntimeError(f"invalid quality filename in research-release manifest: {name!r}")
        if not isinstance(raw_expected, bool):
            raise RuntimeError(f"invalid quality file inventory value for {name!r}")
        path = directory / name
        if path.is_symlink():
            raise RuntimeError(f"pinned quality file must not be a symlink: {path}")
        if not raw_expected:
            if path.exists():
                raise RuntimeError(f"unexpected quality file in research release: {path}")
            continue
        if not path.is_file():
            raise RuntimeError(f"pinned quality file is missing or invalid: {path}")
        try:
            read_quality_report(
                path,
                expected_report_type=_quality_report_type(name, fundamental_status_name),
            )
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid pinned quality status {path}: {exc}") from exc


def _quality_names(fundamental_status_name: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*DOMAIN_STATUS_FILES.values(), fundamental_status_name)))


def _quality_report_type(name: str, fundamental_status_name: str) -> str:
    if name == fundamental_status_name:
        return DOMAIN_REPORT_TYPES["fundamentals"]
    for domain, filename in DOMAIN_STATUS_FILES.items():
        if filename == name:
            return DOMAIN_REPORT_TYPES[domain]
    if name in LEGACY_STATUS_REPORT_TYPES:
        # Pinned releases published under an earlier taxonomy stay readable.
        return LEGACY_STATUS_REPORT_TYPES[name]
    raise RuntimeError(f"unknown quality status filename: {name!r}")


def _committed_generation_id(generation: dict[str, object]) -> str:
    require_committed_generation(generation)
    generation_id = str(generation.get("generation_id") or "")
    _validate_generation_text(generation_id)
    return generation_id


def _validate_generation_text(generation_id: str) -> None:
    if not _SAFE_GENERATION_ID.fullmatch(generation_id):
        raise RuntimeError(f"invalid raw generation id for research release: {generation_id!r}")


def _read_generation(path: Path) -> dict[str, object]:
    return _read_json(path, description="raw generation record")


def _read_json(path: Path, *, description: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {description}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid {description}: expected object at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _exclusive_flock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _try_shared_flock(path: Path) -> Iterator[bool]:
    with path.open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
