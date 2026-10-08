"""One-shot hardlink of a matching exploration PIT view seed into an experiment.

The seed tree is a repo-adjacent, gitignored snapshot of completed ``decision/``
and ``replay/`` views. It is not a live worker ``cache_root``.
New experiments hardlink those views into ``experiments/<id>/pit_views/`` only
when ``provider.json`` matches the contract this experiment would write.

Which tree an experiment reads is its ``pit_views_seed`` parameter: the default
one carries the default dataset selection, and an arm that selects other
datasets points at a tree prebuilt for exactly its own selection
(``scripts/data/prebuild_pit_views_seed.py``). An experiment that names a tree
pins the research release that tree was built from rather than the newest one,
so a seed stays usable while the nightly chain commits new generations. What a
tree carries is the
``SeedPlan`` of one research geometry (docs/data-documentation.md §3.4).
"""

from __future__ import annotations

import errno
import json
import os
import stat
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.runtime import chmod_tree, rmtree_keeping_file_modes
from autotrade.pipelines.calendar import ResearchGeometry, Slot
from autotrade.pipelines.config import SNAPSHOT_CACHE_FORMAT_VERSION

# What a finished view carries: a snapshot restates its manifest and a bundle
# its data summary. Any other directory in the layout is a level the provider
# still writes into, so it is descended, never published as a view.
_VIEW_MARKERS = ("manifest.json", "data_summary.json")

# Written at the seed root by the prebuild before it binds the tree and removed
# only once the build reports status ok, so a build that is still running, or
# that failed or was killed at any point, leaves it behind.
UNFINISHED_BUILD_MARKER = "prebuild_unfinished.json"

# The provider phase each stage's slots are published under: research years
# are validation replays, and the forward replay reads F and H as held-out data.
RESEARCH_PHASE = "valid"
FORWARD_PHASE = "heldout"


@dataclass(frozen=True)
class SeedPlan:
    """The views a seed carries for one research geometry and one release.

    Research reads the decision views at each research year's anchor and at
    research end -- the latter is the Agent's only data mount and the data
    summary bundle describes it -- plus the research-year slots. The forward
    and Held-out slots belong to the pipeline's forward replay alone, so no
    research item is stamped after research end. The as-of stash follows the
    two continuous replays: the full research chain from the first year's
    anchor, and the forward chain from research end.
    """

    geometry: ResearchGeometry
    research_slots: tuple[Slot, ...]
    forward_slots: tuple[Slot, ...]

    @property
    def decision_times(self) -> tuple[datetime, ...]:
        return (
            *(slot.anchor for slot in self.research_slots),
            self.geometry.research_decision_time,
        )

    @property
    def stash_chains(self) -> tuple[tuple[datetime, tuple[Slot, ...]], ...]:
        return (
            (self.research_slots[0].anchor, self.research_slots),
            (self.geometry.research_decision_time, self.forward_slots),
        )

    @property
    def jobs(self) -> tuple[tuple[str, Slot], ...]:
        """``(phase, slot)`` prepare jobs in anchor order."""

        return (
            *((RESEARCH_PHASE, slot) for slot in self.research_slots),
            *((FORWARD_PHASE, slot) for slot in self.forward_slots),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "geometry": self.geometry.to_record(),
            "decision_views": [value.isoformat() for value in self.decision_times],
            "research_slots": [slot.to_record() for slot in self.research_slots],
            "forward_slots": [slot.to_record() for slot in self.forward_slots],
            "bundle": self.geometry.research_decision_time.isoformat(),
            "asof_stash_chains": [
                {"decision": decision.isoformat(), "slots": [slot.label for slot in chain]}
                for decision, chain in self.stash_chains
            ],
        }


def plan_seed(geometry: ResearchGeometry, trading_days: Sequence[str]) -> SeedPlan:
    """The seed plan of ``geometry`` over a release's daily trading dates."""

    return SeedPlan(
        geometry=geometry,
        research_slots=geometry.research_years,
        forward_slots=(geometry.forward, geometry.heldout(trading_days)),
    )


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


def assert_seed_snapshot_config(
    seed: Path, snapshot_config: SnapshotConfig
) -> tuple[str, str]:
    """Refuse a seed prebuilt for a different snapshot configuration or cache format.

    The create-time half of the contract check: the cache format and the
    snapshot configuration, which is exactly what decides whether a dataset
    selection has views here at all. The other two fields of
    ``pit_cache_provider_record`` — the generation and release path the seed
    was built from — are returned rather than judged: an experiment that names
    this seed pins that very release, so it is the caller that checks the
    release is published and reaches Held-out. ``seed_pit_views`` still
    compares the whole record before it links anything.
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
    # exists yet. The prebuild's own marker does: it outlives a build that is
    # still running, failed or was killed, wherever it stopped.
    if (Path(seed) / UNFINISHED_BUILD_MARKER).exists():
        raise ValueError(
            f"PIT view seed {seed} has an unfinished build: {UNFINISHED_BUILD_MARKER} "
            "is still present, so scripts/data/prebuild_pit_views_seed.py has not "
            "reported status ok for this seed. Wait for a running prebuild; re-run "
            "one that failed or was killed until it reports status ok"
        )
    return str(record.get("generation_id") or ""), str(record.get("release_raw_dir") or "")


def assert_seed_carries_bonus_split(seed: Path) -> None:
    """Refuse a seed whose replay slots cannot tell bonus shares from conversions.

    Every arm charges the dividend tax, which reads ``bonus_per_share`` from
    every slot's ex-date table, and its Broker refuses the first day of a
    replay without it. A seed built before the column existed would therefore fail
    the arm's first Validation mid-session; read from the tables' schemas,
    it is refused when the arm is created instead.
    """

    stale = [
        path
        for path in sorted(Path(seed).glob("replay/**/corporate_actions.parquet"))
        if "bonus_per_share" not in pq.read_schema(path).names
    ]
    if stale:
        raise ValueError(
            f"PIT view seed {seed} predates the bonus-share column the dividend tax reads "
            f"({len(stale)} replay slot table(s), first {stale[0].parent.relative_to(seed)}); "
            "rebuild the seed under a new directory"
        )


def assert_seed_views_share_types(seed: Path) -> int:
    """Refuse a seed whose views type one column of a domain two ways.

    The Timeview merges rows of several views of a domain into one as-of part
    (the first roll after ``continue_into`` takes the earlier slot's pending
    rows with the next slot's, a late start publishes years at once), each
    slice after the ``ReplayRows.take`` round trip through pandas, and that
    merge refuses a column typed two ways (``string`` beside ``large_string``,
    ``int64`` beside ``double``). So, per file of a view (``events.parquet``,
    ``text_library/anns_d.parquet``, ...), every decision and replay view must
    type each column alike after that round trip. A column the round trip
    types by its values (object) is taken at its file type, and ``null`` (no
    value in the file) unifies with any type. Column sets may differ: parts
    are projected onto the frozen part's columns.

    Returns the number of files checked.
    """

    seed = Path(seed)
    seen: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    checked = 0
    for view in _completed_seed_views(seed):
        for path in sorted(view.rglob("*.parquet")):
            checked += 1
            name = path.relative_to(view).as_posix()
            for column, data_type in _round_trip_types(path).items():
                if data_type != pa.null():
                    seen[(name, column)].setdefault(str(data_type), path)
    conflicts = [
        f"{name} {column}: "
        + ", ".join(f"{data_type} ({path.parent.relative_to(seed)})" for data_type, path in types.items())
        for (name, column), types in seen.items()
        if len(types) > 1
    ]
    if conflicts:
        raise ValueError(
            f"PIT view seed {seed} types {len(conflicts)} column(s) differently across its views "
            "after the replay round trip, so the Timeview cannot merge a part that spans them; "
            "rebuild the seed under a new directory. First: " + "; ".join(conflicts[:5])
        )
    return checked


def _round_trip_types(path: Path) -> dict[str, pa.DataType]:
    """Each column's Arrow type after ``ReplayRows.take``'s pandas round trip.

    The round trip of a zero-row table gives the type a column's pandas dtype
    fixes; an object column comes back ``null`` there and takes its file
    type, which its values infer. A column pandas restores as the index is
    not a column of the round trip.
    """

    schema = pq.read_schema(path)
    back = pa.Table.from_pandas(schema.empty_table().to_pandas(), preserve_index=False).schema
    types: dict[str, pa.DataType] = {}
    for field in schema:
        index = back.get_field_index(field.name)
        if index >= 0:
            restored = back.field(index).type
            types[field.name] = field.type if restored == pa.null() else restored
    return types


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


def _publish_seed_entry(
    source: Path, seed: Path, dest_root: Path, *, dir_mode: int
) -> None:
    """Hardlink one seed entry into place, or leave an existing one alone.

    The staging tree is a hardlink copy, so its files are the seed's files and
    the files of every experiment already seeded from them. Locking it to
    0444/``dir_mode`` before the rename is what publishes the view read-only;
    discarding it must therefore touch the directories only
    (:func:`rmtree_keeping_file_modes`). Unlocking the payload on the way out
    unfroze the shared inodes instead, and every decision snapshot on the host
    failed ``_require_read_only_tree`` until the next seeding re-locked them.
    """

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
            # Another writer published this view first; its tree is the one
            # that stands and this staging copy is simply discarded.
            if not target.exists():
                raise
    finally:
        if staging.exists():
            rmtree_keeping_file_modes(staging)


def _completed_seed_views(seed: Path) -> list[Path]:
    """Every completed view in the seed, wherever the layout puts it.

    Views sit at different depths: ``decision/<slot>``, the unphased
    ``replay/<slot>`` source and the ``replay/<phase>/<slot>`` views hardlinked
    from it. A view is therefore recognised by the marker the provider writes
    when it publishes one, never by its depth.
    Publishing a layout level instead of a view would freeze that level
    read-only in the experiment, and the provider could then neither take the
    lock beside a seeded slot nor stage a new slot next to it.
    """

    views: list[Path] = []
    for name in ("decision", "replay"):
        views.extend(_marked_seed_views(seed / name))
    return views


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
    "FORWARD_PHASE",
    "RESEARCH_PHASE",
    "UNFINISHED_BUILD_MARKER",
    "SeedPlan",
    "assert_seed_carries_bonus_split",
    "assert_seed_snapshot_config",
    "assert_seed_views_share_types",
    "pit_cache_provider_record",
    "plan_seed",
    "seed_pit_views",
]
