from __future__ import annotations

import errno
import json
import os
import shutil
import stat
import uuid
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.runtime import chmod_tree
from autotrade.environment.strategy import StrategySchedule
from autotrade.pipelines import pit_backend, pit_views_seed
from autotrade.pipelines.calendar import ResearchGeometry
from autotrade.pipelines.config import (
    DEFAULT_RESEARCH_GEOMETRY,
    SNAPSHOT_CACHE_FORMAT_VERSION,
    SnapshotBundle,
)
from autotrade.pipelines.pit_backend import prebuild_asof_stash
from autotrade.pipelines.pit_views_seed import (
    FORWARD_PHASE,
    RESEARCH_PHASE,
    pit_cache_provider_record,
    plan_seed,
    seed_pit_views,
)


def _record(raw_dir: Path, *, events: bool = True) -> dict[str, object]:
    config = SnapshotConfig(
        include_intraday=False,
        events_datasets=("margin",) if events else (),
        macro_datasets=(),
        text_datasets=(),
        fundamental_datasets=(),
        replay_include_events=events,
        replay_include_text=False,
        replay_include_minutes=False,
        replay_include_macro=False,
        replay_include_fundamentals=False,
    )
    return pit_cache_provider_record(
        generation_id="generation_test",
        release_raw_dir=raw_dir,
        snapshot_config=config,
    )


SEED_DECISION_KEY = "20240101T235959+0800"
SEED_SLOT = "20240102_20240103_20240101T235959+0800"
SEED_HELDOUT_SLOT = "20240201_20240202_20240101T235959+0800"

SEED_STASH_LEAF = (
    f"asof_stash/decision/{SEED_DECISION_KEY}/replay/"
    f"{SEED_SLOT}/schedule/period=day/"
    "inference_time/hour=08/minute=30"
)

# The view layout a prebuild leaves behind: the decision snapshot, the unphased
# replay source of each region and the phase views hardlinked from those sources.
SEED_VIEWS = (
    f"decision/{SEED_DECISION_KEY}",
    f"replay/{SEED_SLOT}",
    f"replay/{SEED_HELDOUT_SLOT}",
    f"replay/paper/{SEED_SLOT}",
    f"replay/valid/{SEED_SLOT}",
    f"replay/heldout/{SEED_HELDOUT_SLOT}",
)


def _write_seed(seed: Path, record: dict[str, object]) -> Path:
    stash = seed / SEED_STASH_LEAF
    partial = seed / "asof_stash" / "no_contract_yet"
    for directory in (stash / "daily", partial):
        directory.mkdir(parents=True)
    for name in SEED_VIEWS:
        view = seed / name
        view.mkdir(parents=True)
        kind = name.split("/", 1)[0]
        (view / "manifest.json").write_text(json.dumps({"kind": kind}), encoding="utf-8")
        (view / "daily.parquet").write_bytes(f"{kind}-bytes".encode())
        # The provider leaves its own slot lock beside every published view.
        view.with_suffix(".lock").touch()
    (stash / "contract.json").write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    (stash / "daily" / "part_0001.parquet").write_bytes(b"asof-part")
    (partial / "scratch.bin").write_bytes(b"stash")
    (seed / "provider.json").write_text(json.dumps(record), encoding="utf-8")
    return seed / f"decision/{SEED_DECISION_KEY}" / "daily.parquet"


def _freeze_seed(seed: Path) -> None:
    """Leave the fake seed as the prebuild leaves a real one.

    Every published view is read-only, the stash parts are read-only, and the
    layout directories around them keep the mode ``mkdir`` gave them.
    """

    for name in SEED_VIEWS:
        chmod_tree(seed / name, file_mode=0o444, dir_mode=0o555)
    for part in (seed / SEED_STASH_LEAF).rglob("part_*.parquet"):
        part.chmod(0o444)


def test_matching_seed_hardlinks_views_and_prebuilt_stash(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    experiment = tmp_path / "exp"
    dest = experiment / "pit_views"
    raw = tmp_path / "raw"
    record = _record(raw)
    source_file = _write_seed(seed, record)

    assert seed_pit_views(dest, seed, expected_provider=record) is True

    linked = dest / "decision" / "20240101T235959+0800" / "daily.parquet"
    assert linked.is_file()
    assert linked.read_bytes() == b"decision-bytes"
    assert os.stat(linked).st_ino == os.stat(source_file).st_ino
    assert os.stat(linked).st_nlink >= 2
    assert (dest / "replay" / "20240102_20240103_20240101T235959+0800" / "daily.parquet").is_file()
    # A prebuilt stash comes across so the first backtest hardlinks the as-of
    # parts; its parts stay immutable but its directories stay writable,
    # because a replay can still reach a day the prebuild did not cover.
    part = dest / SEED_STASH_LEAF / "daily" / "part_0001.parquet"
    assert part.is_file()
    assert os.stat(part).st_ino == os.stat(seed / SEED_STASH_LEAF / "daily" / "part_0001.parquet").st_ino
    assert not os.access(part, os.W_OK)
    assert os.access(part.parent, os.W_OK)
    assert (dest / SEED_STASH_LEAF / "contract.json").is_file()
    # A stash directory without a published contract is not a finished stash.
    assert not (dest / "asof_stash" / "no_contract_yet").exists()
    outside = [path for path in tmp_path.rglob("*") if path.is_file()]
    for path in outside:
        resolved = path.resolve()
        assert resolved.is_relative_to(seed.resolve()) or resolved.is_relative_to(
            experiment.resolve()
        )


def test_seeded_tree_is_indistinguishable_from_a_cold_build(tmp_path: Path) -> None:
    """A seeded cache must still be a cache the provider can write into.

    Before touching a slot the provider takes an exclusive lock beside it and
    stages a new slot in the same directory. Publishing a layout level — a
    replay phase directory, for one — as if it were a view freezes it
    read-only, and the worker then dies on its own lock at startup.
    """

    seed = tmp_path / "seed"
    dest = tmp_path / "exp" / "pit_views"
    record = _record(tmp_path / "raw")
    _write_seed(seed, record)
    _freeze_seed(seed)

    assert seed_pit_views(dest, seed, expected_provider=record) is True

    # A seeded view is an immutable read-only hardlink of the seed's own.
    view = dest / "replay" / "paper" / SEED_SLOT
    linked = view / "daily.parquet"
    assert (
        os.stat(linked).st_ino
        == os.stat(seed / "replay" / "paper" / SEED_SLOT / "daily.parquet").st_ino
    )
    assert not os.access(view, os.W_OK)
    assert not os.access(linked, os.W_OK)

    # Every directory that can still receive an entry stays writable.
    for directory in (
        dest,
        dest / "decision",
        dest / "replay",
        dest / "replay" / "paper",
        dest / "replay" / "valid",
        dest / "replay" / "heldout",
        (dest / SEED_STASH_LEAF).parent,
    ):
        assert os.access(directory, os.W_OK), directory

    # The provider's lock beside each seeded slot, in every phase directory and
    # beside the decision snapshot.
    for slot in (
        dest / "replay" / "paper" / SEED_SLOT,
        dest / "replay" / "valid" / SEED_SLOT,
        dest / "replay" / "heldout" / SEED_HELDOUT_SLOT,
        dest / "replay" / SEED_SLOT,
        dest / "decision" / SEED_DECISION_KEY,
    ):
        with pit_backend._exclusive_lock(slot.with_suffix(".lock")):
            assert slot.with_suffix(".lock").is_file()

    # A slot the seed does not carry still cold-builds beside the seeded ones:
    # staging directory in the same phase directory, then the atomic rename.
    fresh = dest / "replay" / "paper" / "20240301_20240302_20240101T235959+0800"
    with pit_backend._exclusive_lock(fresh.with_suffix(".lock")):
        staging = fresh.with_name(f".{fresh.name}.{uuid.uuid4().hex}.tmp")
        staging.mkdir()
        (staging / "manifest.json").write_text("{}", encoding="utf-8")
        os.replace(staging, fresh)
    assert (fresh / "manifest.json").is_file()


def test_losing_the_publication_race_does_not_unfreeze_the_view_that_won(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A discarded staging copy must not change the mode of anything it links.

    The staging tree is a hardlink copy, so its files ARE the seed's files and
    the files of every experiment already seeded from them. Unlocking the
    payload to remove the copy unfroze all of them at once, and from then on
    every formal replay on the host died in ``_require_read_only_tree`` with
    ``decision snapshot is not read-only`` until the next seeding re-locked
    them. Only the directories may be opened on the way out.
    """

    seed = tmp_path / "seed"
    dest = tmp_path / "exp" / "pit_views"
    record = _record(tmp_path / "raw")
    _write_seed(seed, record)
    _freeze_seed(seed)

    target = dest / "decision" / SEED_DECISION_KEY
    source = seed / "decision" / SEED_DECISION_KEY
    real_chmod_tree = pit_views_seed.chmod_tree

    def chmod_tree_then_lose_the_race(root: Path, **modes: int) -> None:
        real_chmod_tree(root, **modes)
        if not root.name.startswith(f".{SEED_DECISION_KEY}") or target.exists():
            return
        # The rival writer publishes between this lock and the rename below,
        # exactly as it would: a complete, read-only, hardlinked view. The
        # rename then fails because the destination is a non-empty directory.
        target.mkdir(parents=True)
        for child in sorted(source.iterdir()):
            os.link(child, target / child.name)
        real_chmod_tree(target, file_mode=0o444, dir_mode=0o555)

    monkeypatch.setattr(pit_views_seed, "chmod_tree", chmod_tree_then_lose_the_race)

    assert seed_pit_views(dest, seed, expected_provider=record) is True

    # The winner stands and nothing of the discarded copy is left beside it.
    assert (target / "daily.parquet").read_bytes() == b"decision-bytes"
    assert not [path for path in dest.rglob("*") if ".tmp" in path.name]
    # Every shared inode is still frozen: the seed's own tree, the view that
    # won, and therefore every other experiment linked to the same files.
    for tree in (source, target):
        for path in sorted(tree.rglob("*")):
            assert stat.S_IMODE(path.stat().st_mode) == 0o444, path


def test_reseeding_an_already_seeded_experiment_changes_nothing(tmp_path: Path) -> None:
    """A restarted worker seeds again over a cache the first run finished."""

    seed = tmp_path / "seed"
    dest = tmp_path / "exp" / "pit_views"
    record = _record(tmp_path / "raw")
    _write_seed(seed, record)
    _freeze_seed(seed)

    assert seed_pit_views(dest, seed, expected_provider=record) is True
    before = _tree_state(dest)

    assert seed_pit_views(dest, seed, expected_provider=record) is True
    assert _tree_state(dest) == before
    assert not [path for path in dest.rglob("*") if ".tmp" in path.name]


def _tree_state(root: Path) -> dict[str, tuple[int, int]]:
    """Inode and mode of every entry, so a re-link or a rewrite is visible."""

    return {
        str(path.relative_to(root)): (
            path.stat().st_ino,
            stat.S_IMODE(path.stat().st_mode),
        )
        for path in sorted(root.rglob("*"))
    }


def test_missing_seed_is_a_noop(tmp_path: Path) -> None:
    dest = tmp_path / "exp" / "pit_views"
    record = _record(tmp_path / "raw")
    assert (
        seed_pit_views(dest, tmp_path / "missing-seed", expected_provider=record) is False
    )
    assert not dest.exists()


def test_mismatch_does_not_mix_views(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    dest = tmp_path / "exp" / "pit_views"
    raw = tmp_path / "raw"
    _write_seed(seed, _record(raw, events=True))
    assert (
        seed_pit_views(dest, seed, expected_provider=_record(raw, events=False)) is False
    )
    assert not dest.exists() or not any(dest.rglob("*"))


def test_required_mismatch_fails_fast(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    dest = tmp_path / "exp" / "pit_views"
    raw = tmp_path / "raw"
    _write_seed(seed, _record(raw, events=True))
    with pytest.raises(RuntimeError, match="refusing to mix"):
        seed_pit_views(
            dest,
            seed,
            expected_provider=_record(raw, events=False),
            required=True,
        )
    assert not dest.exists() or not any(dest.rglob("*"))


def test_required_missing_seed_fails_fast(tmp_path: Path) -> None:
    dest = tmp_path / "exp" / "pit_views"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        seed_pit_views(
            dest,
            tmp_path / "missing-seed",
            expected_provider=_record(tmp_path / "raw"),
            required=True,
        )
    assert not dest.exists()


def test_cross_filesystem_hardlink_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = tmp_path / "seed"
    dest = tmp_path / "exp" / "pit_views"
    record = _record(tmp_path / "raw")
    _write_seed(seed, record)

    def boom(src: str, dst: str) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(os, "link", boom)
    with pytest.raises(RuntimeError, match="different filesystem"):
        seed_pit_views(dest, seed, expected_provider=record)


def test_partial_hardlink_failure_is_cleaned_and_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = tmp_path / "seed"
    dest = tmp_path / "exp" / "pit_views"
    record = _record(tmp_path / "raw")
    _write_seed(seed, record)
    (seed / "decision" / "20240101T235959+0800" / "second.parquet").write_bytes(
        b"second"
    )
    real_link = os.link
    calls = 0

    def fail_second_link(src: str, dst: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.EIO, "injected link failure")
        real_link(src, dst)

    monkeypatch.setattr(os, "link", fail_second_link)
    with pytest.raises(OSError, match="injected link failure"):
        seed_pit_views(dest, seed, expected_provider=record)

    target = dest / "decision" / "20240101T235959+0800"
    assert not target.exists()
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))

    monkeypatch.setattr(os, "link", real_link)
    assert seed_pit_views(dest, seed, expected_provider=record) is True
    assert (target / "daily.parquet").read_bytes() == b"decision-bytes"
    assert (target / "second.parquet").read_bytes() == b"second"


def test_provider_record_carries_the_cache_contract(tmp_path: Path) -> None:
    record = _record(tmp_path / "raw")
    assert record["schema_version"] == SNAPSHOT_CACHE_FORMAT_VERSION
    assert record["generation_id"] == "generation_test"
    assert "snapshot_config" in record


def _daily_frame(days: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": days,
            "ts_code": ["000001.SZ"] * len(days),
            "close": [10.0] * len(days),
            "available_at": [
                f"{day[:4]}-{day[4:6]}-{day[6:]}T17:30:00+08:00" for day in days
            ],
        }
    )


def _stash_slots(tmp_path: Path) -> tuple[Path, Path]:
    """A minimal PIT cache root: one decision slot and a three-day replay slot."""

    cache_root = tmp_path / "pit_views"
    snapshot = cache_root / "decision" / "20240101T235959+0800"
    replay = cache_root / "replay" / "valid" / "20240102_20240104_20240101T235959+0800"
    snapshot.mkdir(parents=True)
    replay.mkdir(parents=True)
    (cache_root / "provider.json").write_text(
        json.dumps(_record(tmp_path / "raw")), encoding="utf-8"
    )
    raw_generation = {"generation_id": "generation_test"}
    (snapshot / "manifest.json").write_text(
        json.dumps({"kind": "decision_input", "raw_generation": raw_generation}),
        encoding="utf-8",
    )
    (replay / "manifest.json").write_text(
        json.dumps(
            {"kind": "replay_slot", "label": "valid", "raw_generation": raw_generation}
        ),
        encoding="utf-8",
    )
    _daily_frame(["20240101"]).to_parquet(snapshot / "daily.parquet", index=False)
    _daily_frame(["20240102", "20240103", "20240104"]).to_parquet(
        replay / "daily.parquet", index=False
    )
    return snapshot, replay


def _prebuild_stash(tmp_path: Path, slots: tuple[Path, Path], host: str) -> dict[str, object]:
    snapshot, replay = slots
    return prebuild_asof_stash(
        snapshot_dir=snapshot,
        replay_dir=replay,
        schedule=StrategySchedule("day", "08:30"),
        phase="valid",
        generation_id="generation_test",
        start="20240102",
        end="20240104",
        host_dir=tmp_path / "host" / host,
    )


def test_finished_stash_is_reused_instead_of_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A seed prebuild is idempotent: the second run replays nothing.

    The prebuild is the expensive half of seeding a region, and a killed run is
    restarted from the top, so a stash this contract already finished must be
    detected rather than encoded again.
    """

    slots = _stash_slots(tmp_path)
    built = _prebuild_stash(tmp_path, slots, "first")
    stash = Path(str(built["stash_dir"]))
    assert built["reused"] is False
    assert built["refresh_calls"] == 3
    # Part 0000 is the frozen decision snapshot, which the stash never holds:
    # only the rolling parts the replay encodes are shared.
    parts = sorted(path.name for path in (stash / "daily").glob("part_*.parquet"))
    assert parts == ["part_0001.parquet", "part_0002.parquet"]
    inodes = {path: path.stat().st_ino for path in (stash / "daily").iterdir()}

    def no_replay(*args: object, **kwargs: object) -> object:
        raise AssertionError("a finished stash must not be replayed again")

    monkeypatch.setattr(pit_backend, "Timeview", no_replay)
    reused = _prebuild_stash(tmp_path, slots, "second")
    assert reused["reused"] is True
    assert reused["refresh_calls"] == 0
    assert reused["trade_days"] == built["trade_days"]
    assert reused["stash_dir"] == built["stash_dir"]
    assert {path: path.stat().st_ino for path in (stash / "daily").iterdir()} == inodes


def test_incomplete_stash_is_never_taken_for_a_finished_one(tmp_path: Path) -> None:
    """A prebuild killed mid-window, or a stash that lost a part, rebuilds."""

    slots = _stash_slots(tmp_path)
    built = _prebuild_stash(tmp_path, slots, "first")
    stash = Path(str(built["stash_dir"]))

    # Killed before the window finished: parts on disk, no finished record.
    record = stash / "prebuild.json"
    kept = record.read_text(encoding="utf-8")
    record.unlink()
    assert _prebuild_stash(tmp_path, slots, "second")["reused"] is False
    assert record.is_file()

    # A part has since gone missing: the record alone does not make it complete.
    record.write_text(kept, encoding="utf-8")
    missing = stash / "daily" / "part_0002.parquet"
    missing.unlink()
    rebuilt = _prebuild_stash(tmp_path, slots, "third")
    assert rebuilt["reused"] is False
    assert rebuilt["refresh_calls"] == 3
    assert missing.is_file()


def _release_days(end: str = "20260911") -> list[str]:
    """Weekday daily dates of a release ending ``end``."""

    return [day.strftime("%Y%m%d") for day in pd.bdate_range("2019-01-02", end)]


def _cn(day: str) -> str:
    return f"{day[:4]}-{day[4:6]}-{day[6:]}T23:59:59+08:00"


def test_the_seed_plan_is_exactly_the_research_forward_and_heldout_views() -> None:
    """Five decision views, four research years, the forward and the clipped
    Held-out slot, one bundle at research end and two as-of chains."""

    plan = plan_seed(DEFAULT_RESEARCH_GEOMETRY, _release_days())
    assert [value.isoformat() for value in plan.decision_times] == [
        _cn(day) for day in ("20210630", "20220630", "20230630", "20240630", "20250630")
    ]
    assert [(slot.label, slot.start, slot.end) for slot in plan.research_slots] == [
        ("Y1", "20210701", "20220630"),
        ("Y2", "20220701", "20230630"),
        ("Y3", "20230701", "20240630"),
        ("Y4", "20240701", "20250630"),
    ]
    assert [
        (slot.label, slot.start, slot.end, slot.requested_end, slot.truncation_reason)
        for slot in plan.forward_slots
    ] == [
        ("F", "20250701", "20260630", "20260630", None),
        ("H", "20260701", "20260911", "20260930", "release_ends_20260911"),
    ]
    # Every slot is prepared at its own anchor, research years as validations
    # and the forward replay's two slots as held-out data.
    assert [(phase, slot.label, slot.anchor.isoformat()) for phase, slot in plan.jobs] == [
        (RESEARCH_PHASE, "Y1", _cn("20210630")),
        (RESEARCH_PHASE, "Y2", _cn("20220630")),
        (RESEARCH_PHASE, "Y3", _cn("20230630")),
        (RESEARCH_PHASE, "Y4", _cn("20240630")),
        (FORWARD_PHASE, "F", _cn("20250630")),
        (FORWARD_PHASE, "H", _cn("20260630")),
    ]
    record = json.loads(json.dumps(plan.to_record()))
    assert record["bundle"] == _cn("20250630")
    assert record["asof_stash_chains"] == [
        {"decision": _cn("20210630"), "slots": ["Y1", "Y2", "Y3", "Y4"]},
        {"decision": _cn("20250630"), "slots": ["F", "H"]},
    ]


@pytest.mark.parametrize(
    ("geometry", "release_end"),
    [
        (DEFAULT_RESEARCH_GEOMETRY, "20260911"),
        (
            ResearchGeometry("20200701", "20220630", "20230630", "20231231"),
            "20260911",
        ),
    ],
)
def test_nothing_planned_for_research_is_stamped_after_research_end(
    geometry: ResearchGeometry, release_end: str
) -> None:
    """Research reads the decision views, their bundle and the research-year
    slots. A decision view holds rows up to its anchor and a slot rows up to its
    last day, so all of them end by research end; the forward and Held-out slots
    start after it and are planned only for the forward replay."""

    plan = plan_seed(geometry, _release_days(release_end))
    research_end = geometry.research_decision_time
    assert research_end.isoformat() == _cn(geometry.research_end)
    assert max(plan.decision_times) == research_end
    assert plan.research_slots == geometry.research_years
    assert all(slot.end <= geometry.research_end for slot in plan.research_slots)
    assert [slot for phase, slot in plan.jobs if phase == RESEARCH_PHASE] == list(
        plan.research_slots
    )
    assert all(
        slot.start > geometry.research_end and slot.anchor >= research_end
        for slot in plan.forward_slots
    )
    assert [slot.label for slot in plan.forward_slots] == ["F", "H"]
    assert not {slot.label for slot in plan.research_slots} & {"F", "H"}


class _FakeProvider:
    """The provider surface the prebuild reads, recording what it is asked."""

    calls: list[tuple[str, str, str, str]] = []

    def __init__(self, **_kwargs: object) -> None:
        self.trading_days = _release_days()
        self.release = SimpleNamespace(generation_id="generation_test", raw_dir=Path("raw"))

    def prepare(self, *, phase, start, end, decision_time):  # noqa: ANN001
        self.calls.append((phase, start, end, decision_time.isoformat()))
        return SnapshotBundle(
            snapshot_id="snapshot",
            decision_ref=f"decision/{decision_time:%Y%m%d}",
            replay_ref=f"replay/{phase}/{start}_{end}",
            generation_id="generation_test",
        )


def _run_prebuild(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str]
) -> list[dict[str, object]]:
    from scripts.data import prebuild_pit_views_seed as prebuild

    stashed: list[dict[str, object]] = []

    def fake_stash(**kwargs: object) -> dict[str, object]:
        stashed.append(kwargs)
        return {"reused": False, "trade_days": 1}

    _FakeProvider.calls = []
    monkeypatch.setattr(prebuild, "ResearchPITSnapshotProvider", _FakeProvider)
    monkeypatch.setattr(prebuild, "prebuild_asof_stash", fake_stash)
    assert prebuild.main(["--repo-root", str(tmp_path), *argv]) == 0
    return stashed


def test_the_prebuild_prepares_every_planned_slot_and_stashes_each_chain_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stashed = _run_prebuild(monkeypatch, tmp_path, ["--heldout-end", "20260831"])
    assert _FakeProvider.calls == [
        ("valid", "20210701", "20220630", _cn("20210630")),
        ("valid", "20220701", "20230630", _cn("20220630")),
        ("valid", "20230701", "20240630", _cn("20230630")),
        ("valid", "20240701", "20250630", _cn("20240630")),
        ("heldout", "20250701", "20260630", _cn("20250630")),
        ("heldout", "20260701", "20260831", _cn("20260630")),
    ]
    # Only the first slot of each chain continues nothing, so only those two
    # are encoded offline; the rest is named as built on first use.
    assert [
        (call["snapshot_dir"], call["replay_dir"], call["phase"], call["start"], call["end"])
        for call in stashed
    ] == [
        ("decision/20210630", "replay/valid/20210701_20220630", "valid", "20210701", "20220630"),
        ("decision/20250630", "replay/heldout/20250701_20260630", "heldout", "20250701", "20260630"),
    ]
    out = capsys.readouterr().out
    assert "on first use: Y2,Y3,Y4" in out and "on first use: H" in out
    assert json.loads(out.strip().splitlines()[-1])["status"] == "ok"


def test_the_dry_run_prints_the_plan_and_builds_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stashed = _run_prebuild(monkeypatch, tmp_path, ["--dry-run"])
    assert _FakeProvider.calls == [] and stashed == []
    lines = capsys.readouterr().out.strip().splitlines()
    header = json.loads(lines[0])
    assert header["jobs"] == 6
    assert header["plan"] == json.loads(
        json.dumps(plan_seed(DEFAULT_RESEARCH_GEOMETRY, _release_days()).to_record())
    )
    assert lines[1].startswith("[1/6] Y1 valid 20210701..20220630")
    assert json.loads(lines[-1])["status"] == "planned"
    assert not (tmp_path / "data/pit_views_seed/explore").exists()


def test_the_prebuild_refuses_a_misaligned_geometry_before_pinning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts.data import prebuild_pit_views_seed as prebuild

    def refuse(**_kwargs: object) -> None:
        raise AssertionError("no release may be pinned for a refused geometry")

    monkeypatch.setattr(prebuild, "ResearchPITSnapshotProvider", refuse)
    with pytest.raises(ValueError, match="twelve months after research"):
        prebuild.main(["--repo-root", str(tmp_path), "--forward-end", "20261231", "--dry-run"])


def test_replay_manifest_matches_requires_phase_label():
    from datetime import datetime

    from autotrade.environment.data.contracts import CN_TZ
    from autotrade.pipelines.pit_backend import _replay_manifest_matches

    decision = datetime(2021, 12, 31, 23, 59, 59, tzinfo=CN_TZ)
    manifest = {
        "kind": "replay_slot",
        "period_start": "20220101",
        "period_end": "20220331",
        "available_from": decision.isoformat(),
        "label": "heldout",
    }
    assert _replay_manifest_matches(
        manifest,
        start="20220101",
        end="20220331",
        decision=decision,
        phase="heldout",
    )
    assert not _replay_manifest_matches(
        manifest,
        start="20220101",
        end="20220331",
        decision=decision,
        phase="valid",
    )
    assert _replay_manifest_matches(
        manifest,
        start="20220101",
        end="20220331",
        decision=decision,
        phase=None,
    )


def _prebuild_identity(argv: list[str]) -> dict[str, object]:
    """The seed contract `prebuild_pit_views_seed.py` would write for `argv`."""

    from scripts.data import prebuild_pit_views_seed as prebuild

    args = prebuild.build_parser().parse_args(argv)
    return prebuild._snapshot_config(prebuild.create_parameters(args)).to_record()


def _experiment_identity(params: dict[str, object]) -> dict[str, object]:
    """The seed contract an experiment created with `params` would write."""

    from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS
    from autotrade.pipelines.worker import _snapshot_config

    return _snapshot_config({**WEB_CREATE_DEFAULTS, **params}).to_record()


# The 2026-09-14 derivatives arm: eight macro series and one events series on
# top of the default scope. Spelled out because the selection REPLACES the
# domain default rather than adding to it.
EXTRA_MACRO = (
    "fut_basic",
    "fut_mapping",
    "fut_daily",
    "opt_basic",
    "opt_daily",
    "cb_basic",
    "cb_daily",
    "cb_call",
)


def test_prebuild_overrides_build_the_identity_the_experiment_asks_for() -> None:
    """A seed is reusable only on whole-record equality, so the prebuild's
    parameters and the experiment's must resolve through one function, not two
    readings that agree today. Asserted on the record itself, and against the
    default identity, so an override that silently did nothing would fail."""

    from autotrade.environment.data.snapshot import DEFAULT_DATASETS

    macro = tuple(DEFAULT_DATASETS["macro"]) + EXTRA_MACRO
    events = tuple(DEFAULT_DATASETS["events"]) + ("stk_surv",)
    prebuilt = _prebuild_identity(
        [
            "--macro-datasets",
            ",".join(macro),
            "--events-datasets",
            ",".join(events),
            "--no-include-intraday",
        ]
    )
    assert prebuilt == _experiment_identity(
        {
            "macro_datasets": list(macro),
            "events_datasets": list(events),
            "include_intraday": False,
        }
    )
    assert prebuilt["datasets"]["macro"] == list(macro)  # type: ignore[index]
    assert prebuilt["datasets"]["events"] == list(events)  # type: ignore[index]
    assert prebuilt != _prebuild_identity([])


def test_prebuild_carries_every_snapshot_identity_knob_it_offers() -> None:
    """Each override has to reach the record; one that parsed but never
    resolved would build a seed no experiment can use."""

    prebuilt = _prebuild_identity(
        [
            "--no-include-text",
            "--include-intraday",
            "--intraday-trade-days",
            "5",
            "--macro-window-months",
            "36",
        ]
    )
    assert prebuilt == _experiment_identity(
        {
            "include_text": False,
            "include_intraday": True,
            "intraday_trade_days": 5,
            "macro_window_months": 36,
        }
    )
    assert prebuilt["datasets"]["text"] == []  # type: ignore[index]
    assert prebuilt["include_intraday"] is True
    assert prebuilt["replay"]["include_minutes"] is True  # type: ignore[index]
    assert prebuilt["decision_windows"]["macro_months"] == 36  # type: ignore[index]


def test_prebuild_refuses_a_dataset_name_the_domain_cannot_load() -> None:
    """Fail at parameter resolution, not after hours of building views."""

    with pytest.raises(ValueError, match="unknown macro_datasets"):
        _prebuild_identity(["--macro-datasets", "cn_gdp,not_a_dataset"])
    with pytest.raises(ValueError, match="unknown macro_datasets"):
        # A macro-domain name is still wrong for the events domain.
        _prebuild_identity(["--macro-datasets", "stk_surv"])


def test_a_seed_built_for_another_selection_is_refused_by_name(tmp_path: Path) -> None:
    """The create-time half of the contract check.

    Its whole point is that a mismatch is loud: an experiment silently
    cold-building every view is hours of runtime that looks like slowness
    rather than a wrong parameter."""

    from autotrade.pipelines.pit_views_seed import assert_seed_snapshot_config

    seed = tmp_path / "seed"
    seed.mkdir()
    wanted = SnapshotConfig(macro_datasets=("cn_gdp", "fut_daily"))
    (seed / "provider.json").write_text(
        json.dumps(_record(tmp_path / "raw")), encoding="utf-8"
    )
    with pytest.raises(ValueError) as excinfo:
        assert_seed_snapshot_config(seed, wanted)
    message = str(excinfo.value)
    # Both identities, so the operator can see which one to rebuild.
    assert "fut_daily" in message and "margin" in message

    (seed / "provider.json").write_text(
        json.dumps(
            pit_cache_provider_record(
                generation_id="other_generation",
                release_raw_dir=tmp_path / "elsewhere",
                snapshot_config=wanted,
            )
        ),
        encoding="utf-8",
    )
    # Generation and release path are not judged here: they name the release
    # an experiment on this seed pins, handed back for the caller to check.
    assert assert_seed_snapshot_config(seed, wanted) == (
        "other_generation",
        str(tmp_path / "elsewhere"),
    )


def test_a_seed_from_an_older_cache_format_is_refused_at_create_time(tmp_path: Path) -> None:
    """A seed built under an older on-disk contract holds views this code would
    never link (seed_pit_views refuses them at run time); naming it must fail
    the create instead of cold-building every view after a misleading accept."""

    from autotrade.pipelines.pit_views_seed import assert_seed_snapshot_config

    seed = tmp_path / "seed"
    seed.mkdir()
    wanted = SnapshotConfig(macro_datasets=("cn_gdp", "fut_daily"))
    stale = pit_cache_provider_record(
        generation_id="generation_test", release_raw_dir=tmp_path / "raw", snapshot_config=wanted
    ) | {"schema_version": SNAPSHOT_CACHE_FORMAT_VERSION - 1}
    (seed / "provider.json").write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="cache format") as excinfo:
        assert_seed_snapshot_config(seed, wanted)
    message = str(excinfo.value)
    assert str(SNAPSHOT_CACHE_FORMAT_VERSION - 1) in message and str(SNAPSHOT_CACHE_FORMAT_VERSION) in message
    assert seed_pit_views(tmp_path / "exp" / "pit_views", seed, expected_provider=stale | {
        "schema_version": SNAPSHOT_CACHE_FORMAT_VERSION
    }) is False


def test_a_seed_without_a_contract_is_refused(tmp_path: Path) -> None:
    from autotrade.pipelines.pit_views_seed import assert_seed_snapshot_config

    seed = tmp_path / "seed"
    seed.mkdir()
    with pytest.raises(ValueError, match="missing provider.json"):
        assert_seed_snapshot_config(seed, SnapshotConfig())


def test_a_seed_whose_build_is_still_staging_a_slot_is_refused(tmp_path: Path) -> None:
    """A create must not accept a tree a prebuild is still filling.

    ``provider.json`` is written when the build binds its cache root, so the
    contract matches from the first minute of a three-hour build: the round
    script's dry-run passed against a seed that had one of its fourteen
    regions. The provider's own staging directory is the evidence, and it
    outlives a killed build too.
    """

    from autotrade.pipelines.pit_views_seed import assert_seed_snapshot_config

    seed = tmp_path / "seed"
    seed.mkdir()
    wanted = SnapshotConfig()
    (seed / "provider.json").write_text(
        json.dumps(
            pit_cache_provider_record(
                generation_id="generation_test",
                release_raw_dir=tmp_path / "raw",
                snapshot_config=wanted,
            )
        ),
        encoding="utf-8",
    )
    finished = seed / "decision" / SEED_DECISION_KEY
    finished.mkdir(parents=True)
    (finished / "manifest.json").write_text('{"kind": "decision"}', encoding="utf-8")
    finished.with_suffix(".lock").touch()
    # A finished tree is accepted; the same tree with the slot the provider is
    # writing right now is not.
    assert_seed_snapshot_config(seed, wanted)

    staging = seed / "decision" / f".{SEED_SLOT}.{uuid.uuid4().hex}.tmp"
    (staging / "daily").mkdir(parents=True)
    with pytest.raises(ValueError) as excinfo:
        assert_seed_snapshot_config(seed, wanted)
    message = str(excinfo.value)
    assert str(seed) in message
    assert staging.name in message
    assert "unfinished build" in message

    # A staged view deeper in the layout is the same evidence.
    shutil.rmtree(staging)
    assert_seed_snapshot_config(seed, wanted)
    deep = seed / "replay" / "paper" / f".{SEED_SLOT}.{uuid.uuid4().hex}.tmp"
    deep.mkdir(parents=True)
    with pytest.raises(ValueError, match="unfinished build"):
        assert_seed_snapshot_config(seed, wanted)
