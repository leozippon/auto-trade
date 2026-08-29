from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

import pandas as pd

from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.pipelines.config import SNAPSHOT_CACHE_FORMAT_VERSION
from autotrade.pipelines.pit_views_seed import (
    iter_plan_pit_jobs,
    pit_cache_provider_record,
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


SEED_STASH_LEAF = (
    "asof_stash/decision/20240101T235959+0800/replay/"
    "20240102_20240103_20240101T235959+0800/schedule/period=day/"
    "inference_time/hour=08/minute=30"
)


def _write_seed(seed: Path, record: dict[str, object]) -> Path:
    decision = seed / "decision" / "20240101T235959+0800"
    replay = seed / "replay" / "20240102_20240103_20240101T235959+0800"
    bundles = seed / "bundles" / "valid" / "20240102_20240103_20240101T235959+0800"
    stash = seed / SEED_STASH_LEAF
    partial = seed / "asof_stash" / "no_contract_yet"
    for directory in (decision, replay, bundles, stash / "daily", partial):
        directory.mkdir(parents=True)
    (decision / "daily.parquet").write_bytes(b"decision-bytes")
    (replay / "daily.parquet").write_bytes(b"replay-bytes")
    (bundles / "data_summary.json").write_text("{}", encoding="utf-8")
    (stash / "contract.json").write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    (stash / "daily" / "part_0001.parquet").write_bytes(b"asof-part")
    (partial / "scratch.bin").write_bytes(b"stash")
    (seed / "provider.json").write_text(json.dumps(record), encoding="utf-8")
    return decision / "daily.parquet"


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
    assert (
        dest / "bundles" / "valid" / "20240102_20240103_20240101T235959+0800" / "data_summary.json"
    ).is_file()
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


def _business_days() -> list[str]:
    return [
        day.strftime("%Y%m%d")
        for day in pd.date_range("2021-01-04", "2026-06-30", freq="B")
    ]


def test_rolling_test_stage_plan_covers_every_region_and_heldout() -> None:
    jobs = iter_plan_pit_jobs(
        _business_days(),
        development_first_period="2021Q4",
        development_last_period="2025Q4",
        heldout_first_period="2026Q1",
        heldout_last_period="2026Q2",
        fold_period="quarter",
        window_months=21,
        min_region_trade_days=2,
        test_stage=True,
    )
    phases = {phase for phase, _start, _end, _decision in jobs}
    assert phases == {"meta", "valid", "frozen_test", "heldout"}
    assert any(
        phase == "valid" and start == "20211001" and end == "20211231"
        for phase, start, end, _decision in jobs
    )
    assert any(
        phase == "heldout" and start == "20260401"
        for phase, start, _end, _decision in jobs
    )
    # 17 quarter labels -> 16 rolling folds, three phases each, plus 2 held-out.
    assert len(jobs) == 16 * 3 + 2
    # Consecutive folds share a region: one fold's test is the next one's
    # validation, so the plan asks for far fewer regions than jobs.
    regions = {(start, end, decision) for _phase, start, end, decision in jobs}
    assert len(regions) == 17 + 2


def test_default_plan_is_the_console_calendar_and_shares_regions() -> None:
    """The seed plan follows whatever calendar the console creates today.

    Calendar-independent invariants only: what must hold for the prebuild to
    match a new experiment and for the provider to reuse regions across phases.
    """

    from autotrade.pipelines.pit_views_seed import plan_parameters

    plan = plan_parameters()
    jobs = iter_plan_pit_jobs(_business_days(), **plan)
    assert jobs
    assert {phase for phase, *_rest in jobs} <= {"meta", "valid", "frozen_test", "heldout"}
    # Decision-time order is what lets each decision snapshot extend the
    # previous one's events instead of rebuilding the whole window.
    decisions = [decision for *_rest, decision in jobs]
    assert decisions == sorted(decisions)
    # Meta and Validation always name the same region, so the plan always has
    # strictly fewer distinct regions to build than jobs to prepare.
    regions = {(start, end, decision) for _phase, start, end, decision in jobs}
    assert len(regions) < len(jobs)
    meta = {(start, end, d) for phase, start, end, d in jobs if phase == "meta"}
    valid = {(start, end, d) for phase, start, end, d in jobs if phase == "valid"}
    assert meta == valid
    heldout = [job for job in jobs if job[0] == "heldout"]
    assert len(heldout) == 1
    assert heldout[0][1], heldout[0][2]


def test_single_window_development_fold_plans_no_frozen_test() -> None:
    """One development Fold over the whole window, judged by Held-out alone.

    Two regions and two decision anchors: the four-year validation region that
    Meta and Validation share, and the explicit held-out range.
    """

    jobs = iter_plan_pit_jobs(
        _business_days(),
        development_first_period="2022",
        development_last_period="2025",
        heldout_first_period="20260101..20260630",
        heldout_last_period="20260101..20260630",
        fold_period="year",
        window_months=24,
        min_region_trade_days=2,
        test_stage=False,
    )
    assert [phase for phase, *_rest in jobs] == ["meta", "valid", "heldout"]
    assert jobs[0][1:3] == ("20220101", "20251231")
    assert jobs[1][1:3] == ("20220101", "20251231")
    assert jobs[2][1:3] == ("20260101", "20260630")
    assert jobs[0][3].isoformat() == "2021-12-31T23:59:59+08:00"
    assert jobs[2][3].isoformat() == "2025-12-31T23:59:59+08:00"
    assert len({(start, end, decision) for _p, start, end, decision in jobs}) == 2


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
        "label": "frozen_test",
    }
    assert _replay_manifest_matches(
        manifest,
        start="20220101",
        end="20220331",
        decision=decision,
        phase="frozen_test",
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
