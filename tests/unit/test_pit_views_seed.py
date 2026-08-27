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


def _write_seed(seed: Path, record: dict[str, object]) -> Path:
    decision = seed / "decision" / "20240101T235959+0800"
    replay = seed / "replay" / "20240102_20240103_20240101T235959+0800"
    bundles = seed / "bundles" / "valid" / "20240102_20240103_20240101T235959+0800"
    stash = seed / "asof_stash" / "should_not_copy"
    for directory in (decision, replay, bundles, stash):
        directory.mkdir(parents=True)
    (decision / "daily.parquet").write_bytes(b"decision-bytes")
    (replay / "daily.parquet").write_bytes(b"replay-bytes")
    (bundles / "data_summary.json").write_text("{}", encoding="utf-8")
    (stash / "scratch.bin").write_bytes(b"stash")
    (seed / "provider.json").write_text(json.dumps(record), encoding="utf-8")
    return decision / "daily.parquet"


def test_matching_seed_hardlinks_views_and_skips_asof_stash(tmp_path: Path) -> None:
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
    assert not (dest / "asof_stash").exists()
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


def test_explore_plan_jobs_cover_2021_lookback_and_heldout() -> None:
    trading_days = [
        day.strftime("%Y%m%d")
        for day in pd.date_range("2021-01-04", "2026-06-30", freq="B")
    ]
    jobs = iter_plan_pit_jobs(
        trading_days,
        first_test_period="2022Q1",
        last_test_period="2025Q4",
        heldout_first_period="2026Q1",
        heldout_last_period="2026Q2",
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
    assert len(jobs) == 16 * 3 + 2


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
        phase="valid",
        require_label=False,
    )
