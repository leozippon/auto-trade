from __future__ import annotations

import json
import shutil
import stat
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.executor import docker_available
from autotrade.environment.nl import NLConfig
from autotrade.environment.runtime import chmod_tree
from autotrade.environment.strategy import StrategySchedule
from autotrade.pipelines.config import (
    SNAPSHOT_CACHE_FORMAT_VERSION,
    ArtifactRevision,
    EvaluationRequest,
    SnapshotBundle,
)
from autotrade.pipelines.pit_backend import (
    HistoricalMinuteSource,
    PITDailyEvaluationBackend,
    ResearchPITSnapshotProvider,
    _asof_stash_dir,
    _bind_asof_stash_contract,
    _load_replay_frames,
)


@pytest.mark.skipif(not docker_available(), reason="Docker is unavailable")
def test_real_sandbox_daily_evaluation_reads_parquet_with_default_limits(
    tmp_path: Path,
) -> None:
    snapshot, replay = _pit_slot_paths(
        tmp_path,
        decision="sandbox",
        replay="sandbox",
        generation_id="generation_sandbox",
    )
    revision = tmp_path / "revision"
    revision.mkdir()
    (snapshot / "text_library").mkdir()
    (replay / "text_library").mkdir()
    _write_domains(snapshot, replay)
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": "snap_sandbox",
                "kind": "decision_input",
                "raw_generation": {"generation_id": "generation_sandbox"},
            }
        ),
        encoding="utf-8",
    )
    (replay / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": "replay_sandbox",
                "kind": "replay_slot",
                "label": "valid",
                "period_start": "20240102",
                "period_end": "20240103",
                "available_from": "2024-01-01T23:59:59+08:00",
                "raw_generation": {"generation_id": "generation_sandbox"},
            }
        ),
        encoding="utf-8",
    )
    chmod_tree(snapshot, file_mode=0o444, dir_mode=0o555)
    (revision / "main.py").write_text(
        """import pandas as pd

def generate_orders(context):
    daily = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"])
    if daily.empty:
        raise RuntimeError("PIT daily view is empty")
    return []
""",
        encoding="utf-8",
    )

    result = PITDailyEvaluationBackend(
        tmp_path / "results",
        execution_mode="sandbox",
    ).evaluate(
        EvaluationRequest(
            ArtifactRevision("revision_sandbox", revision),
            SnapshotBundle(
                "snap_sandbox",
                str(snapshot),
                str(replay),
                generation_id="generation_sandbox",
            ),
            "valid",
            "20240102",
            "20240103",
            StrategySchedule("day", "08:30"),
            BrokerProfile(initial_cash=100_000),
        )
    )

    record = json.loads(Path(result.result_ref).read_text(encoding="utf-8"))
    assert record["inference_dates"] == [
        "2024-01-02T08:30:00+08:00",
        "2024-01-03T08:30:00+08:00",
    ]
    assert record["executions"] == []


def test_pit_daily_evaluation_rolls_all_domains_once_without_loading_future_minutes(
    tmp_path: Path,
) -> None:
    snapshot, replay = _pit_slot_paths(
        tmp_path,
        decision="test",
        replay="test",
        generation_id="generation_test",
    )
    (snapshot / "text_library").mkdir()
    (replay / "text_library").mkdir()

    _write_domains(snapshot, replay)
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": "snap_test",
                "kind": "decision_input",
                "raw_generation": {"generation_id": "generation_test"},
            }
        ),
        encoding="utf-8",
    )
    (replay / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": "replay_test",
                "kind": "replay_slot",
                "label": "valid",
                "period_start": "20240102",
                "period_end": "20240103",
                "available_from": "2024-01-01T23:59:59+08:00",
                "raw_generation": {"generation_id": "generation_test"},
            }
        ),
        encoding="utf-8",
    )
    chmod_tree(snapshot, file_mode=0o444, dir_mode=0o555)

    revision = tmp_path / "revision"
    revision.mkdir()
    (revision / "main.py").write_text(
        """import pandas as pd

def generate_orders(context):
    visible = {
        "daily": len(pd.read_parquet(context.asof_dir + "/daily")),
        "minutes": len(pd.read_parquet(context.asof_dir + "/intraday_1min")),
        "auction": len(pd.read_parquet(context.asof_dir + "/auction")),
        "events": len(pd.read_parquet(context.asof_dir + "/events")),
        "macro": len(pd.read_parquet(context.asof_dir + "/macro")),
        "fundamentals": len(pd.read_parquet(context.asof_dir + "/fundamentals")),
        "text": len(pd.read_parquet(context.asof_dir + "/text_index")),
        "universe": len(pd.read_parquet(context.asof_dir + "/universe")),
        "nl": len(context.nl(query="visibletoken", mode="search")["evidence"]),
    }
    return [{
        "symbol": "000001.SZ",
        "action": "buy",
        "quantity": 100,
        "execute_at": "2099-01-01T09:30:00+08:00",
        "visible": visible,
        "asof_version": context.asof_version,
    }]
""",
        encoding="utf-8",
    )
    backend = PITDailyEvaluationBackend(
        tmp_path / "results",
        execution_mode="trusted",
        nl_config=NLConfig(max_calls_per_decision=1, max_total_calls=2),
        max_intraday_row_group_rows=1,
    )
    result = backend.evaluate(
        EvaluationRequest(
            ArtifactRevision("revision_test", revision),
            SnapshotBundle(
                "snap_test",
                str(snapshot),
                str(replay),
                generation_id="generation_test",
            ),
            "valid",
            "20240102",
            "20240103",
            StrategySchedule("day", "09:28"),
            BrokerProfile(initial_cash=100_000),
        )
    )
    record = json.loads(Path(result.result_ref).read_text(encoding="utf-8"))
    style = json.loads(
        (Path(result.result_ref).parent / "style_analysis.json").read_text(encoding="utf-8")
    )
    assert style["schema_version"] == 1 and style["mode"] == "valid"
    assert style["benchmark_regression"]["reason"] == "benchmark_unavailable"
    assert style["style"]["reason"] == "style_columns_unavailable"
    assert len(record["inference_dates"]) == 2
    first, second = record["pending_orders"]
    assert first["visible"] == {
        "daily": 1,
        "minutes": 1,
        "auction": 1,
        "events": 1,
        "macro": 1,
        "fundamentals": 1,
        "text": 1,
        "universe": 1,
        "nl": 1,
    }
    assert second["visible"] == {
        "daily": 2,
        "minutes": 2,
        "auction": 2,
        "events": 2,
        "macro": 2,
        "fundamentals": 2,
        "text": 2,
        "universe": 1,
        "nl": 2,
    }
    assert record["pit"]["refresh_calls"] == 2
    assert record["pit"]["minute_total_rows"] == 2
    assert record["pit"]["minute_row_groups_loaded"] == 1
    assert record["pit"]["minute_rows_loaded"] == 1
    assert record["pit"]["minute_max_loaded_partition_rows"] == 1
    result_dir = Path(result.result_ref).parent
    assert (result_dir / "result.json").is_file()
    assert (result_dir / "style_analysis.json").is_file()
    assert not (result_dir / "asof").exists()
    assert all(
        not stat.S_IMODE(path.stat().st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        for path in (snapshot, *snapshot.rglob("*"))
    )


def test_first_month_inference_can_have_empty_bars_with_long_pit_daily_history(
    tmp_path: Path,
) -> None:
    snapshot, replay = _pit_slot_paths(
        tmp_path,
        decision="history",
        replay="history",
        generation_id="generation_history",
    )
    revision = tmp_path / "revision"
    revision.mkdir()

    history_days = pd.bdate_range(end="2024-01-31", periods=141)
    pd.DataFrame(
        {
            "trade_date": [stamp.strftime("%Y%m%d") for stamp in history_days],
            "ts_code": ["600000.SH"] * len(history_days),
            "open": [10.0] * len(history_days),
            "close": [10.0] * len(history_days),
            "available_at": [
                f"{stamp.strftime('%Y-%m-%d')}T17:30:00+08:00" for stamp in history_days
            ],
        }
    ).to_parquet(snapshot / "daily.parquet", index=False)
    pd.DataFrame(
        {
            "trade_date": ["20240201", "20240202"],
            "ts_code": ["600000.SH", "600000.SH"],
            "open": [10.0, 10.0],
            "close": [10.0, 10.0],
            "available_at": [
                "2024-02-01T17:30:00+08:00",
                "2024-02-02T17:30:00+08:00",
            ],
        }
    ).to_parquet(replay / "daily.parquet", index=False)
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": "snap_history",
                "kind": "decision_input",
                "raw_generation": {"generation_id": "generation_history"},
            }
        ),
        encoding="utf-8",
    )
    (replay / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": "replay_history",
                "kind": "replay_slot",
                "label": "valid",
                "period_start": "20240201",
                "period_end": "20240202",
                "available_from": "2024-01-31T23:59:59+08:00",
                "raw_generation": {"generation_id": "generation_history"},
            }
        ),
        encoding="utf-8",
    )
    chmod_tree(snapshot, file_mode=0o444, dir_mode=0o555)
    (revision / "main.py").write_text(
        """import pandas as pd

def generate_orders(context):
    daily = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=["trade_date", "ts_code", "close"],
    )
    if context.bars:
        raise RuntimeError("first interval inference unexpectedly had visible bars")
    if len(daily) < 141:
        raise RuntimeError("PIT daily history is incomplete")
    return []
""",
        encoding="utf-8",
    )

    result = PITDailyEvaluationBackend(
        tmp_path / "results",
        execution_mode="trusted",
    ).evaluate(
        EvaluationRequest(
            ArtifactRevision("revision_history", revision),
            SnapshotBundle(
                "snap_history",
                str(snapshot),
                str(replay),
                generation_id="generation_history",
            ),
            "valid",
            "20240201",
            "20240202",
            StrategySchedule("month", "08:30"),
            BrokerProfile(initial_cash=100_000),
        )
    )

    record = json.loads(Path(result.result_ref).read_text(encoding="utf-8"))
    assert record["inference_dates"] == ["2024-02-01T08:30:00+08:00"]
    assert record["executions"] == []
    assert record["pending_orders"] == []


def test_research_pit_provider_reuses_completed_semantic_views(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    for dataset in ("daily", "daily_basic", "adj_factor", "stk_limit", "suspend_d"):
        target = raw / dataset / "trade_date=20240102.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"trade_date": ["20240102"], "ts_code": ["000001.SZ"]}).to_parquet(
            target,
            index=False,
        )
    events = tmp_path / "data" / "pit" / "fundamental_events"
    events.mkdir(parents=True)
    status = tmp_path / "results" / "data_quality" / "fundamental_events_status.json"
    status.parent.mkdir(parents=True)
    status.write_text("{}", encoding="utf-8")
    provider = ResearchPITSnapshotProvider(
        experiment_dir=tmp_path / "experiment",
        raw_dir=raw,
        fundamental_events_root=events,
        fundamental_events_status=status,
        config=SnapshotConfig(
            include_intraday=False,
            events_datasets=(),
            macro_datasets=(),
            text_datasets=(),
            fundamental_datasets=(),
            replay_include_events=False,
            replay_include_text=False,
            replay_include_minutes=False,
            replay_include_macro=False,
            replay_include_fundamentals=False,
        ),
    )

    class FakeBuilder:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def build_decision_snapshot(self, decision, output, config, **_kwargs):
            del config
            self.calls.append("decision")
            output = Path(output)
            output.mkdir(parents=True)
            pd.DataFrame(
                {
                    "trade_date": ["20240101"],
                    "ts_code": ["000001.SZ"],
                    "open": [10.0],
                    "close": [10.0],
                    "available_at": ["2024-01-01T17:30:00+08:00"],
                }
            ).to_parquet(output / "daily.parquet", index=False)
            manifest = {
                "snapshot_id": "snap_stable",
                "kind": "decision_input",
                "decision_time": decision.isoformat(),
                "domains": {"daily": {"rows": 1}},
            }
            (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            return manifest

        def build_replay_slot(self, start, end, output, *, label, config, available_from):
            del config
            self.calls.append("replay")
            output = Path(output)
            output.mkdir(parents=True)
            pd.DataFrame({"trade_date": [start]}).to_parquet(output / "daily.parquet", index=False)
            manifest = {
                "snapshot_id": "replay_stable",
                "kind": "replay_slot",
                "label": label,
                "period_start": start,
                "period_end": end,
                "available_from": available_from.isoformat(),
            }
            (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            return manifest

    fake = FakeBuilder()
    provider.builder = fake  # type: ignore[assignment]
    decision = datetime.fromisoformat("2024-01-01T23:59:59+08:00")
    first = provider.prepare(
        fold=None,
        phase="valid",
        start="20240102",
        end="20240103",
        decision_time=decision,
    )
    second = provider.prepare(
        fold=None,
        phase="valid",
        start="20240102",
        end="20240103",
        decision_time=decision,
    )
    assert first == second
    frozen = provider.prepare(
        fold=None,
        phase="frozen_test",
        start="20240102",
        end="20240103",
        decision_time=decision,
    )
    heldout = provider.prepare(
        fold=None,
        phase="heldout",
        start="20240102",
        end="20240103",
        decision_time=decision,
    )
    assert len({first.replay_ref, frozen.replay_ref, heldout.replay_ref}) == 3
    assert first.decision_ref == frozen.decision_ref == heldout.decision_ref
    assert json.loads(Path(first.replay_ref, "manifest.json").read_text(encoding="utf-8"))[
        "label"
    ] == "valid"
    assert json.loads(Path(frozen.replay_ref, "manifest.json").read_text(encoding="utf-8"))[
        "label"
    ] == "frozen_test"
    assert json.loads(Path(heldout.replay_ref, "manifest.json").read_text(encoding="utf-8"))[
        "label"
    ] == "heldout"
    assert fake.calls == ["decision", "replay", "replay", "replay"]


def test_evaluation_rejects_replay_from_another_phase(tmp_path: Path) -> None:
    snapshot, replay = _pit_slot_paths(
        tmp_path,
        decision="phase",
        replay="phase",
        generation_id="generation_phase",
        phase="valid",
    )
    decision_manifest, replay_manifest = _stash_manifests(
        "generation_phase", phase="valid"
    )
    (snapshot / "manifest.json").write_text(
        json.dumps(decision_manifest), encoding="utf-8"
    )
    (replay / "manifest.json").write_text(
        json.dumps(replay_manifest), encoding="utf-8"
    )
    request = EvaluationRequest(
        ArtifactRevision("revision_phase", tmp_path / "revision"),
        SnapshotBundle(
            "snapshot_one",
            str(snapshot),
            str(replay),
            generation_id="generation_phase",
        ),
        "heldout",
        "20240102",
        "20240103",
        StrategySchedule("day", "08:30"),
        BrokerProfile(initial_cash=100_000),
    )
    with pytest.raises(ValueError, match="mode does not match"):
        PITDailyEvaluationBackend._validate_bundle(request, snapshot, replay)
    with pytest.raises(RuntimeError, match="phase"):
        _bind_asof_stash_contract(
            snapshot_dir=snapshot,
            replay_dir=replay,
            schedule=request.schedule,
            phase=request.mode,
            generation_id="generation_phase",
            decision_manifest=decision_manifest,
            replay_manifest=replay_manifest,
        )


def test_historical_minutes_resolve_only_the_exact_pit_price(tmp_path: Path) -> None:
    path = tmp_path / "intraday_1min.parquet"
    frame = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240102"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_time": [
                "2024-01-02T10:00:00+08:00",
                "2024-01-02T10:01:00+08:00",
            ],
            "close": [10.25, 10.5],
            "available_at": [
                "2024-01-02T10:00:00+08:00",
                "2024-01-02T10:01:00+08:00",
            ],
        }
    )
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, row_group_size=2)
    source = HistoricalMinuteSource(path, max_row_group_rows=2)

    assert source.price_at(
        "000001.SZ", datetime.fromisoformat("2024-01-02T10:00:00+08:00")
    ) == 10.25
    assert source.price_at(
        "000001.SZ", datetime.fromisoformat("2024-01-02T10:00:30+08:00")
    ) is None
    assert source.price_at(
        "000001.SZ", datetime.fromisoformat("2024-01-02T10:02:00+08:00")
    ) is None


def test_load_replay_frames_reuses_cached_parquets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay = tmp_path / "replay"
    replay.mkdir()
    pd.DataFrame({"trade_date": ["20240102"], "close": [1.0]}).to_parquet(
        replay / "daily.parquet", index=False
    )
    pd.DataFrame({"trade_date": ["20240102"]}).to_parquet(
        replay / "events.parquet", index=False
    )
    manifest = {
        "snapshot_id": "replay_one",
        "kind": "replay_slot",
        "raw_generation": {"generation_id": "generation_one"},
    }
    cache = {}
    calls = {"n": 0}
    real_read = pd.read_parquet

    def counting_read_parquet(path, *args, **kwargs):
        calls["n"] += 1
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", counting_read_parquet)
    first = _load_replay_frames(
        replay,
        generation_id="generation_one",
        replay_manifest=manifest,
        cache=cache,
    )
    first_reads = calls["n"]
    assert first_reads >= 1
    first["daily"].loc[:, "close"] = 99.0
    second = _load_replay_frames(
        replay,
        generation_id="generation_one",
        replay_manifest=manifest,
        cache=cache,
    )
    assert calls["n"] == first_reads
    assert second["daily"] is not first["daily"]
    assert list(second["daily"]["close"]) == [1.0]
    assert second["events"] is first["events"]


def test_replay_frame_cache_reloads_rebuilt_path_for_new_generation(tmp_path: Path) -> None:
    replay = tmp_path / "replay"
    replay.mkdir()
    pd.DataFrame({"trade_date": ["20240102"], "close": [1.0]}).to_parquet(
        replay / "daily.parquet", index=False
    )
    cache = {}
    first = _load_replay_frames(
        replay,
        generation_id="generation_one",
        replay_manifest={"raw_generation": {"generation_id": "generation_one"}},
        cache=cache,
    )
    assert first["daily"]["close"].tolist() == [1.0]

    shutil.rmtree(replay)
    replay.mkdir()
    pd.DataFrame({"trade_date": ["20240102"], "close": [2.0]}).to_parquet(
        replay / "daily.parquet", index=False
    )
    second = _load_replay_frames(
        replay,
        generation_id="generation_two",
        replay_manifest={"raw_generation": {"generation_id": "generation_two"}},
        cache=cache,
    )
    assert second["daily"]["close"].tolist() == [2.0]


def test_asof_stash_uses_complete_schedule_hierarchy(tmp_path: Path) -> None:
    snapshot, replay = _pit_slot_paths(
        tmp_path,
        decision="20240101T235959+0800",
        replay="20240102_20240103_20240101T235959+0800",
        generation_id="generation_one",
    )
    target = _asof_stash_dir(
        snapshot, replay, StrategySchedule("day", "08:30"), "valid"
    )
    assert target == (
        tmp_path
        / "pit_views"
        / "asof_stash"
        / "decision"
        / snapshot.name
        / "replay"
        / replay.name
        / "phase"
        / "valid"
        / "schedule"
        / "period=day"
        / "inference_time"
        / "hour=08"
        / "minute=30"
    )


def test_different_schedules_never_share_a_stash(tmp_path: Path) -> None:
    snapshot, replay = _pit_slot_paths(
        tmp_path,
        decision="decision_one",
        replay="replay_one",
        generation_id="generation_one",
    )
    decision_manifest, replay_manifest = _stash_manifests("generation_one")
    daily = _bind_asof_stash_contract(
        snapshot_dir=snapshot,
        replay_dir=replay,
        schedule=StrategySchedule("day", "08:30"),
        phase="valid",
        generation_id="generation_one",
        decision_manifest=decision_manifest,
        replay_manifest=replay_manifest,
    )
    monthly = _bind_asof_stash_contract(
        snapshot_dir=snapshot,
        replay_dir=replay,
        schedule=StrategySchedule("month", "08:30"),
        phase="valid",
        generation_id="generation_one",
        decision_manifest=decision_manifest,
        replay_manifest=replay_manifest,
    )
    assert daily != monthly
    assert json.loads((daily / "contract.json").read_text(encoding="utf-8"))["schedule"] == {
        "period": "day",
        "inference_time": "08:30",
    }
    assert json.loads((monthly / "contract.json").read_text(encoding="utf-8"))["schedule"] == {
        "period": "month",
        "inference_time": "08:30",
    }


def test_stash_contract_refuses_changed_config_or_generation(tmp_path: Path) -> None:
    snapshot, replay = _pit_slot_paths(
        tmp_path,
        decision="decision_one",
        replay="replay_one",
        generation_id="generation_one",
    )
    decision_manifest, replay_manifest = _stash_manifests("generation_one")
    schedule = StrategySchedule("day", "08:30")
    _bind_asof_stash_contract(
        snapshot_dir=snapshot,
        replay_dir=replay,
        schedule=schedule,
        phase="valid",
        generation_id="generation_one",
        decision_manifest=decision_manifest,
        replay_manifest=replay_manifest,
    )

    changed_config = SnapshotConfig(text_body_chars=123).to_record()
    _write_provider_contract(
        snapshot.parent.parent,
        generation_id="generation_one",
        config=changed_config,
    )
    with pytest.raises(RuntimeError, match="conflicts with requested semantics"):
        _bind_asof_stash_contract(
            snapshot_dir=snapshot,
            replay_dir=replay,
            schedule=schedule,
            phase="valid",
            generation_id="generation_one",
            decision_manifest=decision_manifest,
            replay_manifest=replay_manifest,
        )

    _write_provider_contract(snapshot.parent.parent, generation_id="generation_two")
    changed_decision, changed_replay = _stash_manifests("generation_two")
    with pytest.raises(RuntimeError, match="conflicts with requested semantics"):
        _bind_asof_stash_contract(
            snapshot_dir=snapshot,
            replay_dir=replay,
            schedule=schedule,
            phase="valid",
            generation_id="generation_two",
            decision_manifest=changed_decision,
            replay_manifest=changed_replay,
        )


def test_stash_contract_corruption_or_mismatch_fails(tmp_path: Path) -> None:
    snapshot, replay = _pit_slot_paths(
        tmp_path,
        decision="decision_one",
        replay="replay_one",
        generation_id="generation_one",
    )
    decision_manifest, replay_manifest = _stash_manifests("generation_one")
    kwargs = {
        "snapshot_dir": snapshot,
        "replay_dir": replay,
        "schedule": StrategySchedule("day", "08:30"),
        "phase": "valid",
        "generation_id": "generation_one",
        "decision_manifest": decision_manifest,
        "replay_manifest": replay_manifest,
    }
    stash = _bind_asof_stash_contract(**kwargs)
    contract = stash / "contract.json"
    contract.write_text("{broken", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid PIT cache record"):
        _bind_asof_stash_contract(**kwargs)

    contract.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="conflicts with requested semantics"):
        _bind_asof_stash_contract(**kwargs)


def test_asof_stash_rejects_slots_outside_one_cache_root(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="decision/replay slots"):
        _asof_stash_dir(
            tmp_path / "one" / "decision" / "slot",
            tmp_path / "two" / "replay" / "valid" / "slot",
            StrategySchedule(),
            "valid",
        )


def _pit_slot_paths(
    root: Path,
    *,
    decision: str,
    replay: str,
    generation_id: str,
    phase: str = "valid",
) -> tuple[Path, Path]:
    cache_root = root / "pit_views"
    snapshot = cache_root / "decision" / decision
    replay_slot = cache_root / "replay" / phase / replay
    snapshot.mkdir(parents=True)
    replay_slot.mkdir(parents=True)
    _write_provider_contract(cache_root, generation_id=generation_id)
    return snapshot, replay_slot


def _write_provider_contract(
    cache_root: Path,
    *,
    generation_id: str,
    config: dict[str, object] | None = None,
) -> None:
    (cache_root / "provider.json").write_text(
        json.dumps(
            {
                "schema_version": SNAPSHOT_CACHE_FORMAT_VERSION,
                "generation_id": generation_id,
                "release_raw_dir": str(cache_root / "release" / generation_id / "raw"),
                "snapshot_config": config or SnapshotConfig().to_record(),
            }
        ),
        encoding="utf-8",
    )


def _stash_manifests(
    generation_id: str, *, phase: str = "valid"
) -> tuple[dict[str, object], dict[str, object]]:
    raw_generation = {"generation_id": generation_id}
    return (
        {
            "snapshot_id": "snapshot_one",
            "kind": "decision_input",
            "decision_time": "2024-01-01T23:59:59+08:00",
            "raw_generation": raw_generation,
        },
        {
            "snapshot_id": "replay_one",
            "kind": "replay_slot",
            "label": phase,
            "period_start": "20240102",
            "period_end": "20240103",
            "available_from": "2024-01-01T23:59:59+08:00",
            "raw_generation": raw_generation,
        },
    )


def _write_domains(snapshot: Path, replay: Path) -> None:
    pd.DataFrame(
        {
            "trade_date": ["20240101"],
            "ts_code": ["000001.SZ"],
            "open": [10.0],
            "close": [10.0],
            "available_at": ["2024-01-01T17:30:00+08:00"],
        }
    ).to_parquet(snapshot / "daily.parquet", index=False)
    pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.0],
            "close": [10.0, 10.0],
            "available_at": ["2024-01-02T17:30:00+08:00", "2024-01-03T17:30:00+08:00"],
        }
    ).to_parquet(replay / "daily.parquet", index=False)
    minute_columns = {
        "trade_date": ["20240101"],
        "ts_code": ["000001.SZ"],
        "trade_time": ["2024-01-01T15:00:00+08:00"],
        "close": [10.0],
        "available_at": ["2024-01-01T15:00:00+08:00"],
    }
    pd.DataFrame(minute_columns).to_parquet(snapshot / "intraday_1min.parquet", index=False)
    minute_replay = pd.DataFrame(
        {
            key: [
                value[0].replace("2024-01-01", "2024-01-02").replace("20240101", "20240102"),
                value[0].replace("2024-01-01", "2024-01-03").replace("20240101", "20240103"),
            ]
            if isinstance(value[0], str)
            else [value[0], value[0]]
            for key, value in minute_columns.items()
        }
    )
    pq.write_table(pa.Table.from_pandas(minute_replay, preserve_index=False), replay / "intraday_1min.parquet", row_group_size=1)

    _write_simple_domain(snapshot, replay, "auction", dataset=None, time="09:29:00")
    _write_simple_domain(snapshot, replay, "events", dataset="moneyflow", time="10:00:00")
    _write_simple_domain(snapshot, replay, "macro", dataset="cn_cpi", time="10:00:00")
    _write_simple_domain(snapshot, replay, "fundamentals", dataset="income_vip", time="10:00:00")

    snapshot_index = pd.DataFrame(
        {
            "dataset": ["news"],
            "text_id": ["old"],
            "title": ["visibletoken old"],
            "ts_codes": ["000001.SZ"],
            "library_file": ["news.parquet"],
            "available_at": ["2024-01-01T10:00:00+08:00"],
        }
    )
    replay_index = pd.DataFrame(
        {
            "dataset": ["news", "news"],
            "text_id": ["day1", "future"],
            "title": ["visibletoken day1", "visibletoken future"],
            "ts_codes": ["000001.SZ", "000001.SZ"],
            "library_file": ["news.parquet", "news.parquet"],
            "available_at": ["2024-01-02T10:00:00+08:00", "2024-01-03T10:00:00+08:00"],
        }
    )
    snapshot_index.to_parquet(snapshot / "text_index.parquet", index=False)
    replay_index.to_parquet(replay / "text_index.parquet", index=False)
    pd.DataFrame({"text_id": ["old"], "body": ["visibletoken old body"]}).to_parquet(
        snapshot / "text_library" / "news.parquet",
        index=False,
    )
    pd.DataFrame(
        {"text_id": ["day1", "future"], "body": ["visibletoken day1 body", "visibletoken future body"]}
    ).to_parquet(replay / "text_library" / "news.parquet", index=False)
    pd.DataFrame({"ts_code": ["000001.SZ"]}).to_parquet(snapshot / "universe.parquet", index=False)


def _write_simple_domain(
    snapshot: Path,
    replay: Path,
    name: str,
    *,
    dataset: str | None,
    time: str,
) -> None:
    base = {
        "trade_date": ["20240101"],
        "ts_code": ["000001.SZ"],
        "value": [1.0],
        "available_at": [f"2024-01-01T{time}+08:00"],
    }
    if dataset is not None:
        base["dataset"] = [dataset]
    pd.DataFrame(base).to_parquet(snapshot / f"{name}.parquet", index=False)
    current = {key: [value[0], value[0]] for key, value in base.items()}
    current["trade_date"] = ["20240102", "20240103"]
    current["available_at"] = [f"2024-01-02T{time}+08:00", f"2024-01-03T{time}+08:00"]
    pd.DataFrame(current).to_parquet(replay / f"{name}.parquet", index=False)
