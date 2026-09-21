from __future__ import annotations

import errno
import io
import json
import os
import stat
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.executor import docker_available, raised_by_strategy
from autotrade.environment.nl import NLConfig
from autotrade.environment.replay import timeview as timeview_module
from autotrade.environment.replay.engine import BacktestError
from autotrade.environment.replay.null_control import NullControlSetupError
from autotrade.environment.replay.timeview import Timeview
from autotrade.environment.runtime import (
    AGENT_VISIBLE_BACKTEST_SUMMARY_KEYS,
    HOST_PATH_RE,
    _agent_visible_backtest_summary,
    chmod_tree,
)
from autotrade.environment.strategy import StrategySchedule
from autotrade.pipelines import pit_backend
from autotrade.pipelines.config import (
    SNAPSHOT_CACHE_FORMAT_VERSION,
    ArtifactRevision,
    EvaluationRequest,
    SnapshotBundle,
)
from autotrade.pipelines.pit_backend import (
    REPLAY_SLOTS_SIDECAR,
    REPLAY_SOURCE_LABEL,
    HistoricalMinuteSource,
    PITDailyEvaluationBackend,
    ResearchPITSnapshotProvider,
    _asof_stash_dir,
    _AsOfReadOnlyView,
    _bind_asof_stash_contract,
    prebuild_asof_stash,
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
                "domains": {"corporate_actions": {"rows": 0}},
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
                "domains": {"corporate_actions": {"rows": 0}},
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
            "pre_close": [10.0] * len(history_days),
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
            "pre_close": [10.0, 10.0],
            "available_at": [
                "2024-02-01T17:30:00+08:00",
                "2024-02-02T17:30:00+08:00",
            ],
        }
    ).to_parquet(replay / "daily.parquet", index=False)
    _write_corporate_actions(replay)
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
                "domains": {"corporate_actions": {"rows": 0}},
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


def test_evaluation_summary_carries_the_whole_agent_visible_field_set(
    tmp_path: Path,
) -> None:
    """A whitelisted summary key must actually arrive.

    The Agent-visible projection is a fixed allowlist; a key nothing populates
    advertises telemetry that never shows up in a run manifest. This pins the
    timing and NL cost block the whitelist promises, on a real two-day replay.
    """
    snapshot, replay = _pit_slot_paths(
        tmp_path,
        decision="timing",
        replay="timing",
        generation_id="generation_timing",
    )
    (snapshot / "text_library").mkdir()
    (replay / "text_library").mkdir()
    _write_domains(snapshot, replay)
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": "snap_timing",
                "kind": "decision_input",
                "raw_generation": {"generation_id": "generation_timing"},
            }
        ),
        encoding="utf-8",
    )
    (replay / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": "replay_timing",
                "kind": "replay_slot",
                "label": "valid",
                "period_start": "20240102",
                "period_end": "20240103",
                "available_from": "2024-01-01T23:59:59+08:00",
                "raw_generation": {"generation_id": "generation_timing"},
                "domains": {"corporate_actions": {"rows": 0}},
            }
        ),
        encoding="utf-8",
    )
    chmod_tree(snapshot, file_mode=0o444, dir_mode=0o555)

    revision = tmp_path / "revision"
    revision.mkdir()
    (revision / "main.py").write_text(
        """def generate_orders(context):
    context.nl(query="visibletoken", mode="search")
    return [{
        "symbol": "000001.SZ",
        "action": "buy",
        "quantity": 100,
        "execute_at": context.inference_at.replace(hour=15, minute=0).isoformat(),
    }]
""",
        encoding="utf-8",
    )
    result = PITDailyEvaluationBackend(
        tmp_path / "results",
        execution_mode="trusted",
        nl_config=NLConfig(),
    ).evaluate(
        EvaluationRequest(
            ArtifactRevision("revision_timing", revision),
            SnapshotBundle(
                "snap_timing",
                str(snapshot),
                str(replay),
                generation_id="generation_timing",
            ),
            "valid",
            "20240102",
            "20240103",
            StrategySchedule("day", "09:28"),
            BrokerProfile(initial_cash=100_000),
        )
    )

    summary = result.summary
    # result_name/mode/span/status/complete_validation/error belong to the
    # session tool layer, which adds them when it appends the manifest entry;
    # benchmark depends on the slot carrying index rows, which this one
    # deliberately does not (see test_style_analysis for the producer/report
    # round trip); resources is the strategy container's own telemetry and this
    # replay runs the strategy in-process, with no container to measure.
    conditional = {
        "result_name",
        "mode",
        "span",
        "status",
        "complete_validation",
        "error",
        "benchmark",
        "resources",
    }
    expected = set(AGENT_VISIBLE_BACKTEST_SUMMARY_KEYS) - conditional
    assert expected <= set(summary), sorted(expected - set(summary))
    assert "benchmark" not in summary
    assert "resources" not in summary

    assert summary["replayed_trade_days"] == 2
    assert summary["decision_calls"] == 2
    assert summary["started_at"] < summary["finished_at"]
    assert 0.0 < float(summary["replay_wall_seconds"])
    phases = summary["phase_seconds"]
    # Backend setup and replay-loop phases share one breakdown, and the loop
    # phases cannot exceed the loop they were measured inside.
    assert {"replay_frames", "timeview_init", "style_analysis"} <= set(phases)
    assert {"market_build", "data_view", "strategy", "broker", "nl"} <= set(phases)
    loop = sum(phases[name] for name in ("data_view", "strategy", "broker"))
    assert loop <= float(summary["replay_wall_seconds"]) + 0.05
    assert phases["nl"] <= phases["strategy"] + 0.05

    # No LLM is configured, so the calls are counted but none reached a model.
    assert summary["nl_calls"] == 2
    assert summary["nl_executed_calls"] == 0
    assert summary["nl_llm_calls"] == 0
    assert summary["nl_budget_rejected_calls"] == 0
    # The total ceiling is derived from this replay's own trading days, so the
    # effective budget the summary echoes is the one the window earned.
    assert summary["nl_max_total_calls"] == (
        NLConfig().for_replay(int(summary["replayed_trade_days"])).max_total_calls
    )
    assert summary["nl_wall_seconds"] >= 0.0

    # The same block reaches the persisted result and the Agent-visible view.
    attachment = Path(result.result_ref).read_text(encoding="utf-8")
    record = json.loads(attachment)
    assert record["stats"]["phase_seconds"] == phases
    # This file IS the Step attachment the Agent reads through the mounted
    # `steps` root (local_backend.VALIDATION_RESULT_ATTACHMENT), so the host
    # layout must not appear anywhere in it. The slots are named opaquely.
    assert HOST_PATH_RE.search(attachment) is None, attachment
    assert str(tmp_path) not in attachment
    assert record["pit"]["decision_slot"] == snapshot.name
    assert record["pit"]["replay_slots"] == [replay.name]
    assert not {"decision_ref", "replay_ref"} & set(record["pit"])
    assert set(_agent_visible_backtest_summary(dict(summary))) == expected

    # Every charged NL call is explained by exactly one outcome bucket.
    assert summary["nl_search_calls"] == 2
    assert summary["nl_calls"] == (
        summary["nl_executed_calls"]
        + summary["nl_search_calls"]
        + summary["nl_evidence_gated_calls"]
    )


def test_pit_evaluation_credits_a_cash_dividend_from_the_slot_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The formal replay settles both ex-date legs from the slot.

    ``000001.SZ`` goes ex a 0.5 CNY cash dividend on the second day: the bar's
    ``pre_close`` drops by exactly the dividend, so the share count stays and
    the slot's ``corporate_actions.parquet`` cash is credited. The table is
    Broker truth only and never becomes an as-of domain; a slot whose manifest
    declares it but lacks the file is refused.
    """
    snapshot, replay = _pit_slot_paths(
        tmp_path, decision="dividend", replay="dividend", generation_id="generation_dividend"
    )

    def _daily(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
        frame = []
        for day, pre_close, close in rows:
            stamp = f"{day[:4]}-{day[4:6]}-{day[6:]}T17:30:00+08:00"
            frame.append(
                {
                    "trade_date": day, "ts_code": "000001.SZ", "open": pre_close, "close": close,
                    "pre_close": pre_close, "up_limit": round(pre_close * 1.1, 2),
                    "down_limit": round(pre_close * 0.9, 2), "available_at": stamp,
                }
            )
            # A second, action-free name: the null control needs a replacement.
            frame.append(
                {
                    "trade_date": day, "ts_code": "000002.SZ", "open": 5.0, "close": 5.0,
                    "pre_close": 5.0, "up_limit": 5.5, "down_limit": 4.5, "available_at": stamp,
                }
            )
        return pd.DataFrame(frame)

    _daily([("20240101", 10.0, 10.0)]).to_parquet(snapshot / "daily.parquet", index=False)
    _daily([("20240102", 10.0, 10.0), ("20240103", 9.5, 9.5)]).to_parquet(
        replay / "daily.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ", "ex_date": "20240103", "record_date": "20240102",
                "pay_date": "20240103", "div_listdate": "", "cash_per_share": 0.5,
                "stock_per_share": 0.0,
            }
        ]
    ).to_parquet(replay / "corporate_actions.parquet", index=False)
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": "snap_dividend",
                "kind": "decision_input",
                "raw_generation": {"generation_id": "generation_dividend"},
            }
        ),
        encoding="utf-8",
    )
    replay_manifest = {
        "snapshot_id": "replay_dividend",
        "kind": "replay_slot",
        "label": "valid",
        "period_start": "20240102",
        "period_end": "20240103",
        "available_from": "2024-01-01T23:59:59+08:00",
        "raw_generation": {"generation_id": "generation_dividend"},
        "domains": {"daily": {"rows": 4}, "corporate_actions": {"rows": 1}},
    }
    (replay / "manifest.json").write_text(json.dumps(replay_manifest), encoding="utf-8")
    chmod_tree(snapshot, file_mode=0o444, dir_mode=0o555)
    revision = tmp_path / "revision"
    revision.mkdir()
    (revision / "main.py").write_text(
        """def generate_orders(context):
    if context.account.positions:
        return []
    return [{
        "symbol": "000001.SZ",
        "action": "buy",
        "quantity": 100,
        "execute_at": context.inference_at.replace(hour=15, minute=0).isoformat(),
    }]
""",
        encoding="utf-8",
    )
    request = EvaluationRequest(
        ArtifactRevision("revision_dividend", revision),
        SnapshotBundle(
            "snap_dividend", str(snapshot), str(replay), generation_id="generation_dividend"
        ),
        "valid",
        "20240102",
        "20240103",
        StrategySchedule("day", "09:28"),
        BrokerProfile(initial_cash=100_000),
    )
    backend = PITDailyEvaluationBackend(tmp_path / "results", execution_mode="trusted")
    result = backend.evaluate(request)

    record = json.loads(Path(result.result_ref).read_text(encoding="utf-8"))
    first, second = record["equity_curve"]
    assert first["positions"] == {"000001.SZ": 100}
    assert second["positions"] == {"000001.SZ": 100}
    assert second["cash"] == pytest.approx(first["cash"] + 100 * 0.5)
    [action] = record["corporate_actions"]
    assert (action["trade_date"], action["symbol"]) == ("20240103", "000001.SZ")
    assert (action["quantity_before"], action["quantity_after"]) == (100, 100)
    assert action["cash_per_share"] == 0.5
    assert action["cash_credit"] == pytest.approx(50.0)
    assert "corporate_actions" not in record["pit"]["asof_domains"]

    # A formal replay carries the zero-skill panel the verdict grades it
    # against: twenty draws of its own skeleton, the one other name standing
    # in, matched on the float-cap decile because this slot mounts no
    # index_weight. The smoke window below is not a Validation and has none.
    style = json.loads(
        (Path(result.result_ref).parent / "style_analysis.json").read_text(encoding="utf-8")
    )
    assert style["panel"]["k"] == 20 and style["panel"]["round_trips"] == 1
    assert style["panel"]["matched"] == "circ_mv_decile+affordability"
    assert style["panel"]["dropped_trips_mean"] == 0.0
    assert [day for day, _value in style["panel_daily"]] == ["20240102", "20240103"]
    assert [day for day, _value in style["panel_draw_sd"]] == ["20240102", "20240103"]
    smoke = backend.evaluate(request, max_days=1)
    assert "panel_daily" not in json.loads(
        (Path(smoke.result_ref).parent / "style_analysis.json").read_text(encoding="utf-8")
    )
    # The panel is drawn after the replay it grades and through Brokers of its
    # own, so that replay's record is the same bytes with or without it; only
    # wall-clock fields differ between two runs.
    monkeypatch.setattr(
        "autotrade.pipelines.pit_backend.run_null_control",
        lambda *_args, **_kwargs: {"k": 0, "panel_daily": [], "panel_draw_sd": []},
    )
    bare = json.loads(Path(backend.evaluate(request).result_ref).read_text(encoding="utf-8"))
    monkeypatch.undo()

    def without_clock(payload: dict[str, object]) -> str:
        stats = {
            key: value
            for key, value in payload["stats"].items()  # type: ignore[union-attr]
            if key not in ("phase_seconds", "replay_wall_seconds", "started_at", "finished_at")
        }
        return json.dumps({**payload, "stats": stats}, sort_keys=True)

    assert without_clock(bare) == without_clock(record)

    # The null control replays its draws through the same slot table, and it
    # finds that table on disk: the backend instance that evaluated the result
    # is gone by the time a restarted worker ranks the node it recorded.
    restarted = PITDailyEvaluationBackend(tmp_path / "results", execution_mode="trusted")
    null_control_call = {
        "start": "20240102",
        "end": "20240103",
        "profile": BrokerProfile(initial_cash=100_000),
        "schedule": StrategySchedule("day", "09:28"),
        "seed": 1,
        "k": 1,
    }
    block = restarted.null_control(result.result_ref, **null_control_call)
    assert block["k"] == 1 and block["rejects_mean"] == 0.0
    # Where the slots live stays host-only: the Agent-readable record names
    # them and nothing more, so the two say the same slots in two ways.
    sidecar = Path(result.result_ref).parent / REPLAY_SLOTS_SIDECAR
    slots = json.loads(sidecar.read_text(encoding="utf-8"))["replay_slots"]
    assert [Path(slot).name for slot in slots] == record["pit"]["replay_slots"]
    assert HOST_PATH_RE.search(json.dumps(record)) is None
    # A result whose slots cannot be resolved is refused before it draws
    # anything, so the caller knows it measured nothing and owes nothing.
    sidecar.unlink()
    with pytest.raises(NullControlSetupError, match="replay slots are unknown"):
        restarted.null_control(result.result_ref, **null_control_call)

    (replay / "corporate_actions.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="declares corporate_actions"):
        PITDailyEvaluationBackend(tmp_path / "results_missing", execution_mode="trusted").evaluate(
            request
        )
    # A slot cached before ex-dates were settled declares no such domain: it
    # is refused as stale rather than replayed without dividends.
    del replay_manifest["domains"]["corporate_actions"]
    (replay / "manifest.json").write_text(json.dumps(replay_manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="must be rebuilt"):
        PITDailyEvaluationBackend(tmp_path / "results_stale", execution_mode="trusted").evaluate(
            request
        )


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
        def __init__(self, generation_id: str) -> None:
            self.calls: list[str] = []
            self.generation_id = generation_id

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
                "raw_generation": {"generation_id": self.generation_id},
            }
            (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            return manifest

    fake = FakeBuilder(provider.release.generation_id)
    provider.builder = fake  # type: ignore[assignment]
    decision = datetime.fromisoformat("2024-01-01T23:59:59+08:00")
    first = provider.prepare(
        phase="valid",
        start="20240102",
        end="20240103",
        decision_time=decision,
    )
    second = provider.prepare(
        phase="valid",
        start="20240102",
        end="20240103",
        decision_time=decision,
    )
    assert first == second
    frozen = provider.prepare(
        phase="paper",
        start="20240102",
        end="20240103",
        decision_time=decision,
    )
    heldout = provider.prepare(
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
    ] == "paper"
    assert json.loads(Path(heldout.replay_ref, "manifest.json").read_text(encoding="utf-8"))[
        "label"
    ] == "heldout"
    # One region, one build: the three phase views are hardlinks of the single
    # unphased store, each carrying its own immutable label.
    assert fake.calls == ["decision", "replay"]


def test_unphased_forward_replay_is_cloned_into_valid_phase(tmp_path: Path) -> None:
    provider, fake = _provider_with_fake_builder(tmp_path)
    decision = datetime.fromisoformat("2024-01-01T23:59:59+08:00")
    slot = "20240102_20240103_20240101T235959+0800"
    unphased = provider.cache_root / "replay" / slot
    _write_unphased_replay(
        unphased,
        label="forward",
        start="20240102",
        end="20240103",
        available_from=decision,
        generation_id=provider.release.generation_id,
    )
    bundle = provider.prepare(
        phase="valid",
        start="20240102",
        end="20240103",
        decision_time=decision,
    )
    replay = Path(bundle.replay_ref)
    assert replay == provider.cache_root / "replay" / "valid" / slot
    source_manifest = json.loads(
        (unphased / "manifest.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((replay / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["label"] == "valid"
    assert manifest["period_start"] == "20240102"
    assert manifest["snapshot_id"] != source_manifest["snapshot_id"]
    assert str(manifest["snapshot_id"]).startswith("replay_")
    assert manifest["raw_generation"] == source_manifest["raw_generation"]
    assert os.stat(replay / "daily.parquet").st_ino == os.stat(
        unphased / "daily.parquet"
    ).st_ino
    assert source_manifest["label"] == "forward"
    assert fake.calls == ["decision"]
    snapshot = Path(bundle.decision_ref)
    stash = _asof_stash_dir(
        snapshot, replay, StrategySchedule("day", "08:30"), "valid"
    )
    # The stash hangs off the region, not the phase view that materialized it.
    assert stash.parts[-7:] == (
        "replay",
        replay.name,
        "schedule",
        "period=day",
        "inference_time",
        "hour=08",
        "minute=30",
    )
    again = provider.prepare(
        phase="valid",
        start="20240102",
        end="20240103",
        decision_time=decision,
    )
    assert again.replay_ref == bundle.replay_ref
    assert fake.calls == ["decision"]


def test_concurrent_valid_prepare_from_unphased_is_phase_safe(tmp_path: Path) -> None:
    provider, fake = _provider_with_fake_builder(tmp_path)
    decision = datetime.fromisoformat("2024-01-01T23:59:59+08:00")
    slot = "20240102_20240103_20240101T235959+0800"
    _write_unphased_replay(
        provider.cache_root / "replay" / slot,
        label="forward",
        start="20240102",
        end="20240103",
        available_from=decision,
        generation_id=provider.release.generation_id,
    )
    barrier = threading.Barrier(8)
    bundles: list[SnapshotBundle] = []
    errors: list[BaseException] = []
    guard = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            bundle = provider.prepare(
                phase="valid",
                start="20240102",
                end="20240103",
                decision_time=decision,
            )
        except BaseException as exc:
            with guard:
                errors.append(exc)
            return
        with guard:
            bundles.append(bundle)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    refs = {bundle.replay_ref for bundle in bundles}
    assert len(refs) == 1
    replay = Path(next(iter(refs)))
    assert replay == provider.cache_root / "replay" / "valid" / slot
    assert json.loads((replay / "manifest.json").read_text(encoding="utf-8"))[
        "label"
    ] == "valid"
    parent = replay.parent
    assert not list(parent.glob(".*.tmp"))
    assert fake.calls == ["decision"]


def test_failed_unphased_clone_leaves_no_phased_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, fake = _provider_with_fake_builder(tmp_path)
    decision = datetime.fromisoformat("2024-01-01T23:59:59+08:00")
    slot = "20240102_20240103_20240101T235959+0800"
    unphased = provider.cache_root / "replay" / slot
    _write_unphased_replay(
        unphased,
        label="forward",
        start="20240102",
        end="20240103",
        available_from=decision,
        generation_id=provider.release.generation_id,
    )

    def boom(src: str, dst: str) -> None:
        raise OSError(errno.EIO, "injected link failure")

    monkeypatch.setattr(os, "link", boom)
    with pytest.raises(OSError, match="injected link failure"):
        provider.prepare(
            phase="valid",
            start="20240102",
            end="20240103",
            decision_time=decision,
        )
    target = provider.cache_root / "replay" / "valid" / slot
    assert not target.exists()
    parent = target.parent
    if parent.exists():
        assert not list(parent.glob(".*.tmp"))
    assert "replay" not in fake.calls

    monkeypatch.undo()
    bundle = provider.prepare(
        phase="valid",
        start="20240102",
        end="20240103",
        decision_time=decision,
    )
    assert Path(bundle.replay_ref) == target
    assert json.loads((target / "manifest.json").read_text(encoding="utf-8"))[
        "label"
    ] == "valid"
    assert os.stat(target / "daily.parquet").st_ino == os.stat(
        unphased / "daily.parquet"
    ).st_ino
    assert fake.calls == ["decision"]


def test_replay_source_with_wrong_identity_is_refused(tmp_path: Path) -> None:
    """A store whose manifest contradicts its own key is a corrupt cache.

    The store is now the canonical build target for a region, so a conflicting
    one fails the request instead of being quietly rebuilt around — the same
    rule the decision snapshot and the phase view already follow.
    """

    provider, fake = _provider_with_fake_builder(tmp_path)
    decision = datetime.fromisoformat("2024-01-01T23:59:59+08:00")
    slot = "20240102_20240103_20240101T235959+0800"
    unphased = provider.cache_root / "replay" / slot
    _write_unphased_replay(
        unphased,
        label="forward",
        start="19990101",
        end="19990102",
        available_from=decision,
        generation_id=provider.release.generation_id,
    )
    with pytest.raises(RuntimeError, match="conflicting cached replay source"):
        provider.prepare(
            phase="valid",
            start="20240102",
            end="20240103",
            decision_time=decision,
        )
    assert not (provider.cache_root / "replay" / "valid" / slot).exists()
    assert "replay" not in fake.calls


def test_one_region_is_built_once_and_shared_by_every_phase(tmp_path: Path) -> None:
    """Every phase that asks for the same region shares one build.

    The same (start, end, decision) window requested under several phases is
    replayed once and relabelled, not rebuilt per phase.
    """

    provider, fake = _provider_with_fake_builder(tmp_path)
    decision = datetime.fromisoformat("2024-01-01T23:59:59+08:00")
    bundles = [
        provider.prepare(
            phase=phase,
            start="20240102",
            end="20240103",
            decision_time=decision,
        )
        for phase in ("valid", "heldout", "paper")
    ]
    assert fake.calls == ["decision", "replay"]
    source = provider.cache_root / "replay" / "20240102_20240103_20240101T235959+0800"
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    assert source_manifest["label"] == REPLAY_SOURCE_LABEL
    assert len({bundle.replay_ref for bundle in bundles}) == 3
    for phase, bundle in zip(("valid", "heldout", "paper"), bundles, strict=True):
        replay = Path(bundle.replay_ref)
        assert replay == provider.cache_root / "replay" / phase / source.name
        manifest = json.loads((replay / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["label"] == phase
        assert manifest["snapshot_id"] != source_manifest["snapshot_id"]
        assert (
            os.stat(replay / "daily.parquet").st_ino
            == os.stat(source / "daily.parquet").st_ino
        )


def test_unphased_clone_refuses_cross_filesystem_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, fake = _provider_with_fake_builder(tmp_path)
    decision = datetime.fromisoformat("2024-01-01T23:59:59+08:00")
    slot = "20240102_20240103_20240101T235959+0800"
    _write_unphased_replay(
        provider.cache_root / "replay" / slot,
        label="forward",
        start="20240102",
        end="20240103",
        available_from=decision,
        generation_id=provider.release.generation_id,
    )

    def boom(src: str, dst: str) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(os, "link", boom)
    with pytest.raises(RuntimeError, match="different filesystem"):
        provider.prepare(
            phase="valid",
            start="20240102",
            end="20240103",
            decision_time=decision,
        )
    target = provider.cache_root / "replay" / "valid" / slot
    assert not target.exists()
    if target.parent.exists():
        assert not list(target.parent.glob(".*.tmp"))
    assert "replay" not in fake.calls


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
        PITDailyEvaluationBackend._validate_bundle(request, snapshot, (replay,))
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
        / "schedule"
        / "period=day"
        / "inference_time"
        / "hour=08"
        / "minute=30"
    )


def test_every_phase_of_one_region_shares_a_stash(tmp_path: Path) -> None:
    """Two phases replaying one region encode it once.

    Their replay views are hardlinks of one store, so the as-of parts are the
    same bytes; the stash contract states that data identity and never the
    phase or the path of the view that happened to build it.
    """

    cache_root = tmp_path / "pit_views"
    snapshot = cache_root / "decision" / "20211231T235959+0800"
    snapshot.mkdir(parents=True)
    _write_provider_contract(cache_root, generation_id="generation_one")
    slot = "20220101_20251231_20211231T235959+0800"
    schedule = StrategySchedule("day", "08:30")
    contracts: list[Path] = []
    for phase in ("valid", "heldout"):
        replay = cache_root / "replay" / phase / slot
        replay.mkdir(parents=True)
        decision_manifest, replay_manifest = _stash_manifests(
            "generation_one", phase=phase
        )
        contracts.append(
            _bind_asof_stash_contract(
                snapshot_dir=snapshot,
                replay_dir=replay,
                schedule=schedule,
                phase=phase,
                generation_id="generation_one",
                decision_manifest=decision_manifest,
                replay_manifest=replay_manifest,
            )
        )
    assert contracts[0] == contracts[1]
    record = json.loads((contracts[0] / "contract.json").read_text(encoding="utf-8"))
    assert record["decision_slot"] == snapshot.name
    assert record["replay_slot"] == slot
    # No build identity: a stash prebuilt offline and hardlinked into an
    # experiment must bind to exactly the same contract.
    assert "phase" not in record
    assert "/" not in record["decision_slot"] and "/" not in record["replay_slot"]
    assert "snapshot_id" not in json.dumps(record)


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


def _write_corporate_actions(replay: Path, rows: list[dict[str, object]] | None = None) -> None:
    """The slot's ex-date table; every built slot carries it, dividends or not."""

    columns = [
        "ts_code", "ex_date", "record_date", "pay_date", "div_listdate",
        "cash_per_share", "stock_per_share",
    ]
    pd.DataFrame(rows or [], columns=columns).to_parquet(
        replay / "corporate_actions.parquet", index=False
    )


def _write_domains(snapshot: Path, replay: Path) -> None:
    pd.DataFrame(
        {
            "trade_date": ["20240101"],
            "ts_code": ["000001.SZ"],
            "open": [10.0],
            "close": [10.0],
            "pre_close": [10.0],
            "available_at": ["2024-01-01T17:30:00+08:00"],
        }
    ).to_parquet(snapshot / "daily.parquet", index=False)
    pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.0],
            "close": [10.0, 10.0],
            "pre_close": [10.0, 10.0],
            "available_at": ["2024-01-02T17:30:00+08:00", "2024-01-03T17:30:00+08:00"],
        }
    ).to_parquet(replay / "daily.parquet", index=False)
    _write_corporate_actions(replay)
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


class _FakeReplayBuilder:
    def __init__(self, generation_id: str = "") -> None:
        self.calls: list[str] = []
        # The real builder stamps every manifest with the raw generation it
        # read; the cache refuses a view that cannot prove its release.
        self.generation_id = generation_id

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
        pd.DataFrame({"trade_date": [start]}).to_parquet(
            output / "daily.parquet", index=False
        )
        manifest = {
            "snapshot_id": "replay_stable",
            "kind": "replay_slot",
            "label": label,
            "period_start": start,
            "period_end": end,
            "available_from": available_from.isoformat(),
            "raw_generation": {"generation_id": self.generation_id},
        }
        (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest


def _provider_with_fake_builder(
    tmp_path: Path,
) -> tuple[ResearchPITSnapshotProvider, _FakeReplayBuilder]:
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
    fake = _FakeReplayBuilder(provider.release.generation_id)
    provider.builder = fake  # type: ignore[assignment]
    return provider, fake


def _write_unphased_replay(
    path: Path,
    *,
    label: str,
    start: str,
    end: str,
    available_from: datetime,
    generation_id: str,
) -> None:
    path.mkdir(parents=True)
    (path / "text_library").mkdir()
    pd.DataFrame({"trade_date": [start], "close": [10.0]}).to_parquet(
        path / "daily.parquet", index=False
    )
    (path / "text_library" / "news.parquet").write_bytes(b"news")
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": "replay_unphased",
                "kind": "replay_slot",
                "label": label,
                "period_start": start,
                "period_end": end,
                "available_from": available_from.isoformat(),
                "raw_generation": {"generation_id": generation_id},
            }
        ),
        encoding="utf-8",
    )


def test_asof_view_is_a_directory_per_domain_and_truncates_to_max_days(
    tmp_path: Path,
) -> None:
    """The layout hint a smoke run reports has to be the real one.

    Strategies kept reading ``asof_dir/daily.parquet`` because the mounted
    decision snapshot is flat while the rolling view is a directory of parts.
    Pin both the recorded domain names and the short-window truncation the
    unofficial rehearsal rides on.
    """
    snapshot, replay = _pit_slot_paths(
        tmp_path, decision="layout", replay="layout", generation_id="generation_layout"
    )
    (snapshot / "text_library").mkdir()
    (replay / "text_library").mkdir()
    _write_domains(snapshot, replay)
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": "snap_layout",
                "kind": "decision_input",
                "raw_generation": {"generation_id": "generation_layout"},
            }
        ),
        encoding="utf-8",
    )
    (replay / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": "replay_layout",
                "kind": "replay_slot",
                "label": "valid",
                "period_start": "20240102",
                "period_end": "20240103",
                "available_from": "2024-01-01T23:59:59+08:00",
                "raw_generation": {"generation_id": "generation_layout"},
                "domains": {"corporate_actions": {"rows": 0}},
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
    # The directory form is the only one that exists in the rolling view.
    pd.read_parquet(context.asof_dir + "/daily")
    return []
""",
        encoding="utf-8",
    )

    def _run(max_days, start_day=None):
        return PITDailyEvaluationBackend(
            tmp_path / f"results_{max_days}_{start_day}", execution_mode="trusted"
        ).evaluate(
            EvaluationRequest(
                ArtifactRevision("revision_layout", revision),
                SnapshotBundle(
                    "snap_layout",
                    str(snapshot),
                    str(replay),
                    generation_id="generation_layout",
                ),
                "valid",
                "20240102",
                "20240103",
                StrategySchedule("day", "09:28"),
                BrokerProfile(initial_cash=100_000),
            ),
            max_days=max_days,
            start_day=start_day,
        )

    full = _run(None)
    record = json.loads(Path(full.result_ref).read_text(encoding="utf-8"))
    domains = record["pit"]["asof_domains"]
    assert {"daily", "events", "macro", "fundamentals", "universe"} <= set(domains)
    # The slot identity check still sees the full window; only the replay frame
    # is short, so a truncated run can never masquerade as a full Validation.
    assert full.summary["replayed_trade_days"] == 2

    short = _run(1)
    assert short.summary["replayed_trade_days"] == 1
    assert short.summary["decision_calls"] == 1

    # The same truncation from the other end: a probe that opens late replays
    # the days from there on, through the same rolling as-of view, and the
    # record says which day it actually covered.
    late = _run(None, start_day="20240103")
    assert late.summary["replayed_trade_days"] == 1
    assert late.summary["decision_calls"] == 1
    late_record = json.loads(Path(late.result_ref).read_text(encoding="utf-8"))
    assert [row["trade_date"] for row in late_record["equity_curve"]] == ["20240103"]

    for days in (0, -1):
        with pytest.raises(ValueError, match="max_days must be a positive integer"):
            _run(days)


def test_incremental_asof_lock_matches_a_full_chmod_tree(tmp_path: Path) -> None:
    """The cheap lock must leave exactly the modes the expensive one left.

    Re-chmod'ing the whole as-of tree every decision day was the dominant cost
    of the data_view phase (it grows with every part that lands). The
    replacement only touches what newly appeared, which is only safe because the
    tree is append-only — so the end state has to be identical, day by day.
    """
    incremental = tmp_path / "incremental"
    reference = tmp_path / "reference"
    for root in (incremental, reference):
        root.mkdir()
        (root / "daily").mkdir()
        (root / "text_library").mkdir()
    view = _AsOfReadOnlyView(incremental)

    for day in range(1, 6):
        # Real order: the view reopens its directories, Timeview appends, the
        # view locks again.
        view.unlock_directories()
        chmod_tree(reference, file_mode=0o644, dir_mode=0o755)
        for root in (incremental, reference):
            for name in ("daily", "text_library"):
                (root / name / f"part_{day:04d}.parquet").write_bytes(b"x")
            if day == 3:  # a domain directory can appear mid-replay
                (root / "auction").mkdir(exist_ok=True)
                (root / "auction" / "part_0000.parquet").write_bytes(b"x")
        view.lock()
        chmod_tree(reference, file_mode=0o444, dir_mode=0o555)

        left = {
            str(path.relative_to(incremental)): stat.S_IMODE(path.stat().st_mode)
            for path in incremental.rglob("*")
        }
        right = {
            str(path.relative_to(reference)): stat.S_IMODE(path.stat().st_mode)
            for path in reference.rglob("*")
        }
        assert left == right, f"day {day}: {sorted(set(left.items()) ^ set(right.items()))}"
        assert stat.S_IMODE(incremental.stat().st_mode) == stat.S_IMODE(
            reference.stat().st_mode
        )

    # Unlocking must leave every file read-only: only directories reopen.
    view.unlock_directories()
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o444
        for path in incremental.rglob("*")
        if path.is_file()
    )
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o755
        for path in incremental.rglob("*")
        if path.is_dir()
    )
    chmod_tree(incremental, file_mode=0o644, dir_mode=0o755)
    chmod_tree(reference, file_mode=0o644, dir_mode=0o755)


# A span of consecutive replay slots runs as one book. The synthetic release has
# slot A (2024-01-02/03), slot B (2024-01-04/05) continuing it, and the long
# slot AB whose every file holds A's rows then B's, as one slot over both would.
_SPAN_DECISION = "20231231T235959+0800"
_SPAN_GENERATION = "generation_span"
_SPAN_SCHEDULE = StrategySchedule("day", "09:28")
_SPAN_SLOTS = {
    "a": ("20240101", "20240103", "20231231T235959+0800", ("20240102", "20240103")),
    "b": ("20240104", "20240105", "20240103T235959+0800", ("20240104", "20240105")),
    "ab": (
        "20240101",
        "20240105",
        "20231231T235959+0800",
        ("20240102", "20240103", "20240104", "20240105"),
    ),
}
# (pre_close, open, close); 000001.SZ goes ex a 0.2 cash dividend on 2024-01-05.
_SPAN_PRICES = {
    "000001.SZ": {
        "20231229": (10.0, 10.0, 10.0),
        "20240102": (10.0, 10.0, 10.5),
        "20240103": (10.5, 10.5, 11.0),
        "20240104": (11.0, 11.0, 11.2),
        "20240105": (11.0, 11.0, 11.5),
    },
    "000002.SZ": {
        "20231229": (5.0, 5.0, 5.0),
        "20240102": (5.0, 5.0, 5.1),
        "20240103": (5.1, 5.1, 5.2),
        "20240104": (5.2, 5.2, 5.0),
        "20240105": (5.0, 5.0, 5.3),
    },
}
# The as-of universe each decision anchor carries: a vintage table, not an
# event stream. Between the two anchors 000001.SZ is renamed into ST and
# 000003.SZ lists, so a replay that never rolls it judges 2024-01-04 on
# 2023-12-31 names.
_SPAN_UNIVERSE = {
    "20231231T235959+0800": (("000001.SZ", "平安银行"), ("000002.SZ", "万科A")),
    "20240103T235959+0800": (
        ("000001.SZ", "*ST平安"),
        ("000002.SZ", "万科A"),
        ("000003.SZ", "新上市"),
    ),
}
_SPAN_STRATEGY = '''import numpy as np
import pandas as pd

REFIT_PERIOD = "month"
DOMAINS = ("daily", "intraday_1min", "auction", "events", "macro", "fundamentals", "text_index")
TRADES = {
    "20240102": ("000001.SZ", "buy", 1000, 15),
    "20240103": ("000002.SZ", "buy", 1000, 10),
    "20240104": ("000001.SZ", "sell", 500, 10),
    "20240105": ("000002.SZ", "sell", 1000, 15),
}


def fit(context):
    try:
        fits = np.load(context.state_dir + "/fits.npy")
    except FileNotFoundError:
        fits = np.zeros(0)
    np.save(context.state_dir + "/fits.npy", np.append(fits, 1.0))


def generate_orders(context):
    now = pd.Timestamp(context.inference_at)
    seen = {}
    for name in DOMAINS:
        frame = pd.read_parquet(context.asof_dir + "/" + name)
        if len(frame) and pd.to_datetime(frame["available_at"], format="ISO8601").max() > now:
            raise RuntimeError("a future row is visible in " + name)
        seen[name] = len(frame)
    if any(pd.Timestamp(bar["available_at"]) > now for bar in context.bars):
        raise RuntimeError("a future bar is visible")
    universe = pd.read_parquet(context.asof_dir + "/universe")
    vintage = sorted(universe["ts_code"] + ":" + universe["name"])
    day = context.inference_at.strftime("%Y%m%d")
    symbol, action, quantity, hour = TRADES[day]
    return [
        {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "execute_at": context.inference_at.replace(hour=hour, minute=0).isoformat(),
        },
        {
            "symbol": "000001.SZ",
            "action": "buy",
            "quantity": 100,
            "execute_at": "2099-01-01T09:30:00+08:00",
            "seen": seen,
            "universe": vintage,
            "bars": len(context.bars),
            "fits": int(len(np.load(context.state_dir + "/fits.npy"))),
            "cash": context.account.cash,
            "positions": dict(context.account.positions),
            "asof_version": context.asof_version,
        },
    ]
'''


def _stamp(day: str, clock: str) -> str:
    return f"{day[:4]}-{day[4:6]}-{day[6:]}T{clock}+08:00"


def _write_span_universe(decision_dir: Path, vintage: tuple[tuple[str, str], ...]) -> None:
    decision_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(vintage), columns=["ts_code", "name"]).to_parquet(
        decision_dir / "universe.parquet", index=False
    )


def _span_frames(days: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Every domain of the synthetic release over ``days``, day by day."""

    rows: dict[str, list[dict[str, object]]] = {
        name: []
        for name in ("daily", "intraday_1min", "auction", "events", "fundamentals", "macro", "text_index", "news")
    }
    for day in days:
        for symbol, prices in _SPAN_PRICES.items():
            pre_close, open_, close = prices[day]
            rows["daily"].append(
                {
                    "trade_date": day, "ts_code": symbol, "open": open_, "close": close,
                    "pre_close": pre_close, "pct_chg": round(close / pre_close - 1.0, 6),
                    "circ_mv": 1e6, "up_limit": round(pre_close * 1.1, 2),
                    "down_limit": round(pre_close * 0.9, 2), "available_at": _stamp(day, "17:30:00"),
                }
            )
            rows["intraday_1min"].append(
                {
                    "trade_date": day, "ts_code": symbol, "trade_time": _stamp(day, "10:00:00"),
                    "close": round(open_ + 0.05, 2), "available_at": _stamp(day, "10:00:00"),
                }
            )
            rows["auction"].append(
                {"ts_code": symbol, "trade_date": day, "price": open_, "available_at": _stamp(day, "09:29:00")}
            )
            # Two datasets on their own nodes: margin_secs lands before the
            # morning decision, moneyflow only after the evening one, so a slot's
            # last moneyflow rows first show in the next slot.
            for dataset, clock in (("margin_secs", "09:00:00"), ("moneyflow", "19:00:00")):
                rows["events"].append(
                    {
                        "dataset": dataset, "ts_code": symbol, "trade_date": day,
                        "value": close, "available_at": _stamp(day, clock),
                    }
                )
            rows["fundamentals"].append(
                {"dataset": "income_vip", "ts_code": symbol, "value": pre_close, "available_at": _stamp(day, "18:00:00")}
            )
        # Stamped at the end of its day: the last one of a slot sits exactly on
        # the next slot's anchor and belongs to the earlier slot.
        rows["macro"].append(
            {
                "dataset": "index_daily", "ts_code": "000300.SH", "trade_date": day,
                "pct_chg": 0.5, "available_at": _stamp(day, "23:59:59"),
            }
        )
        rows["text_index"].append(
            {
                "text_id": f"news_{day}", "dataset": "news", "ts_codes": "000001.SZ",
                "title": f"title {day}", "available_at": _stamp(day, "22:00:00"),
                "library_file": "news.parquet",
            }
        )
        rows["news"].append({"text_id": f"news_{day}", "body": f"body {day}"})
    return {name: pd.DataFrame(items) for name, items in rows.items()}


def _write_span_release(root: Path) -> tuple[Path, dict[str, Path]]:
    cache_root = root / "pit_views"
    snapshot = cache_root / "decision" / _SPAN_DECISION
    (snapshot / "text_library").mkdir(parents=True)
    _write_provider_contract(cache_root, generation_id=_SPAN_GENERATION)
    raw_generation = {"generation_id": _SPAN_GENERATION}

    def write_frames(target: Path, frames: dict[str, pd.DataFrame], *, minute_groups: int | None) -> None:
        for name, frame in frames.items():
            if name == "news":
                frame.to_parquet(target / "text_library" / "news.parquet", index=False)
            elif name == "intraday_1min":
                pq.write_table(
                    pa.Table.from_pandas(frame, preserve_index=False),
                    target / "intraday_1min.parquet",
                    row_group_size=minute_groups,
                )
            else:
                frame.to_parquet(target / f"{name}.parquet", index=False)

    write_frames(snapshot, _span_frames(("20231229",)), minute_groups=None)
    _write_span_universe(snapshot, _SPAN_UNIVERSE[_SPAN_DECISION])
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": "snap_span",
                "kind": "decision_input",
                "decision_time": "2023-12-31T23:59:59+08:00",
                "raw_generation": raw_generation,
            }
        ),
        encoding="utf-8",
    )
    chmod_tree(snapshot, file_mode=0o444, dir_mode=0o555)
    # Every slot is prepared beside the decision view of its own anchor, and
    # that view's universe is the vintage the rolling view publishes for the
    # slot. B's anchor falls inside A, so it already carries the rename and the
    # listing that happened there.
    for anchor, vintage in _SPAN_UNIVERSE.items():
        if anchor != _SPAN_DECISION:
            _write_span_universe(cache_root / "decision" / anchor, vintage)
    slots: dict[str, Path] = {}
    for key, (start, end, anchor, days) in _SPAN_SLOTS.items():
        slot = cache_root / "replay" / "valid" / f"{start}_{end}_{anchor}"
        (slot / "text_library").mkdir(parents=True)
        write_frames(slot, _span_frames(days), minute_groups=1)
        dividends = (
            [{"ts_code": "000001.SZ", "ex_date": "20240105", "record_date": "20240104",
              "pay_date": "20240105", "div_listdate": "", "cash_per_share": 0.2, "stock_per_share": 0.0}]
            if "20240105" in days
            else []
        )
        _write_corporate_actions(slot, dividends)
        (slot / "manifest.json").write_text(
            json.dumps(
                {
                    "snapshot_id": f"replay_span_{key}",
                    "kind": "replay_slot",
                    "label": "valid",
                    "period_start": start,
                    "period_end": end,
                    "available_from": datetime.strptime(anchor, "%Y%m%dT%H%M%S%z").isoformat(),
                    "raw_generation": raw_generation,
                    "domains": {"corporate_actions": {"rows": len(dividends)}},
                }
            ),
            encoding="utf-8",
        )
        slots[key] = slot
    return snapshot, slots


def _span_revision(root: Path, source: str = _SPAN_STRATEGY) -> Path:
    revision = root / "revision"
    revision.mkdir(parents=True)
    (revision / "main.py").write_text(source, encoding="utf-8")
    return revision


def _span_request(snapshot: Path, *replay_dirs: Path, revision: Path) -> EvaluationRequest:
    first, *rest = replay_dirs
    manifests = [json.loads((path / "manifest.json").read_text(encoding="utf-8")) for path in replay_dirs]
    return EvaluationRequest(
        ArtifactRevision("revision_span", revision),
        SnapshotBundle("snap_span", str(snapshot), str(first), generation_id=_SPAN_GENERATION),
        "valid",
        manifests[0]["period_start"],
        manifests[-1]["period_end"],
        _SPAN_SCHEDULE,
        BrokerProfile(initial_cash=1_000_000),
        continuation=tuple(str(path) for path in rest),
    )


class _CountingParquet:
    """``pyarrow.parquet`` as the Timeview sees it, recording each part it encodes."""

    def __init__(self) -> None:
        self.written: list[str] = []

    def write_table(self, table: pa.Table, where: object, **kwargs: object) -> None:
        self.written.append(Path(str(where)).name)
        pq.write_table(table, where, **kwargs)  # type: ignore[arg-type]

    def __getattr__(self, name: str) -> object:
        return getattr(pq, name)


def _span_stash_parts(
    snapshot: Path, *replay_dirs: Path, window_start: str = ""
) -> dict[str, bytes]:
    """Every as-of part a replay published, by domain and part name, over its slots' stashes.

    ``window_start`` reads the stash a window opening on that day keys for
    itself instead of the span's own.
    """

    parts: dict[str, bytes] = {}
    for replay_dir in replay_dirs:
        stash = _asof_stash_dir(snapshot, replay_dir, _SPAN_SCHEDULE, "valid", window_start)
        if not stash.is_dir():
            continue
        for path in sorted(stash.rglob("*.parquet")):
            key = str(path.relative_to(stash))
            assert key not in parts, key
            parts[key] = path.read_bytes()
    return parts


def _without_universe(orders: list[dict[str, object]]) -> list[dict[str, object]]:
    """The strategy's observations without the two things a vintage roll moves."""

    dropped = {"universe", "asof_version"}
    return [{key: value for key, value in order.items() if key not in dropped} for order in orders]


def _visible_universes(record: dict[str, object]) -> list[list[str]]:
    """The universe each decision of a replay actually read, in order."""

    return [row["universe"] for row in record["pending_orders"] if "universe" in row]


def test_a_span_rolls_the_universe_to_each_slots_own_vintage(tmp_path: Path) -> None:
    """Each slot shows the vintage of its own anchor -- never a later one.

    A's two decisions run on the 2023-12-31 view although B's decision view is
    already on disk; B's first decision picks up the rename and the listing
    that happened inside A. One long slot over the same days has only its own
    anchor, so it never sees them.
    """

    snapshot, slots = _write_span_release(tmp_path)
    revision = _span_revision(tmp_path)
    chain = PITDailyEvaluationBackend(tmp_path / "results_chain", execution_mode="trusted").evaluate(
        _span_request(snapshot, slots["a"], slots["b"], revision=revision)
    )
    single = PITDailyEvaluationBackend(tmp_path / "results_single", execution_mode="trusted").evaluate(
        _span_request(snapshot, slots["ab"], revision=revision)
    )
    base = ["000001.SZ:平安银行", "000002.SZ:万科A"]
    rolled = ["000001.SZ:*ST平安", "000002.SZ:万科A", "000003.SZ:新上市"]
    chain_record, single_record = (
        json.loads(Path(result.result_ref).read_text(encoding="utf-8")) for result in (chain, single)
    )
    # 20240102, 20240103 | 20240104, 20240105
    assert _visible_universes(chain_record) == [base, base, rolled, rolled]
    assert _visible_universes(single_record) == [base] * 4
    # The rolled read is one vintage, not two concatenated (that would repeat
    # 000001.SZ), and the roll is a new global view: a feature cached on the
    # old names is invalidated.
    versions = [int(row["asof_version"]) for row in chain_record["pending_orders"] if "universe" in row]
    assert versions[2] > versions[1]


def test_a_span_refuses_a_slot_whose_anchor_has_no_decision_universe(tmp_path: Path) -> None:
    snapshot, slots = _write_span_release(tmp_path)
    (snapshot.parent / _SPAN_SLOTS["b"][2] / "universe.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="has no decision-view universe"):
        PITDailyEvaluationBackend(tmp_path / "results", execution_mode="trusted").evaluate(
            _span_request(snapshot, slots["a"], slots["b"], revision=_span_revision(tmp_path))
        )


def test_a_span_of_slots_is_one_book_equal_to_one_long_slot(tmp_path: Path) -> None:
    """Account, positions, fit state and the as-of view carry across a slot boundary.

    The chain A→B and the long slot AB must replay identically: the same fills
    (one of them priced from each slot's minute file), the same dividend on a
    position bought in A, one fit under a monthly refit, and byte-equal as-of
    parts, including the part that lands after the boundary and carries A's
    evening rows. The strategy itself fails on any row or bar later than its
    decision, in both replays.
    """

    snapshot, slots = _write_span_release(tmp_path)
    revision = _span_revision(tmp_path)
    chain = PITDailyEvaluationBackend(tmp_path / "results_chain", execution_mode="trusted").evaluate(
        _span_request(snapshot, slots["a"], slots["b"], revision=revision)
    )
    single = PITDailyEvaluationBackend(tmp_path / "results_single", execution_mode="trusted").evaluate(
        _span_request(snapshot, slots["ab"], revision=revision)
    )
    chain_record, single_record = (
        json.loads(Path(result.result_ref).read_text(encoding="utf-8")) for result in (chain, single)
    )
    for key in ("equity_curve", "executions", "corporate_actions", "inference_dates"):
        assert chain_record[key] == single_record[key], key
    # Everything the strategy saw is equal except the universe vintage and the
    # version that its roll bumps: the chain has one decision anchor per slot
    # and rolls to B's at B's first decision, while one long slot only ever has
    # its own anchor's (test_a_span_rolls_the_universe_to_each_slots_own_vintage).
    assert _without_universe(chain_record["pending_orders"]) == _without_universe(
        single_record["pending_orders"]
    )
    for key in ("total_return", "max_drawdown", "turnover", "trade_count", "sub_windows", "benchmark"):
        assert chain.summary[key] == single.summary[key], key
    assert chain_record["pit"]["replay_slots"] == [slots["a"].name, slots["b"].name]
    assert [row["status"] for row in chain_record["executions"]] == ["filled"] * 4
    # Priced off B's own minute file (10:00 close 11.05, less sell slippage).
    assert chain_record["executions"][2]["price"] == pytest.approx(11.05, abs=0.01)
    [dividend] = chain_record["corporate_actions"]
    assert (dividend["trade_date"], dividend["quantity_before"]) == ("20240105", 500)

    observations = [row for row in chain_record["pending_orders"] if "seen" in row]
    assert [row["fits"] for row in observations] == [1, 1, 1, 1]
    # The first decision of slot B holds what slot A bought.
    assert observations[2]["positions"] == {"000001.SZ": 1000, "000002.SZ": 1000}
    assert observations[2]["bars"] == 4
    # A's evening moneyflow rows first show at B's first decision.
    assert observations[2]["seen"]["events"] - observations[1]["seen"]["events"] == 4

    parts = _span_stash_parts(snapshot, slots["a"], slots["b"])
    assert parts == _span_stash_parts(snapshot, slots["ab"])
    assert {key.split("/")[0] for key in parts} == {
        "daily", "intraday_1min", "auction", "events", "macro", "fundamentals", "text_index", "text_library",
    }
    boundary = pd.read_parquet(_asof_stash_dir(snapshot, slots["b"], _SPAN_SCHEDULE, "valid") / "events")
    assert "20240103" in set(boundary["trade_date"])
    contracts = [
        json.loads((_asof_stash_dir(snapshot, slot, _SPAN_SCHEDULE, "valid") / "contract.json").read_text())
        for slot in (slots["a"], slots["b"])
    ]
    assert "preceding_replay_slots" not in contracts[0]
    assert contracts[1]["preceding_replay_slots"] == [slots["a"].name]

    style = [
        json.loads((Path(result.result_ref).parent / "style_analysis.json").read_text(encoding="utf-8"))
        for result in (chain, single)
    ]
    assert style[0]["benchmark_daily"] == style[1]["benchmark_daily"]
    assert len(style[0]["benchmark_daily"]) == 4


def test_a_late_opening_window_never_publishes_the_spans_parts(tmp_path: Path) -> None:
    """A replay that opens inside the span writes a different part stream.

    An unofficial window given a ``start_day`` decodes every slot up to its
    first decision WITHOUT refreshing, so that first decision publishes
    everything accumulated before it as one part, numbered from the beginning.
    The span's own replay numbers the same slot's parts on from where the
    preceding slots left off. Both streams are right views of their own
    instants, but only the span's may be published: where a domain wrote no
    part before the opening slot -- ``auction``, whose rows start inside a
    later slot of the real four-year span -- both streams claim ``part_0001``
    and the span's replay dies hours in on the footer row-count check.
    """

    snapshot, slots = _write_span_release(tmp_path)
    revision = _span_revision(tmp_path)
    request = _span_request(snapshot, slots["a"], slots["b"], revision=revision)

    late = PITDailyEvaluationBackend(tmp_path / "results_late", execution_mode="trusted").evaluate(
        request, start_day="20240103"
    )
    assert late.summary["replayed_trade_days"] == 3
    assert _span_stash_parts(snapshot, slots["a"], slots["b"]) == {}

    # Truncated only at the end, the window IS the span's own prefix: it keeps
    # the stash, so a rehearsal from day one still costs the span nothing.
    head = PITDailyEvaluationBackend(tmp_path / "results_head", execution_mode="trusted").evaluate(
        request, max_days=1
    )
    assert head.summary["replayed_trade_days"] == 1
    prefix = _span_stash_parts(snapshot, slots["a"], slots["b"])
    assert prefix

    full = PITDailyEvaluationBackend(tmp_path / "results_full", execution_mode="trusted").evaluate(request)
    assert full.summary["replayed_trade_days"] == 4
    parts = _span_stash_parts(snapshot, slots["a"], slots["b"])
    assert {key: parts[key] for key in prefix} == prefix
    # The stash the late window ran against still holds exactly the parts one
    # long slot writes.
    PITDailyEvaluationBackend(tmp_path / "results_long", execution_mode="trusted").evaluate(
        _span_request(snapshot, slots["ab"], revision=revision)
    )
    assert parts == _span_stash_parts(snapshot, slots["ab"])


def test_a_late_opening_window_reuses_the_stash_it_keys_by_its_opening_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its stream is its own, but it is the same stream every time.

    Rebuilding it means decoding every slot before the opening day and encoding
    everything accumulated since the span's start -- 2 h 41 min on the real
    four-year span, against 165 s for the same three-day probe from day one. So
    the window binds a stash keyed by the day it opens on: the span's stash
    still holds only the span's parts, a probe repeated on that window encodes
    nothing at all, and a probe opening on another day builds its own.
    """

    snapshot, slots = _write_span_release(tmp_path)
    revision = _span_revision(tmp_path)
    request = _span_request(snapshot, slots["a"], slots["b"], revision=revision)

    first = PITDailyEvaluationBackend(tmp_path / "results_first", execution_mode="trusted").evaluate(
        request, start_day="20240103"
    )
    assert first.summary["replayed_trade_days"] == 3
    assert _span_stash_parts(snapshot, slots["a"], slots["b"]) == {}
    window = _span_stash_parts(snapshot, slots["a"], slots["b"], window_start="20240103")
    assert window
    contract = json.loads(
        (
            _asof_stash_dir(snapshot, slots["a"], _SPAN_SCHEDULE, "valid", "20240103")
            / "contract.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["window_start"] == "20240103"

    counter = _CountingParquet()
    monkeypatch.setattr(timeview_module, "pq", counter)
    again = PITDailyEvaluationBackend(tmp_path / "results_again", execution_mode="trusted").evaluate(
        request, start_day="20240103"
    )
    # Every part of the second probe was hardlinked out of the window's stash:
    # no slot was re-encoded, and the replay is the same book.
    assert counter.written == []
    assert again.summary["total_return"] == first.summary["total_return"]
    assert again.summary["order_count"] == first.summary["order_count"]
    assert _span_stash_parts(snapshot, slots["a"], slots["b"]) == {}
    assert _span_stash_parts(snapshot, slots["a"], slots["b"], window_start="20240103") == window

    # A window opening on another day is another stream: it must not read this
    # one's parts, so the opening day is part of the key, not a label on it.
    monkeypatch.undo()
    PITDailyEvaluationBackend(tmp_path / "results_other", execution_mode="trusted").evaluate(
        request, start_day="20240104"
    )
    later = _span_stash_parts(snapshot, slots["a"], slots["b"], window_start="20240104")
    assert later and later != window
    assert _span_stash_parts(snapshot, slots["a"], slots["b"]) == {}


def test_a_span_opens_a_slot_at_its_first_decision_and_reads_only_the_rows_it_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slot B is opened at its first decision, and no slot is ever decoded whole.

    A's two decisions have refreshed the view by then. What each domain reads
    from a slot's file is exactly the rows its parts publish -- the rows a
    stash hit would not read at all -- never the domain.
    """

    snapshot, slots = _write_span_release(tmp_path)
    revision = _span_revision(tmp_path)
    real_open = pit_backend._open_replay_rows
    real_refresh = Timeview.refresh
    real_take = timeview_module.ReplayRows.take
    refreshes: list[pd.Timestamp] = []
    at_open: dict[str, int] = {}
    taken: dict[str, int] = {}

    def counting_refresh(self: Timeview, when: pd.Timestamp) -> tuple[str, str]:
        refreshes.append(when)
        return real_refresh(self, when)

    def tracking_open(replay_dir: Path) -> dict[str, timeview_module.ReplayRows]:
        at_open[replay_dir.name] = len(refreshes)
        return real_open(replay_dir)

    def counting_take(self: timeview_module.ReplayRows, indices):
        for table in real_take(self, indices):
            taken[self.path.stem] = taken.get(self.path.stem, 0) + table.num_rows
            yield table

    monkeypatch.setattr(Timeview, "refresh", counting_refresh)
    monkeypatch.setattr(pit_backend, "_open_replay_rows", tracking_open)
    monkeypatch.setattr(timeview_module.ReplayRows, "take", counting_take)
    PITDailyEvaluationBackend(tmp_path / "results", execution_mode="trusted").evaluate(
        _span_request(snapshot, slots["a"], slots["b"], revision=revision)
    )
    assert at_open == {slots["a"].name: 0, slots["b"].name: 2}
    published: dict[str, int] = {}
    for key, content in _span_stash_parts(snapshot, slots["a"], slots["b"]).items():
        domain = key.split("/")[0]
        if domain != "text_library":
            published[domain] = published.get(domain, 0) + pq.ParquetFile(io.BytesIO(content)).metadata.num_rows
    assert taken == {domain: rows for domain, rows in published.items() if rows and domain != "intraday_1min"}
    assert taken["events"] < sum(
        pq.ParquetFile(slots[key] / "events.parquet").metadata.num_rows for key in ("a", "b")
    )


def test_a_strategy_exception_in_a_later_slot_is_the_strategys_own(tmp_path: Path) -> None:
    snapshot, slots = _write_span_release(tmp_path)
    source = _SPAN_STRATEGY.replace(
        '    day = context.inference_at.strftime("%Y%m%d")\n',
        '    day = context.inference_at.strftime("%Y%m%d")\n'
        '    if day >= "20240104":\n'
        '        raise ValueError("no edge in slot two")\n',
    )
    revision = _span_revision(tmp_path, source)
    results = tmp_path / "results"
    with pytest.raises(BacktestError, match="no edge in slot two") as raised:
        PITDailyEvaluationBackend(results, execution_mode="trusted").evaluate(
            _span_request(snapshot, slots["a"], slots["b"], revision=revision)
        )
    assert raised_by_strategy(raised.value)
    # The failing decision rides on the error, so a continuous replay can say
    # which of its slices the strategy raised in.
    assert raised.value.inference_at.strftime("%Y%m%d") == "20240104"
    assert list(results.iterdir()) == []


def test_a_span_refuses_slots_that_do_not_partition_its_rows(tmp_path: Path) -> None:
    snapshot, slots = _write_span_release(tmp_path)
    revision = _span_revision(tmp_path)
    backend = PITDailyEvaluationBackend(tmp_path / "results", execution_mode="trusted")
    manifest_path = slots["b"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Overlap: AB already holds B's days and rows.
    with pytest.raises(ValueError, match="does not continue"):
        backend.evaluate(_span_request(snapshot, slots["ab"], slots["b"], revision=revision))
    # Anchored a day early: 2024-01-03's evening rows would be published twice.
    manifest_path.write_text(
        json.dumps({**manifest, "available_from": "2024-01-02T23:59:59+08:00"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not continue"):
        backend.evaluate(_span_request(snapshot, slots["a"], slots["b"], revision=revision))
    # Screened per slot: a held name could vanish from the next slot.
    manifest_path.write_text(
        json.dumps({**manifest, "domains": {**manifest["domains"], "universe_screen": {"active": True}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unscreened universe"):
        backend.evaluate(_span_request(snapshot, slots["a"], slots["b"], revision=revision))


def test_a_span_prebuild_encodes_the_parts_its_replay_publishes(tmp_path: Path) -> None:
    snapshot, slots = _write_span_release(tmp_path / "prebuilt")
    replayed_snapshot, replayed_slots = _write_span_release(tmp_path / "replayed")
    PITDailyEvaluationBackend(tmp_path / "results", execution_mode="trusted").evaluate(
        _span_request(
            replayed_snapshot, replayed_slots["a"], replayed_slots["b"], revision=_span_revision(tmp_path)
        )
    )

    def prebuild(host: str) -> dict[str, object]:
        return prebuild_asof_stash(
            snapshot_dir=snapshot,
            replay_dir=slots["a"],
            continuation=[slots["b"]],
            schedule=_SPAN_SCHEDULE,
            phase="valid",
            generation_id=_SPAN_GENERATION,
            start="20240101",
            end="20240105",
            host_dir=tmp_path / "host" / host,
        )

    built = prebuild("first")
    assert (built["reused"], built["trade_days"], built["refresh_calls"]) == (False, 4, 4)
    assert _span_stash_parts(snapshot, slots["a"], slots["b"]) == _span_stash_parts(
        replayed_snapshot, replayed_slots["a"], replayed_slots["b"]
    )
    records = [
        json.loads((_asof_stash_dir(snapshot, slot, _SPAN_SCHEDULE, "valid") / "prebuild.json").read_text())
        for slot in (slots["a"], slots["b"])
    ]
    assert [(record["start"], record["end"], record["trade_days"]) for record in records] == [
        ("20240101", "20240103", 2),
        ("20240104", "20240105", 2),
    ]
    assert prebuild("second")["reused"] is True
