from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from autotrade.environment.replay.engine import StrategyDataView
from autotrade.environment.strategy import StrategySchedule
from autotrade.paper import DailyPaperEngine, PaperEngineError


class _Executor:
    def __init__(self) -> None: self.calls = 0
    def execute(self, context):
        self.calls += 1
        return [{"symbol": "000001.SZ", "action": "buy", "quantity": 100, "execute_at": "2026-01-02T09:30:00+08:00"}]
    def close(self) -> None: pass


class _TimedExecutor(_Executor):
    def execute(self, context):
        self.calls += 1
        return [{"symbol": "000001.SZ", "action": "buy", "quantity": 100, "execute_at": "2026-01-02T10:00:00+08:00"}]


def _engine(tmp_path: Path, executor: _Executor) -> DailyPaperEngine:
    strategy = tmp_path / "main.py"
    strategy.write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    daily = pd.DataFrame([
        {
            "trade_date": "20260102",
            "symbol": "000001.SZ",
            "open": 10.0,
            "close": 11.0,
            "up_limit": 12.0,
            "down_limit": 8.0,
        }
    ])
    return DailyPaperEngine(strategy_path=strategy, strategy_revision="revision_1", daily=daily, state_root=tmp_path / "paper", executor_factory=lambda _path, _config: executor)


def test_paper_day_is_persistent_and_idempotent(tmp_path: Path):
    executor = _Executor(); engine = _engine(tmp_path, executor)
    first = engine.run_day("20260102"); second = engine.run_day("20260102")
    assert first == second
    assert first["position_count"] == 1
    assert executor.calls == 1
    executions = (tmp_path / "paper/executions_20260102.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(executions) == 1
    assert json.loads(executions[0])["status"] == "filled"


def test_paper_refuses_orphan_journals(tmp_path: Path):
    executor = _Executor(); engine = _engine(tmp_path, executor)
    root = tmp_path / "paper"; root.mkdir(); (root / "orders_20260102.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(PaperEngineError, match="state.*missing"):
        engine.run_day("20260102")


def test_paper_reports_corrupt_state_as_account_restore_failure(tmp_path: Path):
    engine = _engine(tmp_path, _Executor())
    root = tmp_path / "paper"
    root.mkdir()
    (root / ".paper_state.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(PaperEngineError, match="JSON file must contain an object"):
        engine.run_day("20260102")


def test_paper_first_day_must_be_explicitly_present_in_market_view(tmp_path: Path):
    engine = _engine(tmp_path, _Executor())
    with pytest.raises(PaperEngineError, match="daily market has no trading day 20260105"):
        engine.run_day("20260105")


def test_paper_injects_pit_data_view_once_per_daily_decision(tmp_path: Path):
    class ViewExecutor(_Executor):
        def execute(self, context):
            self.calls += 1
            assert context.snapshot_dir == "/snapshot"
            assert context.asof_dir == "/asof"
            assert context.asof_version == "7"
            return []

    executor = ViewExecutor()
    engine = _engine(tmp_path, executor)
    calls: list[str] = []

    def context_data(inference_at):
        calls.append(inference_at.isoformat())
        return StrategyDataView("/snapshot", "/asof", "7")

    engine.context_data = context_data
    result = engine.run_day("20260102")
    assert result["day_complete"] is True
    assert executor.calls == 1
    assert len(calls) == 1


def test_paper_uses_exact_static_price_without_minute_driven_strategy_calls(tmp_path: Path):
    executor = _TimedExecutor()
    engine = _engine(tmp_path, executor)
    quotes: list[tuple[str, str]] = []

    def execution_price(symbol, when):
        quotes.append((symbol, when.isoformat()))
        return 10.5

    engine.execution_price = execution_price
    result = engine.run_day("20260102")
    execution = json.loads(
        (tmp_path / "paper/executions_20260102.jsonl").read_text(encoding="utf-8")
    )
    assert result["day_complete"] is True
    assert executor.calls == 1
    assert quotes == [("000001.SZ", "2026-01-02T10:00:00+08:00")]
    assert execution["matched_at"] == "2026-01-02T10:00:00+08:00"
    assert execution["price"] == pytest.approx(10.50525)


def test_paper_refuses_to_skip_a_fixed_market_day_and_preserves_t_plus_one(tmp_path: Path):
    class RebalanceExecutor(_Executor):
        def execute(self, context):
            self.calls += 1
            trade_date = context.inference_at.strftime("%Y-%m-%d")
            if context.account.positions:
                return [{
                    "symbol": "000001.SZ",
                    "action": "sell",
                    "quantity": 100,
                    "execute_at": f"{trade_date}T09:30:00+08:00",
                }]
            return [
                {
                    "symbol": "000001.SZ",
                    "action": "buy",
                    "quantity": 100,
                    "execute_at": f"{trade_date}T09:30:00+08:00",
                },
                {
                    "symbol": "000001.SZ",
                    "action": "sell",
                    "quantity": 100,
                    "execute_at": f"{trade_date}T15:00:00+08:00",
                },
            ]

    strategy = tmp_path / "main.py"
    strategy.write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    daily = pd.DataFrame([
        {
            "trade_date": day,
            "symbol": "000001.SZ",
            "open": price,
            "close": price,
            "up_limit": price * 1.2,
            "down_limit": price * 0.8,
        }
        for day, price in (("20260102", 10.0), ("20260105", 11.0), ("20260106", 12.0))
    ])
    executor = RebalanceExecutor()
    engine = DailyPaperEngine(
        strategy_path=strategy,
        strategy_revision="revision_1",
        daily=daily,
        state_root=tmp_path / "paper",
        executor_factory=lambda _path, _config: executor,
    )

    engine.run_day("20260102")
    first_day_executions = [
        json.loads(line)
        for line in (tmp_path / "paper/executions_20260102.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(row["action"], row["status"], row["reason"]) for row in first_day_executions] == [
        ("buy", "filled", None),
        ("sell", "rejected", "insufficient_available_position"),
    ]
    with pytest.raises(PaperEngineError, match="next market trading day 20260105"):
        engine.run_day("20260106")
    assert not (tmp_path / "paper/executions_20260106.jsonl").exists()
    result = engine.run_day("20260105")

    assert result["position_count"] == 0
    executions = [
        json.loads(line)
        for line in (tmp_path / "paper/executions_20260105.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(row["action"], row["status"], row["reason"]) for row in executions] == [
        ("sell", "filled", None)
    ]


def test_paper_persists_and_freezes_schedule_and_strategy_path(tmp_path: Path):
    executor = _Executor()
    engine = _engine(tmp_path, executor)
    engine.run_day("20260102")
    daily = pd.DataFrame([
        {"trade_date": "20260102", "symbol": "000001.SZ", "open": 10.0, "close": 11.0}
    ])
    state = json.loads((tmp_path / "paper/.paper_state.json").read_text(encoding="utf-8"))
    assert state["schema_version"] == 3
    assert state["strategy_path"] == str((tmp_path / "main.py").resolve())
    assert state["schedule"] == {"period": "day", "inference_time": "08:30"}

    changed_schedule = DailyPaperEngine(
        strategy_path=tmp_path / "main.py",
        strategy_revision="revision_1",
        daily=daily,
        state_root=tmp_path / "paper",
        schedule=StrategySchedule("month", "09:00"),
        executor_factory=lambda _path, _config: executor,
    )
    with pytest.raises(PaperEngineError, match="strategy schedule differs"):
        changed_schedule.run_day("20260102")

    replacement = tmp_path / "replacement.py"
    replacement.write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    changed_path = DailyPaperEngine(
        strategy_path=replacement,
        strategy_revision="revision_1",
        daily=daily,
        state_root=tmp_path / "paper",
        executor_factory=lambda _path, _config: executor,
    )
    with pytest.raises(PaperEngineError, match="strategy path differs"):
        changed_path.run_day("20260102")


def test_paper_rejects_previous_state_schema_explicitly(tmp_path: Path):
    engine = _engine(tmp_path, _Executor())
    engine.run_day("20260102")
    path = tmp_path / "paper/.paper_state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["schema_version"] = 2
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(PaperEngineError, match="unsupported Paper state schema"):
        engine.run_day("20260102")
