from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from autotrade.agent import StrategyLoadError, load_strategy, validate_strategy_source
from autotrade.environment.broker import BrokerProfile
from autotrade.environment.replay import (
    BacktestError,
    DailyMarketData,
    StrategyDataView,
    run_daily_replay,
)
from autotrade.environment.strategy import (
    CN_TZ,
    AccountSnapshot,
    StrategyContext,
    StrategyContractError,
    StrategySchedule,
    validate_order_payload,
)
from autotrade.pipelines import DailyStrategyPipeline, StrategyExperimentConfig


def _daily() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": "20260102",
                "symbol": "000001.SZ",
                "open": 10.0,
                "close": 11.0,
                "up_limit": 12.0,
                "down_limit": 8.0,
            },
            {
                "trade_date": "20260105",
                "symbol": "000001.SZ",
                "open": 12.0,
                "close": 13.0,
                "up_limit": 14.4,
                "down_limit": 9.6,
            },
        ]
    )


def test_agent_output_template_passes_the_strategy_contract() -> None:
    template = Path("configs/agent_output_template/main.py")
    validate_strategy_source(template.read_text(encoding="utf-8"), filename=str(template))


@pytest.mark.parametrize(
    ("period", "current", "previous", "expected"),
    [
        ("day", "20260202", "20260130", True),
        ("month", "20260202", "20260130", True),
        ("month", "20260203", "20260202", False),
        ("quarter", "20260401", "20260331", True),
        ("quarter", "20260402", "20260401", False),
        ("year", "20260102", "20251231", True),
        ("year", "20260105", "20260102", False),
    ],
)
def test_schedule_runs_on_first_trading_day_of_period(period, current, previous, expected):
    schedule = StrategySchedule(period, "23:59")
    assert schedule.inference_time == "23:59"
    assert schedule.is_due(current, previous) is expected


def test_schedule_rejects_invalid_values():
    with pytest.raises(StrategyContractError, match="period"):
        StrategySchedule("week", "08:30")  # type: ignore[arg-type]
    with pytest.raises(StrategyContractError, match="HH:MM"):
        StrategySchedule("day", "24:00")


def test_order_json_contract_rejects_invalid_and_pre_inference_orders():
    inference = datetime(2026, 1, 2, 8, 30, tzinfo=CN_TZ)
    valid = {
        "symbol": "000001.SZ",
        "action": "buy",
        "quantity": 100,
        "execute_at": "2026-01-02T09:30:00+08:00",
    }
    assert validate_order_payload([valid], inference_at=inference)[0].symbol == "000001.SZ"
    for update, message in (
        ({"action": "hold"}, "action"),
        ({"quantity": 0}, "positive integer"),
        ({"quantity": True}, "positive integer"),
        ({"execute_at": "2026-01-02T08:29:00+08:00"}, "earlier"),
    ):
        with pytest.raises(StrategyContractError, match=message):
            validate_order_payload([{**valid, **update}], inference_at=inference)
    order = validate_order_payload([{**valid, "reason": "rebalance"}], inference_at=inference)[0]
    assert order.to_record()["reason"] == "rebalance"
    with pytest.raises(StrategyContractError, match="JSON array"):
        validate_order_payload({"orders": [valid]}, inference_at=inference)


def test_daily_market_hides_rows_until_available_at():
    market = DailyMarketData(_daily())
    morning = datetime(2026, 1, 2, 8, 30, tzinfo=CN_TZ)
    evening = datetime(2026, 1, 2, 17, 30, tzinfo=CN_TZ)
    assert market.visible_at(morning) == ()
    first = market.visible_at(evening)
    complete = market.visible_at(datetime(2026, 1, 5, 17, 30, tzinfo=CN_TZ))
    assert [row["trade_date"] for row in first] == ["20260102"]
    assert [row["trade_date"] for row in complete] == ["20260102", "20260105"]
    assert first[0] is complete[0]  # visibility slices reuse the once-normalized record
    assert not hasattr(first[0], "available_at")
    assert not hasattr(complete, "max_available_at")
    agent_context = StrategyContext(
        inference_at=datetime(2026, 1, 5, 17, 30, tzinfo=CN_TZ),
        bars=complete,
        account=AccountSnapshot(cash=1000, positions={}),
    )
    assert type(agent_context.bars) is tuple
    assert type(agent_context.bars[0]).__name__ == "mappingproxy"
    for bars in (complete, complete[::-1]):
        with pytest.raises(StrategyContractError, match="not visible"):
            StrategyContext(
                inference_at=evening,
                bars=bars,
                account=AccountSnapshot(cash=1000, positions={}),
            )


def test_daily_market_normalizes_missing_scalars_to_strict_json_null():
    daily = _daily().iloc[:1].assign(up_limit=float("nan"))
    market = DailyMarketData(daily)
    evening = datetime(2026, 1, 2, 17, 30, tzinfo=CN_TZ)
    assert market.visible_at(evening)[0]["up_limit"] is None


def test_context_rejects_future_data_and_exposes_no_environment_handles():
    with pytest.raises(StrategyContractError, match="not visible"):
        StrategyContext(
            inference_at=datetime(2026, 1, 2, 8, 30, tzinfo=CN_TZ),
            bars=(
                {
                    "symbol": "000001.SZ",
                    "available_at": "2026-01-02T17:30:00+08:00",
                },
            ),
            account=AccountSnapshot(cash=1000, positions={}),
        )
    context = StrategyContext(
        inference_at=datetime(2026, 1, 2, 8, 30, tzinfo=CN_TZ),
        bars=(),
        account=AccountSnapshot(cash=1000, positions={}),
    )
    assert not hasattr(context, "broker")
    assert not hasattr(context, "shell")
    assert not hasattr(context, "web")


def test_context_record_round_trip_and_host_nl_pit_validation():
    inference_at = datetime(2026, 1, 2, 18, 0, tzinfo=CN_TZ)
    observed = []

    def nl_query(request, *, inference_at):
        observed.append((request, inference_at))
        return {
            "text": "known",
            "available_at": "2026-01-02T17:30:00+08:00",
        }

    context = StrategyContext(
        inference_at=inference_at,
        bars=(
            {
                "symbol": "000001.SZ",
                "available_at": "2026-01-02T17:30:00+08:00",
                "close": 11.0,
            },
        ),
        account=AccountSnapshot(cash=1000, positions={"000001.SZ": 100}),
        snapshot_dir="/strategy-data/snapshot",
        asof_dir="/strategy-data/asof",
        asof_version="3",
        _nl_query=nl_query,
    )
    record = context.to_record()
    assert "_nl_query" not in record
    restored = StrategyContext.from_record(record, nl_query=nl_query)
    assert restored.to_record() == record
    assert restored.snapshot_dir == "/strategy-data/snapshot"
    assert restored.asof_dir == "/strategy-data/asof"
    assert restored.asof_version == "3"
    nl_result = restored.nl(question="status")
    assert type(nl_result) is dict
    assert nl_result["text"] == "known"
    assert observed == [({"question": "status"}, inference_at)]

    def future_query(_request, *, inference_at):
        return {"available_at": "2026-01-03T17:30:00+08:00"}

    future = StrategyContext.from_record(record, nl_query=future_query)
    with pytest.raises(StrategyContractError, match="not visible"):
        future.nl(question="future")
    with pytest.raises(StrategyContractError, match="not configured"):
        StrategyContext.from_record(record).nl(question="missing")
    with pytest.raises(StrategyContractError, match="unsupported fields"):
        StrategyContext.from_record({**record, "broker": {}})


def test_scheduled_replay_executes_orders_at_their_exact_requested_times_with_pit():
    observed: list[tuple[str, int]] = []

    def generate_orders(context):
        observed.append((context.inference_at.date().isoformat(), len(context.bars)))
        position = context.account.positions.get("000001.SZ", 0)
        action = "sell" if position else "buy"
        return [
            {
                "symbol": "000001.SZ",
                "action": action,
                "quantity": 100,
                "execute_at": context.inference_at.replace(
                    hour=10 if position else 9,
                    minute=0 if position else 30,
                ).isoformat(),
            }
        ]

    result = run_daily_replay(
        daily=_daily(),
        strategy=generate_orders,
        schedule=StrategySchedule("day", "08:30"),
        profile=BrokerProfile(initial_cash=10_000, min_commission_cny=0, slippage_bps=0),
        execution_price=lambda symbol, when: (
            12.5
            if symbol == "000001.SZ" and when.isoformat() == "2026-01-05T10:00:00+08:00"
            else None
        ),
    )
    assert observed == [("2026-01-02", 0), ("2026-01-05", 1)]
    assert [(row["action"], row["matched_at"][11:16], row["status"]) for row in result.executions] == [
        ("buy", "09:30", "filled"),
        ("sell", "10:00", "filled"),
    ]
    assert result.pending_orders == ()


def test_daily_replay_injects_read_only_pit_directory_references():
    observed = []

    def generate_orders(context):
        observed.append((context.snapshot_dir, context.asof_dir, context.asof_version))
        return []

    run_daily_replay(
        daily=_daily().iloc[:1],
        strategy=generate_orders,
        schedule=StrategySchedule("day", "08:30"),
        context_data=lambda _when: StrategyDataView("/snapshot", "/asof", "7"),
    )
    assert observed == [("/snapshot", "/asof", "7")]


def test_daily_replay_enforces_t_plus_one_and_buy_lots():
    def generate_orders(context):
        return [
            {
                "symbol": "000001.SZ",
                "action": "buy",
                "quantity": 50,
                "execute_at": "2026-01-02T09:30:00+08:00",
            },
            {
                "symbol": "000001.SZ",
                "action": "sell",
                "quantity": 50,
                "execute_at": "2026-01-02T15:00:00+08:00",
            },
        ]

    result = run_daily_replay(
        daily=_daily().iloc[:1],
        strategy=generate_orders,
        schedule=StrategySchedule("month", "08:30"),
    )
    assert [row["reason"] for row in result.executions] == [
        "invalid_buy_lot",
        "insufficient_available_position",
    ]


def test_order_without_an_exact_price_is_rejected_instead_of_shifted_to_an_event():
    calls = 0

    def generate_orders(context):
        nonlocal calls
        calls += 1
        if calls > 1:
            return []
        return [
            {
                "symbol": "000001.SZ",
                "action": "buy",
                "quantity": 100,
                "execute_at": context.inference_at.isoformat(),
            }
        ]

    result = run_daily_replay(
        daily=_daily(),
        strategy=generate_orders,
        schedule=StrategySchedule("day", "20:00"),
    )
    assert result.executions[0]["matched_at"].startswith("2026-01-02T20:00")
    assert result.executions[0]["status"] == "rejected"
    assert result.executions[0]["reason"] == "missing_execution_price"


def test_order_without_a_daily_bar_uses_the_missing_price_contract():
    def generate_orders(context):
        return [{
            "symbol": "000002.SZ",
            "action": "buy",
            "quantity": 100,
            "execute_at": context.inference_at.replace(hour=9, minute=30).isoformat(),
        }]

    result = run_daily_replay(
        daily=_daily().iloc[:1],
        strategy=generate_orders,
        schedule=StrategySchedule("day", "08:30"),
    )
    assert result.executions[0]["status"] == "rejected"
    assert result.executions[0]["reason"] == "missing_execution_price"


def test_strategy_loader_blocks_environment_and_network_modules(tmp_path: Path):
    with pytest.raises(StrategyLoadError, match="unsupported module"):
        validate_strategy_source("import requests\ndef generate_orders(context): return []")
    with pytest.raises(StrategyLoadError, match="forbidden builtin"):
        validate_strategy_source("def generate_orders(context):\n    open('x')\n    return []")
    with pytest.raises(StrategyLoadError, match="external I/O"):
        validate_strategy_source("import pandas as pd\ndef generate_orders(context): return pd.read_json('x')")
    validate_strategy_source(
        "import pandas as pd\ndef generate_orders(context):\n"
        "    pd.read_parquet(context.asof_dir + '/daily.parquet')\n    return []"
    )
    with pytest.raises(StrategyLoadError, match="only below"):
        validate_strategy_source(
            "import pandas as pd\ndef generate_orders(context):\n"
            "    pd.read_parquet('/etc/passwd')\n    return []"
        )
    path = tmp_path / "main.py"
    path.write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    assert load_strategy(path)(object()) == []


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np\ndef generate_orders(context): return np.load('x.npy')",
        "import numpy as np\ndef generate_orders(context): return np.genfromtxt('x.csv')",
        "import numpy as np\ndef generate_orders(context): np.savetxt('x.csv', [])\n",
        "import pandas as pd\ndef generate_orders(context): return pd.read_fwf('x.txt')",
    ],
)
def test_strategy_loader_blocks_known_numpy_and_pandas_file_io(source: str):
    with pytest.raises(StrategyLoadError, match="external I/O"):
        validate_strategy_source(source)


def test_pipeline_loads_strategy_and_returns_in_memory_result(tmp_path: Path):
    path = tmp_path / "main.py"
    path.write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    config = StrategyExperimentConfig(
        strategy_path=path,
        schedule=StrategySchedule("quarter", "12:00"),
        execution_mode="trusted",
    )
    result = DailyStrategyPipeline(config).run(_daily())
    assert len(result.inference_dates) == 1
    assert result.executions == ()


def test_strategy_failure_is_explicit():
    def generate_orders(_context):
        return [{"symbol": "x"}]

    with pytest.raises(BacktestError, match="generate_orders failed"):
        run_daily_replay(
            daily=_daily().iloc[:1],
            strategy=generate_orders,
            schedule=StrategySchedule(),
        )
