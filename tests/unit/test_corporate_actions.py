"""Ex-date settlement through the replay, the null control and Paper.

The Broker's own rules are pinned in ``test_broker_engine``; here the slot's
``corporate_actions`` table has to reach the Broker on the right day through
every path that carries a position across days, and a frame that cannot say
what the exchange reference price was must stop the replay instead of
silently marking a bonus issue as a loss.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.replay import DailyMarketData, run_daily_replay
from autotrade.environment.replay.null_control import run_null_control
from autotrade.environment.strategy import StrategySchedule
from autotrade.paper import DailyPaperEngine

DAYS = ("20260105", "20260106", "20260107")
SYMBOL = "000001.SZ"
PROFILE = BrokerProfile(
    initial_cash=100_000.0, min_commission_cny=0, slippage_bps=0, transfer_fee_bps=0
)


def _frame() -> pd.DataFrame:
    """Two names; ``000001.SZ`` goes 10 送 10 with 0.5 CNY cash on the second day."""

    rows = []
    for day, close, pre_close in zip(DAYS, (10.0, 4.9, 5.0), (9.9, 4.75, 4.9), strict=True):
        rows.append(
            {
                "ts_code": SYMBOL,
                "trade_date": day,
                "open": pre_close * 1.01,
                "close": close,
                "pre_close": pre_close,
                "up_limit": round(pre_close * 1.1, 2),
                "down_limit": round(pre_close * 0.9, 2),
                "circ_mv": 1.0e9,
            }
        )
    for day, close, pre_close in zip(DAYS, (20.0, 20.2, 20.1), (19.9, 20.0, 20.2), strict=True):
        rows.append(
            {
                "ts_code": "000002.SZ",
                "trade_date": day,
                "open": pre_close,
                "close": close,
                "pre_close": pre_close,
                "up_limit": round(pre_close * 1.1, 2),
                "down_limit": round(pre_close * 0.9, 2),
                "circ_mv": 1.0e9,
            }
        )
    return pd.DataFrame(rows)


def _actions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": SYMBOL,
                "ex_date": DAYS[1],
                "record_date": DAYS[0],
                "pay_date": DAYS[1],
                "div_listdate": DAYS[1],
                "cash_per_share": 0.5,
                "stock_per_share": 1.0,
            }
        ]
    )


def _buy_once(context):
    if context.account.positions:
        return []
    return [
        {
            "symbol": SYMBOL,
            "action": "buy",
            "quantity": 1000,
            "execute_at": context.inference_at.replace(hour=15, minute=0).isoformat(),
        }
    ]


def test_replay_settles_the_ex_date_from_pre_close_and_the_slot_cash_table():
    result = run_daily_replay(
        daily=_frame(),
        strategy=_buy_once,
        schedule=StrategySchedule("day", "08:30"),
        profile=PROFILE,
        corporate_actions=_actions(),
    )
    # 1000 shares bought at the day-1 close of 10.0 for 10 000 plus 1 bp fee.
    cash_after_buy = 100_000.0 - 10_000.0 - 1.0
    assert result.equity_curve[0]["positions"] == {SYMBOL: 1000}
    assert result.equity_curve[0]["cash"] == pytest.approx(cash_after_buy)
    # Ex-date: (10 - 0.5) / 4.75 = 2 shares per share, 500 CNY of cash.
    assert result.equity_curve[1]["positions"] == {SYMBOL: 2000}
    assert result.equity_curve[1]["cash"] == pytest.approx(cash_after_buy + 500.0)
    assert result.equity_curve[1]["equity"] == pytest.approx(cash_after_buy + 500.0 + 2000 * 4.9)
    assert result.corporate_actions == (
        {
            "trade_date": DAYS[1],
            "symbol": SYMBOL,
            "last_close": 10.0,
            "pre_close": 4.75,
            "cash_per_share": 0.5,
            "quantity_before": 1000,
            "quantity_after": 2000,
            "cash_credit": pytest.approx(500.0),
        },
    )
    record = result.to_record()
    assert record["corporate_actions"] == list(result.corporate_actions)
    # A day-3 reference price equal to the day-2 close settles nothing.
    assert result.equity_curve[2]["positions"] == {SYMBOL: 2000}


def test_the_cash_table_is_validated_and_only_read_for_its_cash_leg():
    market = DailyMarketData(_frame(), _actions())
    assert market.cash_dividends_for_day(DAYS[1]) == {SYMBOL: 0.5}
    assert market.cash_dividends_for_day(DAYS[0]) == {}
    # A pure bonus issue carries no cash and nothing the Broker reads.
    assert DailyMarketData(_frame(), _actions().assign(cash_per_share=0.0)).cash_dividends_for_day(DAYS[1]) == {}
    # Two events of one name on one ex-date share the record-date base.
    doubled = pd.concat([_actions(), _actions().assign(cash_per_share=0.25)], ignore_index=True)
    assert DailyMarketData(_frame(), doubled).cash_dividends_for_day(DAYS[1]) == {SYMBOL: 0.75}
    with pytest.raises(ValueError, match="missing columns"):
        DailyMarketData(_frame(), _actions().drop(columns=["cash_per_share"]))
    for bad in (float("nan"), -0.1, None):
        with pytest.raises(ValueError, match="cash_per_share"):
            DailyMarketData(_frame(), _actions().assign(cash_per_share=bad))
    with pytest.raises(ValueError, match="prebuilt"):
        run_daily_replay(
            daily=market,
            strategy=_buy_once,
            schedule=StrategySchedule("day", "08:30"),
            profile=PROFILE,
            corporate_actions=_actions(),
        )


def test_the_null_control_replays_its_draws_through_the_same_ex_dates():
    result = run_daily_replay(
        daily=_frame(),
        strategy=_buy_once,
        schedule=StrategySchedule("day", "08:30"),
        profile=PROFILE,
        corporate_actions=_actions(),
    )
    block = run_null_control(
        result,
        _frame(),
        {day: 0.0 for day in DAYS},
        PROFILE,
        StrategySchedule("day", "08:30"),
        k=2,
        seed=5,
        corporate_actions=_actions(),
    )
    # The only replacement name never goes ex, so the observed run, which
    # kept its bonus shares and cash, ranks at the top of the null.
    assert block["rejects_mean"] == 0.0
    assert block["excess_percentile"] == 1.0
    assert block["unpaired_sell_shares"] == 0


def test_selling_settlement_created_shares_keeps_the_null_control_available():
    """Buy 1000, 10 送 10, sell 2000: the extra 1000 shares came from the
    ex-date, not a fill, so the skeleton keeps the bought 1000 as its round
    trip, reports the rest, and the null still replays."""

    def strategy(context):
        day = context.inference_at.strftime("%Y%m%d")
        if day == DAYS[0]:
            return _buy_once(context)
        if day == DAYS[2]:
            return [
                {
                    "symbol": SYMBOL,
                    "action": "sell",
                    "quantity": 2000,
                    "execute_at": context.inference_at.replace(hour=15, minute=0).isoformat(),
                }
            ]
        return []

    result = run_daily_replay(
        daily=_frame(),
        strategy=strategy,
        schedule=StrategySchedule("day", "08:30"),
        profile=PROFILE,
        corporate_actions=_actions(),
    )
    assert [row["status"] for row in result.executions] == ["filled", "filled"]
    assert result.equity_curve[-1]["positions"] == {}

    block = run_null_control(
        result,
        _frame(),
        {day: 0.0 for day in DAYS},
        PROFILE,
        StrategySchedule("day", "08:30"),
        k=2,
        seed=3,
        corporate_actions=_actions(),
    )
    assert block["unpaired_sell_shares"] == 1000
    assert block["k"] == 2 and block["rejects_mean"] == 0.0


def test_a_frame_without_pre_close_is_refused_up_front():
    with pytest.raises(ValueError, match=r"missing columns: \['pre_close'\]"):
        DailyMarketData(_frame().drop(columns=["pre_close"]))


def test_paper_settles_the_ex_date_when_the_day_opens_and_journals_it(tmp_path: Path):
    class BuyOnce:
        def execute(self, context):
            return _buy_once(context)

        def close(self) -> None:
            pass

    strategy = tmp_path / "main.py"
    strategy.write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    engine = DailyPaperEngine(
        strategy_path=strategy,
        strategy_revision="revision_1",
        daily=_frame(),
        corporate_actions=_actions(),
        state_root=tmp_path / "paper",
        profile=PROFILE,
        executor_factory=lambda *_mounts: BuyOnce(),
    )
    engine.run_day(DAYS[0])
    summary = engine.run_day(DAYS[1])
    assert summary["position_count"] == 1
    state = json.loads((tmp_path / "paper" / ".paper_state.json").read_text(encoding="utf-8"))
    [position] = state["account"]["positions"]
    assert position["quantity"] == 2000
    assert position["available_quantity"] == 1000
    assert state["account"]["cash"] == pytest.approx(100_000.0 - 10_001.0 + 500.0)
    journal = tmp_path / "paper" / f"corporate_actions_{DAYS[1]}.jsonl"
    [line] = journal.read_text(encoding="utf-8").splitlines()
    entry = json.loads(line)
    assert entry["kind"] == "corporate_action"
    assert (entry["quantity_before"], entry["quantity_after"]) == (1000, 2000)
    assert entry["cash_credit"] == pytest.approx(500.0)
    # Re-running the finished day is idempotent: no second settlement.
    assert engine.run_day(DAYS[1]) == summary
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 1
