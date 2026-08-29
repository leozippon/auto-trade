"""Columnar replay market data: exact Broker/strategy values, lazily.

The golden record in ``data/replay_market_golden.json`` was produced by the
earlier per-row implementation of ``DailyMarketData`` on the synthetic frame
below (``python -m tests.unit.test_replay_market --write-golden`` regenerates it
only when the replay contract itself changes). The replay exercises every
Broker gate that reads a bar (suspension, missing bar, NaN limit, limit-up,
lot ladders, T+1, the stamp-duty cutover) and pins the strategy-visible
``bars``/``history``/``latest`` values through order metadata, so the columnar
store must reproduce the equity curve, fills, rejections and statistics
bit for bit.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.replay import DailyMarketData, run_daily_replay
from autotrade.environment.strategy import (
    CN_TZ,
    AccountSnapshot,
    StrategyContext,
    StrategyContractError,
    StrategySchedule,
)

GOLDEN = Path(__file__).with_name("data") / "replay_market_golden.json"
SYMBOLS = ("000001.SZ", "600000.SH", "688001.SH", "830000.BJ", "300001.SZ")
DAYS = tuple(stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2023-08-21", periods=14))


def synthetic_daily() -> pd.DataFrame:
    """Five names over fourteen days across the 2023-08-28 stamp-duty cutover."""

    rows: list[dict[str, object]] = []
    close = {symbol: 10.0 + 5.0 * index for index, symbol in enumerate(SYMBOLS)}
    for day_index, trade_date in enumerate(DAYS):
        for index, symbol in enumerate(SYMBOLS):
            if day_index == 4 and symbol == "830000.BJ":
                continue  # no bar at all that day
            pre_close = close[symbol]
            drift = ((day_index * 7 + index * 3) % 11 - 5) / 100.0
            close[symbol] = round(pre_close * (1.0 + drift), 2)
            band = 0.2 if symbol.startswith("688") else 0.3 if symbol.endswith(".BJ") else 0.1
            up_limit = round(pre_close * (1.0 + band), 2)
            down_limit = round(pre_close * (1.0 - band), 2)
            open_price = round(pre_close * (1.0 + drift / 2.0), 2)
            if day_index == 5 and symbol == "000001.SZ":
                open_price = up_limit  # opens limit-up
            if day_index == 3 and symbol == "600000.SH":
                up_limit = float("nan")
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    "open": open_price,
                    "high": max(open_price, close[symbol]),
                    "low": min(open_price, close[symbol]),
                    "close": close[symbol],
                    "vol": float(1_000_000 + 1_000 * day_index + index),
                    "up_limit": up_limit,
                    "down_limit": down_limit,
                    "adj_factor": 1.0 + day_index / 100.0,
                    "is_suspended": day_index == 2 and symbol == "300001.SZ",
                    "rank": index + 1,
                    "name": None if symbol == "300001.SZ" else f"name-{symbol}",
                    "note": None,
                    "available_at": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}T17:30:00+08:00",
                }
            )
    return pd.DataFrame(rows)


_SCRIPT: dict[str, tuple[tuple[str, str, int, str, int], ...]] = {
    # trade_date -> (symbol, action, quantity, execute time, day offset)
    DAYS[0]: (("000001.SZ", "buy", 100, "09:30", 0),),
    DAYS[1]: (
        ("688001.SH", "buy", 200, "15:00", 0),
        ("830000.BJ", "buy", 150, "15:00", 0),
        ("000001.SZ", "sell", 100, "09:30", 0),
    ),
    DAYS[2]: (("300001.SZ", "buy", 100, "15:00", 0), ("000001.SZ", "buy", 100, "10:00", 0)),
    DAYS[3]: (("600000.SH", "buy", 100, "09:30", 0),),
    DAYS[4]: (("830000.BJ", "buy", 100, "15:00", 0), ("688001.SH", "sell", 100, "09:30", 0)),
    DAYS[5]: (("000001.SZ", "buy", 100, "09:30", 0), ("688001.SH", "sell", 200, "15:00", 0)),
    DAYS[6]: (("300001.SZ", "buy", 100, "09:30", 0), ("600000.SH", "buy", 300, "15:00", 0)),
    DAYS[7]: (("830000.BJ", "sell", 150, "09:30", 0),),
    DAYS[8]: (("000001.SZ", "buy", 1_000_000, "15:00", 0),),
    DAYS[9]: (("300001.SZ", "sell", 100, "15:00", 0), ("600000.SH", "sell", 100, "15:00", 0)),
    DAYS[10]: (("688001.SH", "buy", 201, "09:30", 0),),
    DAYS[11]: (("000001.SZ", "buy", 150, "09:30", 0),),
    DAYS[12]: (("600000.SH", "buy", 100, "09:30", 1),),
    DAYS[13]: (("600000.SH", "sell", 200, "15:00", 0), ("000001.SZ", "buy", 100, "09:30", 1)),
}


def scripted_strategy(context: StrategyContext) -> list[dict[str, object]]:
    trade_date = context.inference_at.strftime("%Y%m%d")
    orders = []
    for symbol, action, quantity, clock, offset in _SCRIPT.get(trade_date, ()):
        day = pd.Timestamp(trade_date) + pd.offsets.BDay(offset)
        hour, minute = (int(part) for part in clock.split(":"))
        latest = context.latest(symbol)
        orders.append(
            {
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "execute_at": datetime(day.year, day.month, day.day, hour, minute, tzinfo=CN_TZ).isoformat(),
                "seen": len(context.bars),
                "history": len(context.history(symbol)),
                "latest_close": None if latest is None else latest["close"],
                "latest_name": None if latest is None else latest.get("name"),
            }
        )
    return orders


def replay_record() -> dict[str, object]:
    result = run_daily_replay(
        daily=synthetic_daily(),
        strategy=scripted_strategy,
        schedule=StrategySchedule("day", "08:30"),
        profile=BrokerProfile(initial_cash=200_000.0),
    )
    record = result.to_record()
    stats = record["stats"]
    assert isinstance(stats, dict)
    for key in ("replay_wall_seconds", "phase_seconds"):
        stats.pop(key, None)
    return json.loads(json.dumps(record, allow_nan=False))


def test_columnar_market_reproduces_the_golden_replay_exactly() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    record = replay_record()
    assert record["executions"] == golden["executions"]
    assert record["equity_curve"] == golden["equity_curve"]
    assert record["pending_orders"] == golden["pending_orders"]
    assert record["inference_dates"] == golden["inference_dates"]
    assert record["stats"] == golden["stats"]
    # The scenario really covers the bar-reading gates it claims to.
    reasons = {item["reason"] for item in record["executions"] if item["status"] == "rejected"}
    assert {
        "suspended",
        "missing_execution_price",
        "missing_daily_price_limit",
        "daily_price_limit",
        "amount_below_lot_size",
        "insufficient_cash",
        "invalid_buy_lot",
    } <= reasons
    assert len(record["pending_orders"]) == 1


def test_day_bars_and_visibility_are_lazy_views_over_one_columnar_store() -> None:
    frame = synthetic_daily()
    market = DailyMarketData(frame)
    assert market.trade_dates == DAYS
    day = market.bars_for_day(DAYS[3])
    assert len(day) == len(SYMBOLS)
    assert set(day) == set(SYMBOLS)
    bar = day["600000.SH"]
    assert bar["up_limit"] is None  # NaN became a strict JSON null
    assert bar["is_suspended"] is False
    assert bar["rank"] == 2 and type(bar["rank"]) is int
    assert bar["note"] is None and bar["name"] == "name-600000.SH"
    assert list(bar) == list(frame.columns.str.replace("ts_code", "symbol"))
    assert day.get("missing") is None
    assert market.bars_for_day("19990101") == {}
    assert "830000.BJ" not in market.bars_for_day(DAYS[4])
    with pytest.raises(TypeError):
        bar["close"] = 1.0  # type: ignore[index]
    with pytest.raises(StrategyContractError, match="timezone"):
        market.visible_at(datetime(2023, 8, 22, 8, 30))  # noqa: DTZ001 - naive on purpose

    cutoff = datetime(2023, 8, 23, 17, 30, tzinfo=CN_TZ)
    visible = market.visible_at(cutoff)
    assert [row["trade_date"] for row in visible] == sorted(row["trade_date"] for row in visible)
    assert len(visible) == 3 * len(SYMBOLS)
    assert visible[-1] is market.visible_at(cutoff)[-1]
    context = StrategyContext(
        inference_at=cutoff,
        bars=visible,
        account=AccountSnapshot(cash=1.0, positions={}),
    )
    assert type(context.bars) is tuple
    brute = tuple(row for row in context.bars if row["symbol"] == "688001.SH")
    assert context.history("688001.SH") == brute
    assert context.history("688001.SH")[0] is brute[0]
    assert context.latest("688001.SH") is brute[-1]
    assert context.history("nope") == ()
    assert dict(context.bars[0]) == json.loads(json.dumps(dict(context.bars[0])))
    with pytest.raises(StrategyContractError, match="not visible"):
        StrategyContext(
            inference_at=datetime(2023, 8, 23, 8, 30, tzinfo=CN_TZ),
            bars=visible,
            account=AccountSnapshot(cash=1.0, positions={}),
        )


def test_market_normalizes_keys_and_rejects_duplicates_and_non_json_values() -> None:
    frame = synthetic_daily().iloc[:4].copy()
    frame["trade_date"] = ["2023-08-21", "20230821", " 2023-08-21", "2023-08-21"]
    frame["ts_code"] = [" 000001.SZ", "600000.SH ", "688001.SH", "830000.BJ"]
    frame = frame.drop(columns=["available_at"])
    market = DailyMarketData(frame)
    assert market.trade_dates == ("20230821",)
    assert set(market.bars_for_day("20230821")) == {"000001.SZ", "600000.SH", "688001.SH", "830000.BJ"}
    assert market.visible_at(datetime(2023, 8, 21, 17, 29, tzinfo=CN_TZ)) == ()
    first = market.visible_at(datetime(2023, 8, 21, 17, 30, tzinfo=CN_TZ))
    assert [row["symbol"] for row in first] == ["000001.SZ", "600000.SH", "688001.SH", "830000.BJ"]
    assert first[0]["available_at"] == "2023-08-21T17:30:00+08:00"

    duplicated = frame.copy()
    duplicated.loc[1, "ts_code"] = "000001.SZ"
    with pytest.raises(ValueError, match="duplicate business keys"):
        DailyMarketData(duplicated)
    empty = frame.copy()
    empty.loc[1, "ts_code"] = "  "
    with pytest.raises(ValueError, match="empty symbol"):
        DailyMarketData(empty)
    infinite = frame.copy()
    infinite.loc[1, "close"] = np.inf
    with pytest.raises(StrategyContractError, match="JSON"):
        DailyMarketData(infinite)
    with pytest.raises(ValueError, match="missing columns"):
        DailyMarketData(frame.drop(columns=["open"]))
    with pytest.raises(ValueError, match="invalid trade_date"):
        DailyMarketData(frame.assign(trade_date="2023-13-01"))
    naive = frame.assign(available_at="2023-08-21T17:30:00")
    with pytest.raises(ValueError, match="timezone"):
        DailyMarketData(naive)
    empty = DailyMarketData(frame.iloc[:0])
    assert empty.trade_dates == () and empty.bars_for_day("20230821") == {}
    assert empty.visible_at(datetime(2023, 8, 21, 17, 30, tzinfo=CN_TZ)) == ()


def test_bar_values_are_strict_json_scalars_including_nan_and_infinity_paths() -> None:
    frame = synthetic_daily().iloc[:2].copy()
    frame["flag"] = pd.array([True, None], dtype="boolean")
    frame["count"] = pd.array([1, None], dtype="Int64")
    frame["stamp"] = pd.to_datetime(["2023-08-21 09:00", "2023-08-21 10:00"]).tz_localize("UTC")
    frame["small"] = np.array([0.1, 0.2], dtype=np.float32)
    market = DailyMarketData(frame)
    bars = market.bars_for_day("20230821")
    first, second = bars["000001.SZ"], bars["600000.SH"]
    assert first["flag"] is True and second["flag"] is None
    assert first["count"] == 1 and type(first["count"]) is int and second["count"] is None
    assert first["stamp"] == "2023-08-21T17:00:00+08:00"
    assert first["small"] == float(np.float32(0.1))
    assert math.isfinite(first["vol"]) and type(first["vol"]) is float


if __name__ == "__main__":
    if sys.argv[1:] != ["--write-golden"]:
        raise SystemExit("usage: python -m tests.unit.test_replay_market --write-golden")
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(replay_record(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN}")
