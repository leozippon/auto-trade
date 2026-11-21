"""Broker orders and bars, and the random-baseline draws over a board-mixed market."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from autotrade.environment.replay.market import DailyMarketData
from autotrade.environment.replay.null_control import (
    RoundTrip,
    _candidate_pool,
    _orders_from_pools,
    _Universe,
)
from autotrade.environment.strategy import CN_TZ, StrategyOrder

DAYS = tuple(pd.bdate_range("2026-01-05", periods=20).strftime("%Y%m%d"))


def _sellable(universe, frame: pd.DataFrame, corporate_actions=None):
    market = DailyMarketData(frame, corporate_actions)
    return lambda trip, symbol, quantity: universe.sellable_at_exit(
        trip, symbol, quantity, market.cash_dividends_for_day
    )


def _at(day: str, clock: str) -> datetime:
    hour, minute = (int(part) for part in clock.split(":"))
    return datetime(int(day[:4]), int(day[4:6]), int(day[6:8]), hour, minute, tzinfo=CN_TZ)


BOARD_NAMES = (
    "000001.SZ", "000002.SZ", "600001.SH",  # main
    "300001.SZ", "300002.SZ",  # ChiNext
    "688001.SH", "688002.SH",  # STAR
    "830001.BJ", "830002.BJ",  # Beijing
)


def _board_frame(days: int = 6) -> pd.DataFrame:
    """``days`` days of names on every board, one size bucket, no ex-date."""

    rows = []
    for offset, day in enumerate(DAYS[:days]):
        for index, symbol in enumerate(BOARD_NAMES):
            price = 10.0 + index + 0.1 * offset
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": day,
                    "open": price,
                    "close": price,
                    "pre_close": 10.0 + index + 0.1 * (offset - 1) if offset else price,
                    "up_limit": price * 1.5,
                    "down_limit": price * 0.5,
                    "is_suspended": False,
                    "circ_mv": 1.0e9,
                }
            )
    return pd.DataFrame(rows)


_BOARD_SKELETON = (
    RoundTrip("000001.SZ", 1000, 10.0, _at(DAYS[0], "09:30"), _at(DAYS[3], "15:00")),
    RoundTrip("300001.SZ", 500, 13.1, _at(DAYS[1], "09:30"), None),
)


def _board_draws(permitted_boards, draws: int, seed: int) -> list[list[tuple[str, str, int]]]:
    frame = _board_frame()
    universe = _Universe(frame)
    pools = [_candidate_pool(trip, universe, None, permitted_boards) for trip in _BOARD_SKELETON]
    result = []
    for draw in range(draws):
        orders, _dropped = _orders_from_pools(
            _BOARD_SKELETON, pools, seed, draw, _sellable(universe, frame), universe.carried_limit
        )
        result.append(
            [(order["symbol"], order["action"], order["quantity"]) for day in sorted(orders) for order in orders[day]]
        )
    return result


MATCHED_AT = datetime(2026, 1, 5, 9, 30, tzinfo=CN_TZ)


def _order(action: str = "buy", *, symbol: str = "000001.SZ", quantity: int = 100) -> StrategyOrder:
    return StrategyOrder(
        symbol=symbol,
        action=action,
        quantity=quantity,
        execute_at=MATCHED_AT,
    )


def _bar(**overrides: object) -> dict[str, object]:
    bar = {
        "open": 10.0,
        "close": 10.0,
        "pre_close": 10.0,
        "up_limit": 11.0,
        "down_limit": 9.0,
        "is_suspended": False,
    }
    bar.update(overrides)
    return bar
