"""Random-portfolio null control: skeleton pairing, matched draws, replay block.

The synthetic market is five names over 20 trading days in two size groups, so
a replacement is always drawn from the original's own ``circ_mv`` decile and
never from the other group.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.replay import run_daily_replay
from autotrade.environment.replay.null_control import (
    RoundTrip,
    _distribution,
    run_null_control,
    sample_null_orders,
    trade_skeleton,
)
from autotrade.environment.replay.stats import ReplayResult
from autotrade.environment.strategy import CN_TZ, StrategySchedule

DAYS = tuple(pd.bdate_range("2026-01-05", periods=20).strftime("%Y%m%d"))
SMALL = ("000001.SZ", "000002.SZ", "000003.SZ")
LARGE = ("600001.SH", "600002.SH")
CIRC_MV = {**{name: 1.0e9 for name in SMALL}, **{name: 5.0e10 for name in LARGE}}
BENCHMARK = {day: 0.001 for day in DAYS}


def _at(day: str, clock: str) -> datetime:
    hour, minute = (int(part) for part in clock.split(":"))
    return datetime(int(day[:4]), int(day[4:6]), int(day[6:8]), hour, minute, tzinfo=CN_TZ)


def _frame() -> pd.DataFrame:
    rows = []
    for offset, day in enumerate(DAYS):
        for index, (symbol, circ_mv) in enumerate(CIRC_MV.items()):
            price = 10.0 + 2.0 * index + 0.1 * offset
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": day,
                    "open": price,
                    "close": price * 1.01,
                    "up_limit": price * 1.5,
                    "down_limit": price * 0.5,
                    "is_suspended": False,
                    "circ_mv": circ_mv,
                }
            )
    return pd.DataFrame(rows)


_SCRIPT = {
    DAYS[2]: (("000001.SZ", "buy", 1000, _at(DAYS[2], "09:30")),),
    DAYS[4]: (("600001.SH", "buy", 500, _at(DAYS[4], "09:30")),),
    DAYS[9]: (("000001.SZ", "sell", 1000, _at(DAYS[9], "15:00")),),
}


def _observed_result():
    def strategy(context):
        day = context.inference_at.strftime("%Y%m%d")
        return [
            {
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "execute_at": when.isoformat(),
            }
            for symbol, action, quantity, when in _SCRIPT.get(day, ())
        ]

    return run_daily_replay(
        daily=_frame(),
        strategy=strategy,
        schedule=StrategySchedule("day", "08:30"),
        profile=BrokerProfile(),
    )


def _expected_quantity(frame: pd.DataFrame, symbol: str, day: str, notional: float) -> int:
    bar = frame[(frame["ts_code"] == symbol) & (frame["trade_date"] == day)]
    shares = int(notional // float(bar["open"].iloc[0]))
    return shares - shares % 100


def test_trade_skeleton_pairs_fills_fifo_and_keeps_the_open_position():
    executions = [
        _fill("000001.SZ", "buy", 1000, DAYS[0], "09:30", 10.0),
        _fill("600001.SH", "buy", 500, DAYS[1], "09:30", 20.0),
        _fill("000001.SZ", "buy", 600, DAYS[2], "09:30", 12.0),
        {**_fill("000001.SZ", "sell", 900, DAYS[3], "15:00", 11.0), "status": "rejected"},
        _fill("000001.SZ", "sell", 900, DAYS[4], "15:00", 11.0),
        _fill("000001.SZ", "sell", 700, DAYS[6], "15:00", 13.0),
    ]

    skeleton = trade_skeleton(executions)

    assert [
        (trip.symbol, trip.quantity, trip.price, trip.entry_date, trip.exit_date)
        for trip in skeleton
    ] == [
        ("000001.SZ", 900, 10.0, DAYS[0], DAYS[4]),
        ("000001.SZ", 100, 10.0, DAYS[0], DAYS[6]),
        ("600001.SH", 500, 20.0, DAYS[1], None),
        ("000001.SZ", 600, 12.0, DAYS[2], DAYS[6]),
    ]
    assert skeleton[0].notional == 9000.0
    assert skeleton[2].exit_at is None
    assert skeleton[0].exit_at == _at(DAYS[4], "15:00")


def test_trade_skeleton_rejects_a_sell_without_a_matching_buy():
    executions = [
        _fill("000001.SZ", "buy", 500, DAYS[0], "09:30", 10.0),
        _fill("000001.SZ", "sell", 900, DAYS[2], "15:00", 11.0),
    ]

    with pytest.raises(ValueError, match="exceeds its filled buys"):
        trade_skeleton(executions)


def test_null_draws_replace_each_name_inside_its_size_decile():
    frame = _frame()
    skeleton = trade_skeleton(_observed_result().executions)
    assert [(trip.symbol, trip.entry_date, trip.exit_date) for trip in skeleton] == [
        ("000001.SZ", DAYS[2], DAYS[9]),
        ("600001.SH", DAYS[4], None),
    ]
    rng = np.random.default_rng(11)

    seen: set[str] = set()
    for _ in range(50):
        orders = sample_null_orders(skeleton, frame, rng)
        assert sorted(orders) == [DAYS[2], DAYS[4], DAYS[9]]
        entry, holding, exit_ = orders[DAYS[2]][0], orders[DAYS[4]][0], orders[DAYS[9]][0]
        assert entry["symbol"] in set(SMALL) - {"000001.SZ"}
        assert holding["symbol"] in set(LARGE) - {"600001.SH"}
        assert (entry["action"], exit_["action"], holding["action"]) == ("buy", "sell", "buy")
        assert entry["symbol"] == exit_["symbol"]
        assert entry["quantity"] == exit_["quantity"]
        assert [order["execute_at"] for order in (entry, holding, exit_)] == [
            _at(DAYS[2], "09:30").isoformat(),
            _at(DAYS[4], "09:30").isoformat(),
            _at(DAYS[9], "15:00").isoformat(),
        ]
        assert entry["quantity"] == _expected_quantity(
            frame, str(entry["symbol"]), DAYS[2], skeleton[0].notional
        )
        assert holding["quantity"] == _expected_quantity(
            frame, str(holding["symbol"]), DAYS[4], skeleton[1].notional
        )
        seen.update({str(entry["symbol"]), str(holding["symbol"])})
    # Uniform over the matched pool, so both alternatives show up.
    assert seen == {"000002.SZ", "000003.SZ", "600002.SH"}


def test_null_quantities_follow_the_star_declaration_ladder():
    rows = []
    for offset, day in enumerate(DAYS[:4]):
        for symbol, price in (("000001.SZ", 500.0), ("688001.SH", 50.0 * (offset + 1))):
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": day,
                    "open": price,
                    "close": price,
                    "up_limit": price * 1.5,
                    "down_limit": price * 0.5,
                    "is_suspended": False,
                    "circ_mv": 1.0e9,
                }
            )
    skeleton = [
        RoundTrip("000001.SZ", 20, 500.0, _at(DAYS[0], "09:30"), _at(DAYS[2], "15:00")),
        RoundTrip("000001.SZ", 20, 500.0, _at(DAYS[1], "09:30"), _at(DAYS[2], "15:00")),
    ]

    orders = sample_null_orders(skeleton, pd.DataFrame(rows), np.random.default_rng(3))

    # 10 000 / 50 = 200 shares clears the STAR minimum declaration; the second
    # trip buys at 100 and its 100 shares do not, so it places no order at all.
    assert orders[DAYS[0]] == [
        {
            "symbol": "688001.SH",
            "action": "buy",
            "quantity": 200,
            "execute_at": _at(DAYS[0], "09:30").isoformat(),
        }
    ]
    assert DAYS[1] not in orders
    assert [order["quantity"] for order in orders[DAYS[2]]] == [200]


def test_a_round_trip_too_small_to_buy_a_lot_is_counted_not_hidden():
    """A partial exit splits one entry into independent round trips, so each
    exit's share of the money is rounded down to a board lot on its own. A
    share that cannot buy one lot deploys nothing, and the null then runs with
    less capital than the result it is ranked against; the block reports it
    instead of letting the null distribution shrink silently."""
    script = {
        DAYS[2]: (("000001.SZ", "buy", 1000, _at(DAYS[2], "09:30")),),
        DAYS[9]: (("000001.SZ", "sell", 900, _at(DAYS[9], "15:00")),),
        DAYS[12]: (("000001.SZ", "sell", 100, _at(DAYS[12], "15:00")),),
    }

    def strategy(context):
        day = context.inference_at.strftime("%Y%m%d")
        return [
            {
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "execute_at": when.isoformat(),
            }
            for symbol, action, quantity, when in script.get(day, ())
        ]

    result = run_daily_replay(
        daily=_frame(),
        strategy=strategy,
        schedule=StrategySchedule("day", "08:30"),
        profile=BrokerProfile(),
    )
    skeleton = trade_skeleton(result.executions)
    assert [trip.quantity for trip in skeleton] == [900, 100]

    block = run_null_control(
        result,
        _frame(),
        BENCHMARK,
        BrokerProfile(),
        StrategySchedule("day", "08:30"),
        k=3,
        seed=17,
    )

    # Both replacements in the decile open above 000001.SZ, so the 100-share
    # tail (about 1 020 CNY) never reaches one lot: one trip of two is dropped
    # in every draw.
    assert block["dropped_trips_mean"] == 1.0


def test_excess_percentile_is_the_share_of_null_runs_at_or_below_the_observed():
    null = [-0.10, -0.05, 0.0, 0.05, 0.20]

    assert _distribution(0.05, null) == {
        "observed_excess": 0.05,
        "null_excess_mean": 0.02,
        "null_excess_p05": -0.09,
        "null_excess_p95": 0.17,
        "excess_percentile": 0.8,
    }
    assert _distribution(-0.5, null)["excess_percentile"] == 0.0
    assert _distribution(1.0, null)["excess_percentile"] == 1.0


def test_run_null_control_replays_scripted_nulls_and_reports_one_block():
    frame = _frame()
    result = _observed_result()
    benchmark_return = 1.001 ** len(DAYS) - 1.0

    block = run_null_control(
        result,
        frame,
        BENCHMARK,
        BrokerProfile(),
        StrategySchedule("day", "08:30"),
        k=3,
        seed=17,
    )

    assert block["k"] == 3
    assert block["seed"] == 17
    assert block["matched"] == "circ_mv_decile"
    assert block["observed_excess"] == pytest.approx(
        result.equity_curve[-1]["equity"] / 1_000_000.0 - 1.0 - benchmark_return, abs=1e-6
    )
    assert block["null_excess_p05"] <= block["null_excess_mean"] <= block["null_excess_p95"]
    assert 0.0 <= block["excess_percentile"] <= 1.0
    assert block["rejects_mean"] == 0.0
    assert block["dropped_trips_mean"] == 0.0
    assert "step" not in block

    assert (
        run_null_control(
            result, frame, BENCHMARK, BrokerProfile(), StrategySchedule("day", "08:30"),
            k=3, seed=17,
        )
        == block
    )

    stepped = run_null_control(
        result,
        frame,
        BENCHMARK,
        BrokerProfile(),
        StrategySchedule("day", "08:30"),
        k=3,
        seed=17,
        step=(DAYS[10], DAYS[19]),
    )
    step = stepped["step"]
    assert step["start"] == DAYS[10] and step["end"] == DAYS[19]
    assert set(step) == {
        "start",
        "end",
        "observed_excess",
        "null_excess_mean",
        "null_excess_p05",
        "null_excess_p95",
        "excess_percentile",
    }
    # The step starts after the only exit, so the observed step return is the
    # frozen 600001.SH holding measured from the last close before the window.
    equity = {str(row["trade_date"]): float(row["equity"]) for row in result.equity_curve}
    assert step["observed_excess"] == pytest.approx(
        equity[DAYS[19]] / equity[DAYS[9]] - 1.0 - (1.001**10 - 1.0), abs=1e-6
    )


def test_run_null_control_rejects_a_frame_from_another_window():
    result = _observed_result()

    with pytest.raises(ValueError, match="different trading days"):
        run_null_control(
            result,
            _frame()[lambda frame: frame["trade_date"] != DAYS[5]],
            BENCHMARK,
            BrokerProfile(),
            StrategySchedule("day", "08:30"),
            k=1,
            seed=1,
        )


def test_a_result_with_no_filled_trade_has_no_null_and_runs_no_replay(monkeypatch):
    """An idle cash account picked no names: every null draw would be the same
    empty replay and the right-inclusive percentile would read 1.0, the mark of
    a result that beat every random portfolio. The block says unavailable and
    spends no replay."""

    def unexpected_replay(**_kwargs):
        raise AssertionError("an empty skeleton must not spend a null replay")

    monkeypatch.setattr(
        "autotrade.environment.replay.null_control.run_daily_replay", unexpected_replay
    )
    result = ReplayResult(
        equity_curve=tuple(
            {"trade_date": day, "initial_equity": 1_000_000.0, "equity": 1_000_000.0}
            for day in DAYS
        ),
        executions=(),
        inference_dates=(),
        pending_orders=(),
    )
    block = run_null_control(
        result, _frame(), BENCHMARK, BrokerProfile(), StrategySchedule("day", "08:30"), k=500, seed=3
    )
    assert block == {
        "status": "unavailable",
        "reason": "no_filled_trades",
        "k": 0,
        "seed": 3,
        "matched": "circ_mv_decile",
        "excess_percentile": None,
    }


def _fill(
    symbol: str, action: str, quantity: int, day: str, clock: str, price: float
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "execute_at": _at(day, clock).isoformat(),
        "matched_at": _at(day, clock).isoformat(),
        "status": "filled",
        "price": price,
    }
