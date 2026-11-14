"""Random-portfolio null control: skeleton pairing, matched draws, replay block.

The synthetic market is five names over 20 trading days in two size groups, so
without a membership table a replacement is always drawn from the original's
own ``circ_mv`` decile and never from the other group; with one, the side of
the benchmark's membership takes the decile's place.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from autotrade.environment.broker import BrokerProfile, default_permitted_boards
from autotrade.environment.broker_core import board_of
from autotrade.environment.replay import run_daily_replay
from autotrade.environment.replay.market import DailyMarketData
from autotrade.environment.replay.null_control import (
    BASELINE_DRAWS,
    MATCHED_DECILE,
    MATCHED_MEMBERSHIP,
    UNCONSTRAINED_CASH,
    NullControlSetupError,
    RoundTrip,
    _candidate_pool,
    _distribution,
    _draw_profile,
    _Membership,
    _orders_from_pools,
    _scripted_strategy,
    _Universe,
    run_null_control,
    trade_skeleton,
)
from autotrade.environment.replay.stats import ReplayResult
from autotrade.environment.strategy import CN_TZ, StrategySchedule

DAYS = tuple(pd.bdate_range("2026-01-05", periods=20).strftime("%Y%m%d"))
SMALL = ("000001.SZ", "000002.SZ", "000003.SZ")
LARGE = ("600001.SH", "600002.SH")
CIRC_MV = {**{name: 1.0e9 for name in SMALL}, **{name: 5.0e10 for name in LARGE}}
BENCHMARK = {day: 0.001 for day in DAYS}
# The random baseline draws only on the boards a profile pins.
PROFILE = BrokerProfile(permitted_boards=("main", "gem"))
DAYS_ISO = tuple(f"{day[:4]}-{day[4:6]}-{day[6:]}" for day in DAYS)


def _sample_null_orders(skeleton, frame: pd.DataFrame, seed: int, draw: int) -> dict[str, list[dict]]:
    """One null draw of the whole skeleton, as ``run_null_control`` draws it."""

    universe = _Universe(frame)
    pools = [_candidate_pool(trip, universe) for trip in skeleton]
    return _orders_from_pools(skeleton, pools, seed, draw, _sellable(universe, frame), universe.carried_limit)[0]


def _sellable(universe, frame: pd.DataFrame, corporate_actions=None):
    market = DailyMarketData(frame, corporate_actions)
    return lambda trip, symbol, quantity: universe.sellable_at_exit(
        trip, symbol, quantity, market.cash_dividends_for_day
    )


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
                    # The previous day's close: no ex-date anywhere in the window.
                    "pre_close": (10.0 + 2.0 * index + 0.1 * (offset - 1)) * 1.01,
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
        profile=PROFILE,
    )


def _pools(skeleton, frame: pd.DataFrame, membership=None):
    universe = _Universe(frame)
    members = None if membership is None else _Membership(membership)
    return [_candidate_pool(trip, universe, members) for trip in skeleton]


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

    skeleton, unpaired = trade_skeleton(executions)

    assert unpaired == 0
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


def test_trade_skeleton_counts_sold_shares_no_filled_buy_accounts_for():
    """Shares an ex-date settlement created are sold without a buy of their
    own: they are not a round trip, only counted, so the null stays available."""
    executions = [
        _fill("000001.SZ", "buy", 500, DAYS[0], "09:30", 10.0),
        _fill("000001.SZ", "sell", 900, DAYS[2], "15:00", 11.0),
    ]

    skeleton, unpaired = trade_skeleton(executions)

    assert [(trip.quantity, trip.exit_date) for trip in skeleton] == [(500, DAYS[2])]
    assert unpaired == 400


def test_null_draws_replace_each_name_inside_its_size_decile():
    frame = _frame()
    skeleton, _unpaired = trade_skeleton(_observed_result().executions)
    assert [(trip.symbol, trip.entry_date, trip.exit_date) for trip in skeleton] == [
        ("000001.SZ", DAYS[2], DAYS[9]),
        ("600001.SH", DAYS[4], None),
    ]
    seen: set[str] = set()
    for draw in range(50):
        orders = _sample_null_orders(skeleton, frame, 11, draw)
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
        # The previous close: the STAR name rises every day, with no ex-date.
        for symbol, price, previous in (
            ("000001.SZ", 500.0, 500.0),
            ("688001.SH", 50.0 * (offset + 1), 50.0 * max(offset, 1)),
        ):
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": day,
                    "open": price,
                    "close": price,
                    "pre_close": previous,
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

    frame = pd.DataFrame(rows)
    orders = _sample_null_orders(skeleton, frame, 3, 0)

    # 10 000 / 50 = 200 shares clears the STAR minimum declaration; the second
    # trip would buy at 100 and its 100 shares do not, so the name is not in
    # that trip's pool at all and the trip places no order.
    assert [len(pool) for pool in _pools(skeleton, frame)] == [1, 0]
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


def test_the_random_baseline_draws_only_on_the_boards_the_account_may_buy(monkeypatch):
    drawn = {symbol for draw in _board_draws(("main", "gem"), 200, 1) for symbol, _action, _quantity in draw}
    assert drawn == {"000001.SZ", "000002.SZ", "600001.SH", "300001.SZ", "300002.SZ"}
    # Without the restriction the same pool reaches STAR and Beijing names.
    unrestricted = {symbol for draw in _board_draws(None, 200, 1) for symbol, _action, _quantity in draw}
    assert {"688001.SH", "688002.SH", "830001.BJ", "830002.BJ"} <= unrestricted

    # End to end through a restricted Broker: no draw is ever refused for its board.
    def strategy(context):
        day = context.inference_at.strftime("%Y%m%d")
        script = {
            DAYS[0]: [("000001.SZ", "buy", 1000, _at(DAYS[0], "09:30"))],
            DAYS[1]: [("300001.SZ", "buy", 500, _at(DAYS[1], "09:30"))],
            DAYS[3]: [("000001.SZ", "sell", 1000, _at(DAYS[3], "15:00"))],
        }
        return [
            {"symbol": s, "action": a, "quantity": q, "execute_at": w.isoformat()}
            for s, a, q, w in script.get(day, ())
        ]

    profile = BrokerProfile(initial_cash=100_000, permitted_boards=("main", "gem"))
    observed = run_daily_replay(
        daily=_board_frame(), strategy=strategy, schedule=StrategySchedule("day", "08:30"), profile=profile
    )
    runs = _spy_draws(monkeypatch)
    block = run_null_control(
        observed, _board_frame(), BENCHMARK, profile, StrategySchedule("day", "08:30"), k=20, seed=5
    )
    assert block["rejects_mean"] == 0.0
    assert block["dropped_trips_mean"] == 0.0

    # A draw's account opens with more than any board's asset threshold, yet
    # buys only where the candidate's own account could: its boards are the
    # profile's, never derived from its balance.
    assert default_permitted_boards(UNCONSTRAINED_CASH) == ("main", "gem", "star", "bj")
    assert {(draw.initial_cash, draw.permitted_boards) for draw, _run in runs} == {
        (UNCONSTRAINED_CASH, ("main", "gem"))
    }
    bought = {row["symbol"] for _draw, run in runs for row in run.executions if row["action"] == "buy"}
    assert {board_of(symbol) for symbol in bought} == {"main", "gem"}
    with pytest.raises(NullControlSetupError, match="pins none"):
        run_null_control(
            observed, _board_frame(), BENCHMARK, replace(profile, permitted_boards=None),
            StrategySchedule("day", "08:30"), k=1, seed=5,
        )
    # A single-name cap is a weight of the opening balance, so a draw's stays
    # the candidate's CNY cap.
    capped = _draw_profile(replace(profile, max_single_name_weight=0.1))
    assert capped.max_single_name_weight * capped.initial_cash == pytest.approx(0.1 * profile.initial_cash)


def test_the_draws_are_pinned_to_the_seed_the_draw_and_the_trip():
    """Pinned literals: a replacement is seeded from a blake2b digest of the
    random baseline seed, the draw's index and the round trip, never from Python's
    per-process ``hash``, so every process -- a rerun, a paired experiment, a later
    audit -- draws exactly these names."""

    assert _board_draws(None, 6, 7) == [
        [("000002.SZ", "buy", 900), ("688002.SH", "buy", 406), ("000002.SZ", "sell", 900)],
        [("688001.SH", "buy", 666), ("000001.SZ", "buy", 600), ("688001.SH", "sell", 666)],
        [("830002.BJ", "buy", 555), ("688002.SH", "buy", 406), ("830002.BJ", "sell", 555)],
        [("688002.SH", "buy", 625), ("600001.SH", "buy", 500), ("688002.SH", "sell", 625)],
        [("830001.BJ", "buy", 588), ("600001.SH", "buy", 500), ("830001.BJ", "sell", 588)],
        [("830001.BJ", "buy", 588), ("830001.BJ", "buy", 383), ("830001.BJ", "sell", 588)],
    ]


def _replacement_names(skeleton, frame: pd.DataFrame, seed: int) -> list[dict[str, str]]:
    """Per random baseline draw, the name each round trip bought, by its entry instant."""

    universe = _Universe(frame)
    pools = [_candidate_pool(trip, universe) for trip in skeleton]
    sellable = _sellable(universe, frame)
    names = []
    for draw in range(BASELINE_DRAWS):
        orders, _dropped = _orders_from_pools(skeleton, pools, seed, draw, sellable, universe.carried_limit)
        names.append(
            {
                str(order["execute_at"]): str(order["symbol"])
                for day in orders.values()
                for order in day
                if order["action"] == "buy"
            }
        )
    return names


def test_a_changed_round_trip_redraws_that_trip_alone():
    """The defect this replaces: every trip of every draw took the next number
    of one shared stream, so a replay with one buy more than its record (2 238
    against 2 237) redrew all twenty draws from that buy on and moved the
    random baseline's cumulative return by 23 points while the book itself moved 0.37
    pp/yr. Now an added trip and a resized one change their own replacements,
    and every other trip keeps its name in every draw."""

    frame = _board_frame(12)
    fills = [
        _fill("000001.SZ", "buy", 1000, DAYS[1], "09:30", 10.1),
        _fill("300001.SZ", "buy", 700, DAYS[2], "09:30", 13.2),
        _fill("600001.SH", "buy", 800, DAYS[3], "09:30", 12.3),
        _fill("000002.SZ", "buy", 900, DAYS[4], "09:30", 11.4),
        _fill("300002.SZ", "buy", 600, DAYS[5], "09:30", 14.5),
        _fill("000001.SZ", "sell", 1000, DAYS[8], "15:00", 10.8),
        _fill("600001.SH", "sell", 800, DAYS[10], "15:00", 13.3),
    ]
    resized = [
        _fill("600001.SH", "buy", 900, DAYS[3], "09:30", 12.3)
        if fill["symbol"] == "600001.SH" and fill["action"] == "buy"
        else _fill("600001.SH", "sell", 900, DAYS[10], "15:00", 13.3)
        if fill["symbol"] == "600001.SH"
        else fill
        for fill in fills
    ]
    # The earliest entry of the window: in the shared stream every later
    # trip of every draw would have moved one number down.
    added = _fill("688001.SH", "buy", 300, DAYS[0], "09:30", 15.0)
    base, _unpaired = trade_skeleton(fills)
    variant, _unpaired = trade_skeleton([added, *resized])

    before = _replacement_names(base, frame, 29)
    after = _replacement_names(variant, frame, 29)

    touched = {_at(DAYS[0], "09:30").isoformat(), _at(DAYS[3], "09:30").isoformat()}
    for old, new in zip(before, after, strict=True):
        assert set(new) == set(old) | touched
        assert {key: name for key, name in new.items() if key not in touched} == {
            key: name for key, name in old.items() if key not in touched
        }
    resized_entry = _at(DAYS[3], "09:30").isoformat()
    assert any(old[resized_entry] != new[resized_entry] for old, new in zip(before, after))
    # The identical trade list meets the identical draws, in whatever order
    # its trips arrive.
    assert _replacement_names(base, frame, 29) == before
    assert _replacement_names(list(reversed(base)), frame, 29) == before


def test_a_round_trip_draws_uniformly_and_afresh_in_every_draw():
    """Across draws one trip's replacement is uniform over its pool, and the
    equal parts of one entry sold in two exits still draw a name each."""

    frame = _board_frame()
    universe = _Universe(frame)
    sellable = _sellable(universe, frame)
    trip = _BOARD_SKELETON[0]
    pool = _candidate_pool(trip, universe)
    assert len(pool) == 8

    picks = [
        _orders_from_pools([trip], [pool], 13, draw, sellable, universe.carried_limit)[0][trip.entry_date][0]["symbol"]
        for draw in range(1600)
    ]
    counts = Counter(picks)
    assert set(counts) == {symbol for symbol, _quantity in pool}
    # 200 expected per name; 60 is about four and a half standard deviations.
    assert all(abs(count - 200) < 60 for count in counts.values())
    assert len(set(picks[:BASELINE_DRAWS])) > 1

    halves, _unpaired = trade_skeleton(
        [
            _fill("000001.SZ", "buy", 1000, DAYS[0], "09:30", 10.0),
            _fill("000001.SZ", "sell", 500, DAYS[3], "15:00", 10.3),
            _fill("000001.SZ", "sell", 500, DAYS[5], "15:00", 10.5),
        ]
    )
    pools = [_candidate_pool(half, universe) for half in halves]
    pairs = [
        tuple(
            order["symbol"]
            for order in _orders_from_pools(halves, pools, 13, draw, sellable, universe.carried_limit)[0][DAYS[0]]
        )
        for draw in range(BASELINE_DRAWS)
    ]
    assert all(len(pair) == 2 for pair in pairs)
    assert any(first != second for first, second in pairs)


def test_identical_trades_meet_the_identical_random_baseline():
    """Two replays of one script are two results with one trade list, so
    their random baselines agree to the last digit."""

    arguments = (_frame(), BENCHMARK, PROFILE, StrategySchedule("day", "08:30"))
    first = run_null_control(_observed_result(), *arguments, seed=17, baselines=True)
    second = run_null_control(_observed_result(), *arguments, seed=17, baselines=True)

    assert first["k"] == BASELINE_DRAWS
    assert first == second


def _spy_draws(monkeypatch) -> list[tuple[BrokerProfile, ReplayResult]]:
    """Every draw ``run_null_control`` replays, with the profile it ran under."""

    runs: list[tuple[BrokerProfile, ReplayResult]] = []

    def spy(**kwargs):
        run = run_daily_replay(**kwargs)
        runs.append((kwargs["profile"], run))
        return run

    monkeypatch.setattr("autotrade.environment.replay.null_control.run_daily_replay", spy)
    return runs


def _script_strategy(script):
    """A strategy that sends ``script[day]``'s ``(symbol, action, quantity, clock)`` orders."""

    def strategy(context):
        day = context.inference_at.strftime("%Y%m%d")
        return [
            {"symbol": symbol, "action": action, "quantity": quantity, "execute_at": _at(day, clock).isoformat()}
            for symbol, action, quantity, clock in script.get(day, ())
        ]

    return strategy


def _growth_frame() -> pd.DataFrame:
    """Six names in one size bucket over 16 days: 000001.SZ climbs from 10 to
    64 by DAYS[9] and holds there, the others stay flat at 10 to 14."""

    rows = []
    for offset, day in enumerate(DAYS[:16]):
        for index, symbol in enumerate(f"00000{number}.SZ" for number in range(1, 7)):
            if index == 0:
                price, previous = 10.0 + 6.0 * min(offset, 9), 10.0 + 6.0 * min(max(offset - 1, 0), 9)
            else:
                price = previous = 9.0 + index
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": day,
                    "open": price,
                    "close": price,
                    "pre_close": previous,
                    "up_limit": price * 1.5,
                    "down_limit": price * 0.5,
                    "is_suspended": False,
                    "circ_mv": 1.0e9,
                }
            )
    return pd.DataFrame(rows)


def test_a_draw_holds_every_trip_of_a_candidate_that_outgrew_its_opening_cash(monkeypatch):
    """The defect this replaces: each draw was a CNY 200k account of its own
    while its trips kept the candidate's CNY, so once the candidate's equity
    pulled away a draw's buys were refused for cash. A draw now trades on the
    candidate's capital: a candidate that grew six-fold and then deploys a
    million meets draws that hold every one of its trips."""

    frame = _growth_frame()
    profile = BrokerProfile(initial_cash=200_000, permitted_boards=("main", "gem"))
    script = {
        DAYS[0]: (("000001.SZ", "buy", 19_000, "09:30"),),
        DAYS[9]: (("000001.SZ", "sell", 19_000, "15:00"),),
        DAYS[10]: tuple((symbol, "buy", 30_000, "09:30") for symbol in ("000002.SZ", "000003.SZ", "000004.SZ")),
        DAYS[15]: tuple((symbol, "sell", 30_000, "15:00") for symbol in ("000002.SZ", "000003.SZ", "000004.SZ")),
    }
    schedule = StrategySchedule("day", "08:30")
    result = run_daily_replay(daily=frame, strategy=_script_strategy(script), schedule=schedule, profile=profile)
    assert all(row["status"] == "filled" for row in result.executions)
    assert result.equity_curve[9]["equity"] > 6 * profile.initial_cash

    runs = _spy_draws(monkeypatch)
    block = run_null_control(result, frame, BENCHMARK, profile, schedule, k=5, seed=11)

    assert (block["round_trips"], block["rejects_mean"], block["dropped_trips_mean"]) == (4, 0.0, 0.0)
    assert len(runs) == 5
    for _draw, run in runs:
        assert all(row["status"] == "filled" for row in run.executions)
        assert sum(row["action"] == "buy" for row in run.executions) == 4
        assert run.equity_curve[-1]["positions"] == {}
    # The same orders in an account of the candidate's opening cash, as each
    # draw used to be, are refused for cash.
    skeleton, _unpaired = trade_skeleton(result.executions)
    universe = _Universe(frame)
    pools = [_candidate_pool(trip, universe, None, profile.permitted_boards) for trip in skeleton]
    orders, _dropped = _orders_from_pools(skeleton, pools, 11, 0, _sellable(universe, frame), universe.carried_limit)
    own_account = run_daily_replay(
        daily=frame, strategy=_scripted_strategy(orders), schedule=StrategySchedule("year", "08:30"), profile=profile
    )
    assert any(row["reason"] == "insufficient_cash" for row in own_account.executions)


def test_a_draws_daily_return_is_its_pnl_over_the_candidates_previous_close():
    """Worked by hand, without costs: the candidate buys 1 000 of 600001.SH
    at 10 and sells at the third close; its only replacement, 600002.SH, takes
    the same 10 000 CNY as 500 shares at 20, closes 21 and 22 and is sold at
    19. Each draw earns +500, +500 and -1 500 CNY on those days, read on the
    candidate's equity at the previous close: 100 000, 102 000 and 105 000 --
    not on what an account of the draw's own would hold (100 500, 101 000)."""

    prices = {  # (open, close) per day; each day's pre_close is the last close.
        "600001.SH": ((10.0, 12.0), (12.0, 15.0), (15.0, 16.0)),
        "600002.SH": ((20.0, 21.0), (21.0, 22.0), (22.0, 19.0)),
    }
    frame = pd.DataFrame(
        {
            "ts_code": symbol,
            "trade_date": day,
            "open": opening,
            "close": closing,
            "pre_close": bars[offset - 1][1] if offset else opening,
            "up_limit": 100.0,
            "down_limit": 1.0,
            "is_suspended": False,
            "circ_mv": 5.0e10,
        }
        for symbol, bars in prices.items()
        for offset, (day, (opening, closing)) in enumerate(zip(DAYS[:3], bars, strict=True))
    )
    free = BrokerProfile(
        initial_cash=100_000,
        commission_bps=0.0,
        min_commission_cny=0.0,
        transfer_fee_bps=0.0,
        stamp_duty_sell_bps_before_cutover=0.0,
        stamp_duty_sell_bps_from_cutover=0.0,
        slippage_bps=0.0,
        permitted_boards=("main",),
    )
    script = {DAYS[0]: (("600001.SH", "buy", 1000, "09:30"),), DAYS[2]: (("600001.SH", "sell", 1000, "15:00"),)}
    schedule = StrategySchedule("day", "08:30")
    result = run_daily_replay(daily=frame, strategy=_script_strategy(script), schedule=schedule, profile=free)
    assert [row["equity"] for row in result.equity_curve] == [102_000.0, 105_000.0, 106_000.0]

    flat = {day: 0.0 for day in DAYS[:3]}
    block = run_null_control(result, frame, flat, free, schedule, k=2, seed=1, baselines=True)

    expected = [500 / 100_000, 500 / 102_000, -1_500 / 105_000]
    assert [day for day, _value in block["baseline_daily"]] == list(DAYS[:3])
    assert [value for _day, value in block["baseline_daily"]] == pytest.approx(expected, abs=1e-12)
    assert [value for _day, value in block["baseline_draw_sd"]] == pytest.approx([0.0] * 3, abs=1e-12)
    assert block["null_excess_mean"] == pytest.approx(np.prod(1 + np.asarray(expected)) - 1, abs=1e-6)
    assert block["observed_excess"] == pytest.approx(0.06, abs=1e-6)


def test_the_same_trips_meet_the_same_names_in_a_shorter_replay():
    """A replay cut short of another meets, on every day the two share, the
    same draws: the same names, fills and composite, read on the same
    capital, so both replays grade the round trips they share alike. (A trip
    still open at the shorter replay's end is the exception: its pool skips
    the exit-day filter, so it may draw another name.)"""

    script = {
        DAYS[1]: (("000001.SZ", "buy", 1000, "09:30"),),
        DAYS[3]: (("600001.SH", "buy", 500, "09:30"),),
        DAYS[5]: (("000001.SZ", "sell", 1000, "15:00"),),
        DAYS[9]: (("600001.SH", "sell", 500, "15:00"),),
        DAYS[14]: (("000002.SZ", "buy", 800, "09:30"),),
    }
    schedule = StrategySchedule("day", "08:30")

    def replay(days: int) -> tuple[dict, list[tuple[BrokerProfile, ReplayResult]]]:
        frame = _frame()[lambda rows: rows["trade_date"].isin(DAYS[:days])]
        result = run_daily_replay(daily=frame, strategy=_script_strategy(script), schedule=schedule, profile=PROFILE)
        with pytest.MonkeyPatch.context() as patch:
            runs = _spy_draws(patch)
            block = run_null_control(
                result, frame, BENCHMARK, PROFILE, schedule, k=4, seed=17, step=(DAYS[0], DAYS[11]), baselines=True
            )
        return block, runs

    short, short_runs = replay(12)
    long, long_runs = replay(20)

    assert (short["round_trips"], long["round_trips"]) == (2, 3)
    assert short["baseline_daily"] == long["baseline_daily"][:12]
    assert short["baseline_draw_sd"] == long["baseline_draw_sd"][:12]
    assert short["step"] == long["step"]
    for (_short_profile, short_run), (_long_profile, long_run) in zip(short_runs, long_runs, strict=True):
        assert short_run.executions == tuple(row for row in long_run.executions if row["matched_at"] < DAYS_ISO[12])


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
        profile=PROFILE,
    )
    skeleton, _unpaired = trade_skeleton(result.executions)
    assert [trip.quantity for trip in skeleton] == [900, 100]

    block = run_null_control(
        result,
        _frame(),
        BENCHMARK,
        PROFILE,
        StrategySchedule("day", "08:30"),
        k=3,
        seed=17,
    )

    # Both replacements in the decile open above 000001.SZ, so the 100-share
    # tail (about 1 020 CNY) reaches one lot of neither: its pool is empty and
    # one trip of two is dropped in every draw.
    assert block["round_trips"] == 2
    assert block["dropped_trips_mean"] == 1.0


def test_affordability_is_settled_in_the_pool_so_an_expensive_name_drops_no_trip():
    """The defect this replaces: a replacement was drawn first and the trip
    dropped when one lot of it cost more than the trip's money, so a book of
    cheap names ran its null with a quarter of its trips missing. A name the
    money cannot buy is never a candidate, so the trip always deploys."""
    rows = []
    for day in DAYS[:6]:
        for symbol, price in (("000001.SZ", 10.0), ("000002.SZ", 11.0), ("000003.SZ", 900.0)):
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": day,
                    "open": price,
                    "close": price,
                    "pre_close": price,
                    "up_limit": price * 1.5,
                    "down_limit": price * 0.5,
                    "is_suspended": False,
                    "circ_mv": 1.0e9,
                }
            )
    frame = pd.DataFrame(rows)
    skeleton = [RoundTrip("000001.SZ", 1000, 10.0, _at(DAYS[1], "09:30"), _at(DAYS[4], "15:00"))]

    # One lot of 000003.SZ costs 90 000 against the trip's 10 000.
    assert _pools(skeleton, frame) == [(("000002.SZ", 900),)]
    for draw in range(20):
        orders, dropped = _orders_from_pools(
            skeleton, _pools(skeleton, frame), 5, draw, _sellable(_Universe(frame), frame), _Universe(frame).carried_limit
        )
        assert dropped == 0
        assert orders[DAYS[1]][0]["symbol"] == "000002.SZ"


def test_membership_matches_a_name_to_its_own_side_of_the_benchmark():
    """With the benchmark's membership mounted a replacement comes from the
    original's side of it as the cross-section dated before the entry day
    states it, whatever its float-cap decile; the membership of a later
    cross-section is not read."""
    frame = _frame()
    membership = {
        DAYS[0]: {"000001.SZ", "600001.SH"},
        # Published on the entry day's own close: not readable that morning.
        DAYS[2]: {"000002.SZ", "000003.SZ", "600002.SH"},
    }
    skeleton = [
        RoundTrip("000001.SZ", 1000, 10.2, _at(DAYS[2], "09:30"), _at(DAYS[9], "15:00")),
        RoundTrip("000002.SZ", 1000, 12.2, _at(DAYS[2], "09:30"), _at(DAYS[9], "15:00")),
    ]

    inside, outside = _pools(skeleton, frame, membership)

    assert {symbol for symbol, _quantity in inside} == {"600001.SH"}
    assert {symbol for symbol, _quantity in outside} == {"000003.SZ", "600002.SH"}
    with pytest.raises(NullControlSetupError, match="no benchmark membership is dated before"):
        _pools(
            [RoundTrip("000001.SZ", 1000, 10.0, _at(DAYS[0], "09:30"), None)],
            frame,
            membership,
        )


def test_a_mounted_membership_that_misses_the_span_fails_before_any_draw():
    """An experiment that mounts ``index_weight`` is graded on membership in every
    span. Where its release holds no section before the span's entries -- an
    eight-year experiment on a release whose constituents start in 2020 -- the random baseline
    must fail by name, never quietly draw the float-cap decile for this span
    while the experiment's other spans draw membership."""
    arguments = (
        _observed_result(),
        _frame(),
        BENCHMARK,
        PROFILE,
        StrategySchedule("day", "08:30"),
    )

    with pytest.raises(NullControlSetupError, match="no section of the benchmark"):
        run_null_control(*arguments, k=2, seed=3, membership={})
    with pytest.raises(NullControlSetupError, match="first section of the benchmark is"):
        run_null_control(*arguments, k=2, seed=3, membership={DAYS[3]: set(SMALL)})
    assert run_null_control(*arguments, k=2, seed=3)["matched"] == MATCHED_DECILE


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
        PROFILE,
        StrategySchedule("day", "08:30"),
        k=3,
        seed=17,
    )

    assert block["k"] == 3
    assert block["seed"] == 17
    assert block["matched"] == MATCHED_DECILE
    assert block["observed_excess"] == pytest.approx(
        result.equity_curve[-1]["equity"] / 1_000_000.0 - 1.0 - benchmark_return, abs=1e-6
    )
    assert block["null_excess_p05"] <= block["null_excess_mean"] <= block["null_excess_p95"]
    assert 0.0 <= block["excess_percentile"] <= 1.0
    assert block["rejects_mean"] == 0.0
    assert block["dropped_trips_mean"] == 0.0
    assert block["unpaired_sell_shares"] == 0
    assert "step" not in block
    assert "baseline_daily" not in block

    assert (
        run_null_control(
            result, frame, BENCHMARK, PROFILE, StrategySchedule("day", "08:30"),
            k=3, seed=17,
        )
        == block
    )

    stepped = run_null_control(
        result,
        frame,
        BENCHMARK,
        PROFILE,
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


def test_the_random_baseline_is_the_draws_mean_daily_return_and_leaves_the_result_alone():
    """``baselines`` keeps what the percentile throws away: every draw's daily
    return, as their equal-weight mean and cross-draw spread per trading day.
    The draws replay through Brokers of their own, so the graded result is the
    same object before and after."""
    frame = _frame()
    result = _observed_result()
    before = (result.equity_curve, result.executions)
    arguments = (result, frame, BENCHMARK, PROFILE, StrategySchedule("day", "08:30"))

    block = run_null_control(
        *arguments, k=4, seed=17, membership={"20251231": set(SMALL)}, baselines=True
    )

    assert (result.equity_curve, result.executions) == before
    assert block["matched"] == MATCHED_MEMBERSHIP
    assert [date for date, _value in block["baseline_daily"]] == list(DAYS)
    assert [date for date, _value in block["baseline_draw_sd"]] == list(DAYS)
    draws = [
        run_null_control(*arguments, k=1, seed=seed, membership={"20251231": set(SMALL)}, baselines=True)
        for seed in range(40)
    ]
    # One draw's random baseline is that draw itself, so no two seeds need agree, and the
    # composite of four is flat before the first entry and moves after it.
    assert len({tuple(value for _date, value in draw["baseline_daily"]) for draw in draws}) > 1
    values = np.asarray([value for _date, value in block["baseline_daily"]])
    assert np.all(values[:2] == 0.0) and np.any(values[2:] != 0.0)
    assert all(value >= 0.0 for _date, value in block["baseline_draw_sd"])
    # The percentile block is the same with or without the series.
    plain = run_null_control(*arguments, k=4, seed=17, membership={"20251231": set(SMALL)})
    assert plain == {
        key: value for key, value in block.items() if key not in ("baseline_daily", "baseline_draw_sd")
    }


def test_run_null_control_rejects_a_frame_from_another_window():
    result = _observed_result()

    with pytest.raises(ValueError, match="different trading days"):
        run_null_control(
            result,
            _frame()[lambda frame: frame["trade_date"] != DAYS[5]],
            BENCHMARK,
            PROFILE,
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
        result, _frame(), BENCHMARK, PROFILE, StrategySchedule("day", "08:30"), k=500, seed=3
    )
    assert block == {
        "status": "unavailable",
        "reason": "no_filled_trades",
        "k": 0,
        "seed": 3,
        "matched": MATCHED_DECILE,
        "excess_percentile": None,
    }
    # The zero-skill copy of a book that never traded is an idle cash account.
    idle = run_null_control(
        result, _frame(), BENCHMARK, PROFILE, StrategySchedule("day", "08:30"),
        seed=3, baselines=True,
    )
    assert idle["baseline_daily"] == [[day, 0.0] for day in DAYS]
    assert idle["baseline_draw_sd"] == [[day, 0.0] for day in DAYS]


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


def _bonus_frame() -> pd.DataFrame:
    """Two names in one decile; 000002.SZ goes ex a 3-for-10 bonus on DAYS[5]:
    its 13.00 close resets to a 10.00 pre_close, a share multiplier of 1.3."""

    rows = []
    for offset, day in enumerate(DAYS):
        for symbol in ("000001.SZ", "000002.SZ"):
            price = 10.0 if symbol == "000002.SZ" and offset >= 5 else 13.0
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": day,
                    "open": price,
                    "close": price,
                    "pre_close": price,
                    "up_limit": price * 1.1,
                    "down_limit": price * 0.9,
                    "is_suspended": False,
                    "circ_mv": 1.0e9,
                }
            )
    return pd.DataFrame(rows)


def test_a_null_exit_sells_the_bonus_shares_its_replacement_earned():
    """The Broker grows a held lot of 100 to 130 shares on the bonus ex-date,
    so the exit sells 130: selling the bought 100 would strand 30 shares to
    the end of the window. Shares created on the ex-date are locked through
    it, so an exit that day sells the lot as it was. A lot held alongside
    another lot of the same name sells whole lots only: their odd tails merged,
    and an odd tail is sold only whole."""

    frame = _bonus_frame()
    universe = _Universe(frame)
    sellable = _sellable(universe, frame)

    def replay(orders):
        return run_daily_replay(
            daily=frame,
            strategy=_scripted_strategy(orders),
            schedule=StrategySchedule("year", "08:30"),
            profile=PROFILE,
        )

    single = [RoundTrip("000001.SZ", 100, 13.0, _at(DAYS[2], "09:30"), _at(DAYS[8], "15:00"))]
    pools = _pools(single, frame)
    assert pools == [(("000002.SZ", 100),)]
    orders, _ = _orders_from_pools(single, pools, 1, 0, sellable, universe.carried_limit)
    assert [order["quantity"] for order in orders[DAYS[8]]] == [130]
    run = replay(orders)
    assert [record["status"] for record in run.executions] == ["filled", "filled"]
    assert run.equity_curve[-1]["positions"] == {}

    on_ex_date = RoundTrip("000001.SZ", 100, 13.0, _at(DAYS[2], "09:30"), _at(DAYS[5], "15:00"))
    assert sellable(on_ex_date, "000002.SZ", 100) == 100

    overlapping = [
        *single,
        RoundTrip("000001.SZ", 100, 13.0, _at(DAYS[3], "09:30"), _at(DAYS[10], "15:00")),
    ]
    orders, _ = _orders_from_pools(
        overlapping, _pools(overlapping, frame), 1, 0, sellable, universe.carried_limit
    )
    assert [order["quantity"] for order in orders[DAYS[8]]] == [100]
    assert [order["quantity"] for order in orders[DAYS[10]]] == [100]
    run = replay(orders)
    assert all(record["status"] == "filled" for record in run.executions)
    # 200 shares became 260; the merged 60-share tail is what stays behind.
    assert run.equity_curve[-1]["positions"] == {"000002.SZ": 60}


def test_a_replacement_meets_the_market_in_its_originals_session_and_limit():
    """Both legs keep their session, and a limit price moves to the drawn name
    at the same distance from its own previous close, on a whole tick; the
    draw then replays under the same rules."""

    frame = _frame().assign(
        high=lambda rows: rows["open"] * 1.02,
        low=lambda rows: rows["open"] * 0.98,
        vol=1.0e6,
        amount=lambda rows: rows["open"] * 1.0e6,
    )
    [trip], _unpaired = trade_skeleton(
        [
            {**_fill("000001.SZ", "buy", 1000, DAYS[2], "09:30", 10.2), "session": "intraday", "limit_price": 10.2},
            {**_fill("000001.SZ", "sell", 1000, DAYS[9], "15:00", 11.0), "session": "close"},
        ]
    )
    assert (trip.entry_session, trip.entry_limit, trip.exit_session, trip.exit_limit) == (
        "intraday", 10.2, "close", None
    )
    universe = _Universe(frame)
    orders, _dropped = _orders_from_pools(
        [trip], [_candidate_pool(trip, universe)], 3, 0, _sellable(universe, frame), universe.carried_limit
    )
    [buy], [sell] = orders[DAYS[2]], orders[DAYS[9]]
    pre_close = frame.set_index(["ts_code", "trade_date"])["pre_close"]
    carried = 10.2 / pre_close["000001.SZ", DAYS[2]] * pre_close[buy["symbol"], DAYS[2]]
    assert buy["session"] == "intraday"
    assert buy["limit_price"] == round(carried, 2) and abs(buy["limit_price"] - carried) <= 0.005
    assert sell["session"] == "close" and "limit_price" not in sell
    run = run_daily_replay(
        daily=frame,
        strategy=_scripted_strategy(orders),
        schedule=StrategySchedule("year", "08:00"),
        profile=PROFILE,
    )
    assert [(row["session"], row["status"]) for row in run.executions] == [("intraday", "filled"), ("close", "filled")]


def test_the_draws_meet_an_order_at_any_minute_through_the_results_own_price_source():
    """An order without a session at a minute other than 09:30 and 15:00 is
    priced only by the replay's minute source (``engine.resolve_execution_price``):
    the draws meet their copies of it through the same source, or every one of
    them is rejected for want of a price."""

    frame = _frame()
    opens = frame.set_index(["ts_code", "trade_date"])["open"]
    script = {DAYS[2]: "buy", DAYS[9]: "sell"}

    def minute_price(symbol: str, when: datetime) -> float:
        return float(opens[symbol, when.strftime("%Y%m%d")])

    def strategy(context):
        day = context.inference_at.strftime("%Y%m%d")
        if day not in script:
            return []
        return [{"symbol": "000001.SZ", "action": script[day], "quantity": 1000, "execute_at": _at(day, "10:00").isoformat()}]

    schedule = StrategySchedule("day", "08:30")
    result = run_daily_replay(
        daily=frame, strategy=strategy, schedule=schedule, profile=PROFILE, execution_price=minute_price
    )
    assert [row["status"] for row in result.executions] == ["filled", "filled"]
    arguments = (result, frame, BENCHMARK, PROFILE, schedule)
    assert run_null_control(*arguments, k=3, seed=5, execution_price=minute_price)["rejects_mean"] == 0.0
    # Without it each draw's buy finds no price and its exit nothing to sell.
    assert run_null_control(*arguments, k=3, seed=5)["rejects_mean"] == 2.0
