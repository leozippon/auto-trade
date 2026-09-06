"""Daily Broker primitives: eligibility gates, lot rules and cost math.

``configs/agent_output_template/README.md`` promises the Agent that suspension,
``missing_execution_price``, missing or non-finite daily price limits, daily
price-limit touches, insufficient cash, insufficient
sellable quantity and invalid buy lots reject the whole order, and that
commission applies on both sides with a minimum fee while stamp duty applies on
sells. Those are the numbers every backtest is scored on, so each is pinned here
against the real Broker rather than a projection of it.
"""

from __future__ import annotations

import math
import unittest
from datetime import datetime

from autotrade.environment.broker import (
    EX_DATE_PRICE_TOLERANCE,
    BrokerProfile,
    DailyBroker,
)
from autotrade.environment.broker_core import (
    LOT_SIZE,
    STAMP_DUTY_CUTOVER,
    STAR_MIN_LOT_SIZE,
    CostModel,
    is_bse_market,
    is_star_market,
    reduce_amount_reject,
    validate_buy_lot,
)
from autotrade.environment.strategy import CN_TZ, StrategyOrder

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


def _broker(**profile_fields: object) -> DailyBroker:
    broker = DailyBroker(BrokerProfile(**profile_fields))
    broker.open_day("20260105", {})
    return broker


class BrokerEligibilityTest(unittest.TestCase):
    def test_long_buy_hold_and_close_follows_t_plus_one(self) -> None:
        broker = _broker(initial_cash=100_000, min_commission_cny=0, slippage_bps=0)
        filled = broker.execute(_order(), _bar(), matched_at=MATCHED_AT, raw_price=10.0)
        self.assertEqual(filled.status, "filled")
        self.assertEqual(broker.positions["000001.SZ"].quantity, 100)
        # Bought today, not sellable today.
        self.assertEqual(broker.positions["000001.SZ"].available_quantity, 0)
        same_day = broker.execute(_order("sell"), _bar(), matched_at=MATCHED_AT, raw_price=10.0)
        self.assertEqual(same_day.status, "rejected")
        self.assertEqual(same_day.reason, "insufficient_available_position")
        # The next trading day releases it.
        broker.open_day("20260106", {"000001.SZ": _bar()})
        self.assertEqual(broker.positions["000001.SZ"].available_quantity, 100)
        closed = broker.execute(_order("sell"), _bar(), matched_at=MATCHED_AT, raw_price=10.0)
        self.assertEqual(closed.status, "filled")
        self.assertNotIn("000001.SZ", broker.positions)

    def test_suspension_and_missing_price_reject_the_whole_order(self) -> None:
        broker = _broker()
        suspended = broker.execute(
            _order(), _bar(is_suspended=True), matched_at=MATCHED_AT, raw_price=10.0
        )
        self.assertEqual((suspended.status, suspended.reason), ("rejected", "suspended"))
        no_bar = broker.execute(_order(), None, matched_at=MATCHED_AT, raw_price=10.0)
        self.assertEqual((no_bar.status, no_bar.reason), ("rejected", "missing_execution_price"))
        no_price = broker.execute(_order(), _bar(), matched_at=MATCHED_AT, raw_price=None)
        self.assertEqual((no_price.status, no_price.reason), ("rejected", "missing_execution_price"))
        for bad in (float("nan"), float("inf"), 0.0, -1.0, True):
            with self.subTest(price=bad):
                rejected = broker.execute(_order(), _bar(), matched_at=MATCHED_AT, raw_price=bad)
                self.assertEqual(rejected.reason, "missing_execution_price")
        self.assertEqual(broker.cash, broker.profile.initial_cash)

    def test_limit_up_blocks_buy_and_limit_down_blocks_sell(self) -> None:
        broker = _broker(initial_cash=1_000_000, slippage_bps=0)
        blocked = broker.execute(
            _order(), _bar(up_limit=10.0), matched_at=MATCHED_AT, raw_price=10.0
        )
        self.assertEqual((blocked.status, blocked.reason), ("rejected", "daily_price_limit"))
        broker.execute(_order(), _bar(), matched_at=MATCHED_AT, raw_price=10.0)
        broker.open_day("20260106", {"000001.SZ": _bar()})
        sell_blocked = broker.execute(
            _order("sell"), _bar(down_limit=10.0), matched_at=MATCHED_AT, raw_price=10.0
        )
        self.assertEqual((sell_blocked.status, sell_blocked.reason), ("rejected", "daily_price_limit"))
        self.assertEqual(broker.positions["000001.SZ"].quantity, 100)

    def test_missing_or_non_finite_daily_limit_rejects_the_whole_order(self) -> None:
        broker = _broker(initial_cash=1_000_000, slippage_bps=0)
        missing_key = _bar()
        del missing_key["up_limit"]
        for label, bar in (
            ("missing_key", missing_key),
            ("none", _bar(up_limit=None)),
            ("nan", _bar(up_limit=float("nan"))),
            ("inf", _bar(up_limit=float("inf"))),
        ):
            with self.subTest(side="buy", limit=label):
                rejected = broker.execute(_order(), bar, matched_at=MATCHED_AT, raw_price=10.0)
                self.assertEqual(
                    (rejected.status, rejected.reason),
                    ("rejected", "missing_daily_price_limit"),
                )
                self.assertEqual(broker.positions, {})
                self.assertEqual(broker.cash, broker.profile.initial_cash)
        # Only the buy-side cap is required; a missing down_limit does not block a buy.
        filled = broker.execute(
            _order(), _bar(down_limit=None), matched_at=MATCHED_AT, raw_price=10.0
        )
        self.assertEqual(filled.status, "filled")
        broker.open_day("20260106", {"000001.SZ": _bar()})
        for label, bar in (
            ("none", _bar(down_limit=None)),
            ("nan", _bar(down_limit=float("nan"))),
            ("inf", _bar(down_limit=float("inf"))),
        ):
            with self.subTest(side="sell", limit=label):
                rejected = broker.execute(
                    _order("sell"), bar, matched_at=MATCHED_AT, raw_price=10.0
                )
                self.assertEqual(
                    (rejected.status, rejected.reason),
                    ("rejected", "missing_daily_price_limit"),
                )
                self.assertEqual(broker.positions["000001.SZ"].quantity, 100)
        closed = broker.execute(
            _order("sell"), _bar(up_limit=None), matched_at=MATCHED_AT, raw_price=10.0
        )
        self.assertEqual(closed.status, "filled")
        self.assertNotIn("000001.SZ", broker.positions)

    def test_insufficient_cash_rejects_instead_of_partially_filling(self) -> None:
        broker = _broker(initial_cash=500.0, min_commission_cny=0, slippage_bps=0)
        rejected = broker.execute(_order(), _bar(), matched_at=MATCHED_AT, raw_price=10.0)
        self.assertEqual((rejected.status, rejected.reason), ("rejected", "insufficient_cash"))
        self.assertEqual(broker.cash, 500.0)
        self.assertEqual(broker.positions, {})

    def test_execute_before_open_day_fails_fast(self) -> None:
        broker = DailyBroker(BrokerProfile())
        with self.assertRaisesRegex(RuntimeError, "open_day"):
            broker.execute(_order(), _bar(), matched_at=MATCHED_AT, raw_price=10.0)


class BuyLotLadderTest(unittest.TestCase):
    def test_board_detection(self) -> None:
        self.assertTrue(is_star_market("688001.SH"))
        self.assertTrue(is_star_market("689009.SH"))
        self.assertFalse(is_star_market("600000.SH"))
        self.assertTrue(is_bse_market("830000.BJ"))
        self.assertFalse(is_bse_market("000001.SZ"))

    def test_main_board_requires_whole_lots(self) -> None:
        validate_buy_lot(200, "000001.SZ")
        with self.assertRaisesRegex(ValueError, f"multiple of {LOT_SIZE}"):
            validate_buy_lot(150, "000001.SZ")

    def test_star_market_minimum_then_one_share_increment(self) -> None:
        validate_buy_lot(STAR_MIN_LOT_SIZE, "688001.SH")
        validate_buy_lot(STAR_MIN_LOT_SIZE + 1, "688001.SH")
        with self.assertRaisesRegex(ValueError, "at least 200"):
            validate_buy_lot(199, "688001.SH")

    def test_bse_minimum_then_one_share_increment(self) -> None:
        validate_buy_lot(LOT_SIZE, "830000.BJ")
        validate_buy_lot(LOT_SIZE + 1, "830000.BJ")
        with self.assertRaisesRegex(ValueError, "at least 100"):
            validate_buy_lot(99, "830000.BJ")

    def test_invalid_buy_lot_rejects_at_the_broker(self) -> None:
        broker = _broker()
        rejected = broker.execute(
            _order(quantity=150), _bar(), matched_at=MATCHED_AT, raw_price=10.0
        )
        self.assertEqual((rejected.status, rejected.reason), ("rejected", "invalid_buy_lot"))

    def test_odd_lot_tail_must_exit_in_one_declaration(self) -> None:
        # 零股必须一次性申报卖出: corporate actions legitimately create odd
        # positions, so reduces cannot reuse the strict buy ladder.
        self.assertIsNone(reduce_amount_reject(100, 150, "000001.SZ"))
        self.assertIsNone(reduce_amount_reject(150, 150, "000001.SZ"))
        # The 50-share tail of a 150-share holding may be declared on its own.
        self.assertIsNone(reduce_amount_reject(50, 150, "000001.SZ"))
        # A sub-lot slice of a whole-lot holding has no odd tail to carry.
        self.assertEqual(reduce_amount_reject(50, 200, "000001.SZ"), "amount_below_lot_size")
        self.assertEqual(reduce_amount_reject(120, 150, "000001.SZ"), "amount_not_lot_aligned")
        # A whole odd position below the board minimum is exitable in full.
        self.assertIsNone(reduce_amount_reject(50, 50, "688001.SH"))
        self.assertEqual(reduce_amount_reject(50, 150, "688001.SH"), "amount_below_lot_size")
        self.assertIsNone(reduce_amount_reject(50, 50, "830000.BJ"))


class CostModelTest(unittest.TestCase):
    def test_slippage_is_directional(self) -> None:
        costs = CostModel(slippage_bps=10.0)
        self.assertAlmostEqual(costs.fill_price(100.0, action="buy"), 100.1)
        self.assertAlmostEqual(costs.fill_price(100.0, action="sell"), 99.9)
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(price=bad), self.assertRaises(ValueError):
                costs.fill_price(bad, action="buy")

    def test_commission_has_a_minimum_and_transfer_fee_applies_both_sides(self) -> None:
        costs = CostModel(commission_bps=1.0, min_commission_cny=5.0, transfer_fee_bps=0.1)
        self.assertAlmostEqual(costs.commission(1_000.0), 5.0)  # minimum wins
        self.assertAlmostEqual(costs.commission(100_000.0), 10.0)
        self.assertAlmostEqual(costs.transfer_fee(100_000.0), 1.0)
        self.assertAlmostEqual(costs.trade_fee(100_000.0), 11.0)
        buy_fee, buy_duty = costs.fees(100_000.0, action="buy", trade_date="20260105")
        self.assertAlmostEqual(buy_fee, 11.0)
        self.assertEqual(buy_duty, 0.0)  # 印花税 sell-side only

    def test_stamp_duty_follows_the_2023_08_28_cutover(self) -> None:
        costs = CostModel()
        self.assertEqual(STAMP_DUTY_CUTOVER, "20230828")
        self.assertAlmostEqual(costs.stamp_duty_on_sale(100_000.0, "20230827"), 100.0)  # 10 bps
        self.assertAlmostEqual(costs.stamp_duty_on_sale(100_000.0, STAMP_DUTY_CUTOVER), 50.0)
        self.assertAlmostEqual(costs.stamp_duty_on_sale(100_000.0, "20260105"), 50.0)

    def test_broker_prices_the_duty_from_the_trading_day(self) -> None:
        for trade_date, expected in (("20230825", 100.0), ("20230828", 50.0)):
            with self.subTest(trade_date=trade_date):
                broker = DailyBroker(
                    BrokerProfile(initial_cash=1_000_000, slippage_bps=0, min_commission_cny=0)
                )
                broker.open_day(trade_date, {})
                broker.execute(
                    _order(quantity=10_000), _bar(up_limit=99.0), matched_at=MATCHED_AT, raw_price=10.0
                )
                broker.open_day(trade_date, {})
                broker.positions["000001.SZ"].available_quantity = 10_000
                sold = broker.execute(
                    _order("sell", quantity=10_000),
                    _bar(down_limit=1.0),
                    matched_at=MATCHED_AT,
                    raw_price=10.0,
                )
                self.assertEqual(sold.status, "filled")
                self.assertAlmostEqual(sold.stamp_duty, expected)

    def test_fees_reject_a_non_positive_or_non_finite_notional(self) -> None:
        costs = CostModel()
        for bad in (0.0, -1.0, float("nan")):
            with self.subTest(notional=bad), self.assertRaises(ValueError):
                costs.fees(bad, action="buy", trade_date="20260105")


class BrokerProfileValidationTest(unittest.TestCase):
    def test_record_round_trips_every_constructor_field_and_names_the_cutover(self) -> None:
        profile = BrokerProfile(
            initial_cash=250_000.0,
            commission_bps=2.5,
            min_commission_cny=1.0,
            stamp_duty_sell_bps_before_cutover=10.0,
            stamp_duty_sell_bps_from_cutover=5.0,
            transfer_fee_bps=0.2,
            slippage_bps=3.0,
        )
        record = profile.to_record()
        self.assertEqual(record["initial_cash"], 250_000.0)
        self.assertEqual(record["commission_bps"], 2.5)
        self.assertEqual(record["transfer_fee_bps"], 0.2)
        self.assertEqual(record["stamp_duty_cutover_date"], STAMP_DUTY_CUTOVER)
        # Every constructor field surfaces in the manifest record.
        for name in (
            "initial_cash", "commission_bps", "min_commission_cny",
            "stamp_duty_sell_bps_before_cutover", "stamp_duty_sell_bps_from_cutover",
            "transfer_fee_bps", "slippage_bps",
        ):
            self.assertIn(name, record)

    def test_initial_cash_must_be_a_positive_finite_number(self) -> None:
        for value in (0, -1.0, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "initial_cash"):
                BrokerProfile(initial_cash=value)

    def test_cost_fields_reject_negatives_booleans_and_non_finite_values(self) -> None:
        for name in (
            "commission_bps", "min_commission_cny", "stamp_duty_sell_bps_before_cutover",
            "stamp_duty_sell_bps_from_cutover", "transfer_fee_bps", "slippage_bps",
        ):
            for value in (-1.0, float("nan"), float("inf"), True):
                with self.subTest(field=name, value=value), self.assertRaisesRegex(ValueError, name):
                    BrokerProfile(**{name: value})

    def test_default_profile_matches_the_documented_a_share_costs(self) -> None:
        profile = BrokerProfile()
        self.assertEqual(profile.commission_bps, 1.0)
        self.assertEqual(profile.min_commission_cny, 5.0)
        self.assertEqual(profile.slippage_bps, 5.0)
        self.assertEqual(profile.transfer_fee_bps, 0.1)
        self.assertEqual(profile.stamp_duty_sell_bps_before_cutover, 10.0)
        self.assertEqual(profile.stamp_duty_sell_bps_from_cutover, 5.0)


class BrokerAccountingTest(unittest.TestCase):
    def test_cash_position_and_cost_accounting_are_consistent_after_a_round_trip(self) -> None:
        broker = _broker(initial_cash=100_000, min_commission_cny=0, slippage_bps=0, transfer_fee_bps=0)
        bought = broker.execute(_order(), _bar(), matched_at=MATCHED_AT, raw_price=10.0)
        commission = 1_000.0 * 1.0 / 10_000.0
        self.assertAlmostEqual(bought.commission, commission)
        self.assertAlmostEqual(broker.cash, 100_000 - 1_000.0 - commission)
        broker.open_day("20260106", {"000001.SZ": _bar()})
        sold = broker.execute(_order("sell"), _bar(), matched_at=MATCHED_AT, raw_price=10.0)
        self.assertAlmostEqual(sold.stamp_duty, 1_000.0 * 5.0 / 10_000.0)
        self.assertAlmostEqual(
            broker.cash, 100_000 - 2 * commission - sold.stamp_duty, places=6
        )
        self.assertAlmostEqual(broker.traded_notional, 2_000.0)
        self.assertAlmostEqual(broker.fees_paid, 2 * commission)
        self.assertAlmostEqual(broker.stamp_duty_paid, sold.stamp_duty)

    def test_rejected_orders_move_no_money_and_are_counted_separately(self) -> None:
        broker = _broker(initial_cash=1_000)
        broker.execute(_order(quantity=1_000), _bar(), matched_at=MATCHED_AT, raw_price=10.0)
        self.assertEqual(broker.cash, 1_000)
        self.assertEqual(broker.traded_notional, 0.0)
        self.assertEqual(broker.fees_paid, 0.0)
        self.assertEqual(broker.reject_counts, {"insufficient_cash": 1})

    def test_equity_marks_to_the_latest_visible_close(self) -> None:
        broker = _broker(initial_cash=100_000, min_commission_cny=0, slippage_bps=0, transfer_fee_bps=0)
        broker.execute(_order(), _bar(), matched_at=MATCHED_AT, raw_price=10.0)
        broker.mark({"000001.SZ": _bar(close=12.0)})
        self.assertAlmostEqual(broker.positions["000001.SZ"].last_price, 12.0)
        self.assertAlmostEqual(broker.equity(), broker.cash + 100 * 12.0)
        self.assertTrue(math.isfinite(broker.equity()))

    def test_account_snapshot_exposes_cash_and_quantities_only(self) -> None:
        broker = _broker(initial_cash=100_000, min_commission_cny=0, slippage_bps=0)
        broker.execute(_order(), _bar(), matched_at=MATCHED_AT, raw_price=10.0)
        cash, positions = broker.account_snapshot()
        self.assertAlmostEqual(cash, broker.cash)
        self.assertEqual(positions, {"000001.SZ": 100})


class BrokerCorporateActionTest(unittest.TestCase):
    """Ex-dates are settled from the exchange's reference price (``pre_close``).

    A held name whose ``pre_close`` differs from its last close is reset so the
    account is worth at ``pre_close`` exactly what it was worth at the last
    close: the cash dividend is credited, the share count follows the reset,
    fractional shares are paid in cash, and created shares stay locked until
    the next day. Without this, every bonus issue read as a loss and every cash
    dividend vanished.
    """

    def _holding(self, quantity: int = 1000) -> DailyBroker:
        broker = _broker(initial_cash=100_000, min_commission_cny=0, slippage_bps=0, transfer_fee_bps=0)
        filled = broker.execute(_order(quantity=quantity), _bar(), matched_at=MATCHED_AT, raw_price=10.0)
        self.assertEqual(filled.status, "filled")
        broker.mark({"000001.SZ": _bar(close=10.0)})
        return broker

    def test_bonus_issue_with_a_cash_dividend_preserves_equity_at_pre_close(self) -> None:
        # 10 送 10 plus 0.5 CNY cash: pre_close = (10 - 0.5) / 2 = 4.75.
        broker = self._holding(1000)
        cash_before, equity_before = broker.cash, broker.equity()
        broker.open_day("20260106", {"000001.SZ": _bar(pre_close=4.75)}, {"000001.SZ": 0.5})
        position = broker.positions["000001.SZ"]
        self.assertEqual(position.quantity, 2000)
        self.assertAlmostEqual(broker.cash, cash_before + 0.5 * 1000)
        self.assertAlmostEqual(position.last_price, 4.75)
        self.assertAlmostEqual(broker.equity(), equity_before, places=6)
        self.assertAlmostEqual(broker.cash + 2000 * 4.75, equity_before, places=6)
        [action] = broker.corporate_actions
        self.assertEqual(
            (action.trade_date, action.symbol, action.quantity_before, action.quantity_after),
            ("20260106", "000001.SZ", 1000, 2000),
        )
        self.assertAlmostEqual(action.cash_credit, 500.0)
        self.assertEqual(action.cash_per_share, 0.5)

    def test_created_shares_are_locked_until_the_next_day(self) -> None:
        broker = self._holding(1000)
        broker.open_day("20260106", {"000001.SZ": _bar(pre_close=4.75)}, {"000001.SZ": 0.5})
        self.assertEqual(broker.positions["000001.SZ"].available_quantity, 1000)
        ex_day = _bar(pre_close=4.75, up_limit=5.2, down_limit=4.3)
        blocked = broker.execute(_order("sell", quantity=2000), ex_day, matched_at=MATCHED_AT, raw_price=4.8)
        self.assertEqual((blocked.status, blocked.reason), ("rejected", "insufficient_available_position"))
        allowed = broker.execute(_order("sell", quantity=1000), ex_day, matched_at=MATCHED_AT, raw_price=4.8)
        self.assertEqual(allowed.status, "filled")
        self.assertEqual(broker.positions["000001.SZ"].quantity, 1000)
        broker.open_day("20260107", {"000001.SZ": _bar(pre_close=4.8)})
        self.assertEqual(broker.positions["000001.SZ"].available_quantity, 1000)

    def test_realized_pnl_after_an_ex_date_closes_the_cash_loop(self) -> None:
        # The basis carries into the new share count and the payout reduces
        # it, so selling everything realizes exactly the account's cash change.
        broker = _broker(initial_cash=100_000, slippage_bps=0)
        broker.execute(_order(quantity=1000), _bar(), matched_at=MATCHED_AT, raw_price=10.0)
        broker.mark({"000001.SZ": _bar(close=10.0)})
        broker.open_day("20260106", {"000001.SZ": _bar(pre_close=4.75)}, {"000001.SZ": 0.5})
        broker.open_day("20260107", {"000001.SZ": _bar(pre_close=4.75)})
        sold = broker.execute(
            _order("sell", quantity=2000),
            _bar(pre_close=4.75, up_limit=5.2, down_limit=4.3),
            matched_at=MATCHED_AT,
            raw_price=5.0,
        )
        self.assertEqual(sold.status, "filled")
        self.assertNotIn("000001.SZ", broker.positions)
        self.assertAlmostEqual(sold.realized_pnl, broker.cash - 100_000, places=6)

    def test_pure_cash_dividend_credits_cash_and_keeps_the_shares(self) -> None:
        broker = self._holding(1000)
        cash_before, equity_before = broker.cash, broker.equity()
        broker.open_day("20260106", {"000001.SZ": _bar(pre_close=9.5)}, {"000001.SZ": 0.5})
        self.assertEqual(broker.positions["000001.SZ"].quantity, 1000)
        self.assertAlmostEqual(broker.cash, cash_before + 500.0)
        self.assertAlmostEqual(broker.equity(), equity_before, places=6)

    def test_a_day_without_a_reset_changes_nothing(self) -> None:
        broker = self._holding(1000)
        cash_before = broker.cash
        for day, pre_close in (
            ("20260106", 10.0),
            ("20260107", 10.0 + EX_DATE_PRICE_TOLERANCE),
            ("20260108", 10.0 - EX_DATE_PRICE_TOLERANCE),
        ):
            with self.subTest(pre_close=pre_close):
                broker.open_day(day, {"000001.SZ": _bar(pre_close=pre_close)})
                self.assertEqual(broker.positions["000001.SZ"].quantity, 1000)
                self.assertEqual(broker.cash, cash_before)
        self.assertEqual(broker.corporate_actions, [])

    def test_a_recorded_dividend_without_a_reset_credits_only_the_cash(self) -> None:
        broker = self._holding(1000)
        cash_before = broker.cash
        broker.open_day("20260106", {"000001.SZ": _bar(pre_close=10.0)}, {"000001.SZ": 0.2})
        self.assertEqual(broker.positions["000001.SZ"].quantity, 1000)
        self.assertAlmostEqual(broker.cash, cash_before + 200.0)
        self.assertEqual(broker.corporate_actions[0].quantity_after, 1000)

    def test_fractional_shares_are_paid_out_in_cash(self) -> None:
        # 10 转 5 on 100 shares: the exchange rounds 10 / 1.5 to 6.67, so the
        # implied 149.925 shares become 149 whole shares plus their cash value.
        broker = self._holding(100)
        equity_before = broker.equity()
        broker.open_day("20260106", {"000001.SZ": _bar(pre_close=6.67)})
        position = broker.positions["000001.SZ"]
        self.assertEqual(position.quantity, 149)
        [action] = broker.corporate_actions
        self.assertAlmostEqual(action.cash_credit, 100 * 10.0 - 149 * 6.67, places=6)
        self.assertGreater(action.cash_credit, 0.0)
        self.assertAlmostEqual(broker.cash + 149 * 6.67, equity_before, places=6)

    def test_a_held_name_without_pre_close_fails_fast(self) -> None:
        for label, bar in (
            ("missing_key", {key: value for key, value in _bar().items() if key != "pre_close"}),
            ("none", _bar(pre_close=None)),
            ("nan", _bar(pre_close=float("nan"))),
            ("zero", _bar(pre_close=0.0)),
        ):
            with self.subTest(pre_close=label):
                broker = self._holding(1000)
                with self.assertRaisesRegex(ValueError, "pre_close"):
                    broker.open_day("20260106", {"000001.SZ": bar})
        # A name with no bar at all today (suspended) is simply carried.
        broker = self._holding(1000)
        broker.open_day("20260106", {})
        self.assertEqual(broker.positions["000001.SZ"].quantity, 1000)

    def test_an_impossible_reset_fails_fast(self) -> None:
        broker = self._holding(1000)
        with self.assertRaisesRegex(ValueError, "share multiplier"):
            broker.open_day("20260106", {"000001.SZ": _bar(pre_close=4.75)}, {"000001.SZ": 12.0})
        broker = self._holding(1000)
        with self.assertRaisesRegex(ValueError, "cash dividend"):
            broker.open_day("20260106", {"000001.SZ": _bar(pre_close=4.75)}, {"000001.SZ": float("nan")})


class PositionCapTest(unittest.TestCase):
    """``max_total_holdings`` and ``max_single_name_weight`` are risk limits,
    not advisory settings: an order that breaches one is rejected outright."""

    def test_total_holdings_cap_blocks_a_new_name_but_not_a_top_up(self) -> None:
        broker = _broker(initial_cash=1_000_000, min_commission_cny=0, slippage_bps=0,
                         max_total_holdings=1)
        first = broker.execute(_order(), _bar(), matched_at=MATCHED_AT, raw_price=10.0)
        self.assertEqual(first.status, "filled")
        second = broker.execute(
            _order(symbol="600000.SH"), _bar(), matched_at=MATCHED_AT, raw_price=10.0
        )
        self.assertEqual((second.status, second.reason), ("rejected", "max_holdings_reached"))
        # Adding to an existing holding does not open a new name.
        top_up = broker.execute(_order(), _bar(), matched_at=MATCHED_AT, raw_price=10.0)
        self.assertEqual(top_up.status, "filled")
        self.assertEqual(broker.reject_counts, {"max_holdings_reached": 1})

    def test_single_name_weight_cap_measures_against_initial_equity(self) -> None:
        # 200 shares at 10.0 = 2 000 notional; the cap is 0.001 * 1e6 = 1 000.
        broker = _broker(initial_cash=1_000_000, min_commission_cny=0, slippage_bps=0,
                         max_single_name_weight=0.001)
        rejected = broker.execute(
            _order(quantity=200), _bar(), matched_at=MATCHED_AT, raw_price=10.0
        )
        self.assertEqual((rejected.status, rejected.reason), ("rejected", "single_name_weight_cap"))
        # 100 shares = 1 000 notional sits exactly on the cap and fills.
        allowed = broker.execute(_order(), _bar(), matched_at=MATCHED_AT, raw_price=10.0)
        self.assertEqual(allowed.status, "filled")
        # A further lot would breach it, counting what is already held.
        breach = broker.execute(_order(), _bar(), matched_at=MATCHED_AT, raw_price=10.0)
        self.assertEqual(breach.reason, "single_name_weight_cap")

    def test_the_caps_are_off_by_default(self) -> None:
        profile = BrokerProfile()
        self.assertIsNone(profile.max_total_holdings)
        self.assertIsNone(profile.max_single_name_weight)
        broker = _broker(initial_cash=1_000_000, min_commission_cny=0, slippage_bps=0)
        for symbol in ("000001.SZ", "600000.SH", "000002.SZ"):
            filled = broker.execute(
                _order(symbol=symbol), _bar(), matched_at=MATCHED_AT, raw_price=10.0
            )
            self.assertEqual(filled.status, "filled", symbol)

    def test_cap_values_are_validated_at_construction(self) -> None:
        for value in (0, -1, 1.5, True):
            with self.subTest(max_total_holdings=value):
                with self.assertRaisesRegex(ValueError, "max_total_holdings"):
                    BrokerProfile(max_total_holdings=value)
        for value in (0.0, -0.1, float("nan"), float("inf"), True):
            with self.subTest(max_single_name_weight=value):
                with self.assertRaisesRegex(ValueError, "max_single_name_weight"):
                    BrokerProfile(max_single_name_weight=value)

    def test_the_caps_and_the_profile_identity_reach_the_manifest_record(self) -> None:
        record = BrokerProfile(max_total_holdings=5, max_single_name_weight=0.2).to_record()
        self.assertEqual(record["max_total_holdings"], 5)
        self.assertEqual(record["max_single_name_weight"], 0.2)
        # The Agent is told which cost profile it is authored against.
        self.assertEqual(record["profile_id"], "gjzq_cash")
        self.assertEqual(record["source"], "docs/environment-design.md §3.4")
