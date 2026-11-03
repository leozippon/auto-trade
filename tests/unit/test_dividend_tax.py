"""Record-date historical tax rules, cash settlement, FIFO and Paper recovery."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from autotrade.environment.broker import BrokerProfile, DailyBroker, Lot, Position
from autotrade.environment.distributions import Distribution
from autotrade.environment.dividend_tax import (
    NEW_RULE,
    OLD_RULE,
    dividend_tax_rate,
    dividend_tax_rule,
)
from autotrade.environment.replay import DailyMarketData, run_daily_replay
from autotrade.environment.replay.engine import StrategyDataView
from autotrade.environment.replay.null_control import run_null_control
from autotrade.environment.strategy import StrategySchedule
from autotrade.paper import DailyPaperEngine, PaperEngineError

SYMBOL = "000001.SZ"
OTHER = "000002.SZ"
SCHEDULE = StrategySchedule("day", "08:30")
# Cost-free, so every cash figure below is dividends, prices and the tax alone.
UNTAXED = BrokerProfile(
    initial_cash=100_000.0,
    commission_bps=0.0,
    min_commission_cny=0.0,
    transfer_fee_bps=0.0,
    stamp_duty_sell_bps_before_cutover=0.0,
    stamp_duty_sell_bps_from_cutover=0.0,
    slippage_bps=0.0,
)
TAXED = replace(UNTAXED, dividend_tax=True)


def _frame(prices: dict[str, list[tuple[str, float, float]]]) -> pd.DataFrame:
    """Bars opening at ``pre_close`` and closing at ``close``, per symbol."""

    return pd.DataFrame(
        [
            {
                "ts_code": symbol,
                "trade_date": day,
                "open": pre_close,
                "close": close,
                "pre_close": pre_close,
                "up_limit": round(pre_close * 1.1, 2),
                "down_limit": round(pre_close * 0.9, 2),
                "circ_mv": 1.0e9,
            }
            for symbol, rows in prices.items()
            for day, pre_close, close in rows
        ]
    )


def _actions(rows: list[tuple[str, str, float, float, float]], *, bonus_column: bool = True) -> pd.DataFrame:
    """``(symbol, ex_date, cash, stock, bonus)`` per share; ``stock`` is 送+转."""

    frame = pd.DataFrame(
        [
            {
                "ts_code": symbol,
                "ex_date": ex_date,
                "record_date": ex_date,
                "pay_date": ex_date,
                "div_listdate": "",
                "cash_per_share": cash,
                "stock_per_share": stock,
                "bonus_per_share": bonus,
            }
            for symbol, ex_date, cash, stock, bonus in rows
        ]
    )
    return frame if bonus_column else frame.drop(columns=["bonus_per_share"])


def _script(orders: dict[str, list[tuple[str, int, str]]]):
    """``{day: [(action, quantity, "09:30" | "15:00")]}`` on ``SYMBOL``."""

    def strategy(context):
        day = context.inference_at.strftime("%Y%m%d")
        stamp = context.inference_at.strftime("%Y-%m-%d")
        return [
            {"symbol": SYMBOL, "action": action, "quantity": quantity, "execute_at": f"{stamp}T{at}:00+08:00"}
            for action, quantity, at in orders.get(day, [])
        ]

    return strategy


def _replay(profile: BrokerProfile, frame: pd.DataFrame, actions: pd.DataFrame, orders) -> dict:
    return run_daily_replay(
        daily=frame,
        strategy=_script(orders),
        schedule=SCHEDULE,
        profile=profile,
        corporate_actions=actions,
    ).to_record()


def _filled(record: dict) -> list[dict]:
    return [row for row in record["executions"] if row["status"] == "filled"]


@pytest.mark.parametrize("rule", [OLD_RULE, NEW_RULE])
def test_the_holding_period_tiers_follow_the_calendar_month_and_year(rule):
    # Held from the buy through the day before the sale: a sale on the same
    # day one month on has held exactly one month, which is still 20 %.
    assert dividend_tax_rate("20240110", "20240111", rule) == 0.20
    assert dividend_tax_rate("20240110", "20240210", rule) == 0.20
    assert dividend_tax_rate("20240110", "20240211", rule) == 0.10
    # A month without the day ends at its own last day.
    assert dividend_tax_rate("20240131", "20240301", rule) == 0.20
    assert dividend_tax_rate("20240131", "20240302", rule) == 0.10
    assert dividend_tax_rate("20230703", "20240703", rule) == 0.10
    assert dividend_tax_rate("20230703", "20240704", rule) == rule.rate_beyond
    assert dividend_tax_rate("20240229", "20250301", rule) == 0.10
    assert dividend_tax_rate("20240229", "20250302", rule) == rule.rate_beyond
    assert dividend_tax_rate("20150801", "20150817", OLD_RULE) == 0.20
    assert dividend_tax_rate("20130801", "20150817", OLD_RULE) == 0.05
    assert dividend_tax_rule("20150907") == OLD_RULE
    assert dividend_tax_rule("20150908") == NEW_RULE
    with pytest.raises(ValueError, match="cannot be sold"):
        dividend_tax_rate("20240110", "20240110", rule)


def test_sales_take_fifo_lots_and_pay_each_lots_tier_on_its_own_dividends():
    """Two buys of 1000, a 0.5 dividend on both, then three sales.

    The first sale of 1500 takes all of the January lot (held over a month:
    10 % of its 500) and half the March lot (held under a month: 20 % of its
    250). The March lot's other 500 shares then receive a 0.3 dividend and are
    sold on the day a year is up (10 %) and the day after (nothing).
    """

    days = ["20240110", "20240304", "20240305", "20240320", "20240603", "20250304", "20250305"]
    pre = [10.0, 10.0, 9.5, 9.5, 9.2, 9.2, 9.2]
    frame = _frame({SYMBOL: [(day, price, price) for day, price in zip(days, pre, strict=True)]})
    actions = _actions([(SYMBOL, "20240305", 0.5, 0.0, 0.0), (SYMBOL, "20240603", 0.3, 0.0, 0.0)])
    orders = {
        "20240110": [("buy", 1000, "09:30")],
        "20240304": [("buy", 1000, "09:30")],
        "20240320": [("sell", 1500, "09:30")],
        "20250304": [("sell", 100, "09:30")],
        "20250305": [("sell", 400, "09:30")],
    }
    taxed = _replay(TAXED, frame, actions, orders)
    untaxed = _replay(UNTAXED, frame, actions, orders)

    taxes = [row["dividend_tax"] for row in _filled(taxed)]
    assert taxes == pytest.approx([0.0, 0.0, 0.1 * 500 + 0.2 * 250, 0.1 * 400 / 5, 0.0])
    assert taxed["stats"]["dividend_tax_paid"] == pytest.approx(108.0)
    assert untaxed["stats"]["dividend_tax_paid"] == 0.0
    assert all(row["dividend_tax"] == 0.0 for row in untaxed["executions"])
    # The tax is a cash debit at the sale and nothing else: the dividends are
    # credited gross on the ex-date in both runs, and realized P&L carries it.
    assert taxed["corporate_actions"] == untaxed["corporate_actions"]
    gap = [
        untaxed_row["cash"] - taxed_row["cash"]
        for taxed_row, untaxed_row in zip(taxed["equity_curve"], untaxed["equity_curve"], strict=True)
    ]
    assert gap == pytest.approx([0.0, 0.0, 0.0, 100.0, 100.0, 108.0, 108.0])
    assert [
        untaxed_row["realized_pnl"] - taxed_row["realized_pnl"]
        for taxed_row, untaxed_row in zip(_filled(taxed)[2:], _filled(untaxed)[2:], strict=True)
    ] == pytest.approx([100.0, 8.0, 0.0])
    assert taxed["equity_curve"][-1]["positions"] == {}


def test_selling_at_the_ex_date_open_pays_a_fifth_of_the_dividend():
    """The dividend book's round trip: bought the day before, sold at the open
    of the ex-date, after the Broker credited the gross dividend."""

    frame = _frame({SYMBOL: [("20240102", 10.0, 10.0), ("20240103", 9.5, 9.6)]})
    actions = _actions([(SYMBOL, "20240103", 0.5, 0.0, 0.0)])
    orders = {"20240102": [("buy", 1000, "15:00")], "20240103": [("sell", 1000, "09:30")]}
    taxed = _replay(TAXED, frame, actions, orders)
    untaxed = _replay(UNTAXED, frame, actions, orders)

    [action] = taxed["corporate_actions"]
    assert action["cash_credit"] == pytest.approx(500.0)
    buy, sell = _filled(taxed)
    assert (buy["dividend_tax"], sell["dividend_tax"]) == (0.0, pytest.approx(100.0))
    # 10 000 out, 500 of dividend and 9 500 of sale in, 100 of tax out.
    assert taxed["equity_curve"][-1]["cash"] == pytest.approx(100_000.0 - 100.0)
    assert untaxed["equity_curve"][-1]["cash"] == pytest.approx(100_000.0)


@pytest.mark.parametrize(("bonus", "income"), [(0.3, 1000 * 0.5 + 1000 * 0.3 * 1.0), (0.0, 1000 * 0.5)])
def test_bonus_shares_are_income_at_par_and_conversions_are_not(bonus: float, income: float):
    """10 送3 转7 (or 转10) with 0.5 cash: 1000 shares become 2000. The new
    shares join their lot, so selling half pays half the lot's tax."""

    frame = _frame(
        {
            SYMBOL: [
                ("20240102", 10.0, 10.0),
                ("20240103", 4.75, 4.8),
                ("20240104", 4.8, 4.8),
                ("20240105", 4.8, 4.8),
            ]
        }
    )
    actions = _actions([(SYMBOL, "20240103", 0.5, 1.0, bonus)])
    orders = {
        "20240102": [("buy", 1000, "15:00")],
        "20240104": [("sell", 1000, "09:30")],
        "20240105": [("sell", 1000, "09:30")],
    }
    record = _replay(TAXED, frame, actions, orders)

    [action] = record["corporate_actions"]
    assert (action["quantity_before"], action["quantity_after"]) == (1000, 2000)
    assert [row["dividend_tax"] for row in _filled(record)] == pytest.approx(
        [0.0, 0.2 * income / 2, 0.2 * income / 2]
    )


def test_a_taxed_replay_refuses_an_ex_date_table_without_the_bonus_split():
    frame = _frame({SYMBOL: [("20240102", 10.0, 10.0), ("20240103", 9.5, 9.6)]})
    old_table = _actions([(SYMBOL, "20240103", 0.5, 0.0, 0.0)], bonus_column=False)
    assert DailyMarketData(frame, old_table).distributions_for_day("20240103")[SYMBOL].bonus_per_share is None
    assert DailyMarketData(frame).distributions_for_day("20240103") == {}
    orders = {"20240102": [("buy", 1000, "15:00")]}
    with pytest.raises(ValueError, match="bonus_per_share"):
        _replay(TAXED, frame, old_table, orders)
    # The same slot still replays an arm that does not charge the tax.
    assert _replay(UNTAXED, frame, old_table, orders)["corporate_actions"][0]["cash_credit"] == pytest.approx(500.0)
    with pytest.raises(ValueError, match="bonus_per_share"):
        DailyMarketData(frame, _actions([(SYMBOL, "20240103", 0.5, 1.0, -0.1)]))
    with pytest.raises(ValueError, match="dividend_tax must be a boolean"):
        BrokerProfile(dividend_tax="yes")  # type: ignore[arg-type]


def test_the_zero_skill_panel_charges_its_draws_the_arms_own_tax():
    """The only replacement name pays its own dividend on the exit day, so a
    taxed panel's draws lose a fifth of it and an untaxed panel's do not."""

    frame = _frame(
        {
            SYMBOL: [("20240102", 10.0, 10.0), ("20240103", 9.6, 9.6)],
            OTHER: [("20240102", 10.0, 10.0), ("20240103", 9.7, 9.7)],
        }
    )
    actions = _actions([(SYMBOL, "20240103", 0.4, 0.0, 0.0), (OTHER, "20240103", 0.3, 0.0, 0.0)])
    orders = {"20240102": [("buy", 1000, "15:00")], "20240103": [("sell", 1000, "09:30")]}
    result = run_daily_replay(
        daily=frame, strategy=_script(orders), schedule=SCHEDULE, profile=TAXED, corporate_actions=actions
    )

    def panel(profile: BrokerProfile) -> dict:
        return run_null_control(
            result, frame, {"20240102": 0.0, "20240103": 0.0}, profile, SCHEDULE,
            k=2, seed=1, corporate_actions=actions, panel=True,
        )

    taxed, untaxed = panel(TAXED), panel(UNTAXED)
    assert taxed["rejects_mean"] == untaxed["rejects_mean"] == 0.0
    # 1000 shares of the replacement paid 300 of dividend: the taxed draw
    # keeps 240 of it.
    assert untaxed["null_excess_mean"] - taxed["null_excess_mean"] == pytest.approx(60.0 / 100_000.0)
    assert taxed["observed_excess"] == untaxed["observed_excess"]


class _Data:
    """Committed bars and ex-date table through the session before a decision."""

    def __init__(self, frame: pd.DataFrame, actions: pd.DataFrame, sessions, release_end: str) -> None:
        self.market = DailyMarketData(frame[frame["trade_date"] <= release_end], actions)
        self.sessions = sessions
        self.release_end = release_end
        self.generation_id = release_end
        self.vintages: dict = {}

    def context_data(self, inference_at):
        return StrategyDataView()

    def execution_price(self, symbol, when):
        return None

    def references(self, symbols, before):
        return {}

    def close(self) -> None:
        pass


class _Executor:
    def __init__(self, orders) -> None:
        self._strategy = _script(orders)

    def execute(self, context):
        return self._strategy(context)

    def close(self) -> None:
        pass


PAPER_SESSIONS = ("20260102", "20260105", "20260106", "20260107", "20260108")
PAPER_FRAME = _frame(
    {SYMBOL: [("20260105", 10.0, 10.0), ("20260106", 4.75, 4.8), ("20260107", 4.8, 4.9)]}
)
PAPER_ACTIONS = _actions([(SYMBOL, "20260106", 0.5, 1.0, 0.3)])
PAPER_ORDERS = {"20260105": [("buy", 1000, "15:00")], "20260107": [("sell", 2000, "09:30")]}


def _paper(tmp_path: Path, profile: BrokerProfile) -> DailyPaperEngine:
    strategy = tmp_path / "main.py"
    strategy.write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    return DailyPaperEngine(
        strategy_path=strategy,
        strategy_revision="revision_1",
        state_root=tmp_path / "paper",
        data_factory=lambda start, trade_date: _Data(
            PAPER_FRAME, PAPER_ACTIONS, PAPER_SESSIONS, max(day for day in PAPER_SESSIONS if day < trade_date)
        ),
        profile=profile,
        executor_factory=lambda *_mounts: _Executor(PAPER_ORDERS),
    )


def _state(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "paper" / ".paper_state.json").read_text(encoding="utf-8"))


def test_a_taxed_paper_book_carries_its_lots_across_runs(tmp_path: Path):
    engine = _paper(tmp_path, TAXED)
    for day in PAPER_SESSIONS[1:4]:
        engine.run_day(day)
    # Settled through the ex-date: the lot grew to 2000 shares and holds 800
    # of income (500 cash, 300 bonus at par) in the checkpoint.
    assert _state(tmp_path)["account"]["lots"] == {
        SYMBOL: [{"quantity": 2000, "acquired": "20260105",
                  "dividend_income_new": pytest.approx(800.0), "dividend_income_old": 0.0}]
    }
    # A fresh engine settles the sale from that checkpoint alone.
    _paper(tmp_path, TAXED).run_day(PAPER_SESSIONS[4])
    journal = tmp_path / "paper" / "executions_20260107.jsonl"
    [sale] = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert sale["dividend_tax"] == pytest.approx(160.0)
    state = _state(tmp_path)
    assert state["account"]["lots"] == {} and state["account"]["positions"] == []
    assert state["account"]["cash"] == pytest.approx(100_000.0 - 10_000.0 + 500.0 + 2000 * 4.8 - 160.0)


def test_a_book_checkpointed_before_the_tax_existed_is_refused(tmp_path: Path):
    """Its state's profile has no ``dividend_tax`` field. The profile record is
    read strictly, so the account is refused rather than read as untaxed by
    default, taxed or not, and the state is left as it was."""

    engine = _paper(tmp_path, UNTAXED)
    for day in PAPER_SESSIONS[1:4]:
        engine.run_day(day)
    path = tmp_path / "paper" / ".paper_state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    del state["profile"]["dividend_tax"]
    path.write_text(json.dumps(state), encoding="utf-8")
    for profile in (TAXED, UNTAXED):
        with pytest.raises(PaperEngineError, match=r"missing keys \['dividend_tax'\]"):
            _paper(tmp_path, profile).run_day(PAPER_SESSIONS[4])
    assert json.loads(path.read_text(encoding="utf-8")) == state


def test_a_checkpoint_without_its_lot_ledger_is_refused(tmp_path: Path):
    engine = _paper(tmp_path, UNTAXED)
    for day in PAPER_SESSIONS[1:4]:
        engine.run_day(day)
    path = tmp_path / "paper" / ".paper_state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    del state["account"]["lots"]
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(PaperEngineError, match="persisted lot ledger is invalid"):
        _paper(tmp_path, UNTAXED).run_day(PAPER_SESSIONS[4])


def test_old_rule_withholds_cash_before_a_pre_cutover_short_sale():
    frame = _frame({SYMBOL: [("20150803", 10.0, 10.0), ("20150804", 9.0, 9.0), ("20150817", 9.0, 9.0)]})
    orders = {"20150803": [("buy", 1000, "09:30")], "20150817": [("sell", 1000, "09:30")]}
    record = _replay(TAXED, frame, _actions([(SYMBOL, "20150804", 1.0, 0.0, 0.0)]), orders)
    [action] = record["corporate_actions"]
    assert action["cash_credit"] == 1000.0
    assert action["dividend_tax_withheld"] == 50.0
    assert record["equity_curve"][1]["cash"] == 90_950.0
    assert _filled(record)[-1]["dividend_tax"] == pytest.approx(150.0)
    assert record["stats"]["dividend_tax_paid"] == pytest.approx(200.0)
    assert record["stats"]["fees_paid"] == 0.0
    assert record["stats"]["long_return"] == pytest.approx(-200.0 / 100_000.0)
    assert record["equity_curve"][-1]["cash"] == pytest.approx(99_800.0)


def test_one_lot_keeps_both_record_date_regimes_through_partial_sales():
    frame = _frame({SYMBOL: [
        ("20150820", 10.0, 10.0), ("20150908", 9.0, 9.0), ("20150909", 8.0, 8.0),
        ("20150910", 8.0, 8.0), ("20160822", 8.0, 8.0),
    ]})
    actions = _actions([(SYMBOL, "20150908", 1.0, 0.0, 0.0), (SYMBOL, "20150909", 1.0, 0.0, 0.0)])
    # Both ex-dates and sales are in the new regime, but the first entitlement is old.
    actions["record_date"] = ["20150907", "20150908"]
    orders = {
        "20150820": [("buy", 1000, "09:30")], "20150910": [("sell", 500, "09:30")],
        "20160822": [("sell", 500, "09:30")],
    }
    record = _replay(TAXED, frame, actions, orders)
    assert [row["dividend_tax_withheld"] for row in record["corporate_actions"]] == [50.0, 0.0]
    assert [row["dividend_tax"] for row in _filled(record)] == pytest.approx([0.0, 175.0, 0.0])
    assert record["stats"]["dividend_tax_paid"] == pytest.approx(225.0)
    assert record["equity_curve"][-1]["cash"] == pytest.approx(99_775.0)
    assert record["stats"]["long_return"] == pytest.approx(-225.0 / 100_000.0)


@pytest.mark.parametrize(
    ("buy", "ex_date", "sell", "record_date", "cash", "bonus", "last_close", "pre_close"),
    [
        # 000409.SZ: flat quotes either side of an ex-date for an old record date.
        ("20180801", "20180803", "20180806", "20150717", 0.0, 0.0328041, 4.31, 4.31),
        # A new-rule cash dividend explains the price reset without bonus shares.
        ("20160504", "20160511", "20160512", "20160506", 0.3, 0.08189, 12.23, 11.93),
        # A pure bonus on a quote that does not move at all, new rule.
        ("20160906", "20160909", "20160912", "20160909", 0.0, 1.0, 10.0, 10.0),
        # The same 10 送 10 on a flat quote under the old rule: 1000 shares would
        # have paid 50 of withholding and carried 1000 of income they never got.
        ("20150803", "20150804", "20150805", "20150804", 0.0, 1.0, 10.0, 10.0),
        # Cash plus a bonus on an old-regime record date, resetting by the cash.
        ("20150803", "20150804", "20150805", "20150804", 1.0, 0.5, 10.0, 9.0),
    ],
)
def test_a_bonus_the_exchange_reset_does_not_deliver_is_not_taxed(
    buy: str, ex_date: str, sell: str, record_date: str,
    cash: float, bonus: float, last_close: float, pre_close: float,
):
    """Tax delivered cash, not a bonus that the exchange reset never delivered."""

    frame = _frame({SYMBOL: [
        (buy, last_close, last_close), (ex_date, pre_close, pre_close), (sell, pre_close, pre_close),
    ]})
    actions = _actions([(SYMBOL, ex_date, cash, 0.0, bonus)])
    actions["record_date"] = record_date
    record = _replay(TAXED, frame, actions, {
        buy: [("buy", 1000, "09:30")], sell: [("sell", 1000, "09:30")],
    })

    rule = dividend_tax_rule(record_date)
    withheld = 1000 * cash * rule.withholding_rate
    supplement = 1000 * cash * (dividend_tax_rate(buy, sell, rule) - rule.withholding_rate)
    if cash == 0.0:
        # Nothing was delivered: no corporate action and no tax base to carry.
        assert record["corporate_actions"] == []
    else:
        [action] = record["corporate_actions"]
        assert (action["quantity_before"], action["quantity_after"]) == (1000, 1000)
        assert action["cash_credit"] == pytest.approx(1000 * cash)
        assert action["dividend_tax_withheld"] == pytest.approx(withheld)
    assert [row["dividend_tax"] for row in _filled(record)] == pytest.approx([0.0, supplement])
    assert record["stats"]["dividend_tax_paid"] == pytest.approx(withheld + supplement)
    assert record["equity_curve"][-1]["cash"] == pytest.approx(
        100_000.0 - 1000 * last_close + 1000 * cash - withheld + 1000 * pre_close - supplement
    )
    assert record["equity_curve"][-1]["positions"] == {}


@pytest.mark.parametrize("bonus", [1.0, 0.0])
def test_old_rule_pure_bonus_withholds_but_conversion_has_no_tax(bonus):
    frame = _frame({SYMBOL: [
        ("20150803", 10.0, 10.0), ("20150804", 5.0, 5.0),
        ("20150805", 5.0, 5.0), ("20150806", 5.0, 5.0),
    ]})
    record = _replay(TAXED, frame, _actions([(SYMBOL, "20150804", 0.0, 1.0, bonus)]), {
        "20150803": [("buy", 1000, "09:30")], "20150805": [("sell", 1000, "09:30")],
        "20150806": [("sell", 1000, "09:30")],
    })
    [action] = record["corporate_actions"]
    assert action["cash_credit"] == 0.0
    assert action["dividend_tax_withheld"] == 50.0 * bonus
    assert record["equity_curve"][1]["cash"] == 90_000.0 - 50.0 * bonus
    assert [row["dividend_tax"] for row in _filled(record)] == pytest.approx([0, 75 * bonus, 75 * bonus])
    assert record["equity_curve"][-1]["cash"] == pytest.approx(100_000.0 - 200.0 * bonus)
    assert record["stats"]["dividend_tax_paid"] == pytest.approx(200.0 * bonus)


def test_taxable_distributions_need_one_real_record_date():
    frame = _frame({SYMBOL: [("20150909", 10.0, 10.0)]})
    event = _actions([(SYMBOL, "20150909", 1.0, 0.0, 0.0)])
    for missing in ("", None, "bad"):
        with pytest.raises(ValueError, match="record_date"):
            DailyMarketData(frame, event.assign(record_date=missing))
    with pytest.raises(ValueError, match=r"conflicting.*record_date"):
        DailyMarketData(frame, pd.concat([
            event.assign(record_date="20150907"), event.assign(record_date="20150908")
        ], ignore_index=True))
    # A pure bonus needs its date too; a capital conversion creates no income.
    with pytest.raises(ValueError, match="record_date"):
        DailyMarketData(frame, event.assign(cash_per_share=0.0, bonus_per_share=1.0, record_date=""))
    DailyMarketData(frame, event.assign(cash_per_share=0.0, bonus_per_share=0.0, record_date=""))


def test_lot_records_preserve_both_tax_bases_and_refuse_unknown_nonzero_legacy_income():
    broker = DailyBroker(TAXED)
    broker.positions[SYMBOL] = Position(SYMBOL, 1000, 1000, 10.0, 10.0)
    broker.lots[SYMBOL] = [Lot(1000, "20150820", 200.0, 100.0)]
    records = json.loads(json.dumps(broker.lot_records()))
    restored = DailyBroker(TAXED)
    restored.positions = broker.positions.copy()
    restored.restore_lots(records)
    assert restored.lot_records() == records
    restored._current_day = "20150910"
    assert restored._release_lots(SYMBOL, 500) == pytest.approx(27.5)
    [lot] = restored.lots[SYMBOL]
    assert (lot.quantity, lot.dividend_income_new, lot.dividend_income_old) == (500, 100.0, 50.0)
    for acquired in ("20150820", "20260105"):
        with pytest.raises(ValueError, match="unclassified dividend tax ledger"):
            restored.restore_lots({SYMBOL: [{"quantity": 1000, "acquired": acquired, "dividend_income": 100.0}]})
    restored.restore_lots({SYMBOL: [{"quantity": 1000, "acquired": "20150820", "dividend_income": 0.0}]})
    assert restored.lots[SYMBOL] == [Lot(1000, "20150820")]


def test_acquisition_after_cutover_does_not_prove_income_is_new_rule():
    # Accepted ex-date entitlement approximation: shares bought after an old
    # record date but before its ex-date still receive that distribution.
    broker = DailyBroker(TAXED)
    broker.positions[SYMBOL] = Position(SYMBOL, 1000, 1000, 10.0, 10.0)
    broker.lots[SYMBOL] = [Lot(1000, "20150908")]
    broker.open_day("20150909", {SYMBOL: {"pre_close": 9.0}}, distributions={
        SYMBOL: Distribution(1.0, 0.0, "20150907")
    })
    assert broker.lots[SYMBOL][0].dividend_income_old == 1000.0
    assert broker.corporate_actions[0].dividend_tax_withheld == 50.0


def test_paper_refuses_ambiguous_legacy_tax_basis_without_rewriting_the_checkpoint(tmp_path):
    for day in PAPER_SESSIONS[1:4]:
        _paper(tmp_path, TAXED).run_day(day)
    path = tmp_path / "paper" / ".paper_state.json"
    state = _state(tmp_path)
    [lot] = state["account"]["lots"][SYMBOL]
    lot["dividend_income"] = lot.pop("dividend_income_new")
    del lot["dividend_income_old"]
    path.write_text(json.dumps(state), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(PaperEngineError, match="unclassified dividend tax ledger"):
        _paper(tmp_path, TAXED).run_day(PAPER_SESSIONS[4])
    assert path.read_bytes() == before
