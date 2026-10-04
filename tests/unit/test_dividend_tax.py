"""The individual dividend tax (财税〔2015〕101号) the Broker charges at a sale.

A profile with ``dividend_tax`` keeps its buys as first-in first-out lots, credits
each ex-date's dividend income (gross cash plus bonus shares at par) to the lots
that held the shares, and at each sale deducts 20 / 10 / 0 % of the income the
sold shares received by how long their lot was held. A profile without it, which
is every profile recorded before the tax existed, settles exactly as before.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.broker_core import dividend_tax_rate
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
                "record_date": "",
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


def test_the_holding_period_tiers_follow_the_calendar_month_and_year():
    # Held from the buy through the day before the sale: a sale on the same
    # day one month on has held exactly one month, which is still 20 %.
    assert dividend_tax_rate("20240110", "20240111") == 0.20
    assert dividend_tax_rate("20240110", "20240210") == 0.20
    assert dividend_tax_rate("20240110", "20240211") == 0.10
    # A month without the day ends at its own last day.
    assert dividend_tax_rate("20240131", "20240301") == 0.20
    assert dividend_tax_rate("20240131", "20240302") == 0.10
    assert dividend_tax_rate("20230703", "20240703") == 0.10
    assert dividend_tax_rate("20230703", "20240704") == 0.0
    assert dividend_tax_rate("20240229", "20250301") == 0.10
    assert dividend_tax_rate("20240229", "20250302") == 0.0
    with pytest.raises(ValueError, match="predates the dividend tax rule"):
        dividend_tax_rate("20150801", "20150907")
    with pytest.raises(ValueError, match="cannot be sold"):
        dividend_tax_rate("20240110", "20240110")


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
    assert DailyMarketData(frame, old_table).bonus_shares_for_day("20240103") is None
    assert DailyMarketData(frame).bonus_shares_for_day("20240103") == {}
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
        self.nl_query = None

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
        SYMBOL: [{"quantity": 2000, "acquired": "20260105", "dividend_income": pytest.approx(800.0)}]
    }
    # A fresh engine settles the sale from that checkpoint alone.
    _paper(tmp_path, TAXED).run_day(PAPER_SESSIONS[4])
    journal = tmp_path / "paper" / "executions_20260107.jsonl"
    [sale] = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert sale["dividend_tax"] == pytest.approx(160.0)
    state = _state(tmp_path)
    assert state["account"]["lots"] == {} and state["account"]["positions"] == []
    assert state["account"]["cash"] == pytest.approx(100_000.0 - 10_000.0 + 500.0 + 2000 * 4.8 - 160.0)


def test_a_book_checkpointed_before_the_tax_existed_runs_on_unchanged(tmp_path: Path):
    """Its state has no profile field and no ledger: it is the untaxed book it
    always was, and it may not be reopened as a taxed one."""

    reference = tmp_path / "reference"
    reference.mkdir()
    for day in PAPER_SESSIONS[1:]:
        _paper(reference, UNTAXED).run_day(day)

    engine = _paper(tmp_path, UNTAXED)
    for day in PAPER_SESSIONS[1:4]:
        engine.run_day(day)
    path = tmp_path / "paper" / ".paper_state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    del state["profile"]["dividend_tax"]
    del state["account"]["lots"]
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(PaperEngineError, match="Broker profile differs"):
        _paper(tmp_path, TAXED).run_day(PAPER_SESSIONS[4])
    _paper(tmp_path, UNTAXED).run_day(PAPER_SESSIONS[4])

    assert _state(tmp_path)["account"] == _state(reference)["account"]
    assert _state(tmp_path)["account"]["cash"] == pytest.approx(100_000.0 - 10_000.0 + 500.0 + 2000 * 4.8)
