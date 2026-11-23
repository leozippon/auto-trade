"""The exchange's delisting rules: settlement on the delisting date and the
consolidation-period buy gate, in the Broker, a replay and its random baseline.

The market is a few main-board names over 20 sessions (``DAYS``): a forced
delisting (``FORCED``, *ST and then 退 in its consolidation period from
``DAYS[3]``, last bar ``DAYS[5]``, delisted ``DAYS[8]``), a clean one
(``CLEAN``, a merger), another consolidation-period delisting (``WINDUP``, 退
name from ``DAYS[3]``, announced ``DAYS[1]``, delisted ``DAYS[9]``), an
*ST name delisted without a consolidation period above 1 CNY that no
announcement table lists (``AMBIGUOUS``), a name suspended for four sessions
and back (``HALTED``) and names that trade throughout. A profile without the rules is the Broker every
experiment and Paper book created before them still runs: it must replay as
it always did, table or no table.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime

import pandas as pd
import pytest

from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.environment.broker import BrokerProfile, DailyBroker
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.delisting import ANNOUNCED, Delisting, DelistingTable
from autotrade.environment.executor import TrustedStrategyExecutor
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.replay import null_control, run_daily_replay
from autotrade.environment.replay.engine import StrategyDataView
from autotrade.environment.replay.market import DailyMarketData
from autotrade.environment.replay.null_control import (
    NullControlSetupError,
    run_null_control,
    trade_skeleton,
)
from autotrade.environment.replay.stats import compute_return_stats
from autotrade.environment.strategy import CN_TZ, StrategyOrder, StrategySchedule
from autotrade.paper import DailyPaperEngine
from autotrade.paper import verdict as paper_verdict
from autotrade.paper.consistency import _rebuild
from autotrade.paper.pit import BookPITData
from autotrade.paper.verdict import paper_reading
from autotrade.pipelines import worker
from autotrade.pipelines.ledger import ExperimentLedger
from tests.unit.support.paper_book_fixture import _registered_book
from tests.unit.support.paper_reconcile_fixture import PROFILE as RECONCILE_PROFILE
from tests.unit.support.paper_reconcile_fixture import SESSIONS as RECONCILE_SESSIONS
from tests.unit.support.paper_reconcile_fixture import B, _write_slot
from tests.unit.support.paper_reconcile_fixture import _book as _reconcile_book
from tests.unit.support.replay_fixture import DAYS, _at, _bar
from tests.unit.support.snapshot_fixture import (
    build_fundamental_events,
    build_raw,
    write_fundamental_status,
)
from tests.unit.support.synthetic_experiment import make_experiment

FORCED, CLEAN, WINDUP, HALTED = "000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"
STEADY = ("000005.SZ", "000006.SZ", "000007.SZ")
AMBIGUOUS = "000008.SZ"
LAST_BAR = {FORCED: 5, CLEAN: 5, WINDUP: 6, AMBIGUOUS: 5}
HALT = range(4, 8)
OLD = BrokerProfile(initial_cash=1_000_000.0, permitted_boards=("main", "gem"))
NEW = replace(OLD, delisting_rules=True)
DAYS_ISO = tuple(f"{day[:4]}-{day[4:6]}-{day[6:]}" for day in DAYS)


def _table() -> DelistingTable:
    delisted = pd.DataFrame(
        {"ts_code": [FORCED, CLEAN, WINDUP, AMBIGUOUS], "delist_date": [DAYS[8], DAYS[8], DAYS[9], DAYS[8]]}
    )
    names = pd.DataFrame(
        [
            (FORCED, "甲股份", "20200101", "20200101", "其他"),
            (FORCED, "*ST甲", DAYS[0], DAYS[0], "*ST"),
            (FORCED, "甲退", DAYS[3], DAYS[2], "退市整理期"),
            (AMBIGUOUS, "戊股份", "20200101", "20200101", "其他"),
            (AMBIGUOUS, "*ST戊", DAYS[0], DAYS[0], "*ST"),
            (CLEAN, "乙股份", "20200101", "20200101", "其他"),
            (WINDUP, "丙股份", "20200101", "20200101", "其他"),
            (WINDUP, "*ST丙", "20250101", "20241231", "*ST"),
            (WINDUP, "丙退", DAYS[3], DAYS[1], "退市整理期"),
            (HALTED, "丁股份", "20200101", None, "其他"),
        ],
        columns=["ts_code", "name", "start_date", "ann_date", "change_reason"],
    )
    return DelistingTable.from_frames(delisted, names)


def _price(symbol: str, offset: int) -> float:
    base = {FORCED: 2.0, CLEAN: 12.0, WINDUP: 0.5, HALTED: 20.0, AMBIGUOUS: 5.0}.get(symbol, 10.0 + STEADY.index(symbol) if symbol in STEADY else 10.0)
    return round(base * (1.0 + 0.01 * offset), 2)


def _frame() -> pd.DataFrame:
    rows = []
    for offset, day in enumerate(DAYS):
        for symbol in (FORCED, CLEAN, WINDUP, HALTED, AMBIGUOUS, *STEADY):
            if offset > LAST_BAR.get(symbol, len(DAYS)) or (symbol == HALTED and offset in HALT):
                continue
            previous = 3 if symbol == HALTED and offset == 8 else offset - 1
            price = _price(symbol, offset)
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": day,
                    "open": price,
                    "close": price,
                    "pre_close": _price(symbol, max(previous, 0)),
                    "up_limit": price * 1.5,
                    "down_limit": price * 0.5,
                    "is_suspended": False,
                    "circ_mv": 1.0e9,
                }
            )
    return pd.DataFrame(rows)


SCRIPT = {
    DAYS[0]: ((FORCED, "buy", 1000), (CLEAN, "buy", 1000), (HALTED, "buy", 1000), (STEADY[0], "buy", 1000)),
    DAYS[2]: ((WINDUP, "buy", 1000),),
    DAYS[4]: ((WINDUP, "buy", 1000),),
    DAYS[5]: ((WINDUP, "sell", 1000),),
}


def _replay(profile: BrokerProfile, delistings: DelistingTable | None, script=SCRIPT):
    def strategy(context):
        day = context.inference_at.strftime("%Y%m%d")
        return [
            {"symbol": symbol, "action": action, "quantity": quantity, "execute_at": _at(day, "09:30").isoformat()}
            for symbol, action, quantity in script.get(day, ())
        ]

    return run_daily_replay(
        daily=_frame(),
        strategy=strategy,
        schedule=StrategySchedule("day", "08:30"),
        profile=profile,
        delistings=delistings,
    )


def _held(result, day: str) -> dict[str, int]:
    return next(dict(row["positions"]) for row in result.equity_curve if row["trade_date"] == day)


def _cash(result, day: str) -> float:
    return next(float(row["cash"]) for row in result.equity_curve if row["trade_date"] == day)


def _equity(result, day: str) -> float:
    return next(float(row["equity"]) for row in result.equity_curve if row["trade_date"] == day)


# ---------------------------------------------------------------- the table


def test_a_delisting_is_forced_by_its_consolidation_period_or_the_face_value_and_clean_by_its_name():
    table = _table()
    assert table.delisting(FORCED, DAYS[7]) is None
    forced = table.delisting(FORCED, DAYS[8])
    assert forced == Delisting(FORCED, DAYS[8], True) and table.delisting(FORCED, DAYS[19]) == forced
    assert forced.settlement_price(2.1) == 0.0
    clean = table.delisting(CLEAN, DAYS[8])
    assert clean.forced is False and clean.settlement_price(12.6) == 12.6
    # A clean name that last closed below the face value was delisted by force.
    assert clean.settlement_price(0.99) == 0.0
    assert table.delisting(WINDUP, DAYS[9]).forced is True
    assert table.delisting(HALTED, DAYS[19]) is None


# The nine ST-named delistings since 2015-08 without a consolidation period
# that closed at or above 1 CNY: name, delisting date and last close.
ANNOUNCED_CASES = {
    "000016.SZ": ("*ST康佳A", "20261023", 2.46),
    "000023.SZ": ("*ST深天", "20240902", 1.72),
    "000627.SZ": ("*ST天茂", "20250930", 1.58),
    "000780.SZ": ("ST平能", "20220124", 10.95),
    "600213.SH": ("*ST亚星", "20241017", 6.36),
    "600485.SH": ("*ST信威", "20210601", 1.39),
    "600677.SH": ("*ST航通", "20210318", 3.01),
    "600680.SH": ("*ST上普", "20190523", 7.69),
    "603963.SH": ("*ST大药", "20250321", 1.18),
}


def test_the_nine_st_delistings_without_consolidation_settle_as_their_announcements_say():
    """Six were a merger or voluntary delistings with a cash option (last
    close), three were forced (zero)."""

    table = DelistingTable.from_frames(
        pd.DataFrame({"ts_code": list(ANNOUNCED_CASES), "delist_date": [day for _n, day, _c in ANNOUNCED_CASES.values()]}),
        pd.DataFrame(
            [(code, name, "20180101", "20180101", "*ST") for code, (name, _day, _close) in ANNOUNCED_CASES.items()],
            columns=["ts_code", "name", "start_date", "ann_date", "change_reason"],
        ),
    )
    assert set(ANNOUNCED_CASES) == set(ANNOUNCED)
    settled = {
        code: table.delisting(code, day).settlement_price(close) for code, (_name, day, close) in ANNOUNCED_CASES.items()
    }
    paid = {"000016.SZ", "000627.SZ", "000780.SZ", "600213.SH", "600677.SH", "600680.SH"}
    assert {code for code, price in settled.items() if price == ANNOUNCED_CASES[code][2]} == paid
    assert {code for code, price in settled.items() if price == 0.0} == {"000023.SZ", "600485.SH", "603963.SH"}


def test_an_unlisted_delisting_the_rules_cannot_classify_fails_only_when_it_is_held():
    table = _table()  # loads although AMBIGUOUS is neither classified nor listed
    assert table.delisting(AMBIGUOUS, DAYS[8]).forced is None
    _replay(NEW, table)  # nobody holds it: nothing to settle
    with pytest.raises(ValueError, match=f"add {AMBIGUOUS} to delisting.ANNOUNCED"):
        _replay(NEW, table, {DAYS[0]: ((AMBIGUOUS, "buy", 1000),)})
    # Below the face value the delisting was forced whatever the name.
    assert table.delisting(AMBIGUOUS, DAYS[8]).settlement_price(0.5) == 0.0
    # So is one with no name on record by its delisting date.
    unnamed = DelistingTable.from_frames(
        pd.DataFrame({"ts_code": [AMBIGUOUS], "delist_date": [DAYS[8]]}),
        pd.DataFrame(
            [(AMBIGUOUS, "戊股份", DAYS[9], DAYS[9], "其他")],
            columns=["ts_code", "name", "start_date", "ann_date", "change_reason"],
        ),
    )
    with pytest.raises(ValueError, match="cannot tell a forced delisting"):
        unnamed.delisting(AMBIGUOUS, DAYS[8]).settlement_price(5.0)
    with pytest.raises(ValueError, match="without a valid delist_date"):
        DelistingTable.from_frames(pd.DataFrame({"ts_code": [FORCED], "delist_date": [None]}), pd.DataFrame(
            columns=["ts_code", "name", "start_date", "ann_date", "change_reason"]
        ))


def test_the_consolidation_period_runs_from_when_it_is_public_until_the_delisting():
    table = _table()
    marked = [day for day in DAYS if WINDUP in table.consolidating(day)]
    assert marked == list(DAYS[3:9])
    # Announced after its start, it is acted on only from the announcement.
    late = DelistingTable.from_frames(
        pd.DataFrame({"ts_code": [WINDUP], "delist_date": [DAYS[12]]}),
        pd.DataFrame(
            [
                (WINDUP, "丙股份", "20200101", "20200101", "其他"),
                (WINDUP, "丙退", DAYS[3], DAYS[5], "退市整理期"),
                # A later name in force ends it, however the code goes on.
                (WINDUP, "丙股份", DAYS[10], DAYS[9], "其他"),
            ],
            columns=["ts_code", "name", "start_date", "ann_date", "change_reason"],
        ),
    )
    assert [day for day in DAYS if WINDUP in late.consolidating(day)] == list(DAYS[5:10])


# ---------------------------------------------------------------- the Broker


def test_a_held_forced_delisting_is_written_off_at_the_open_of_its_date():
    new = _replay(NEW, _table(), {DAYS[0]: ((FORCED, "buy", 1000), (STEADY[0], "buy", 1000))})
    [buy, settlement] = [row for row in new.executions if row["symbol"] == FORCED]
    at = _at(DAYS[8], "09:30").isoformat()
    assert settlement == {
        **settlement,
        "action": "sell",
        "quantity": 1000,
        "status": "filled",
        "price": 0.0,
        "commission": 0.0,
        "stamp_duty": 0.0,
        "reason": "delisted",
        "execute_at": at,
        "matched_at": at,
    }
    assert settlement["realized_pnl"] == pytest.approx(-(buy["price"] * 1000 + buy["commission"]))
    assert FORCED in _held(new, DAYS[7]) and FORCED not in _held(new, DAYS[8])
    # Nothing is paid out, and the last close leaves equity on the date
    # (beside the day's move of the name that trades on).
    assert _cash(new, DAYS[8]) == _cash(new, DAYS[7])
    steady_move = 1000 * (_price(STEADY[0], 8) - _price(STEADY[0], 7))
    assert _equity(new, DAYS[8]) - _equity(new, DAYS[7]) == pytest.approx(
        steady_move - 1000 * _price(FORCED, LAST_BAR[FORCED]), abs=1e-6
    )


def test_a_held_clean_delisting_is_paid_out_at_its_last_close():
    new = _replay(NEW, _table())
    [settlement] = [row for row in new.executions if row.get("reason") == "delisted" and row["symbol"] == CLEAN]
    last_close = _price(CLEAN, LAST_BAR[CLEAN])
    assert (settlement["price"], settlement["quantity"]) == (last_close, 1000)
    assert _cash(new, DAYS[8]) - _cash(new, DAYS[7]) == pytest.approx(1000 * last_close)


def test_a_consolidation_buy_is_refused_and_a_sale_inside_it_fills():
    new = _replay(NEW, _table())
    windup = [(row["action"], row["status"], row["reason"]) for row in new.executions if row["symbol"] == WINDUP]
    # Before the period a buy fills; inside it the next buy is refused and the sale fills.
    assert windup == [("buy", "filled", None), ("buy", "rejected", "delisting_consolidation"), ("sell", "filled", None)]
    assert WINDUP not in _held(new, DAYS[9])
    stats = compute_return_stats(new)
    assert stats["reject_counts"] == {"delisting_consolidation": 1}
    # The settlements are exits: realized, but no rejection.
    assert stats["trade_count"] == 3


def test_a_sale_before_the_delisting_date_leaves_nothing_to_settle():
    script = {**SCRIPT, DAYS[3]: ((FORCED, "sell", 1000),)}
    new = _replay(NEW, _table(), script)
    forced = [(row["action"], row["status"], row["reason"]) for row in new.executions if row["symbol"] == FORCED]
    assert forced == [("buy", "filled", None), ("sell", "filled", None)]


def test_a_suspended_name_that_is_not_delisted_is_untouched():
    new, old = _replay(NEW, _table()), _replay(OLD, None)
    for result in (new, old):
        assert [row["symbol"] for row in result.executions if row["symbol"] == HALTED] == [HALTED]
        assert all(_held(result, day)[HALTED] == 1000 for day in DAYS[:12])
    # Marked at its last close while halted and at its own close once back, in both.
    assert [_equity(new, day) - _equity(old, day) for day in DAYS[:3]] == [0.0, 0.0, 0.0]


def test_an_old_profile_replays_as_it_always_did_table_or_no_table():
    with_table, without = _replay(OLD, _table()), _replay(OLD, None)
    assert with_table.equity_curve == without.equity_curve
    assert with_table.executions == without.executions
    # The old Broker: the frozen names stay at their last close to the end and
    # the consolidation buy fills.
    end = _held(without, DAYS[19])
    assert (end[FORCED], end[CLEAN], end[WINDUP]) == (1000, 1000, 1000)
    assert all(row.get("reason") is None for row in without.executions)
    frozen = 1000 * (_price(FORCED, 5) + _price(CLEAN, 5) + _price(WINDUP, 6))
    new = _replay(NEW, _table())
    gap = _equity(without, DAYS[19]) - _equity(new, DAYS[19])
    # Under the rules the forced name is gone, the clean one is cash and the
    # consolidation lot was never bought (its cost stayed in cash).
    windup_lot = next(
        row for row in without.executions if row["symbol"] == WINDUP and row["action"] == "buy" and row["execute_at"].startswith(DAYS_ISO[4])
    )
    assert gap == pytest.approx(
        frozen - 1000 * _price(CLEAN, 5) - (windup_lot["price"] * 1000 + windup_lot["commission"]), abs=1e-6
    )



def test_the_rules_without_their_table_fail_rather_than_replay_the_old_broker():
    broker = DailyBroker(NEW)
    with pytest.raises(ValueError, match="delisting rules require the delisting table"):
        broker.open_day(DAYS[0], {})
    with pytest.raises(ValueError, match="delisting rules require the delisting table"):
        _replay(NEW, None)


def test_a_settlement_drops_the_dividend_tax_lots_with_the_position():
    broker = DailyBroker(replace(NEW, dividend_tax=True))
    table = _table()
    broker.open_day(DAYS[0], {}, distributions={}, delistings=table)
    order = StrategyOrder(symbol=FORCED, action="buy", quantity=1000, execute_at=_at(DAYS[0], "09:30"))
    assert broker.execute(order, _bar(), matched_at=_at(DAYS[0], "09:30"), raw_price=10.0).status == "filled"
    assert set(broker.lots) == {FORCED}
    broker.open_day(DAYS[8], {}, distributions={}, delistings=table)
    assert broker.positions == {} and broker.lots == {}
    assert broker.executions[-1].dividend_tax == 0.0 and broker.executions[-1].reason == "delisted"


def test_a_profile_recorded_before_the_rules_reads_without_them():
    record = json.loads(json.dumps(OLD.to_record()))
    del record["delisting_rules"]
    assert BrokerProfile.from_record(record) == OLD
    assert BrokerProfile.from_record(NEW.to_record()) == NEW
    with pytest.raises(ValueError, match="delisting_rules must be a boolean"):
        BrokerProfile(delisting_rules=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing keys \\['dividend_tax'\\]"):
        BrokerProfile.from_record({key: value for key, value in record.items() if key != "dividend_tax"})


# ---------------------------------------------------------------- the random baseline


def test_a_settlement_closes_the_round_trip_the_random_baseline_replaces():
    new = _replay(NEW, _table())
    trips, _unpaired = trade_skeleton(new.executions)
    forced = next(trip for trip in trips if trip.symbol == FORCED)
    assert forced.exit_at == _at(DAYS[8], "09:30") and forced.exit_session is None
    old = _replay(OLD, None)
    assert next(trip for trip in trade_skeleton(old.executions)[0] if trip.symbol == FORCED).exit_at is None


def _drawn_orders(monkeypatch, result, profile, delistings, membership):
    queued: list[list[dict[str, object]]] = []
    scripted = null_control._scripted_strategy

    def capture(orders):
        queued.append([order for day in sorted(orders) for order in orders[day]])
        return scripted(orders)

    monkeypatch.setattr(null_control, "_scripted_strategy", capture)
    block = run_null_control(
        result,
        _frame(),
        {day: 0.0 for day in DAYS},
        profile,
        StrategySchedule("day", "08:30"),
        k=20,
        seed=7,
        membership=membership,
        baselines=True,
        delistings=delistings,
    )
    return block, queued


def test_the_replacement_of_a_settled_trip_is_sold_at_the_settlements_open(monkeypatch):
    result = _replay(NEW, _table(), {DAYS[0]: ((FORCED, "buy", 1000),)})
    # FORCED is inside the benchmark with the steady names: they are its pool.
    membership = {"20251231": {FORCED, *STEADY}}
    block, queued = _drawn_orders(monkeypatch, result, NEW, _table(), membership)
    assert block["rejects_mean"] == 0.0
    for orders in queued:
        [buy, sell] = orders
        assert buy["symbol"] in STEADY and sell["symbol"] == buy["symbol"]
        assert (sell["action"], sell["execute_at"]) == ("sell", _at(DAYS[8], "09:30").isoformat())


def test_a_drawn_name_that_delists_while_held_is_settled_like_the_candidates(monkeypatch):
    # An open round trip in STEADY[0]; its only same-side replacement is FORCED.
    result = _replay(NEW, _table(), {DAYS[0]: ((STEADY[0], "buy", 1000),)})
    membership = {"20251231": {STEADY[0], FORCED}}
    block, queued = _drawn_orders(monkeypatch, result, NEW, _table(), membership)
    assert {order["symbol"] for orders in queued for order in orders} == {FORCED}
    capital = {row["trade_date"]: float(row["equity"]) for row in result.equity_curve}
    written_off = dict(block["baseline_daily"])[DAYS[8]]
    shares = queued[0][0]["quantity"]
    assert written_off == pytest.approx(-shares * _price(FORCED, 5) / capital[DAYS[7]])
    # The old Broker keeps the drawn name frozen at its last close.
    old_block, _ = _drawn_orders(monkeypatch, _replay(OLD, None, {DAYS[0]: ((STEADY[0], "buy", 1000),)}), OLD, None, membership)
    assert dict(old_block["baseline_daily"])[DAYS[8]] == 0.0


def test_a_name_in_its_consolidation_period_is_no_replacement(monkeypatch):
    result = _replay(NEW, _table(), {DAYS[4]: ((STEADY[0], "buy", 1000),)})
    membership = {"20251231": {STEADY[0], WINDUP, STEADY[1]}}
    _block, queued = _drawn_orders(monkeypatch, result, NEW, _table(), membership)
    assert {orders[0]["symbol"] for orders in queued} == {STEADY[1]}
    old_result = _replay(OLD, None, {DAYS[4]: ((STEADY[0], "buy", 1000),)})
    _block, queued = _drawn_orders(monkeypatch, old_result, OLD, None, membership)
    assert {orders[0]["symbol"] for orders in queued} == {WINDUP, STEADY[1]}


def test_the_random_baseline_under_the_rules_needs_their_table():
    result = _replay(NEW, _table(), {DAYS[0]: ((STEADY[0], "buy", 1000),)})
    with pytest.raises(NullControlSetupError, match="delisting rules"):
        run_null_control(result, _frame(), {}, NEW, StrategySchedule("day", "08:30"), k=1, seed=1)


def test_settlement_times_are_market_time():
    new = _replay(NEW, _table())
    stamp = next(row["matched_at"] for row in new.executions if row.get("reason") == "delisted")
    assert datetime.fromisoformat(stamp).utcoffset().total_seconds() == 8 * 3600


# ---------------------------------------------------------------- Paper


PAPER_STRATEGY = f"""def generate_orders(context):
    day = context.inference_at.strftime("%Y-%m-%d")
    if day == "{DAYS_ISO[1]}":
        return [{{"symbol": "{FORCED}", "action": "buy", "quantity": 1000, "execute_at": day + "T09:30:00+08:00"}}]
    return []
"""


class _PaperData:
    def __init__(self, release_end: str, delistings: DelistingTable) -> None:
        frame = _frame()
        self.sessions = DAYS
        self.release_end = release_end
        self.generation_id = f"gen_{release_end}"
        self.vintages: dict = {}
        self.market = DailyMarketData(frame[frame["trade_date"] <= release_end], None, delistings)

    def context_data(self, inference_at):
        return StrategyDataView("", "", "1")

    def execution_price(self, symbol, when):
        return None

    def references(self, symbols, before):
        return {symbol: {"name": "", "close": 1.0} for symbol in symbols}

    def close(self) -> None:
        pass


def _paper_book(tmp_path, profile: BrokerProfile):
    strategy = tmp_path / "strategy" / "main.py"
    strategy.parent.mkdir(parents=True)
    strategy.write_text(PAPER_STRATEGY, encoding="utf-8")
    release = {"end": DAYS[0]}
    engine = DailyPaperEngine(
        strategy_path=strategy,
        strategy_revision="revision_1",
        state_root=tmp_path / "paper",
        data_factory=lambda _start, _day: _PaperData(release["end"], _table()),
        profile=profile,
        executor_factory=lambda path, _sandbox, _view, state_dir, models_dir: TrustedStrategyExecutor.from_path(
            path, state_dir=state_dir, models_dir=models_dir
        ),
    )
    for day in DAYS[1:11]:
        release["end"] = DAYS[DAYS.index(day) - 1]
        engine.run_day(day)
    return tmp_path / "paper"


def _journal(path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_a_paper_book_journals_the_settlement_as_the_execution_it_is(tmp_path):
    root = _paper_book(tmp_path / "new", replace(NEW, initial_cash=100_000.0))
    [settlement] = _journal(root / f"executions_{DAYS[8]}.jsonl")
    assert (settlement["symbol"], settlement["action"], settlement["price"], settlement["reason"]) == (
        FORCED, "sell", 0.0, "delisted"
    )
    equity = {row["trade_date"]: row for row in _journal(root / "equity_daily.jsonl")}
    assert (equity[DAYS[7]]["position_count"], equity[DAYS[8]]["position_count"]) == (1, 0)
    assert equity[DAYS[8]]["equity"] == equity[DAYS[8]]["cash"] == equity[DAYS[7]]["cash"]
    # The consistency check rebuilds the holdings from these journals alone.
    held: dict[str, int] = {}
    for day in DAYS[1:10]:
        _rebuild(held, [], _journal(root / f"executions_{day}.jsonl"))
        assert len(held) == equity[day]["position_count"]
    old = _paper_book(tmp_path / "old", replace(OLD, initial_cash=100_000.0))
    assert not (old / f"executions_{DAYS[8]}.jsonl").exists()
    assert {row["trade_date"]: row["position_count"] for row in _journal(old / "equity_daily.jsonl")}[DAYS[9]] == 1


# ---------------------------------------------------------------- the worker


def test_the_worker_reads_the_table_from_the_pinned_release_only_under_the_rules(tmp_path):
    def evaluator(root, **params):
        repo, experiment = make_experiment(root, **params)
        raw = repo / "data" / "raw"
        (raw / "stock_basic").mkdir()
        (raw / "namechange").mkdir()
        pd.DataFrame({"ts_code": [FORCED], "delist_date": ["20230105"]}).to_parquet(
            raw / "stock_basic" / "list_status=D.parquet", index=False
        )
        pd.DataFrame(
            [{"ts_code": FORCED, "name": "甲退", "start_date": "20221201", "ann_date": "20221125", "change_reason": "退市整理期"}]
        ).to_parquet(raw / "namechange" / "namechange.parquet", index=False)
        options = worker.resolve_worker_options(
            json.loads((experiment / "hitl" / "params.json").read_text(encoding="utf-8")),
            experiment_dir=experiment,
            repo_root=repo,
        )
        build = worker.build_experiment_pipeline(
            options,
            ledger=ExperimentLedger(options.experiment.ledger_path),
            store=FilesystemArtifactStore(experiment / "artifacts" / "strategy"),
            ref_store=AgentRefStore(experiment),
        )
        return options.experiment.broker_profile, build.pipeline.evaluator

    profile, stamped = evaluator(tmp_path / "stamped", delisting_rules=True)
    assert profile.delisting_rules
    assert stamped.delistings.delisting(FORCED, "20230105") == Delisting(FORCED, "20230105", True)
    profile, recorded = evaluator(tmp_path / "recorded")
    assert not profile.delisting_rules and recorded.delistings is None


def test_a_paper_verdict_draws_its_baseline_under_the_books_own_rules(tmp_path, monkeypatch):
    """A book with the rules hands its random baseline the delisting table of
    its raw lake; one recorded without them hands none, as before."""

    real = _reconcile_book(tmp_path, monkeypatch, plan={"20260105": [(B, "buy", 300, "15:00")]}, real=False)
    for day in ("20260105", "20260106", "20260107"):
        real.run(day)
    registered = _registered_book(tmp_path)
    _write_slot(real.root, registered.benchmark_index, dict.fromkeys(RECONCILE_SESSIONS, 0.1))
    raw = registered.raw_dir
    (raw / "stock_basic").mkdir(parents=True, exist_ok=True)
    (raw / "namechange").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ts_code": [FORCED], "delist_date": ["20230105"]}).to_parquet(
        raw / "stock_basic" / "list_status=D.parquet", index=False
    )
    pd.DataFrame(
        [{"ts_code": FORCED, "name": "甲退", "start_date": "20221201", "ann_date": "20221125", "change_reason": "退市整理期"}]
    ).to_parquet(raw / "namechange" / "namechange.parquet", index=False)
    handed: list[object] = []

    def spy(*args, **kwargs):
        handed.append(kwargs["delistings"])
        return run_null_control(*args, **kwargs)

    monkeypatch.setattr(paper_verdict, "run_null_control", spy)
    profile = replace(RECONCILE_PROFILE, permitted_boards=("main", "gem"))
    for rules in (True, False):
        paper_reading(replace(registered, root=real.root, profile=replace(profile, delisting_rules=rules)))
    assert handed[0].delisting(FORCED, "20230105") == Delisting(FORCED, "20230105", True)
    assert handed[1] is None


def test_a_paper_book_without_the_rules_does_not_read_the_delisting_table(tmp_path):
    """A delisted row without a date stops only a book that acts on the table."""

    raw = tmp_path / "data" / "raw"
    build_raw(raw)
    delisted = raw / "stock_basic" / "list_status=D.parquet"
    pd.read_parquet(delisted).assign(delist_date=None).to_parquet(delisted, index=False)
    events_root = tmp_path / "data" / "pit" / "fundamental_events"
    build_fundamental_events(events_root)
    status = tmp_path / "results" / "data_quality" / "fundamental_events_status.json"
    status.parent.mkdir(parents=True)
    write_fundamental_status(status)

    def data(rules: bool) -> BookPITData:
        return BookPITData(
            state_root=tmp_path / f"paper_{rules}", raw_dir=raw, fundamental_events_root=events_root,
            fundamental_events_status=status,
            snapshot_config=SnapshotConfig(
                events_datasets=(), macro_datasets=(), text_datasets=(), fundamental_datasets=("income_vip",),
                include_intraday=False, include_industry=False,
            ),
            start="20211011", trade_date="20211011",
            due_at=datetime(2021, 10, 11, 8, 30, tzinfo=CN_TZ), schedule={}, delisting_rules=rules,
        )

    without = data(False)
    try:
        assert without.market.delistings is None
    finally:
        without.close()
    with pytest.raises(ValueError, match="without a valid delist_date"):
        data(True)
