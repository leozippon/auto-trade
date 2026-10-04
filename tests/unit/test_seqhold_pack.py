"""The holding switches of the seqhold_alla_100k_8y_20261009 starter.

The pack changes only when the 100k sequence bag's swaps execute and which
names its book may buy, so what must hold is: the label is the bag's own to
the bit while its beta estimator now also serves the beta lane; the switches
run one lane at a time with registered values only; the clock reads review and
fill days from the visible calendar alone; under the reversed clock a review
only sells, at the close, and the next morning only buys, into the empty
seats and with cash that exists at the open; the beta floor gates buys and
never sells a holding; and the placebo's random score is a function of the
decision date. The modules run as a starter runs them, as ``lib.*`` imported
from the pack's own ``starter`` directory.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autotrade.environment.strategy import CN_TZ

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKS = REPO_ROOT / "configs" / "workspace_refs"
STARTER = PACKS / "seqhold_alla_100k_8y_20261009" / "starter"
PARENT_LABEL = PACKS / "seqbag_alla_100k_8y_20261003" / "starter" / "lib" / "label.py"


def _drop_lib() -> None:
    for name in [module for module in sys.modules if module == "lib" or module.startswith("lib.")]:
        sys.modules.pop(name)


@pytest.fixture
def lib(monkeypatch: pytest.MonkeyPatch):
    _drop_lib()
    # The pack is copied into every session's output/: import it without
    # writing bytecode next to it.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(STARTER))
    from lib import beta, book, clock, knobs, label, panel

    yield SimpleNamespace(beta=beta, book=book, clock=clock, knobs=knobs, label=label, panel=panel)
    _drop_lib()


def test_the_label_is_the_100k_bags_and_its_beta_leaves_short_histories_unknown(lib) -> None:
    rng = np.random.default_rng(7)
    rows, names = 260, 30
    dates = pd.bdate_range("2021-01-04", periods=rows).strftime("%Y%m%d").to_numpy()
    level = 1000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, rows)))
    closes = 10.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, (rows, names)), axis=0))
    closes[rng.random((rows, names)) < 0.03] = np.nan
    closes[:200, 0] = np.nan                     # name 0 lists late: 60 bars, 59 returns
    data = {"dates": dates, "close_adj": closes, "open_adj": closes * 1.001}
    benchmark = pd.DataFrame({"open": level, "close": level}, index=pd.Index(dates))
    spec = importlib.util.spec_from_file_location("parent_label", PARENT_LABEL)
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    np.testing.assert_array_equal(lib.label.target(data, benchmark), parent.residual_rank(data, benchmark, 10))

    beta = lib.label.trailing_beta(data, benchmark).to_numpy()
    assert np.isnan(beta[-1, 0])                 # too short: the label fills 1.0, the lane does not buy it
    assert np.isfinite(beta[-1, 1:]).all()
    assert ((beta[-1, 1:] >= 0.3) & (beta[-1, 1:] <= 2.0)).all()
    short = lib.label.trailing_beta(data, benchmark, 60).to_numpy()
    assert not np.array_equal(short[-1, 1:], beta[-1, 1:])


def test_the_switches_run_one_lane_at_a_time_with_registered_values(lib, monkeypatch: pytest.MonkeyPatch) -> None:
    knobs = lib.knobs
    assert knobs.leg() == "c_base"
    monkeypatch.setattr(knobs, "CLOCK", "close_open")
    assert knobs.leg() == "close_open"
    monkeypatch.setattr(knobs, "BETA_FLOOR", 1.0)
    with pytest.raises(ValueError, match="one lane at a time"):
        knobs.leg()
    monkeypatch.setattr(knobs, "CLOCK", "open_close")
    assert knobs.leg() == "beta1"
    monkeypatch.setattr(knobs, "BETA_DAYS", 60)
    assert knobs.leg() == "beta1+w60"
    monkeypatch.setattr(knobs, "BETA_PLACEBO", True)
    assert knobs.leg() == "beta1+w60+shuf"
    monkeypatch.setattr(knobs, "SEED_BASE", 2000)
    with pytest.raises(ValueError, match="seed base"):
        knobs.leg()
    monkeypatch.setattr(knobs, "SEED_BASE", 1000)
    monkeypatch.setattr(knobs, "BETA_FLOOR", 0.0)
    with pytest.raises(ValueError, match="set BETA_FLOOR first"):
        knobs.leg()
    monkeypatch.setattr(knobs, "BETA_FLOOR", 1)            # an int is not the registered 1.0
    with pytest.raises(ValueError, match="BETA_FLOOR"):
        knobs.leg()
    monkeypatch.setattr(knobs, "BETA_FLOOR", 0.9)
    with pytest.raises(ValueError, match="BETA_FLOOR"):
        knobs.leg()
    monkeypatch.setattr(knobs, "BETA_FLOOR", 1.1)
    monkeypatch.setattr(knobs, "BETA_PLACEBO", False)
    assert knobs.leg() == "beta1.1+w60"


def _days(*days: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(day) for day in days]


def _at(day: str, hour: int = 8, minute: int = 30) -> pd.Timestamp:
    return pd.Timestamp(datetime(int(day[:4]), int(day[4:6]), int(day[6:]), hour, minute, tzinfo=CN_TZ))


def test_the_clock_reads_review_and_fill_days_from_the_visible_calendar(lib, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = lib.clock
    # Monday after a full week: a review, not a fill day.
    assert clock.reviews(_days("20241010", "20241011"), _at("20241014")) == (True, False)
    # Tuesday after that review: a fill day.
    assert clock.reviews(_days("20241011", "20241014"), _at("20241015")) == (False, True)
    # The first trading day after National Day, when the Monday before it was the
    # week's only trading day: a review and a fill day at once.
    assert clock.reviews(_days("20240927", "20240930"), _at("20241008")) == (True, True)
    assert clock.reviews([], _at("20241008")) == (True, False)

    def plan(decision, flat, today, yesterday):
        sell, buy = clock.instants(decision, flat, today, yesterday)
        return (sell and sell.strftime("%H:%M"), buy and buy.strftime("%H:%M"))

    monday = _at("20241014")
    assert plan(monday, False, True, False) == ("09:30", "15:00")        # c_base
    assert plan(monday, False, False, True) == (None, None)              # c_base never fills
    assert plan(monday, True, False, False) == (None, "15:00")           # c_base builds at the close
    assert plan(_at("20241014", 10, 0), False, True, False) == ("15:00", "15:00")
    assert plan(_at("20241014", 15, 1), False, True, False) == (None, None)
    monkeypatch.setattr(lib.knobs, "CLOCK", "close_open")
    assert plan(monday, False, True, False) == ("15:00", None)
    assert plan(monday, False, False, True) == (None, "09:30")
    assert plan(monday, False, True, True) == ("15:00", "09:30")
    assert plan(monday, False, False, False) == (None, None)
    assert plan(monday, True, False, False) == (None, "09:30")           # builds at the open


CODES = np.array([f"N{k:02d}" for k in range(30)])


def _book(lib, monkeypatch, day, visible, cash, positions, betas=None):
    """book.run on a synthetic 30-name panel where N00 ranks first and every lot costs 1,000 CNY."""
    rows = 70
    data = {
        "dates": pd.bdate_range(end=pd.Timestamp(day) - pd.Timedelta(days=1), periods=rows).strftime("%Y%m%d").to_numpy(),
        "codes": CODES,
        "close": np.full((rows, len(CODES)), 10.0),
        "has_bar": np.ones((rows, len(CODES)), dtype=bool),
        "nbars": np.cumsum(np.ones((rows, len(CODES))), axis=0),
        "not_st": np.ones(len(CODES), dtype=bool),
    }
    monkeypatch.setattr(lib.panel, "build", lambda context, days, labels: data)
    monkeypatch.setattr(lib.book, "score", lambda context, d: 1.0 - np.arange(len(CODES)) / len(CODES))
    monkeypatch.setattr(lib.clock, "visible_days", lambda context: _days(*visible))
    if betas is not None:
        monkeypatch.setattr(lib.beta, "exante", lambda context, d, days: betas)
    context = SimpleNamespace(
        inference_at=_at(day).to_pydatetime(),
        account=SimpleNamespace(cash=cash, positions=positions),
        asof_dir="unused",
    )
    orders = lib.book.run(context)
    return (
        sorted((o["execute_at"][11:16], o["symbol"]) for o in orders if o["action"] == "sell"),
        sorted((o["execute_at"][11:16], o["symbol"]) for o in orders if o["action"] == "buy"),
        orders,
    )


def test_a_reversed_review_only_sells_at_the_close_and_the_next_open_fills_from_cash(
    lib, monkeypatch: pytest.MonkeyPatch
) -> None:
    held = {f"N{k:02d}": 800 for k in range(10)} | {"N28": 800, "N29": 800}   # N28, N29 rank outside 24
    review, after = ("20241010", "20241011"), ("20241011", "20241014")

    sells, buys, orders = _book(lib, monkeypatch, "20241014", review, 3_000.0, held)
    assert sells == [("09:30", "N28"), ("09:30", "N29")]
    assert buys == [("15:00", "N10"), ("15:00", "N11")]
    # c_base funds its 15:00 buys with the 09:30 proceeds.
    assert sum(o["quantity"] * 10.0 for o in orders if o["action"] == "buy") > 3_000.0

    monkeypatch.setattr(lib.knobs, "CLOCK", "close_open")
    sells, buys, _ = _book(lib, monkeypatch, "20241014", review, 3_000.0, held)
    assert sells == [("15:00", "N28"), ("15:00", "N29")]
    assert buys == []

    kept = {f"N{k:02d}": 800 for k in range(10)}
    sells, buys, orders = _book(lib, monkeypatch, "20241015", after, 18_900.0, kept)
    assert sells == []
    assert buys == [("09:30", "N10"), ("09:30", "N11")]
    spent = sum(o["quantity"] * 10.0 for o in orders)
    assert spent <= 18_900.0 * 0.97
    assert all(o["quantity"] % 100 == 0 for o in orders)
    # Nothing to do on a day that is neither.
    assert _book(lib, monkeypatch, "20241016", ("20241014", "20241015"), 3_000.0, held)[2] == []

    # A review that is also a fill day: the 09:30 buys use only the cash at the
    # open, never the proceeds of the same day's 15:00 sells.
    eleven = {f"N{k:02d}": 800 for k in range(9)} | {"N28": 800, "N29": 800}
    sells, buys, orders = _book(lib, monkeypatch, "20241008", ("20240927", "20240930"), 2_000.0, eleven)
    assert sells == [("15:00", "N28"), ("15:00", "N29")]
    assert buys == [("09:30", "N09")]
    assert sum(o["quantity"] * 10.0 for o in orders if o["action"] == "buy") <= 2_000.0 * 0.97


def test_the_beta_floor_gates_buys_and_never_sells_a_holding(lib, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lib.knobs, "BETA_FLOOR", 1.0)
    betas = np.full(len(CODES), 1.2)
    betas[0] = 0.4                    # a holding under the floor stays while it ranks inside the band
    betas[10] = 0.8                   # the best entrant is under the floor
    betas[11] = np.nan                # unknown beta is not buyable
    held = {f"N{k:02d}": 800 for k in range(10)} | {"N28": 800, "N29": 800}
    sells, buys, orders = _book(lib, monkeypatch, "20241014", ("20241010", "20241011"), 3_000.0, held, betas)
    assert sells == [("09:30", "N28"), ("09:30", "N29")]
    assert buys == [("15:00", "N12"), ("15:00", "N13")]
    assert all(o["beta_exante"] >= 1.0 for o in orders if o["action"] == "buy")


def test_the_placebo_score_is_a_function_of_the_decision_date(lib) -> None:
    data = {
        "codes": CODES,
        "has_bar": np.ones((70, len(CODES)), dtype=bool),
        "nbars": np.cumsum(np.ones((70, len(CODES))), axis=0),
        "not_st": np.arange(len(CODES)) != 3,
    }
    day = SimpleNamespace(inference_at=_at("20241014").to_pydatetime())
    other = SimpleNamespace(inference_at=_at("20241021").to_pydatetime())
    first = lib.beta.placebo_score(day, data, 69)
    np.testing.assert_array_equal(first, lib.beta.placebo_score(day, data, 69))
    assert np.isnan(first[3]) and np.isfinite(np.delete(first, 3)).all()
    assert not np.array_equal(first, lib.beta.placebo_score(other, data, 69), equal_nan=True)
