"""The canonical book a pack starter copies (``configs/starter_lib/``).

The two defects that shipped in every per-pack copy in round 20260927 must
stay fixed: a full book bought a 13th seat at a review, and an empty market
value was compared with 0. The book runs here as a starter runs it -- the
canonical ``trade.py`` inside a ``lib`` package beside the pack's own ``data``,
``score`` and ``knobs`` -- with those three reduced to fixed tables.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from autotrade.environment.strategy import CN_TZ

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_TRADE = REPO_ROOT / "configs" / "starter_lib" / "trade.py"
SEATS = 3

# The pack's own modules, reduced to what the book reads.
DATA = '''
FRAME = None
def pool(context, positions):
    return FRAME, "20240628", "20240628"
'''
SCORE = '''
import pandas as pd
NAME = "probe"
VALUES = None
def score(context, codes):
    return VALUES.reindex(codes), {}
def shuffle(raw, decision_at):
    return raw
def describe(raw, target):
    return {}
'''
KNOBS = f'''
SEATS = {SEATS}
CAPITAL = 0.97
KEEP_BAND = 2.0
INDUSTRY_CAP = 10
REVIEW = "month"
SHUFFLE = False
'''


@pytest.fixture
def book(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The canonical book as ``lib.trade``, with a pool of six names A..F
    ranked in that order at a close of 10, and the as-of calendar's last day
    2024-06-28 (so a decision in July is a monthly review, one on 2024-06-28
    is not)."""

    package = tmp_path / "starter" / "lib"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    shutil.copyfile(CANONICAL_TRADE, package / "trade.py")
    for name, source in (("data", DATA), ("score", SCORE), ("knobs", KNOBS)):
        (package / f"{name}.py").write_text(source, encoding="utf-8")
    daily = tmp_path / "asof" / "daily"
    daily.mkdir(parents=True)
    pd.DataFrame({"trade_date": ["20240627", "20240628"]}).to_parquet(daily / "part.parquet")
    for name in [module for module in sys.modules if module == "lib" or module.startswith("lib.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(tmp_path / "starter"))
    from lib import data, score, trade

    codes = list("ABCDEF")
    data.FRAME = pd.DataFrame(
        {
            "close": 10.0,
            "member": True,
            "tradable": True,
            "weight": 0.0,
            "industry": [f"industry_{code}" for code in codes],
        },
        index=codes,
    )
    score.VALUES = pd.Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], index=codes)
    yield trade, data, tmp_path / "asof"
    # Popped, not monkeypatched: monkeypatch would put these stubs back when it
    # undoes, and every later strategy importing its own ``lib`` would find them.
    for name in [module for module in sys.modules if module == "lib" or module.startswith("lib.")]:
        sys.modules.pop(name)


def _context(asof: Path, day: str, positions: dict[str, int], cash: float) -> SimpleNamespace:
    return SimpleNamespace(
        inference_at=datetime(int(day[:4]), int(day[4:6]), int(day[6:]), 8, 30, tzinfo=CN_TZ),
        account=SimpleNamespace(positions=positions, cash=cash),
        asof_dir=str(asof),
    )


def test_a_full_book_buys_no_extra_seat_at_a_review_and_a_short_one_refills(book) -> None:
    trade, _data, asof = book
    full = {code: 300 for code in "ABC"}
    # A review with every holding inside the keep band and cash for several
    # lots: nothing to sell, and the seat count is read before a name is
    # added, so nothing is bought (the copies that read it after bought D).
    assert trade.run(_context(asof, "20240701", full, cash=5_000.0)) == []
    # A refill between reviews buys exactly the empty seat, the best name not held.
    orders = trade.run(_context(asof, "20240628", {"A": 300, "B": 300}, cash=3_100.0))
    assert [(order["action"], order["symbol"]) for order in orders] == [("buy", "C")]
    assert orders[0]["book_size"] == SEATS


def test_a_holding_without_a_close_is_valued_at_zero_not_compared(book) -> None:
    """A held name the pool has no close for -- a removed name with no bar --
    is a forced exit valued at 0; the seats are sized on the rest of the book."""

    trade, data, asof = book
    data.FRAME.loc["Z"] = {"close": float("nan"), "member": False, "tradable": False, "weight": 0.0, "industry": None}
    positions = {"A": 300, "Z": 500, "Y": 200}
    orders = trade.run(_context(asof, "20240701", positions, cash=6_000.0))
    assert [(order["action"], order["symbol"]) for order in orders] == [
        ("sell", "Y"),
        ("sell", "Z"),
        ("buy", "B"),
        ("buy", "C"),
    ]
    # Y is not in the pool at all and Z has no close: both add 0, so a seat is
    # the cash at 97 % plus A's 300 shares at 10, over three seats.
    assert orders[2]["seat_cash"] == round((6_000.0 * 0.97 + 3_000.0) / SEATS, 2)
