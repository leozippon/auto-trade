"""The STAR pool of the incdraft_alla_500k_8y_20261012 starter.

The 500k sibling of incdraft_alla_8y_20261012 differs only in the pool: the
500k account holds the STAR permission, so STAR drafts are bought there and
never at 100k. What must hold: every buy the book sends is one the Broker's
own lot rule accepts -- STAR whole shares from 200, every other board whole
lots of 100 -- a STAR name whose 200 shares do not fit a seat is skipped as
unaffordable, and a STAR buy cut below 200 shares by the cash left is not
sent. The book runs as a starter runs it, ``lib.*`` imported from the pack's
own ``starter`` directory, over a point-in-time view written to disk.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from autotrade.environment.broker_core import validate_buy_lot
from autotrade.environment.strategy import CN_TZ

REPO_ROOT = Path(__file__).resolve().parents[2]
STARTER = REPO_ROOT / "configs" / "workspace_refs" / "incdraft_alla_500k_8y_20261012" / "starter"

# Code: (T-1 close, Shenwan L1 industry). One fresh draft each, seen last week.
NAMES = {
    "300001.SZ": (33.0, "电子"),
    "600001.SH": (30.0, "机械设备"),
    "688001.SH": (50.0, "计算机"),
    "688002.SH": (70.0, "医药生物"),
}


def _drop_lib() -> None:
    for name in [module for module in sys.modules if module == "lib" or module.startswith("lib.")]:
        sys.modules.pop(name)


@pytest.fixture
def lib(monkeypatch: pytest.MonkeyPatch):
    _drop_lib()
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(STARTER))
    from lib import book, knobs

    yield SimpleNamespace(book=book, knobs=knobs)
    _drop_lib()


def _view(root: Path) -> Path:
    """Daily bars to Friday 2024-03-08, the universe, and one draft title per name stamped
    Tuesday 2024-03-05, as the views a decision on Monday 2024-03-11 reads."""

    days = pd.bdate_range("2023-10-02", "2024-03-08").strftime("%Y%m%d")
    daily = pd.DataFrame(
        [
            {"ts_code": code, "trade_date": day, "close": close, "amount": 5.0e7, "is_suspended": False}
            for code, (close, _) in NAMES.items()
            for day in days
        ]
    )
    universe = pd.DataFrame(
        [{"ts_code": code, "name": f"样本{index}", "l1_name": industry} for index, (code, (_, industry)) in enumerate(NAMES.items())]
    )
    text = pd.DataFrame(
        [
            {
                "dataset": "anns_d",
                "ts_codes": code,
                "title": "关于公司限制性股票激励计划（草案）的公告",
                "available_at": "2024-03-05 23:59:59+08:00",
            }
            for code in NAMES
        ]
    )
    for name, frame in (("daily", daily), ("universe", universe), ("text_index", text)):
        (root / name).mkdir()
        frame.to_parquet(root / name / "part-0.parquet", index=False)
    return root


def _run(lib, root: Path, cash: float) -> list[dict[str, object]]:
    context = SimpleNamespace(
        inference_at=datetime(2024, 3, 11, 8, 30, tzinfo=CN_TZ),
        asof_dir=str(root),
        account=SimpleNamespace(cash=cash, positions={}),
    )
    return lib.book.run(context)


def test_the_500k_book_buys_star_in_units_the_broker_accepts(lib, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lib.knobs, "ACCOUNT", 500_000)
    monkeypatch.setattr(lib.knobs, "SEATS", 40)
    orders = _run(lib, _view(tmp_path), 500_000.0)
    # Seat cash 485,000 / 40 = 12,125: STAR at 50 rounds to the nearest share
    # (242.5 -> 243); STAR at 70 needs 14,000 for 200 shares and is skipped.
    assert {order["symbol"]: order["quantity"] for order in orders} == {
        "300001.SZ": 400,
        "600001.SH": 400,
        "688001.SH": 243,
    }
    assert orders[0]["skipped_unaffordable"] == 1 and orders[0]["skipped_untradable"] == 0
    for order in orders:
        validate_buy_lot(int(order["quantity"]), str(order["symbol"]))

    # A STAR buy the cash left cuts below 200 shares is not sent; a main-board
    # buy is cut to whole lots.
    prices = pd.Series({"688001.SH": 50.0, "600001.SH": 30.0})
    at = datetime(2024, 3, 11, 15, 0, tzinfo=CN_TZ)
    assert lib.book._buy_orders(["688001.SH"], prices, 9_990.0, 12_125.0, 500_000.0, at, "b40", {}) == []
    [cut] = lib.book._buy_orders(["600001.SH"], prices, 9_990.0, 12_125.0, 500_000.0, at, "b40", {})
    assert cut["quantity"] == 300


def test_the_100k_book_never_buys_star(lib, tmp_path: Path) -> None:
    assert (lib.knobs.ACCOUNT, lib.knobs.STAR) == (100_000, {100_000: False, 500_000: True})
    orders = _run(lib, _view(tmp_path), 100_000.0)
    # Seat cash 97,000 / 16 = 6,062.5: two lots of each main-board name.
    assert {order["symbol"]: order["quantity"] for order in orders} == {"300001.SZ": 200, "600001.SH": 200}
    assert orders[0]["skipped_untradable"] == 2
