"""The event rules and the union of the titlemap_alla_100k_8y_20261012 starter.

What must hold: each category's rule admits the act it names and none of the
companion documents filed around it (opinions, adjustments, unlocks, progress
and completion notices, cancellations of plan shares); a fresh event of the
chosen categories is bought at the first weekly review after its visible day
and an event of another category is not; a union applies one quiet period
across its categories, so a stock's second commitment inside it is no new
event; the late-entry control buys nothing at the fresh review; and a leg is
only one of the registered shapes. The book runs as a starter runs it,
``lib.*`` imported from the pack's own ``starter`` directory, over a
point-in-time view written to disk.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from autotrade.environment.strategy import CN_TZ

REPO_ROOT = Path(__file__).resolve().parents[2]
STARTER = REPO_ROOT / "configs" / "workspace_refs" / "titlemap_alla_100k_8y_20261012" / "starter"

# Research-window titles from the vendor's feed, and the categories each belongs to.
TITLES = [
    ("关于向激励对象首次授予限制性股票的公告", {"grant"}),
    ("关于2023年股票期权激励计划授予登记完成的公告", {"grant"}),
    ("关于2017年限制性股票激励计划首次授予部分第一个解锁期解锁条件成就的公告", set()),
    ("关于向激励对象授予预留部分限制性股票的公告", set()),
    ("关于调整2022年限制性股票激励计划首次授予激励对象名单及授予数量的公告", set()),
    ("关于控股股东计划增持公司股份的公告", {"increase"}),
    ("关于控股股东增持公司股份计划实施完成的公告", set()),
    ("关于股东减持计划的预披露公告", set()),
    ("回购报告书", {"buyback"}),
    ("关于以集中竞价交易方式回购公司股份方案的公告", {"buyback"}),
    ("关于收到董事长提议回购公司股份的提示性公告", {"buyback"}),
    ("关于回购注销部分限制性股票的公告", set()),
    ("关于回购股份方案实施完毕暨回购实施结果的公告", set()),
    ("独立董事关于以集中竞价交易方式回购股份预案的独立意见", set()),
    ("关于业绩补偿方案暨回购注销对应补偿股份的公告", set()),
    ("关于2023年限制性股票激励计划（草案）的公告", set()),
    ("关于签订重大合同的公告", set()),
]

# Code: T-1 close. Each name is one category's fresh event (or a decoy).
NAMES = {"000001.SZ": 10.0, "000002.SZ": 12.0, "600003.SH": 9.0, "600004.SH": 11.0, "300005.SZ": 8.0}


def _drop_lib() -> None:
    for name in [module for module in sys.modules if module == "lib" or module.startswith("lib.")]:
        sys.modules.pop(name)


@pytest.fixture
def lib(monkeypatch: pytest.MonkeyPatch):
    _drop_lib()
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(STARTER))
    from lib import book, knobs, titles

    yield SimpleNamespace(book=book, knobs=knobs, titles=titles)
    _drop_lib()


def _view(root: Path, titles: list[tuple[str, str, str]]) -> Path:
    """Daily bars to Friday 2024-03-08, the universe and the given (code, local stamp, title)
    rows, as the views a decision on Monday 2024-03-11 reads."""

    days = pd.bdate_range("2023-06-01", "2024-03-08").strftime("%Y%m%d")
    daily = pd.DataFrame(
        [
            {"ts_code": code, "trade_date": day, "close": close, "amount": 5.0e7, "is_suspended": False}
            for code, close in NAMES.items()
            for day in days
        ]
    )
    universe = pd.DataFrame(
        [{"ts_code": code, "name": f"样本{index}", "l1_name": f"行业{index}"} for index, code in enumerate(NAMES)]
    )
    text = pd.DataFrame(
        [
            {"dataset": "anns_d", "ts_codes": code, "title": title, "available_at": f"{stamp} 23:59:59+08:00"}
            for code, stamp, title in titles
        ]
    )
    for name, frame in (("daily", daily), ("universe", universe), ("text_index", text)):
        (root / name).mkdir()
        frame.to_parquet(root / name / "part-0.parquet", index=False)
    return root


# Fresh at the Monday 2024-03-11 review: visible Wednesday 03-06 (stamped Tuesday).
FRESH = [
    ("000001.SZ", "2024-03-05", "关于向激励对象首次授予限制性股票的公告"),
    ("000002.SZ", "2024-03-05", "关于控股股东计划增持公司股份的公告"),
    ("600003.SH", "2024-03-05", "关于以集中竞价交易方式回购公司股份方案的公告"),
    ("600004.SH", "2024-03-05", "关于2021年限制性股票激励计划首次授予部分第二个解锁期解锁条件成就的公告"),
    # Stamped Friday: visible Monday 03-11, fresh only at the next review.
    ("300005.SZ", "2024-03-08", "关于向激励对象授予股票期权的公告"),
]


def _bought(lib, root: Path) -> list[str]:
    context = SimpleNamespace(
        inference_at=datetime(2024, 3, 11, 8, 30, tzinfo=CN_TZ),
        asof_dir=str(root),
        account=SimpleNamespace(cash=100_000.0, positions={}),
    )
    orders = lib.book.run(context)
    assert all(order["action"] == "buy" and order["quantity"] % 100 == 0 for order in orders)
    return sorted(str(order["symbol"]) for order in orders)


@pytest.mark.parametrize(("title", "categories"), TITLES)
def test_each_rule_admits_the_act_and_none_of_its_companions(lib, title: str, categories: set[str]) -> None:
    series = pd.Series([title])
    assert {name for name in lib.titles.RULES if lib.titles.matched(series, (name,)).iloc[0]} == categories


def test_the_book_buys_the_chosen_categories_at_the_review_after_they_become_visible(
    lib, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _view(tmp_path, FRESH)
    assert _bought(lib, root) == ["000001.SZ"]
    monkeypatch.setattr(lib.knobs, "CATEGORIES", ("increase", "buyback"))
    assert _bought(lib, root) == ["000002.SZ", "600003.SH"]
    monkeypatch.setattr(lib.knobs, "CANDIDATE", "late")
    assert _bought(lib, root) == []


def test_a_union_applies_one_quiet_period_across_its_categories(
    lib, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grant 33 days after the same company's holding-increase plan is the grant book's
    event, but not the union's: there the plan opened the quiet period."""

    root = _view(tmp_path, [("000001.SZ", "2024-02-01", "关于控股股东计划增持公司股份的公告"), *FRESH[:1]])
    assert _bought(lib, root) == ["000001.SZ"]
    monkeypatch.setattr(lib.knobs, "CATEGORIES", ("grant", "increase"))
    assert _bought(lib, root) == []


def test_a_leg_is_one_of_the_registered_shapes(lib, monkeypatch: pytest.MonkeyPatch) -> None:
    assert lib.knobs.leg() == "grant"
    for categories, hold, candidate, name in (
        (("buyback",), 60, "event", "buyback_h60"),
        (("grant", "increase", "buyback"), 40, "late", "c_late_grant+increase+buyback"),
    ):
        monkeypatch.setattr(lib.knobs, "CATEGORIES", categories)
        monkeypatch.setattr(lib.knobs, "HOLD", hold)
        monkeypatch.setattr(lib.knobs, "CANDIDATE", candidate)
        assert lib.knobs.leg() == name
    for categories in (("buyback", "grant"), ("grant", "grant"), ("draft",), (), "grant"):
        monkeypatch.setattr(lib.knobs, "CATEGORIES", categories)
        with pytest.raises(ValueError, match="CATEGORIES"):
            lib.knobs.leg()
