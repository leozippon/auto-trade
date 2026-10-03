"""The fundamentals read helper of the fund_open_100k_8y_20261005 starter.

`lib/fund.py` is the pack's correctness aid for the fundamentals domain, so
what it must never do is the point-in-time mistakes it exists to prevent: mix
another dataset's column into this one, take a later revision for the first
announcement, pick a restated old period as the newest report, difference a
year-to-date flow against a period it cannot see, or date an announcement to a
decision that cannot see it yet (the domain is released at 03:35 Tuesday to
Saturday, so a weekend stamp first reaches Tuesday's decision). It runs here as
a starter runs it, as ``lib.fund`` imported from the pack's own ``starter``
directory, over a small as-of view written to disk.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from autotrade.environment.strategy import CN_TZ

REPO_ROOT = Path(__file__).resolve().parents[2]
STARTER = (
    REPO_ROOT / "configs" / "workspace_refs" / "fund_open_100k_8y_20261005" / "starter"
)


def _drop_lib() -> None:
    for name in [
        module for module in sys.modules if module == "lib" or module.startswith("lib.")
    ]:
        sys.modules.pop(name)


@pytest.fixture
def fund(monkeypatch: pytest.MonkeyPatch):
    _drop_lib()
    # The pack is copied into every session's output/, where a __pycache__
    # makes smoke_backtest refuse the package: import it without writing one.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(STARTER))
    from lib import fund

    yield fund
    # Popped, not monkeypatched: every later strategy importing its own `lib`
    # must not find this one.
    _drop_lib()


def _stamp(day: str, clock: str = "18:00:00") -> str:
    return f"{day[:4]}-{day[4:6]}-{day[6:]} {clock}+08:00"


def _row(
    dataset: str, code: str, end: str, day: str, value: float
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "available_at": _stamp(day),
        "ts_code": code,
        "end_date": end,
        "n_income_attr_p": value,
    }


def _context(
    tmp_path: Path,
    day: str,
    rows: list[dict[str, object]],
    trade_days: tuple[str, ...] = (),
) -> SimpleNamespace:
    fundamentals = tmp_path / "asof" / "fundamentals"
    fundamentals.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(fundamentals / "part_0000.parquet")
    daily = tmp_path / "asof" / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"trade_date": list(trade_days) or [day]}).to_parquet(
        daily / "part_0000.parquet"
    )
    return SimpleNamespace(
        inference_at=datetime(
            int(day[:4]), int(day[4:6]), int(day[6:]), 8, 30, tzinfo=CN_TZ
        ),
        asof_dir=str(tmp_path / "asof"),
    )


def test_a_read_takes_one_dataset_and_orders_versions_by_stamp(
    fund, tmp_path: Path
) -> None:
    rows = [
        # A correction stamped later, written first.
        _row("income_vip", "A", "20231231", "20240420", 12.0),
        _row("income_vip", "A", "20231231", "20240410", 10.0),
        # Another dataset filling the same column.
        _row("express_vip", "A", "20231231", "20240110", 99.0),
        _row("income_vip", "B", "20231231", "20240411", 20.0),
    ]
    context = _context(tmp_path, "20240501", rows)
    income = fund.read(context, "income_vip", ["n_income_attr_p"])
    assert income[["ts_code", "n_income_attr_p"]].values.tolist() == [
        ["A", 10.0],
        ["A", 12.0],
        ["B", 20.0],
    ]
    assert str(income["stamp"].dt.tz) == "UTC"
    with pytest.raises(pa.ArrowInvalid, match="no_such_column"):
        fund.read(context, "income_vip", ["no_such_column"])


def test_the_first_announcement_and_the_newest_version_are_different_rows(
    fund, tmp_path: Path
) -> None:
    rows = [
        _row("income_vip", "A", "20231231", "20240410", 10.0),
        _row("income_vip", "A", "20231231", "20240420", 12.0),
        # A restatement of an older period, stamped after the current report.
        _row("income_vip", "A", "20221231", "20240425", 7.0),
    ]
    income = fund.read(
        _context(tmp_path, "20240501", rows), "income_vip", ["n_income_attr_p"]
    )
    first = fund.per_period(income, keep="first")
    last = fund.per_period(income, keep="last")
    assert first.set_index("end_date")["n_income_attr_p"].to_dict() == {
        "20221231": 7.0,
        "20231231": 10.0,
    }
    assert last.set_index("end_date")["n_income_attr_p"].to_dict() == {
        "20221231": 7.0,
        "20231231": 12.0,
    }
    assert fund.newest_period(last)["end_date"].tolist() == ["20231231"]
    with pytest.raises(ValueError, match="keep"):
        fund.per_period(income, keep="newest")


def _periods(values: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": "A",
            "end_date": list(values),
            "n_income_attr_p": list(values.values()),
        }
    )


def test_a_quarter_is_differenced_only_against_a_visible_quarter_of_its_own_year(
    fund,
) -> None:
    rows = _periods(
        {
            "20230331": 1.0,
            "20230630": 3.0,
            # 20230930 is not visible, so the fourth quarter is unknown.
            "20231231": 10.0,
            "20240331": 4.0,
            "20240630": 9.0,
        }
    )
    quarter = fund.single_quarter(rows, "n_income_attr_p")
    assert quarter.tolist()[:2] == [1.0, 2.0]
    assert np.isnan(quarter.iloc[2])
    assert quarter.tolist()[3:] == [4.0, 5.0]


def test_a_trailing_year_needs_last_years_full_year_and_same_period(fund) -> None:
    rows = _periods(
        {
            "20220630": 2.0,
            "20221231": 8.0,
            "20230630": 3.0,
            "20231231": 10.0,
            "20240630": 9.0,
        }
    )
    trailing = fund.trailing_year(rows, "n_income_attr_p")
    # 2022-06 lacks 2021; a full year is its own value; 2023-06 = 3 + 8 - 2;
    # 2024-06 = 9 + 10 - 3.
    assert np.isnan(trailing.iloc[0])
    assert trailing.tolist()[1:] == [8.0, 9.0, 10.0, 16.0]
    with pytest.raises(ValueError, match="one version per"):
        fund.trailing_year(pd.concat([rows, rows]), "n_income_attr_p")


def test_a_stamp_reaches_the_first_decision_after_the_next_tuesday_to_saturday_release(
    fund, tmp_path: Path
) -> None:
    # Thursday 2024-04-11 .. Wednesday 2024-04-17; the decision is that Wednesday.
    trade_days = ("20240411", "20240412", "20240415", "20240416")
    context = _context(tmp_path, "20240417", [], trade_days=trade_days)
    days = fund.calendar(context, lookback_days=30)
    assert days == [*trade_days, "20240417"]
    stamps = [
        _stamp("20240411"),  # Thursday evening -> Friday
        _stamp("20240412"),  # Friday evening -> released Saturday -> Monday
        _stamp("20240413"),  # Saturday -> released Tuesday
        _stamp("20240414"),  # Sunday -> released Tuesday
        _stamp("20240415"),  # Monday evening -> Tuesday
        _stamp("20240412", "02:00:00"),  # before Friday's run -> Friday
        _stamp("20240417"),  # after the last decision in the calendar
        None,
    ]
    assert fund.event_days(stamps, days) == [
        "20240412",
        "20240415",
        "20240416",
        "20240416",
        "20240416",
        "20240412",
        None,
        None,
    ]
