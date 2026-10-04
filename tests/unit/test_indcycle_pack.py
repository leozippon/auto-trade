"""The industry-cycle starter of indcycle_alla_100k_8y_20261009.

What must hold: the switches take registered values only and the book's fixed
parts stay fixed; a firm's readings are the differences the README names, on
the newest visible version of each period, and nothing for a firm whose
newest report is stale; an industry without enough readings is not ranked;
the score walks the ranked industries in blocks of the first affordable (or
held) names; and c_shuf keeps every block and redraws only the blocks' order,
as a function of the decision day. The modules run as a starter runs them, as
``lib.*`` imported from the pack's own ``starter`` directory.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autotrade.environment.strategy import CN_TZ

REPO_ROOT = Path(__file__).resolve().parents[2]
STARTER = REPO_ROOT / "configs" / "workspace_refs" / "indcycle_alla_100k_8y_20261009" / "starter"


def _drop_lib() -> None:
    for name in [module for module in sys.modules if module == "lib" or module.startswith("lib.")]:
        sys.modules.pop(name)


@pytest.fixture
def lib(monkeypatch: pytest.MonkeyPatch):
    _drop_lib()
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(STARTER))
    from lib import data, industry, knobs, score

    yield SimpleNamespace(data=data, industry=industry, knobs=knobs, score=score)
    _drop_lib()


def _context(day: str, root: Path | str = "unused", cash: float = 100_000.0, positions=None) -> SimpleNamespace:
    return SimpleNamespace(
        inference_at=datetime(int(day[:4]), int(day[4:6]), int(day[6:]), 8, 30, tzinfo=CN_TZ),
        asof_dir=str(root),
        account=SimpleNamespace(cash=cash, positions=positions or {}),
    )


def test_the_switches_take_registered_values_and_the_book_stays_fixed(lib, monkeypatch: pytest.MonkeyPatch) -> None:
    knobs = lib.knobs
    assert knobs.leg() == "G"
    monkeypatch.setattr(knobs, "SIGNAL", "GI")
    monkeypatch.setattr(knobs, "SEATS", 16)
    monkeypatch.setattr(knobs, "WITHIN", "accel")
    assert knobs.leg() == "GI+s16+accel"
    monkeypatch.setattr(knobs, "SHUFFLE", True)
    assert knobs.leg() == "c_shuf+s16+accel"
    monkeypatch.setattr(knobs, "SIGNAL", "px")
    with pytest.raises(ValueError, match="c_px is a control"):
        knobs.leg()
    monkeypatch.setattr(knobs, "SHUFFLE", False)
    assert knobs.leg() == "c_px+s16+accel"
    monkeypatch.setattr(knobs, "INDUSTRY_CAP", 3)
    with pytest.raises(ValueError, match="fixed"):
        knobs.leg()
    monkeypatch.setattr(knobs, "INDUSTRY_CAP", 4)
    monkeypatch.setattr(knobs, "SEATS", 20)
    with pytest.raises(ValueError, match="SEATS"):
        knobs.leg()


def _fundamentals(root: Path, rows: list[dict[str, object]]) -> None:
    directory = root / "fundamentals"
    directory.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "ts_code", "end_date", "available_at",
               "q_sales_yoy", "q_dt_roe", "c_pay_acq_const_fiolta", "total_assets"]
    pd.DataFrame([{name: row.get(name, np.nan) for name in columns} for row in rows]).to_parquet(
        directory / "part_0000.parquet")


def _row(dataset: str, code: str, end: str, stamp: str, **values: float) -> dict[str, object]:
    return {"dataset": dataset, "ts_code": code, "end_date": end, "available_at": stamp, **values}


def test_firm_readings_are_the_named_differences_on_the_newest_versions(lib, tmp_path: Path) -> None:
    rows = [
        # A: fresh, a restated newest period (the later version counts), previous quarter and year ago.
        _row("fina_indicator_vip", "A", "20240331", "2024-04-20T18:00:00+08:00", q_sales_yoy=10.0, q_dt_roe=2.0),
        _row("fina_indicator_vip", "A", "20240331", "2024-04-25T18:00:00+08:00", q_sales_yoy=12.0, q_dt_roe=2.5),
        _row("fina_indicator_vip", "A", "20231231", "2024-03-30T18:00:00+08:00", q_sales_yoy=5.0, q_dt_roe=1.0),
        _row("fina_indicator_vip", "A", "20230331", "2023-04-28T18:00:00+08:00", q_sales_yoy=3.0, q_dt_roe=1.5),
        _row("cashflow_vip", "A", "20240331", "2024-04-25T18:00:00+08:00", c_pay_acq_const_fiolta=30.0),
        _row("cashflow_vip", "A", "20230331", "2023-04-28T18:00:00+08:00", c_pay_acq_const_fiolta=10.0),
        _row("balancesheet_vip", "A", "20240331", "2024-04-25T18:00:00+08:00", total_assets=1000.0),
        _row("balancesheet_vip", "A", "20230331", "2023-04-28T18:00:00+08:00", total_assets=500.0),
        # B: its newest report ended long before the decision.
        _row("fina_indicator_vip", "B", "20230630", "2023-08-30T18:00:00+08:00", q_sales_yoy=50.0, q_dt_roe=5.0),
        _row("fina_indicator_vip", "B", "20230331", "2023-04-28T18:00:00+08:00", q_sales_yoy=1.0, q_dt_roe=1.0),
        # C: no previous quarter visible.
        _row("fina_indicator_vip", "C", "20240331", "2024-04-26T18:00:00+08:00", q_sales_yoy=7.0, q_dt_roe=3.0),
    ]
    _fundamentals(tmp_path, rows)
    readings, audit = lib.industry.firm_readings(_context("20240603", tmp_path), ["A", "B", "C", "D"])
    assert readings.loc["A", "sales_accel"] == pytest.approx(12.0 - 5.0)
    assert readings.loc["A", "roe_change"] == pytest.approx(2.5 - 1.5)
    assert readings.loc["A", "capex_change"] == pytest.approx(30.0 / 1000.0 - 10.0 / 500.0)
    assert readings.loc["B"].isna().all()           # stale
    assert np.isnan(readings.loc["C", "sales_accel"]) and np.isnan(readings.loc["C", "roe_change"])
    assert readings.loc["D"].isna().all()           # no report at all
    assert audit["fund_after_decision"] == 0


def test_an_industry_needs_half_its_names_read_and_g_averages_two_ranks(lib, monkeypatch: pytest.MonkeyPatch) -> None:
    codes = [f"N{k}" for k in range(8)]
    label = pd.Series(["X", "X", "Y", "Y", "Z", "Z", "Z", "Z"], index=codes)
    firm = pd.DataFrame({
        "sales_accel": [5.0, 7.0, 1.0, 3.0, 9.0, np.nan, np.nan, np.nan],   # Z: one of four read
        "roe_change": [1.0, 1.0, 4.0, 4.0, 9.0, np.nan, np.nan, np.nan],
        "capex_change": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    }, index=codes)
    # Z (one of four names read) is left out. X: sales 6 (rank 1.0), roe 1 (rank 0.5) -> 0.75;
    # Y: sales 2 (0.5), roe 4 (1.0) -> 0.75: the tie keeps label order.
    assert lib.industry.ranking(_context("20240603"), codes, "G", label, firm) == ["X", "Y"]
    firm.loc["N2", "sales_accel"] = 10.0
    assert lib.industry.ranking(_context("20240603"), codes, "G", label, firm) == ["Y", "X"]


def test_the_score_walks_industries_in_blocks_of_affordable_names(lib, monkeypatch: pytest.MonkeyPatch) -> None:
    codes = [f"N{k:02d}" for k in range(16)]
    label = pd.Series(["X"] * 8 + ["Y"] * 8, index=codes)
    close = pd.Series(10.0, index=codes)
    close["N00"] = 900.0                                   # one lot is 90,000 CNY: not buyable ...
    cap = pd.Series(np.arange(16, 0, -1, dtype=float), index=codes)   # N00 largest within X, N08 within Y
    firm = pd.DataFrame({"sales_accel": np.nan, "roe_change": 0.0, "capex_change": 0.0}, index=codes)
    firm.loc["N15", "sales_accel"] = 50.0
    monkeypatch.setattr(lib.industry, "labels", lambda context, c: label.reindex(c))
    monkeypatch.setattr(lib.industry, "firm_readings", lambda context, c: (firm.reindex(c), {"fund_after_decision": 0}))
    monkeypatch.setattr(lib.industry, "ranking", lambda context, c, signal, lab, f: ["Y", "X"])
    monkeypatch.setattr(lib.data, "panel", lambda context, columns, days: {
        "close": close.to_frame().T, "circ_mv": cap.to_frame().T})

    values, meta = lib.score.score(_context("20240603"), codes)
    per = lib.score.NAMES_PER_INDUSTRY
    ranked = values.dropna().sort_values(ascending=False).index.tolist()
    assert ranked == [f"N{k:02d}" for k in range(8, 8 + per)] + [f"N{k:02d}" for k in range(1, 1 + per)]
    assert meta["first_industries"] == ["Y", "X"]
    # ... unless it is already held.
    held, _ = lib.score.score(_context("20240603", positions={"N00": 100}), codes)
    assert held["N00"] == -float(per)

    monkeypatch.setattr(lib.knobs, "WITHIN", "accel")
    accel, _ = lib.score.score(_context("20240603"), codes)
    assert accel.dropna().sort_values(ascending=False).index[0] == "N15"   # its own acceleration first


def test_c_shuf_keeps_every_block_and_redraws_their_order_by_quarter(lib) -> None:
    per = lib.score.NAMES_PER_INDUSTRY
    names = [f"N{k:02d}" for k in range(8 * per)]
    values = pd.Series([-float(k) for k in range(8 * per)], index=names)
    blocks = {name: k // per for k, name in enumerate(names)}
    day = pd.Timestamp("2024-04-01 08:30", tz=CN_TZ)
    first = lib.score.shuffle(values, day)
    # The same quarter draws the same order on any of its days; another quarter redraws it.
    pd.testing.assert_series_equal(first, lib.score.shuffle(values, pd.Timestamp("2024-06-03 08:30", tz=CN_TZ)))
    assert not first.equals(lib.score.shuffle(values, pd.Timestamp("2024-07-01 08:30", tz=CN_TZ)))
    for block in range(8):
        members = [name for name in names if blocks[name] == block]
        drawn = first[members]
        assert len(set(np.floor(-drawn / per))) == 1                     # a block stays together
        assert drawn.sort_values(ascending=False).index.tolist() == members   # in its own order
    assert sorted(first.to_numpy()) == sorted(values.to_numpy())
