"""The label and fundamentals switches of the seqaxes_alla_100k_8y_20261006 starter.

The pack's claim is "candidate minus the 100k bag, with one thing changed", so
what must hold is that the baseline is the 100k bag's own label to the bit,
that each label variant is the one change it names, that a fundamentals row
reaches a training date only when the decision opening that date's position
could see it, and that the c_perm control keeps every feature row and moves
only which stock it belongs to. The modules run as a starter runs them, as
``lib.*`` imported from the pack's own ``starter`` directory.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKS = REPO_ROOT / "configs" / "workspace_refs"
STARTER = PACKS / "seqaxes_alla_100k_8y_20261006" / "starter"
PARENT_LABEL = PACKS / "seqbag_alla_100k_8y_20261003" / "starter" / "lib" / "label.py"


def _drop_lib() -> None:
    for name in [module for module in sys.modules if module == "lib" or module.startswith("lib.")]:
        sys.modules.pop(name)


@pytest.fixture
def lib(monkeypatch: pytest.MonkeyPatch):
    _drop_lib()
    # The pack is copied into every session's output/, where a __pycache__
    # makes smoke_backtest refuse the package: import it without writing one.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(STARTER))
    from lib import fusion, knobs, label

    yield {"fusion": fusion, "knobs": knobs, "label": label}
    _drop_lib()


def _panel(rows: int = 260, names: int = 40) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Adjusted prices with gaps, free-float values and a benchmark over business days."""

    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2021-01-04", periods=rows).strftime("%Y%m%d").to_numpy()
    market = rng.normal(0.0, 0.01, rows)
    returns = 0.3 * market[:, None] * rng.uniform(0.5, 1.5, names) + rng.normal(0.0, 0.02, (rows, names))
    closes = 10.0 * np.exp(np.cumsum(returns, axis=0))
    opens = closes * np.exp(rng.normal(0.0, 0.005, (rows, names)))
    gaps = rng.random((rows, names)) < 0.03
    closes[gaps], opens[gaps] = np.nan, np.nan
    level = 1000.0 * np.exp(np.cumsum(market))
    benchmark = pd.DataFrame({"open": level * 0.999, "close": level}, index=pd.Index(dates, name="trade_date"))
    data = {
        "dates": dates,
        "open_adj": opens,
        "close_adj": closes,
        "circ_mv": np.where(gaps, np.nan, rng.lognormal(22.0, 1.0, names)[None, :] * closes),
    }
    return data, benchmark


def test_the_baseline_label_is_the_100k_bags_label_to_the_bit(lib) -> None:
    spec = importlib.util.spec_from_file_location("parent_label", PARENT_LABEL)
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    data, benchmark = _panel()
    assert lib["knobs"].LABEL == "base" and lib["knobs"].FUND == "off"
    assert lib["label"].HOLD == 10 and lib["knobs"].leg() == "c_base"
    np.testing.assert_array_equal(
        lib["label"].target(data, benchmark), parent.residual_rank(data, benchmark, 10)
    )


def test_each_label_variant_is_the_one_change_it_names(lib, monkeypatch: pytest.MonkeyPatch) -> None:
    label, knobs = lib["label"], lib["knobs"]
    data, benchmark = _panel()
    stock, _ = label.forward_returns(data, benchmark, label.HOLD)
    # The index leg is one number per date, so the pure benchmark residual
    # ranks exactly like the raw forward return.
    monkeypatch.setattr(knobs, "LABEL", "bench")
    np.testing.assert_array_equal(label.target(data, benchmark), label.rank_target(stock))
    # Within every date's size quintile the demeaned residual averages zero
    # and keeps its order; across quintiles only the group means moved.
    values = np.where(np.isfinite(stock), stock, np.nan)
    demeaned = label.group_demean(values, data["circ_mv"])
    ranks = pd.DataFrame(np.where(np.isfinite(values), data["circ_mv"], np.nan)).rank(axis=1, pct=True).to_numpy()
    for t in range(0, len(data["dates"]) - label.HOLD - 1, 37):
        group = np.ceil(np.nan_to_num(ranks[t]) * label.SIZE_GROUPS) - 1
        for g in range(label.SIZE_GROUPS):
            member = (group == g) & np.isfinite(values[t])
            assert member.sum() >= 5
            assert abs(demeaned[t, member].mean()) < 1e-12
            np.testing.assert_array_equal(np.argsort(demeaned[t, member]), np.argsort(values[t, member]))
    monkeypatch.setattr(knobs, "LABEL", "size")
    monkeypatch.setattr(knobs, "FUND", "late")
    with pytest.raises(ValueError, match="one lane at a time"):
        label.target(data, benchmark)


def _report(code: str, end: str, stamp: str, q_roe: float) -> dict[str, object]:
    row = {name: 1.0 for name in ("grossprofit_margin", "debt_to_assets", "netprofit_yoy", "q_sales_yoy")}
    available_at = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]} 18:00:00+08:00"
    return {"ts_code": code, "end_date": end, "available_at": available_at,
            "stamp": pd.Timestamp(available_at).tz_convert("UTC"), "q_roe": q_roe, **row}


def test_a_report_reaches_the_row_whose_next_decision_sees_it(lib) -> None:
    # Monday 2024-06-03 .. Friday 2024-06-14 (no holiday), then the decision
    # on Monday 2024-06-17. Row t is what the decision of days[t + 1] sees.
    days = [*pd.bdate_range("2024-06-03", "2024-06-14").strftime("%Y%m%d"), "20240617"]
    rows = pd.DataFrame([
        # Wednesday stamp: usable at Thursday's decision, i.e. on Wednesday's row.
        _report("A", "20240331", "20240605", 5.0),
        # A restated older period, stamped later, never replaces the newer one.
        _report("A", "20231231", "20240607", 9.0),
        # A newer version of the current period lacks the field: it stays missing.
        _report("A", "20240331", "20240611", np.nan),
        # Saturday stamp: Monday's decision does not see it, Tuesday's does.
        _report("B", "20240331", "20240608", 3.0),
        # A name outside the panel is ignored.
        _report("Z", "20240331", "20240604", 1.0),
    ])
    out = lib["fusion"].asof_panel(rows, np.array(["A", "B"]), days)
    q_roe = out["q_roe"]
    row = {day: k for k, day in enumerate(days[:-1])}
    assert np.isnan(q_roe[: row["20240605"], 0]).all()
    assert (q_roe[row["20240605"]: row["20240611"], 0] == 5.0).all()
    assert np.isnan(q_roe[row["20240611"]:, 0]).all()
    assert np.isnan(q_roe[: row["20240610"], 1]).all()
    assert (q_roe[row["20240610"]:, 1] == 3.0).all()
    # Age in calendar days from the stamp to the deciding day.
    assert out["age"][row["20240605"], 0] == 1.0
    assert out["age"][row["20240614"], 1] == 9.0
    assert set(out) == set(lib["fusion"].FEATURES)


def test_c_perm_keeps_each_dates_rows_and_moves_them_between_stocks(
    lib, monkeypatch: pytest.MonkeyPatch
) -> None:
    fusion = lib["fusion"]
    rng = np.random.default_rng(3)
    fx = rng.random((4, 50, len(fusion.FEATURES))).astype(np.float32)
    fx[1, ::7, 2] = np.nan
    data = {"dates": np.array(["20240603", "20240604", "20240605", "20240606"])}
    idx = np.arange(3, 45)
    np.testing.assert_array_equal(fusion.take(fx, data, 1, idx), fx[1, idx])
    monkeypatch.setattr(lib["knobs"], "FUND_PERM", True)
    permuted = fusion.take(fx, data, 1, idx)
    assert not np.array_equal(permuted, fx[1, idx], equal_nan=True)
    key = lambda block: sorted(map(tuple, np.nan_to_num(block, nan=-1.0)))  # noqa: E731
    assert key(permuted) == key(fx[1, idx])
    np.testing.assert_array_equal(permuted, fusion.take(fx, data, 1, idx))
    inputs = fusion.early_inputs(permuted)
    assert inputs.shape == (len(idx), fusion.WIDTH)
    missing = np.isnan(permuted)
    assert (inputs[:, len(fusion.FEATURES):] == missing).all()
    assert (inputs[:, : len(fusion.FEATURES)][missing] == 0.0).all()


def test_the_late_blend_ranks_both_scores_and_leaves_the_bag_alone_at_weight_zero(lib) -> None:
    blend = lib["fusion"].blend
    bag = np.array([0.9, 0.1, np.nan, 0.5, 0.3])
    fund = np.array([0.0, 1.0, 0.4, np.nan, 0.6])
    alone = blend(bag, fund, 0.0)
    assert np.isnan(alone[[2, 3]]).all()
    np.testing.assert_array_equal(np.argsort(alone[[0, 1, 4]]), np.argsort(bag[[0, 1, 4]]))
    mixed = blend(bag, fund, lib["fusion"].WEIGHT)
    assert sorted(mixed[[0, 1, 4]]) == [0.0, 0.5, 1.0]
    np.testing.assert_array_equal(blend(bag, fund, 1.0)[[0, 1, 4]], [0.0, 1.0, 0.5])
