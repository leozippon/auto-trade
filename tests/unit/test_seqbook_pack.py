"""The book switches of the seqbook_alla_100k_8y_20261008 starter.

The pack changes only how the 100k sequence bag's book follows its score, so
what must hold is that the baseline label is the bag's own to the bit, that
the switches run one lane at a time with registered values only, that the
pool filters read nothing after the decision's newest visible row or index
section, that the keep band and the swap cap sell exactly what they say, and
that score smoothing is a correctly weighted average that never revives a name
without a score today. The modules run as a starter runs them, as ``lib.*``
imported from the pack's own ``starter`` directory.
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
STARTER = PACKS / "seqbook_alla_100k_8y_20261008" / "starter"
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
    from lib import book, knobs, label, pool

    yield SimpleNamespace(book=book, knobs=knobs, label=label, pool=pool)
    _drop_lib()


def test_the_label_is_the_100k_bags_and_the_switches_run_one_lane_at_a_time(
    lib, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(7)
    rows, names = 260, 30
    dates = pd.bdate_range("2021-01-04", periods=rows).strftime("%Y%m%d").to_numpy()
    level = 1000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, rows)))
    closes = 10.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, (rows, names)), axis=0))
    closes[rng.random((rows, names)) < 0.03] = np.nan
    data = {"dates": dates, "close_adj": closes, "open_adj": closes * 1.001}
    benchmark = pd.DataFrame({"open": level, "close": level}, index=pd.Index(dates))
    spec = importlib.util.spec_from_file_location("parent_label", PARENT_LABEL)
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    np.testing.assert_array_equal(lib.label.target(data, benchmark), parent.residual_rank(data, benchmark, 10))

    knobs = lib.knobs
    assert knobs.leg() == "c_base"
    monkeypatch.setattr(knobs, "KEEP_BAND", 3.0)
    monkeypatch.setattr(knobs, "SMOOTH_HALFLIFE", 2)
    assert knobs.leg() == "band3+smooth2"
    monkeypatch.setattr(knobs, "LIQ_DROP", 0.4)
    with pytest.raises(ValueError, match="one lane at a time"):
        knobs.leg()
    monkeypatch.setattr(knobs, "KEEP_BAND", 2.0)
    monkeypatch.setattr(knobs, "SMOOTH_HALFLIFE", 0)
    monkeypatch.setattr(knobs, "INDEX_POOL", "csi1500")
    assert knobs.leg() == "liq40+csi1500"
    monkeypatch.setattr(knobs, "MV_DROP", 0.25)
    with pytest.raises(ValueError, match="MV_DROP"):
        knobs.leg()


def test_the_pool_floors_read_only_rows_up_to_the_decision_and_drop_the_named_share(lib) -> None:
    pool = lib.pool
    rng = np.random.default_rng(3)
    amount = rng.lognormal(16.0, 1.0, (40, 10))
    amount[30:, :] = np.nan                     # no bar on the last ten rows for anyone ...
    amount[25:30, 0] = np.nan                   # ... and name 0 has none on five more
    data = {"amount": amount}
    t = 29
    median = pool.median_amount(data, t)
    assert median[0] == pytest.approx(np.median(amount[10:25, 0]))
    assert median[1] == pytest.approx(np.median(amount[10:30, 1]))
    later = dict(data, amount=np.where(np.arange(40)[:, None] > t, 1e12, amount))
    np.testing.assert_array_equal(pool.median_amount(later, t), median)

    values = np.array([5.0, np.nan, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 6.0, 4.0])
    candidates = np.array([True] * 8 + [False, False])
    kept = pool.floor(values, candidates, 0.4)
    # 8 candidates, the 3 lowest (missing counts as lowest) are dropped; non-candidates stay out.
    assert np.nonzero(kept)[0].tolist() == [0, 3, 4, 5, 7]
    assert not kept[8:].any()
    assert (pool.floor(values, candidates, 0.0) == candidates).all()


def _context(tmp_path: Path, day: str, rows: list[dict[str, object]]) -> SimpleNamespace:
    macro = tmp_path / "asof" / "macro"
    macro.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(macro / "part_0000.parquet")
    return SimpleNamespace(
        inference_at=datetime(int(day[:4]), int(day[4:6]), int(day[6:]), 8, 30, tzinfo=CN_TZ),
        asof_dir=str(tmp_path / "asof"),
    )


def _section(index: str, date: str, names: list[str]) -> list[dict[str, object]]:
    stamp = f"{date[:4]}-{date[4:6]}-{date[6:]} 17:30:00+08:00"
    return [{"dataset": "index_weight", "available_at": stamp, "index_code": index, "con_code": name,
             "trade_date": date, "weight": 1.0} for name in names]


def test_the_index_pool_is_the_newest_section_the_decision_can_see(
    lib, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        *_section("000852.SH", "20240430", ["A", "B"]),
        # Stamped 17:30 on the decision day itself: not visible at 08:30.
        *_section("000852.SH", "20240531", ["A", "C"]),
        *_section("000905.SH", "20240430", ["D"]),
    ]
    context = _context(tmp_path, "20240531", rows)
    codes = np.array(["A", "B", "C", "D", "E"])
    monkeypatch.setattr(lib.knobs, "INDEX_POOL", "csi1000")
    assert lib.pool.members(context, codes).tolist() == [True, True, False, False, False]
    monkeypatch.setattr(lib.knobs, "INDEX_POOL", "csi1500")
    assert lib.pool.members(context, codes).tolist() == [True, True, False, True, False]
    later = _context(tmp_path / "later", "20240603", rows)
    monkeypatch.setattr(lib.knobs, "INDEX_POOL", "csi1000")
    assert lib.pool.members(later, codes).tolist() == [True, False, True, False, False]


def test_the_keep_band_and_the_swap_cap_sell_what_they_say(lib, monkeypatch: pytest.MonkeyPatch) -> None:
    sell = lib.book._sell_orders
    at = pd.Timestamp("2024-06-03 09:30", tz=CN_TZ)
    positions = {"A": 100, "B": 100, "C": 100, "D": 100, "E": 100, "F": 100}
    # 12 seats: band 2 keeps ranks below 24, band 3 below 36. F has no rank (forced exit).
    rank = {"A": 3, "B": 25, "C": 30, "D": 40, "E": 50}
    sold = lambda: sorted(order["symbol"] for order in sell(positions, rank, at, "leg", 12))  # noqa: E731
    assert sold() == ["C", "D", "E", "F"]          # 4 swaps: forced F, then E, D, C worst first
    monkeypatch.setattr(lib.knobs, "MAX_SWAPS", 2)
    assert sold() == ["E", "F"]                     # forced F takes one of the two
    monkeypatch.setattr(lib.knobs, "MAX_SWAPS", 4)
    monkeypatch.setattr(lib.knobs, "KEEP_BAND", 3.0)
    assert sold() == ["D", "E", "F"]                # B (25) and C (30) are inside 36


def test_score_smoothing_weights_each_week_by_its_half_life_and_needs_a_score_today(lib) -> None:
    smooth = lib.book.smooth
    scores = np.array([
        [0.9, 0.2, np.nan, 0.5],   # newest row
        [0.1, 0.4, 0.8, np.nan],
        [0.5, np.nan, 0.6, 0.7],
    ])
    out = smooth(scores, 2)
    w = 0.5 ** (np.arange(3) / 2)
    assert out[0] == pytest.approx((0.9 * w[0] + 0.1 * w[1] + 0.5 * w[2]) / w.sum())
    assert out[1] == pytest.approx((0.2 * w[0] + 0.4 * w[1]) / (w[0] + w[1]))
    assert np.isnan(out[2])                          # no score today: not revived by its past
    assert out[3] == pytest.approx((0.5 * w[0] + 0.7 * w[2]) / (w[0] + w[2]))
    np.testing.assert_array_equal(smooth(scores[:1], 2), scores[0])
