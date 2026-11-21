from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from autotrade.environment.data.contracts import benchmark_index_label
from autotrade.environment.replay.stats import ReplayResult, daily_returns_from_curve
from autotrade.environment.replay.style import (
    BENCHMARK_TS_CODE,
    FACTOR_SERIES,
    STYLE_SCHEMA_VERSION,
    _factor_legs,
    _factor_regression,
    _size_factor,
    _turnover_factor,
    _value_factor,
    benchmark_summary_block,
    replay_style_analysis,
    slot_benchmark,
    warmup_bars,
    write_style_rollup,
)
from autotrade.pipelines.session_tools import book_reading

# The boards the account may buy on: a current experiment's.
BOARDS = ("main", "gem")
# Forty names make a cross-section, ordered by float cap (slot 0 smallest).
# Twelve are cheap and twelve dear on book-to-price, each set split evenly
# between the small and the big size legs and between the slots below and
# above 20, so the size and the value spreads move independently of each other.
CHEAP = frozenset({0, 1, 2, 12, 13, 14, 20, 21, 22, 28, 29, 30})
DEAR = frozenset({3, 4, 5, 15, 16, 17, 23, 24, 25, 31, 32, 33})
# Twelve churn little and twelve much. The quiet names sit six in each size
# leg and in neither value set; the busy ones sit six below slot 20 and six
# above it and in neither the cheap set nor the size legs' moves, so a move of
# the quiet names alone is the turnover spread and leaves the size and value
# spreads as they were.
QUIET = frozenset({6, 7, 8, 9, 10, 11, 34, 35, 36, 37, 38, 39})
BUSY = frozenset({3, 4, 15, 16, 18, 19, 23, 24, 26, 27, 31, 32})


def _pb(slot: int) -> float:
    return 0.5 if slot in CHEAP else 4.0 if slot in DEAR else 1.0


def _turnover(slot: int) -> float:
    return 0.005 if slot in QUIET else 0.05 if slot in BUSY else 0.02


def _replay(days: list[str], *, with_holdings: bool = True) -> ReplayResult:
    equity = 100_000.0
    curve = []
    for index, day in enumerate(days):
        equity *= 1.0 + (0.01 if index % 2 else -0.005)
        curve.append(
            {
                "trade_date": day,
                "initial_equity": 100_000.0,
                "equity": equity,
                "cash": equity - (1_000.0 if with_holdings else 0.0),
                "positions": {"000001.SZ": 100} if with_holdings else {},
            }
        )
    return ReplayResult(tuple(curve), (), (), ())


def _daily(days: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": day,
                "ts_code": code,
                "close": close,
                "circ_mv": size,
                "pb": pb,
                "turnover_rate": turnover,
            }
            for day in days
            for code, close, size, pb, turnover in (
                ("000001.SZ", 10.0, 100.0, 1.0, 1.0),
                ("000002.SZ", 20.0, 500.0, 2.0, 2.0),
                ("000003.SZ", 30.0, 2_000.0, 4.0, 3.0),
                ("000004.SZ", 40.0, 9_000.0, 8.0, 4.0),
            )
        ]
    )


def test_daily_style_uses_replay_positions_and_frozen_slot_inputs(tmp_path: Path):
    days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2024-01-02", periods=10)]
    replay_dir = tmp_path / "replay"
    snapshot_dir = tmp_path / "snapshot"
    replay_dir.mkdir()
    snapshot_dir.mkdir()
    pd.DataFrame(
        [
            {
                "dataset": "index_daily",
                "ts_code": "000300.SH",
                "trade_date": day,
                "pct_chg": (-0.25 if index % 2 == 0 else 0.5),
            }
            for index, day in enumerate(days)
        ]
    ).to_parquet(replay_dir / "macro.parquet", index=False)
    pd.DataFrame(
        {"ts_code": ["000001.SZ"], "l1_name": ["银行"]}
    ).to_parquet(snapshot_dir / "universe.parquet", index=False)

    payload = replay_style_analysis(
        _replay(days),
        _daily(days),
        replay_dir=replay_dir,
        universes=[(days[0], snapshot_dir / "universe.parquet")],
        mode="valid",
        benchmark_index=BENCHMARK_TS_CODE,
        permitted_boards=BOARDS,
    )

    assert payload["schema_version"] == STYLE_SCHEMA_VERSION and payload["mode"] == "valid"
    # The slot's own index rows, compounded over the replayed days.
    expected = float(np.prod([1.0 + (-0.0025 if index % 2 == 0 else 0.005) for index in range(10)]) - 1.0)
    assert payload["compact"]["benchmark_return"] == round(expected, 6)
    assert "benchmark_regression" not in payload
    assert payload["style"]["available"] is True
    assert payload["style"]["days"] == 10
    assert payload["style"]["tilts"]["size"] == -0.5
    assert payload["style"]["industries"][0] == {"name": "银行", "weight": 1.0}

    target = write_style_rollup(tmp_path / "result", payload)
    written = json.loads(target.read_text(encoding="utf-8"))
    assert target.name == "style_analysis.json"
    assert written["compact"]["benchmark_return"] == round(expected, 6)


def test_the_agent_reads_its_own_sector_concentration_beside_the_size_tilt(tmp_path: Path):
    """The random baseline draws its names without regard to industry, so a
    single-sector book's sector move scores as selection — the one artifact
    that cleared every gate was 82.5 % banks. ``industries`` stays host-side,
    so the compact block carries the top industry's weight, and the
    Agent-visible projection must keep it."""

    days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2024-01-02", periods=10)]
    replay_dir = tmp_path / "replay"
    snapshot_dir = tmp_path / "snapshot"
    replay_dir.mkdir()
    snapshot_dir.mkdir()
    pd.DataFrame(
        [
            {
                "dataset": "index_daily",
                "ts_code": BENCHMARK_TS_CODE,
                "trade_date": day,
                "pct_chg": (-0.25 if index % 2 == 0 else 0.5),
            }
            for index, day in enumerate(days)
        ]
    ).to_parquet(replay_dir / "macro.parquet", index=False)
    pd.DataFrame(
        {"ts_code": ["000001.SZ", "000003.SZ"], "l1_name": ["银行", "电子"]}
    ).to_parquet(snapshot_dir / "universe.parquet", index=False)
    # 100 shares at 10 元 of a bank against 100 at 30 元 of the other name.
    equity = 100_000.0
    curve = []
    for index, day in enumerate(days):
        equity *= 1.0 + (0.01 if index % 2 else -0.005)
        curve.append(
            {
                "trade_date": day,
                "initial_equity": 100_000.0,
                "equity": equity,
                "cash": equity - 4_000.0,
                "positions": {"000001.SZ": 100, "000003.SZ": 100},
            }
        )

    payload = replay_style_analysis(
        ReplayResult(tuple(curve), (), (), ()),
        _daily(days),
        replay_dir=replay_dir,
        universes=[(days[0], snapshot_dir / "universe.parquet")],
        mode="valid",
        benchmark_index=BENCHMARK_TS_CODE,
        permitted_boards=BOARDS,
    )

    # The share of the largest industry, averaged over decision days — the same
    # number the host-side listing leads with, not a recomputation of it.
    assert payload["style"]["top_industry_weight"] == 0.75
    assert payload["style"]["industries"][0] == {"name": "电子", "weight": 0.75}
    block = benchmark_summary_block(payload)
    assert block["top_industry_weight"] == 0.75
    # What a validate row's book carries of it.
    visible = book_reading({}, {"total_return": 0.1, "benchmark": block}, detail=False)
    assert visible["top_industry_weight"] == 0.75
    assert "size_tilt" in visible


def test_a_span_classifies_each_day_under_its_own_slots_universe_vintage(tmp_path: Path):
    """Every slot of a span runs on its own anchor's universe vintage, so a name
    reclassified between two slots and a name listed after the first anchor
    are read under the vintage in force on each day -- not the first slot's
    for the whole span, which would keep the old class and leave the late
    listing unclassified."""

    days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2024-01-02", periods=10)]
    first, second = tmp_path / "anchor_1", tmp_path / "anchor_2"
    first.mkdir()
    second.mkdir()
    pd.DataFrame({"ts_code": ["000001.SZ"], "l1_name": ["电子"]}).to_parquet(
        first / "universe.parquet", index=False
    )
    pd.DataFrame(
        {"ts_code": ["000001.SZ", "000003.SZ"], "l1_name": ["计算机", "计算机"]}
    ).to_parquet(second / "universe.parquet", index=False)
    # The second slot opens on days[4]; the late listing is held from then on.
    curve = [
        {
            "trade_date": day,
            "initial_equity": 100_000.0,
            "equity": 100_000.0 * (1.0 + 0.001 * index),
            "cash": 90_000.0,
            "positions": {"000001.SZ": 100, **({"000003.SZ": 100} if index >= 4 else {})},
        }
        for index, day in enumerate(days)
    ]

    style = replay_style_analysis(
        ReplayResult(tuple(curve), (), (), ()),
        _daily(days),
        replay_dir=None,
        universes=[(days[0], first / "universe.parquet"), (days[4], second / "universe.parquet")],
        mode="valid",
        benchmark_index=BENCHMARK_TS_CODE,
        permitted_boards=BOARDS,
    )["style"]

    # Four days wholly 电子, six wholly 计算机; nothing falls to 未分类.
    assert style["industries"] == [
        {"name": "计算机", "weight": 0.6},
        {"name": "电子", "weight": 0.4},
    ]
    assert style["top_industry_weight"] == 0.6


def test_style_records_structured_unavailable_values(tmp_path: Path):
    days = ["20240102", "20240103"]
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    pd.DataFrame(
        {
            "dataset": ["index_daily", "index_daily"],
            "ts_code": ["000300.SH", "000300.SH"],
            "trade_date": days,
            "pct_chg": [0.1, -0.1],
        }
    ).to_parquet(replay_dir / "macro.parquet", index=False)
    payload = replay_style_analysis(
        _replay(days, with_holdings=False),
        _daily(days),
        replay_dir=replay_dir,
        universes=[(days[0], tmp_path / "missing-decision-snapshot" / "universe.parquet")],
        mode="valid",
        benchmark_index=BENCHMARK_TS_CODE,
        permitted_boards=BOARDS,
    )

    regression = payload["factor_regression"]
    assert (regression["available"], regression["reason"], regression["n_days"]) == (
        False, "factors_unavailable", 0,
    )
    assert payload["compact"]["benchmark_return"] == -1e-06
    style = payload["style"]
    assert style["available"] is False
    assert style["reason"] == "no_holdings"
    assert style["tilts"] is None and style["industries"] == []

    no_columns = replay_style_analysis(
        _replay(days),
        _daily(days).drop(columns=["circ_mv"]),
        replay_dir=tmp_path / "missing-replay-slot",
        universes=(),
        mode="valid",
        benchmark_index=BENCHMARK_TS_CODE,
        permitted_boards=BOARDS,
    )
    assert no_columns["compact"]["benchmark_return"] is None
    assert no_columns["style"]["reason"] == "style_columns_unavailable"


def test_daily_returns_chain_from_the_initial_equity():
    curve = [
        {"trade_date": "20220104", "initial_equity": 1_000_000.0, "equity": 1_010_000.0},
        {"trade_date": "20220105", "initial_equity": 1_000_000.0, "equity": 999_900.0},
    ]
    returns = daily_returns_from_curve(curve)
    assert [date for date, _ in returns] == ["20220104", "20220105"]
    assert returns[0][1] == pytest.approx(0.01)
    assert returns[1][1] == pytest.approx(999_900.0 / 1_010_000.0 - 1.0)
    # Rows without a usable equity are skipped, never treated as a flat day.
    assert daily_returns_from_curve([]) == []
    assert daily_returns_from_curve(
        [{"trade_date": "20220104", "initial_equity": 0.0, "equity": float("nan")}]
    ) == []


def test_benchmark_block_restates_the_replay_benchmark(tmp_path: Path):
    """The ``benchmark`` block a replay writes into its evaluation summary
    carries the style computation's own CSI 300 figures."""
    days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2024-01-02", periods=12)]
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    pd.DataFrame(
        [
            {
                "dataset": "index_daily",
                "ts_code": "000300.SH",
                "trade_date": day,
                "pct_chg": (-0.25 if index % 2 == 0 else 0.5),
            }
            for index, day in enumerate(days)
        ]
    ).to_parquet(replay_dir / "macro.parquet", index=False)

    analysis = replay_style_analysis(
        _replay(days),
        _daily(days),
        replay_dir=replay_dir,
        universes=(),
        mode="heldout",
        benchmark_index=BENCHMARK_TS_CODE,
        permitted_boards=BOARDS,
    )
    block = benchmark_summary_block(analysis)
    assert block is not None
    assert block["label"] == benchmark_index_label(BENCHMARK_TS_CODE)
    assert block["ts_code"] == BENCHMARK_TS_CODE
    assert block["benchmark_return"] == analysis["compact"]["benchmark_return"]
    assert block["excess_return"] == analysis["compact"]["excess_return"]


def test_benchmark_block_is_absent_when_the_slot_has_no_benchmark(tmp_path: Path):
    days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2024-01-02", periods=6)]
    analysis = replay_style_analysis(
        _replay(days), _daily(days), replay_dir=None, universes=(), mode="valid", benchmark_index=BENCHMARK_TS_CODE, permitted_boards=BOARDS
    )
    # No fabricated zero: a slot without index rows carries no benchmark at all,
    # and the report keeps reporting missing coverage truthfully.
    assert analysis["compact"]["benchmark_return"] is None
    assert analysis["factor_regression"]["reason"] == "factors_unavailable"
    assert benchmark_summary_block(analysis) is None


def test_a_slot_benchmark_read_error_surfaces_instead_of_reading_as_no_benchmark(tmp_path: Path):
    """Only a macro file without index columns means "no benchmark here"; a file
    that has them but cannot be read as index rows is damaged and must say so,
    not degrade into an unbenchmarked slot."""

    no_index = tmp_path / "no_index"
    no_index.mkdir()
    pd.DataFrame({"dataset": ["cn_cpi"], "trade_date": ["20240102"]}).to_parquet(
        no_index / "macro.parquet", index=False
    )
    assert slot_benchmark(no_index, benchmark_index=BENCHMARK_TS_CODE) == {}

    mistyped = tmp_path / "mistyped"
    mistyped.mkdir()
    pd.DataFrame(
        {"dataset": ["index_daily"], "ts_code": [300], "trade_date": ["20240102"], "pct_chg": [1.0]}
    ).to_parquet(mistyped / "macro.parquet", index=False)
    with pytest.raises(pa.ArrowException):
        slot_benchmark(mistyped, benchmark_index=BENCHMARK_TS_CODE)

    truncated = tmp_path / "truncated"
    truncated.mkdir()
    (truncated / "macro.parquet").write_bytes(b"PAR1")
    with pytest.raises(pa.ArrowException):
        slot_benchmark(truncated, benchmark_index=BENCHMARK_TS_CODE)


def _regression_inputs(tmp_path: Path, days: list[str]) -> tuple[Path, pd.DataFrame]:
    """A replay slot whose cross-section has real size, value and turnover spreads."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    pd.DataFrame(
        [
            {
                "dataset": "index_daily",
                "ts_code": BENCHMARK_TS_CODE,
                "trade_date": day,
                "pct_chg": (-0.25 if index % 2 == 0 else 0.5),
            }
            for index, day in enumerate(days)
        ]
    ).to_parquet(replay_dir / "macro.parquet", index=False)
    rows = []
    for index, day in enumerate(days):
        # 40 names per day, and the small half moves opposite the large half so
        # the size factor is not a constant.
        for slot in range(40):
            rows.append(
                {
                    "trade_date": day,
                    "ts_code": f"{slot:06d}.SZ",
                    "close": 10.0,
                    "circ_mv": 100.0 * (slot + 1),
                    "pb": _pb(slot),
                    "turnover_rate": _turnover(slot),
                    # daily.parquet stores pct_chg as a decimal fraction:
                    # +0.6% / -0.4%, not 0.6 / -0.4. The size spread runs on
                    # another period than the market's, the cheap names add
                    # a value spread on a third and the quiet names a
                    # turnover spread on a fourth, so no two legs are
                    # collinear.
                    "pct_chg": (0.006 if slot < 20 else -0.004) * (1.0, 1.0, -1.0, 0.5)[index % 4]
                    + ((0.002, -0.001, 0.0005)[index % 3] if slot in CHEAP else 0.0)
                    + ((0.003, -0.002, 0.001, 0.0, -0.001)[index % 5] if slot in QUIET else 0.0),
                }
            )
    return replay_dir, pd.DataFrame(rows)


def test_the_factor_regression_reads_the_market_and_size_loadings(tmp_path: Path):
    """A raw excess return cannot tell an edge from a small-cap or high-beta
    tilt, so the book's loadings on both ride beside it; the regression's
    intercept grades nothing and is not reported."""
    days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2024-01-02", periods=20)]
    replay_dir, daily = _regression_inputs(tmp_path, days)
    analysis = replay_style_analysis(
        _replay(days),
        daily,
        replay_dir=replay_dir,
        universes=(),
        mode="valid",
        benchmark_index=BENCHMARK_TS_CODE,
        permitted_boards=BOARDS,
    )
    block = analysis["factor_regression"]
    assert block["available"] is True
    # With no bars before the window (a replay opening on the first slot the
    # cache holds) the legs, formed on the prior day's values and the turnover
    # leg on a mean of at least ten days, start the regression on its
    # eleventh day.
    assert block["n_days"] == len(days) - 10
    loadings = [name for name, _series in FACTOR_SERIES]
    assert loadings == ["market_beta", "size_beta", "value_beta", "turnover_beta"]
    assert all(isinstance(block[name], float) for name in loadings)
    assert set(block) == {"method", "available", "reason", "n_days", "tracking_error", "r2", *loadings}
    # The method is stated with the numbers, not left to the reader.
    assert "沪深300" in block["method"] and "244" in block["method"]
    assert "前一交易日" in block["method"] and "20 个交易日" in block["method"]
    assert all(analysis[series] for _name, series in FACTOR_SERIES)
    compact = benchmark_summary_block(analysis)
    assert {name: compact[name] for name in [*loadings, "tracking_error"]} == {
        name: block[name] for name in [*loadings, "tracking_error"]
    }


def test_the_size_loading_is_stored_beside_the_tilt(tmp_path: Path):
    """``size_tilt`` says which rung of the cap ladder the book stands on;
    ``size_beta`` is its loading on the size spread — the one that produced
    the 2024-01 drawdown.
    The two move in opposite directions across measured books, so a budget
    written on the tilt cannot reach the loading. It therefore rides in the
    compact block of every result; a validate row reads the active series'
    loadings beside the book's tilt."""

    days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2024-01-02", periods=20)]
    replay_dir, daily = _regression_inputs(tmp_path, days)
    analysis = replay_style_analysis(
        _replay(days),
        daily,
        replay_dir=replay_dir,
        universes=(),
        mode="valid",
        benchmark_index=BENCHMARK_TS_CODE,
        permitted_boards=BOARDS,
    )

    size_beta = analysis["factor_regression"]["size_beta"]
    assert isinstance(size_beta, float)
    block = benchmark_summary_block(analysis)
    # Projected from the one computation point, not recomputed beside it, and
    # not the tilt under another name.
    assert block["size_beta"] == size_beta
    assert block["size_beta"] != block["size_tilt"]

    # Same slot, a cross-section carrying no returns: the size factor cannot be
    # built, so the loading reads as unmeasured (null) in the stored block
    # rather than standing there as a zero something could gate on.
    unmeasured = replay_style_analysis(
        _replay(days),
        _daily(days),
        replay_dir=replay_dir,
        universes=(),
        mode="valid",
        benchmark_index=BENCHMARK_TS_CODE,
        permitted_boards=BOARDS,
    )
    assert unmeasured["factor_regression"]["size_beta"] is None
    unmeasured_block = benchmark_summary_block(unmeasured)
    assert unmeasured_block["size_beta"] is None


def test_size_beta_is_measured_on_the_decimal_pct_chg_scale(tmp_path: Path):
    """A hand-built panel with a known size beta pins the factor's unit.

    ``daily.parquet`` normalizes ``pct_chg`` to a decimal fraction at snapshot
    load, on the same scale as the strategy and benchmark returns; the size
    factor used to be divided by 100 again, which left ``size_beta`` inflated
    100x (live windows reported 36-91 for what is really 0.36-0.91). Daily
    returns are built here as ``alpha + 0.8 * market + 0.5 * smb + 0.3 * hml
    + 0.2 * lmh`` exactly, so every coefficient is known in advance.
    """
    days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2024-01-02", periods=30)]
    market_percent = [0.5, -0.3, 0.8, -0.6, 0.2, -0.1]
    # Different periods, so no two legs are collinear.
    smb = [0.004, -0.002, -0.003, 0.001]
    hml = [0.002, -0.001, 0.003, -0.002, 0.0005]
    lmh = [0.003, -0.001, 0.002, -0.004, 0.001, 0.0005, -0.002]
    alpha_daily = 0.0002

    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    pd.DataFrame(
        [
            {
                "dataset": "index_daily",
                "ts_code": BENCHMARK_TS_CODE,
                "trade_date": day,
                # macro.parquet keeps index pct_chg in percent.
                "pct_chg": market_percent[index % len(market_percent)],
            }
            for index, day in enumerate(days)
        ]
    ).to_parquet(replay_dir / "macro.parquet", index=False)

    # 40 names ordered by circ_mv: the small 30% all move by the day's spread
    # and the big 30% do not, so small-minus-big is exactly ``smb``; the cheap
    # names alone also move by the value spread, so high-minus-low
    # book-to-price is exactly ``hml``; the quiet names alone move by the
    # turnover spread, so low-minus-high turnover is exactly ``lmh``.
    daily = pd.DataFrame(
        [
            {
                "trade_date": day,
                "ts_code": f"{slot:06d}.SZ",
                "close": 10.0,
                "circ_mv": 100.0 * (slot + 1),
                "pb": _pb(slot),
                "turnover_rate": _turnover(slot),
                "pct_chg": (smb[index % len(smb)] if slot < 20 else 0.0)
                + (hml[index % len(hml)] if slot in CHEAP else 0.0)
                + (lmh[index % len(lmh)] if slot in QUIET else 0.0),
            }
            for index, day in enumerate(days)
            for slot in range(40)
        ]
    )

    equity = 100_000.0
    curve = []
    for index, day in enumerate(days):
        market = market_percent[index % len(market_percent)] / 100.0
        equity *= (
            1.0
            + alpha_daily
            + 0.8 * market
            + 0.5 * smb[index % len(smb)]
            + 0.3 * hml[index % len(hml)]
            + 0.2 * lmh[index % len(lmh)]
        )
        curve.append(
            {
                "trade_date": day,
                "initial_equity": 100_000.0,
                "equity": equity,
                "cash": equity,
                "positions": {},
            }
        )

    analysis = replay_style_analysis(
        ReplayResult(tuple(curve), (), (), ()),
        daily,
        replay_dir=replay_dir,
        universes=(),
        mode="valid",
        benchmark_index=BENCHMARK_TS_CODE,
        permitted_boards=BOARDS,
    )

    # The published factor series is the raw decimal spread, not a percent.
    # With no bars before the window it starts on the second day, the legs
    # formed on the prior day's cap, and the turnover leg once a mean of ten
    # days stands behind it.
    assert analysis["size_factor_daily"][0] == [days[1], pytest.approx(smb[1])]
    assert analysis["value_factor_daily"][0] == [days[1], pytest.approx(hml[1])]
    assert analysis["turnover_factor_daily"][0] == [days[10], pytest.approx(lmh[10 % len(lmh)])]

    block = analysis["factor_regression"]
    assert block["available"] is True and block["n_days"] == len(days) - 10
    assert block["market_beta"] == 0.8
    assert block["size_beta"] == 0.5
    assert block["value_beta"] == 0.3
    assert block["turnover_beta"] == 0.2
    assert block["r2"] == 1.0
    assert block["tracking_error"] == 0.0

    # Rescaling the factor moves only its own coefficient: the market beta
    # and the fit quality are scale-invariant, which is why the old bug
    # corrupted ``size_beta`` alone.
    strategy = [(str(date), float(value)) for date, value in analysis["strategy_daily"]]
    benchmark = {str(date): float(value) for date, value in analysis["benchmark_daily"]}
    size = {str(date): float(value) for date, value in analysis["size_factor_daily"]}
    value = {str(date): float(item) for date, item in analysis["value_factor_daily"]}
    turnover = {str(date): float(item) for date, item in analysis["turnover_factor_daily"]}
    rescaled = _factor_regression(
        strategy,
        {
            "benchmark_daily": benchmark,
            "size_factor_daily": {k: v / 100.0 for k, v in size.items()},
            "value_factor_daily": value,
            "turnover_factor_daily": turnover,
        },
    )
    assert rescaled["size_beta"] == pytest.approx(50.0)
    assert rescaled["market_beta"] == block["market_beta"]
    assert rescaled["r2"] == block["r2"]


def test_size_legs_are_formed_on_the_prior_days_cap():
    """A name's leg is decided by the cap it had before earning the day's return.

    ``circ_mv`` is the end-of-day cap and already embeds ``pct_chg``: sorting on
    it moved the day's winners into the big leg and its losers into the small
    leg, which biased the 2022 replay's spread to -17% when the prior-day sort
    gives +8%, and inflated the neutralized excess of every small-loading book
    (the untouched template read +31%, corrected -4%). Here one name sits at
    the top of the small leg on day one and gains 50%: on the prior-day cap it
    stays in the small leg and the spread is 0.5/18; sorted on the same day's
    cap it would have left the leg and the spread would be exactly 0.
    """
    caps = [float(index + 1) for index in range(30)] + [100.0 + index for index in range(30)]
    winner = "000018.SZ"  # cap 18: the 18th smallest, the last name inside the bottom 30%
    rows = []
    for index, cap in enumerate(caps):
        code = f"{index + 1:06d}.SZ"
        gain = 0.5 if code == winner else 0.0
        rows.append({"ts_code": code, "trade_date": "20240102", "circ_mv": cap, "pct_chg": 0.001})
        rows.append({"ts_code": code, "trade_date": "20240103", "circ_mv": cap * (1.0 + gain), "pct_chg": gain})
    factor = _size_factor(pd.DataFrame(rows))
    # Day one has no prior cap for anyone and therefore no spread.
    assert list(factor) == ["20240103"]
    assert factor["20240103"] == pytest.approx(0.5 / 18)

    # An IPO's listing day is the same situation: no prior cap, so its listing
    # return (quoted against the issue price) never enters a leg.
    rows.append({"ts_code": "301999.SZ", "trade_date": "20240103", "circ_mv": 1.0, "pct_chg": 2.0})
    assert _size_factor(pd.DataFrame(rows))["20240103"] == pytest.approx(0.5 / 18)

    # The legs do not depend on how the slot's rows happen to be stored.
    shuffled = pd.DataFrame(rows).sample(frac=1, random_state=9).reset_index(drop=True)
    assert _size_factor(shuffled) == _size_factor(pd.DataFrame(rows))


def test_value_legs_are_formed_on_the_prior_days_book_to_price():
    """High minus low book-to-price (1 / pb), sorted on the previous day's pb
    as the size legs are on the previous day's cap, and a name whose pb is
    not positive has no book-to-price and sits out."""

    rows = []
    for slot in range(40):
        code = f"{slot:06d}.SZ"
        # pb 1..40: slots 0..11 are the cheap (high book-to-price) leg and
        # 28..39 the dear one; the cheapest name re-rates on day two.
        pb = float(slot + 1)
        rows.append({"ts_code": code, "trade_date": "20240102", "pb": pb, "pct_chg": 0.0})
        gain = 0.5 if slot == 0 else 0.0
        rows.append(
            {"ts_code": code, "trade_date": "20240103", "pb": pb * (1.0 + gain), "pct_chg": gain}
        )
    factor = _value_factor(pd.DataFrame(rows))
    assert list(factor) == ["20240103"]
    # Sorted on day one's pb the winner stays in the cheap leg of twelve.
    assert factor["20240103"] == pytest.approx(0.5 / 12)

    # A negative-book name is no cheapest name: it never enters a leg.
    loss = [{**row, "pb": -3.0, "pct_chg": 9.0} for row in rows if row["ts_code"] == "000039.SZ"]
    kept = [row for row in rows if row["ts_code"] != "000039.SZ"]
    assert "20240103" in _value_factor(pd.DataFrame([*kept, *loss]))
    assert _value_factor(pd.DataFrame([*kept, *loss]))["20240103"] == pytest.approx(0.5 / 12)


def test_turnover_legs_sort_on_the_prior_days_twenty_day_mean():
    """Low minus high turnover: each name sorted on its mean turnover rate
    over the last twenty trading days through the previous day, read
    only where at least ten of those days carry its rate, the 30 % tails
    equal-weighted. Here rates 1..40 put slots 0..11 in the low leg; on the
    window's twelfth day slot 0 churns 1,000 and gains 50 %. Sorted on its
    mean through the day before it stays in the low leg and low minus high
    reads +0.5/12; on the same day's mean it would have joined the high leg
    and read -0.5/12."""

    days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2024-01-02", periods=12)]
    rows = [
        {
            "ts_code": f"{slot:06d}.SZ",
            "trade_date": day,
            "turnover_rate": 1_000.0 if (index, slot) == (11, 0) else slot + 1.0,
            "pct_chg": 0.5 if (index, slot) == (11, 0) else 0.0,
        }
        for index, day in enumerate(days)
        for slot in range(40)
    ]
    factor = _turnover_factor(pd.DataFrame(rows))
    # The first mean of ten days is through day 10 of the window, read on day 11.
    assert list(factor) == days[10:]
    assert factor[days[10]] == 0.0
    assert factor[days[11]] == pytest.approx(0.5 / 12)
    # A name listed on the sixth day has six rates by the eleventh and sits
    # out the twelfth however low it churns and however far it moves.
    listed = [
        {"ts_code": "301999.SZ", "trade_date": day, "turnover_rate": 0.001, "pct_chg": 3.0 if index == 11 else 0.0}
        for index, day in enumerate(days)
        if index >= 5
    ]
    assert _turnover_factor(pd.DataFrame([*rows, *listed])) == factor
    # The legs do not depend on how the slot's rows happen to be stored.
    shuffled = pd.DataFrame(rows).sample(frac=1, random_state=3).reset_index(drop=True)
    assert _turnover_factor(shuffled) == factor


def _leg_panel(days: list[str], codes: list[str], seed: int) -> pd.DataFrame:
    """Daily bars of ``codes`` whose caps, book-to-price, turnover and returns
    all move from day to day, so every leg sorts afresh each day."""

    rng = np.random.default_rng(seed)
    shape = (len(days), len(codes))
    return pd.DataFrame(
        {
            "trade_date": np.repeat(days, len(codes)),
            "ts_code": np.tile(codes, len(days)),
            "close": 10.0,
            "circ_mv": rng.lognormal(8.0, 1.0, shape).ravel(),
            "pb": rng.normal(2.0, 1.5, shape).ravel(),
            "turnover_rate": rng.lognormal(0.0, 1.0, shape).ravel(),
            "pct_chg": rng.normal(0.0, 0.02, shape).ravel(),
        }
    )


# Thirty main-board and ten ChiNext names: the boards a current account may buy on.
_PERMITTED = [f"{slot:06d}.SZ" for slot in range(30)] + [f"3000{slot:02d}.SZ" for slot in range(10)]


def test_a_warm_up_gives_every_leg_its_value_from_the_replays_first_day(tmp_path: Path):
    """A replay whose source holds the trading days before it sorts its first
    days on them, so each leg reads on the replay's first day what a longer
    replay covering both periods reads on that day. They are read only to
    sort: no leg value is emitted for them, and bars on or after the replay's
    first day are refused as a warm-up. With none (a replay opening on the
    first slot the cache holds) the size and value legs begin on the second
    day and the turnover leg on the eleventh."""

    days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2024-01-02", periods=45)]
    panel = _leg_panel(days, _PERMITTED, seed=11)
    before, span = days[:25], days[25:]
    own = panel[panel["trade_date"].isin(span)]
    warmup = warmup_bars(panel[panel["trade_date"].isin(before)])
    # Only what the longest window reads is kept.
    assert sorted(set(warmup["trade_date"])) == before[-20:]

    longer = _factor_legs(panel, None, BOARDS)
    warm = _factor_legs(own, warmup, BOARDS)
    cold = _factor_legs(own, None, BOARDS)
    for series, leg in warm.items():
        assert sorted(leg) == span, series
        assert leg == {day: value for day, value in longer[series].items() if day >= span[0]}, series
    assert min(cold["size_factor_daily"]) == min(cold["value_factor_daily"]) == span[1]
    assert min(cold["turnover_factor_daily"]) == span[10]

    with pytest.raises(ValueError, match="not before the replay's first day"):
        _factor_legs(own, panel[panel["trade_date"].isin(days[20:26])], BOARDS)

    # The sidecar regresses on every day of the replay.
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    pd.DataFrame(
        {
            "dataset": "index_daily",
            "ts_code": BENCHMARK_TS_CODE,
            "trade_date": span,
            "pct_chg": np.random.default_rng(5).normal(0.0, 1.0, len(span)),
        }
    ).to_parquet(replay_dir / "macro.parquet", index=False)
    analysis = replay_style_analysis(
        _replay(span),
        own,
        replay_dir=replay_dir,
        universes=(),
        mode="valid",
        benchmark_index=BENCHMARK_TS_CODE,
        permitted_boards=BOARDS,
        warmup=warmup,
    )
    assert analysis["factor_regression"]["n_days"] == len(span)
    assert analysis["turnover_factor_daily"][0][0] == span[0]


def test_the_legs_are_formed_on_the_boards_the_account_may_buy_on():
    """Beijing-exchange and STAR names the account cannot buy -- micro-caps
    with extreme moves here -- leave the size, value and turnover legs as they
    are without them; read on every board they would move them. A profile
    that pins no boards is refused rather than read on every board."""

    days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2024-01-02", periods=15)]
    permitted = _leg_panel(days, _PERMITTED, seed=3)
    foreign = _leg_panel(
        days, [f"8300{slot:02d}.BJ" for slot in range(12)] + [f"6880{slot:02d}.SH" for slot in range(12)], seed=4
    ).assign(circ_mv=1.0, pb=50.0, turnover_rate=40.0, pct_chg=0.3)
    mixed = pd.concat([permitted, foreign], ignore_index=True).sample(frac=1, random_state=2)

    assert _factor_legs(mixed, None, BOARDS) == _factor_legs(permitted, None, BOARDS)
    assert _factor_legs(mixed, None, ("main", "gem", "star", "bj")) != _factor_legs(permitted, None, BOARDS)
    with pytest.raises(ValueError, match="permitted_boards"):
        replay_style_analysis(
            _replay(days),
            mixed,
            replay_dir=None,
            universes=(),
            mode="valid",
            benchmark_index=BENCHMARK_TS_CODE,
            permitted_boards=None,
        )


@pytest.mark.parametrize("exposure", [0.0, 0.2])
def test_cash_and_a_constant_low_beta_book_load_on_the_market_alone(exposure: float):
    """A book that only scales the market loads on it by its exposure and on
    nothing else, however much its raw excess flatters it because less capital
    was exposed."""

    dates = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2024-01-02", periods=24)]
    market = {day: (-0.012, 0.004, -0.006)[i % 3] for i, day in enumerate(dates)}
    size = {day: (0.005, -0.003, 0.002, -0.002)[i % 4] for i, day in enumerate(dates)}
    value = {day: (0.002, -0.004, 0.001, 0.003, -0.002)[i % 5] for i, day in enumerate(dates)}
    turnover = {day: (0.003, -0.001, 0.002, -0.004, 0.001, 0.0005, -0.002)[i % 7] for i, day in enumerate(dates)}
    strategy = [(day, exposure * market[day]) for day in dates]
    block = _factor_regression(
        strategy,
        {
            "benchmark_daily": market,
            "size_factor_daily": size,
            "value_factor_daily": value,
            "turnover_factor_daily": turnover,
        },
    )
    assert block["available"] is True
    assert block["tracking_error"] == 0.0
    assert block["market_beta"] == exposure
    assert (block["size_beta"], block["value_beta"], block["turnover_beta"]) == (0.0, 0.0, 0.0)
    # The raw benchmark excess is positive solely because less capital was exposed.
    raw = 1.0
    for _, value in strategy:
        raw *= 1.0 + value
    benchmark = 1.0
    for value in market.values():
        benchmark *= 1.0 + value
    assert raw > benchmark


def test_the_factor_regression_reports_missing_factors_instead_of_zero(tmp_path: Path):
    days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2024-01-02", periods=20)]
    # No macro.parquet: no benchmark, so no regression is possible.
    replay_dir = tmp_path / "empty_replay"
    replay_dir.mkdir()
    _, daily = _regression_inputs(tmp_path, days)
    analysis = replay_style_analysis(
        _replay(days),
        daily,
        replay_dir=replay_dir,
        universes=(),
        mode="valid",
        benchmark_index=BENCHMARK_TS_CODE,
        permitted_boards=BOARDS,
    )
    block = analysis["factor_regression"]
    assert block["available"] is False
    assert block["reason"] == "factors_unavailable"
    assert (block["market_beta"], block["tracking_error"]) == (None, None)
    assert benchmark_summary_block(analysis) is None


def test_the_sidecar_carries_the_random_baseline_and_the_active_figures_the_verdict_grades(tmp_path: Path):
    """A formal replay's random baseline is stored beside the strategy's own
    series, and the figures the Agent reads off it are the ones the verdict
    measures: the mean, standard deviation and drawdown of strategy minus
    random baseline, with no regression."""
    from autotrade.environment.replay.stats import attach_sub_window_benchmark
    from autotrade.environment.replay.style import active_analysis
    from autotrade.pipelines.verdict import (
        _mandate,
        _max_slice_drawdown,
        graded_statistics,
    )

    rng = np.random.default_rng(7)
    days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2024-01-02", periods=60)]
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    pd.DataFrame(
        {
            "dataset": "index_daily",
            "ts_code": BENCHMARK_TS_CODE,
            "trade_date": days,
            "pct_chg": rng.normal(0.0, 1.0, len(days)),
        }
    ).to_parquet(replay_dir / "macro.parquet", index=False)
    daily = pd.DataFrame(
        [
            {"trade_date": day, "ts_code": f"{slot:06d}.SZ", "close": 10.0,
             "circ_mv": 100.0 * (slot + 1), "pb": _pb(slot), "turnover_rate": _turnover(slot),
             "pct_chg": float(rng.normal(0.0, 0.01))}
            for day in days
            for slot in range(40)
        ]
    )
    equity = 100_000.0 * np.cumprod(1.0 + rng.normal(0.001, 0.01, len(days)))
    replay = ReplayResult(
        tuple(
            {"trade_date": day, "initial_equity": 100_000.0, "equity": float(value)}
            for day, value in zip(days, equity)
        ),
        (), (), (),
    )
    baselines = {
        "k": 20,
        "seed": 5,
        "matched": "index_membership+affordability",
        "dropped_trips_mean": 0.0,
        "baseline_daily": [[day, float(value)] for day, value in zip(days, rng.normal(0.0005, 0.008, len(days)))],
        "baseline_draw_sd": [[day, 0.004] for day in days],
    }

    def analyse(block):
        return replay_style_analysis(
            replay, daily, replay_dir=replay_dir, universes=(), mode="valid", baselines=block, benchmark_index=BENCHMARK_TS_CODE, permitted_boards=BOARDS
        )

    graded, bare = analyse(baselines), analyse(None)

    assert graded["baseline_daily"] == baselines["baseline_daily"]
    assert graded["baseline_draw_sd"] == baselines["baseline_draw_sd"]
    assert graded["baseline"] == {
        "k": 20, "seed": 5, "matched": "index_membership+affordability", "dropped_trips_mean": 0.0,
    }
    assert graded["strategy_daily"] == bare["strategy_daily"]
    compact = benchmark_summary_block(graded)
    active = graded_statistics(graded)
    assert active["series"] == "active"
    # Without its random baseline the replay has no active series to grade.
    with pytest.raises(ValueError, match="no random baseline"):
        graded_statistics(bare)
    assert compact["active_excess"] == round(active["plain_excess"], 4)
    assert compact["active_tracking_error"] == round(active["tracking_error"], 4)
    assert compact["active_information_ratio"] == round(active["information_ratio"], 4)
    # The mandate's figures are the strategy's own regression's.
    own = _mandate(bare, "", "")
    assert compact["tracking_error"] == round(own["tracking_error"], 4)
    assert compact["market_beta"] == round(own["market_beta"], 3)
    assert compact["baseline_draws"] == 20
    # The active drawdown the verdict holds to its limit is the one the Agent
    # reads: two sessions rebuilt it from proxies because no row had it.
    assert compact["active_max_drawdown"] == _max_slice_drawdown(active_analysis(graded), "", "")
    active_curve = np.concatenate(
        [[1.0], np.cumprod(1.0 + np.asarray(
            [value for _day, value in active_analysis(graded)["strategy_daily"]]
        ))]
    )
    expected = float(np.max(1.0 - active_curve / np.maximum.accumulate(active_curve)))
    assert expected > 0.0
    assert compact["active_max_drawdown"] == pytest.approx(expected, abs=1e-12)
    # A replay without a random baseline states none of it, and nothing else moved.
    without_baselines = benchmark_summary_block(bare)
    assert set(compact) - set(without_baselines) == {
        "active_excess",
        "active_tracking_error",
        "active_information_ratio",
        "active_max_drawdown",
        "baseline_draws",
    }
    assert {key: compact[key] for key in without_baselines} == without_baselines

    def year_rows(analysis):
        summary = {"sub_windows": [{"start": days[0], "end": days[-1], "return": 0.05}]}
        return attach_sub_window_benchmark(summary, analysis)["sub_windows"]

    [with_baselines], [without] = year_rows(graded), year_rows(bare)
    assert with_baselines["active_excess_return"] == round(active["plain_excess"], 6)
    assert "active_excess_return" not in without


def test_membership_is_read_from_the_decision_view_and_every_slot(tmp_path: Path):
    from autotrade.environment.replay.style import slot_membership

    def slot(name: str, rows: list[dict[str, object]]) -> Path:
        directory = tmp_path / name
        directory.mkdir()
        pd.DataFrame(rows).to_parquet(directory / "macro.parquet", index=False)
        return directory

    def weight(day: str, index: str, code: str) -> dict[str, object]:
        return {"dataset": "index_weight", "trade_date": day, "index_code": index, "con_code": code}

    decision = slot("decision", [weight("20240531", BENCHMARK_TS_CODE, "000001.SZ"),
                                 weight("20240531", "000905.SH", "000009.SZ")])
    replay = slot("replay", [weight("20240628", BENCHMARK_TS_CODE, "000001.SZ"),
                             weight("20240628", BENCHMARK_TS_CODE, "000002.SZ")])
    # An experiment that did not select index_weight has no such rows, or columns.
    plain = slot("plain", [{"dataset": "index_daily", "ts_code": BENCHMARK_TS_CODE,
                            "trade_date": "20240628", "pct_chg": 0.1}])
    # A mounted index_weight keeps its columns where its window holds no row
    # of the benchmark (here: only another index's section).
    other = slot("other", [weight("20240531", "000905.SH", "000009.SZ")])

    assert slot_membership([decision, replay, plain, tmp_path / "absent"], benchmark_index=BENCHMARK_TS_CODE) == {
        "20240531": frozenset({"000001.SZ"}),
        "20240628": frozenset({"000001.SZ", "000002.SZ"}),
    }
    # Unmounted and mounted-but-empty are different answers: only the first
    # lets the random baseline fall back to the float-cap decile.
    assert slot_membership([plain], benchmark_index=BENCHMARK_TS_CODE) is None
    assert slot_membership([plain, other], benchmark_index=BENCHMARK_TS_CODE) == {}
