from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from autotrade.environment.replay.stats import TRADING_DAYS_PER_YEAR, ReplayResult
from autotrade.environment.replay.style import (
    BENCHMARK_LABEL,
    BENCHMARK_TS_CODE,
    _benchmark_regression,
    _neutralized_excess,
    _size_factor,
    benchmark_summary_block,
    daily_returns_from_curve,
    replay_style_analysis,
    write_style_rollup,
)
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.reporting import build_experiment_report


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
        snapshot_dir=snapshot_dir,
        mode="valid",
    )

    assert payload["schema_version"] == 1 and payload["mode"] == "valid"
    assert payload["benchmark_regression"]["available"] is True
    assert payload["benchmark_regression"]["n_days"] == 10
    assert payload["benchmark_regression"]["beta"] == 2.0
    assert payload["style"]["available"] is True
    assert payload["style"]["days"] == 10
    assert payload["style"]["tilts"]["size"] == -0.5
    assert payload["style"]["industries"][0] == {"name": "银行", "weight": 1.0}

    target = write_style_rollup(tmp_path / "result", payload)
    written = json.loads(target.read_text(encoding="utf-8"))
    assert target.name == "style_analysis.json"
    assert written["compact"]["beta"] == 2.0


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
        snapshot_dir=tmp_path / "missing-decision-snapshot",
        mode="valid",
    )

    regression = payload["benchmark_regression"]
    assert regression == {
        "available": False,
        "reason": "insufficient_overlapping_days",
        "n_days": 2,
        "benchmark_return": -1e-06,
        "beta": None,
        "alpha_annualized": None,
        "r2": None,
    }
    style = payload["style"]
    assert style["available"] is False
    assert style["reason"] == "no_holdings"
    assert style["tilts"] is None and style["industries"] == []

    no_columns = replay_style_analysis(
        _replay(days),
        _daily(days).drop(columns=["circ_mv"]),
        replay_dir=tmp_path / "missing-replay-slot",
        snapshot_dir=None,
        mode="valid",
    )
    assert no_columns["benchmark_regression"]["reason"] == "benchmark_unavailable"
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


def test_benchmark_regression_math_and_degenerate_inputs():
    strategy = [(f"202201{day:02d}", 0.02 * ((-1) ** day)) for day in range(1, 11)]
    bench = {date: value / 2 for date, value in strategy}
    regression = _benchmark_regression(strategy, bench)
    assert regression["available"] is True
    assert regression["beta"] == 2.0
    assert regression["r2"] == 1.0
    assert regression["n_days"] == 10
    assert regression["alpha_annualized"] == round(0.0 * TRADING_DAYS_PER_YEAR, 4)

    # Fewer overlapping days than the regression minimum: reported, not guessed.
    short = _benchmark_regression(strategy[:3], bench)
    assert short["available"] is False
    assert short["reason"] == "insufficient_overlapping_days"
    assert short["beta"] is None

    # No overlap at all is a different, named reason.
    none = _benchmark_regression(strategy, {})
    assert none["reason"] == "benchmark_unavailable"
    assert none["n_days"] == 0
    assert none["benchmark_return"] is None

    # A flat benchmark has no variance to regress against.
    flat = _benchmark_regression(strategy, {date: 0.0 for date, _ in strategy})
    assert flat["available"] is False
    assert flat["reason"] == "benchmark_variance_zero"


def test_benchmark_block_reaches_the_experiment_report(tmp_path: Path):
    """The summary block a replay writes is what the report reads back.

    The producer and the consumer used to disagree by omission: nothing wrote
    ``benchmark`` into an evaluation summary, so every experiment report was
    permanently ``status="warning"``. Drive the real style computation into a
    ledger record and build the report from it.
    """
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
        snapshot_dir=None,
        mode="frozen_test",
    )
    block = benchmark_summary_block(analysis)
    assert block is not None
    assert block["label"] == BENCHMARK_LABEL and block["ts_code"] == BENCHMARK_TS_CODE
    assert block["benchmark_return"] == analysis["compact"]["benchmark_return"]
    assert block["excess_return"] == analysis["compact"]["excess_return"]

    ledger_path = tmp_path / "ledger.jsonl"
    ledger = ExperimentLedger(ledger_path)
    ledger.append(
        {
            "record_type": "fold",
            "experiment_id": "e",
            "epoch_id": "epoch_001",
            "fold_id": "fold_2024Q1",
            "run_id": "run_1",
            "fold_status": "frozen",
            "test_period": "20240102..20240117",
            "validation_result": {"total_return": 0.02, "sharpe": 1.0, "max_drawdown": 0.05},
            "test_result": {
                "total_return": 0.01,
                "sharpe": 0.8,
                "max_drawdown": 0.07,
                "order_count": 4,
                "benchmark": block,
            },
        }
    )
    summary = build_experiment_report(ledger_path, tmp_path / "report")
    assert summary["benchmark"]["status"] == "ok"
    assert summary["status"] == "ok"
    assert summary["benchmark"]["covered_periods"] == 1


def test_benchmark_block_is_absent_when_the_slot_has_no_benchmark(tmp_path: Path):
    days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2024-01-02", periods=6)]
    analysis = replay_style_analysis(
        _replay(days), _daily(days), replay_dir=None, snapshot_dir=None, mode="valid"
    )
    # No fabricated zero: a slot without index rows carries no benchmark at all,
    # and the report keeps reporting missing coverage truthfully.
    assert analysis["benchmark_regression"]["reason"] == "benchmark_unavailable"
    assert benchmark_summary_block(analysis) is None


def _neutralization_inputs(tmp_path: Path, days: list[str]) -> tuple[Path, pd.DataFrame]:
    """A replay slot whose cross-section has a real small-minus-big spread."""
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
                    "pb": 1.0,
                    "turnover_rate": 1.0,
                    # daily.parquet stores pct_chg as a decimal fraction:
                    # +0.6% / -0.4%, not 0.6 / -0.4.
                    "pct_chg": (0.006 if slot < 20 else -0.004) * (1 if index % 2 else -1),
                }
            )
    return replay_dir, pd.DataFrame(rows)


def test_neutralized_excess_removes_the_market_and_size_contributions(tmp_path: Path):
    """A raw excess return cannot tell an edge from a small-cap or high-beta
    tilt — the audited fold's apparent edge was mostly the latter."""
    days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2024-01-02", periods=20)]
    replay_dir, daily = _neutralization_inputs(tmp_path, days)
    analysis = replay_style_analysis(
        _replay(days),
        daily,
        replay_dir=replay_dir,
        snapshot_dir=None,
        mode="valid",
    )
    block = analysis["neutralized_excess"]
    assert block["available"] is True
    # The size legs are formed on the prior day's cap, so the window's first
    # day has no factor and the regression runs on the remaining days.
    assert block["n_days"] == len(days) - 1
    assert block["market_beta"] is not None
    assert block["size_beta"] is not None
    # The method is stated with the number, not left to the reader.
    assert "沪深300" in block["method"] and "244" in block["method"]
    assert "前一交易日" in block["method"]
    assert analysis["size_factor_daily"]
    compact = benchmark_summary_block(analysis)
    assert compact["neutralized_excess_return"] == block["neutralized_excess_return"]
    assert compact["neutralized_excess_method"] == block["method"]
    # It is a different number from the raw excess, not a relabelling of it.
    assert compact["neutralized_excess_return"] != compact["excess_return"]


def test_size_beta_is_measured_on_the_decimal_pct_chg_scale(tmp_path: Path):
    """A hand-built panel with a known size beta pins the factor's unit.

    ``daily.parquet`` normalizes ``pct_chg`` to a decimal fraction at snapshot
    load, on the same scale as the strategy and benchmark returns; the size
    factor used to be divided by 100 again, which left ``size_beta`` inflated
    100x (live windows reported 36-91 for what is really 0.36-0.91). Daily
    returns are built here as ``alpha + 0.8 * market + 0.5 * smb`` exactly, so
    every coefficient is known in advance.
    """
    days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2024-01-02", periods=24)]
    market_percent = [0.5, -0.3, 0.8, -0.6, 0.2, -0.1]
    # A different period, so market and size are not collinear.
    smb = [0.004, -0.002, -0.003, 0.001]
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
    # and the big 30% do not, so small-minus-big is exactly ``smb``.
    daily = pd.DataFrame(
        [
            {
                "trade_date": day,
                "ts_code": f"{slot:06d}.SZ",
                "close": 10.0,
                "circ_mv": 100.0 * (slot + 1),
                "pb": 1.0,
                "turnover_rate": 1.0,
                "pct_chg": smb[index % len(smb)] if slot < 20 else 0.0,
            }
            for index, day in enumerate(days)
            for slot in range(40)
        ]
    )

    equity = 100_000.0
    curve = []
    for index, day in enumerate(days):
        market = market_percent[index % len(market_percent)] / 100.0
        equity *= 1.0 + alpha_daily + 0.8 * market + 0.5 * smb[index % len(smb)]
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
        snapshot_dir=None,
        mode="valid",
    )

    # The published factor series is the raw decimal spread, not a percent. It
    # starts on the second day: the legs are formed on the prior day's cap.
    assert analysis["size_factor_daily"][0] == [days[1], pytest.approx(smb[1])]

    block = analysis["neutralized_excess"]
    assert block["available"] is True and block["n_days"] == len(days) - 1
    assert block["market_beta"] == 0.8
    assert block["size_beta"] == 0.5
    assert block["r2"] == 1.0
    assert block["neutralized_excess_return"] == round(alpha_daily * TRADING_DAYS_PER_YEAR, 4)

    # Rescaling the factor moves only its own coefficient: the neutralized
    # excess, the market beta and the fit quality are scale-invariant, which is
    # why the old bug corrupted ``size_beta`` alone.
    strategy = [(str(date), float(value)) for date, value in analysis["strategy_daily"]]
    benchmark = {str(date): float(value) for date, value in analysis["benchmark_daily"]}
    size = {str(date): float(value) for date, value in analysis["size_factor_daily"]}
    rescaled = _neutralized_excess(strategy, benchmark, {k: v / 100.0 for k, v in size.items()})
    assert rescaled["size_beta"] == pytest.approx(50.0)
    assert rescaled["market_beta"] == block["market_beta"]
    assert rescaled["r2"] == block["r2"]
    assert rescaled["neutralized_excess_return"] == block["neutralized_excess_return"]


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


@pytest.mark.parametrize("exposure", [0.0, 0.2])
def test_cash_and_a_constant_low_beta_book_have_no_alpha_in_a_bear_market(exposure: float):
    """A book that only scales the market shows no neutralized excess, however
    much its raw excess flatters it because less capital was exposed."""

    dates = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2024-01-02", periods=24)]
    market = {day: (-0.012, 0.004, -0.006)[i % 3] for i, day in enumerate(dates)}
    size = {day: (0.005, -0.003, 0.002, -0.002)[i % 4] for i, day in enumerate(dates)}
    strategy = [(day, exposure * market[day]) for day in dates]
    block = _neutralized_excess(strategy, market, size)
    assert block["available"] is True
    assert block["neutralized_excess_return"] == 0.0
    assert block["market_beta"] == exposure
    assert block["size_beta"] == 0.0
    # The raw benchmark excess is positive solely because less capital was exposed.
    raw = 1.0
    for _, value in strategy:
        raw *= 1.0 + value
    benchmark = 1.0
    for value in market.values():
        benchmark *= 1.0 + value
    assert raw > benchmark


def test_neutralized_excess_reports_missing_factors_instead_of_zero(tmp_path: Path):
    days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2024-01-02", periods=20)]
    # No macro.parquet: no benchmark, so no neutralization is possible.
    replay_dir = tmp_path / "empty_replay"
    replay_dir.mkdir()
    _, daily = _neutralization_inputs(tmp_path, days)
    analysis = replay_style_analysis(
        _replay(days),
        daily,
        replay_dir=replay_dir,
        snapshot_dir=None,
        mode="valid",
    )
    block = analysis["neutralized_excess"]
    assert block["available"] is False
    assert block["reason"] == "factors_unavailable"
    assert block["neutralized_excess_return"] is None
    assert benchmark_summary_block(analysis) is None
