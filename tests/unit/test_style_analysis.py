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
