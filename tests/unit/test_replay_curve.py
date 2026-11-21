"""``replay.curve``: one result file and its style sidecar as the daily curve
the research console and a Paper account draw."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotrade.environment.replay.curve import result_curve
from autotrade.environment.replay.style import STYLE_ARTIFACT_NAME

SIDECAR = {
    "benchmark": {"ts_code": "000905.SH", "label": "中证500"},
    "benchmark_daily": [["20231229", 0.5], ["20240102", 0.01], ["20240103", -0.02], ["20240104", 0.0]],
    "baseline_daily": [["20240102", 0.0], ["20240103", 0.1]],
}


def _result(directory: Path, curve: list[dict], sidecar: dict) -> Path:
    directory.mkdir()
    (directory / STYLE_ARTIFACT_NAME).write_text(json.dumps(sidecar), encoding="utf-8")
    result = directory / "result.json"
    result.write_text(json.dumps({"equity_curve": curve, "stats": {}}), encoding="utf-8")
    return result


def test_the_curve_opens_on_the_initial_equity_and_reads_the_sidecar_on_the_strategy_days(
    tmp_path: Path,
):
    curve = [
        {"trade_date": "20240103", "initial_equity": 100.0, "equity": 99.0, "cash": 49.5},
        {"trade_date": "20240102", "initial_equity": 100.0, "equity": 110.0, "cash": 110.0},
        {"trade_date": "20240104", "initial_equity": 100.0, "equity": 99.0},
    ]

    payload = result_curve(_result(tmp_path / "result", curve, SIDECAR))

    (strategy,) = payload["series"]
    # Day one against the initial 100, then close to close: +10 %, -10 %, flat.
    assert strategy["dates"] == ["20240102", "20240103", "20240104"]
    assert strategy["cum"] == [0.1, -0.01, -0.01]
    assert strategy["drawdown"] == [0.0, -0.1, -0.1]
    # The sidecar's series on the strategy's own days only.
    assert payload["benchmark"]["label"] == "中证500"
    assert payload["benchmark"]["dates"] == ["20240102", "20240103", "20240104"]
    assert payload["baseline"]["dates"] == ["20240102", "20240103"]
    # A day without a cash split has no position weight.
    assert payload["exposure"] == {
        "strategy": {"dates": ["20240102", "20240103"], "long": [0.0, 0.5]}
    }


def test_a_missing_result_is_an_empty_curve_and_an_unlabelled_benchmark_is_refused(
    tmp_path: Path,
):
    assert result_curve(tmp_path / "absent" / "result.json") == {
        "series": [],
        "benchmark": None,
        "baseline": None,
        "exposure": {},
    }
    curve = [{"trade_date": "20240102", "initial_equity": 100.0, "equity": 101.0, "cash": 0.0}]
    unlabelled = {**SIDECAR, "benchmark": {"ts_code": "000905.SH"}}
    with pytest.raises(ValueError, match="names no benchmark label"):
        result_curve(_result(tmp_path / "result", curve, unlabelled))
