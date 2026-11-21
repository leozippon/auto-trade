"""Canned snapshots and evaluation results that stand in for the PIT backend
in the experiment pipeline tests."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from autotrade.environment.replay.null_control import BASELINE_DRAWS
from autotrade.pipelines.calendar import ResearchGeometry
from autotrade.pipelines.config import (
    EvaluationResult,
    ExperimentConfig,
    SnapshotBundle,
)
from tests.unit.support.acceptance_fixture import RULES

GEOMETRY = ResearchGeometry(
    research_start="20220701",
    research_end="20240630",
    forward_end="20250630",
    heldout_end="20250930",
)


# The release ends inside Held-out, so the Held-out slot is clipped.
RELEASE_END = "20250912"


DAYS = [
    stamp.strftime("%Y%m%d")
    for stamp in np.arange(
        np.datetime64("2022-06-01"), np.datetime64("2025-09-13"), dtype="datetime64[D]"
    ).astype(datetime)
    if stamp.weekday() < 5
]


class Snapshots:
    """Bundles named after what was asked for, with every request logged."""

    def __init__(self) -> None:
        self.prepared: list[tuple[str, str, str, datetime]] = []
        self.decisions: list[datetime] = []

    def prepare(self, *, phase, start, end, decision_time):
        self.prepared.append((phase, start, end, decision_time))
        return SnapshotBundle(
            snapshot_id=f"decision_{decision_time:%Y%m%d}",
            decision_ref=f"decision/{decision_time:%Y%m%d}",
            replay_ref=f"replay/{phase}/{start}_{end}",
        )

    def prepare_decision(self, *, decision_time):
        self.decisions.append(decision_time)
        return SnapshotBundle(
            snapshot_id=f"decision_{decision_time:%Y%m%d}",
            decision_ref=f"decision/{decision_time:%Y%m%d}",
            replay_ref="",
        )


def _write_result(directory: Path, days: list[str], *, alpha: float, seed: int) -> EvaluationResult:
    rng = np.random.default_rng(seed)
    benchmark = rng.normal(0.0003, 0.01, len(days))
    size = rng.normal(0.0, 0.004, len(days))
    strategy = alpha + 0.9 * benchmark + 0.2 * size + rng.normal(0.0, 0.004, len(days))
    equity = 1_000_000 * np.cumprod(1 + strategy)
    curve = [
        {"trade_date": day, "equity": float(value), "initial_equity": 1_000_000.0, "cash": float(value) * 0.1}
        for day, value in zip(days, equity)
    ]
    executions = []
    for index in range(0, len(days) - 5, 10):
        executions.append(
            {"status": "filled", "action": "buy", "price": 10.0, "quantity": 1000,
             "matched_at": f"{days[index][:4]}-{days[index][4:6]}-{days[index][6:]}T09:30:00+08:00"}
        )
        executions.append(
            {"status": "filled", "action": "sell", "price": 10.5, "quantity": 1000, "realized_pnl": 500.0,
             "matched_at": f"{days[index + 5][:4]}-{days[index + 5][4:6]}-{days[index + 5][6:]}T09:30:00+08:00"}
        )
    directory.mkdir(parents=True)
    total_return = float(equity[-1] / 1_000_000 - 1)
    summary = {
        "total_return": total_return,
        "max_drawdown": 0.1,
        "sharpe": 1.0,
        "order_count": len(executions),
        "sub_windows": [],
        "benchmark": {"excess_return": total_return - float(np.prod(1 + benchmark) - 1)},
        # What the registration check's cost stress reads: the slippage, charged
        # again on the replay's turnover.
        "turnover": 1.0,
        "cost_sensitivity": {"slippage_bps": 5.0, "cost_per_bp_per_side": 1e-4},
    }
    (directory / "result.json").write_text(
        json.dumps(
            {
                "equity_curve": curve,
                "executions": executions,
                "inference_dates": [f"{day[:4]}-{day[4:6]}-{day[6:]}T08:30:00+08:00" for day in days],
                "stats": summary,
            }
        ),
        encoding="utf-8",
    )
    (directory / "style_analysis.json").write_text(
        json.dumps(
            {
                "strategy_daily": [[day, float(value)] for day, value in zip(days, strategy)],
                "benchmark_daily": [[day, float(value)] for day, value in zip(days, benchmark)],
                "size_factor_daily": [[day, float(value)] for day, value in zip(days, size)],
                # A value leg the book does not load on, from its own draws.
                "value_factor_daily": [
                    [day, float(value)]
                    for day, value in zip(
                        days, np.random.default_rng(seed + 10_000).normal(0.0, 0.003, len(days))
                    )
                ],
                # The random baseline: random names on the book's skeleton,
                # riding the market as the book does.
                "baseline_daily": [[day, float(0.9 * value)] for day, value in zip(days, benchmark)],
                "baseline": {"k": BASELINE_DRAWS},
            }
        ),
        encoding="utf-8",
    )
    return EvaluationResult(summary, str(directory / "result.json"))


_DEFAULT = ExperimentConfig(experiment_id="experiment", experiments_root=Path("unused"), acceptance=RULES)


CONFIG_SCHEDULE = _DEFAULT.schedule


CONFIG_PROFILE = _DEFAULT.broker_profile
