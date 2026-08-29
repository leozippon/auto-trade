"""``smoke_backtest``: the unofficial rehearsal that must not lie.

Seven of nine official backtests died on their first decision because each Fold
hand-rolled a shell smoke test against the flat frozen snapshot and a fake
account object. These tests pin the two properties that make the real tool worth
using instead: it runs the REAL replay path, and it stays outside every official
accounting surface.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.strategy import StrategySchedule
from autotrade.environment.tools import ToolError
from autotrade.environment.tools.modification_check import ModificationCheckTool
from autotrade.pipelines.config import FoldSessionRequest, SnapshotBundle
from autotrade.pipelines.folds import FoldSpec
from autotrade.pipelines.local_backend import (
    SMOKE_BACKTEST_MAX_DAYS,
    LocalDailyEvaluationBackend,
    SmokeBacktestTool,
)

DAYS = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2025-10-01", periods=12)]

WORKING_STRATEGY = """def generate_orders(context):
    # The real ABI: AccountSnapshot is an object, not a mapping.
    cash = context.account.cash
    held = dict(context.account.positions)
    if cash <= 0 or held:
        return []
    return [{
        "symbol": "000001.SZ",
        "action": "buy",
        "quantity": 100,
        "execute_at": context.inference_at.replace(hour=15, minute=0).isoformat(),
    }]
"""

SUBSCRIPT_STRATEGY = """def generate_orders(context):
    cash = context.account["cash"]
    return []
"""


def _fold() -> FoldSpec:
    moment = datetime(2025, 9, 30, 23, 59, 59, tzinfo=UTC)
    return FoldSpec(
        fold_id="fold_2026Q1",
        input_window_start="20240101",
        input_window_end="20250930",
        validation_start=DAYS[0],
        validation_end=DAYS[-1],
        test_start="20260101",
        test_end="20260331",
        valid_decision_time=moment,
        test_decision_time=moment,
    )


def _tool(root: Path, strategy: str, *, check=None, evaluator=None) -> SmokeBacktestTool:
    daily = root / "daily.parquet"
    pd.DataFrame(
        {
            "trade_date": DAYS,
            "symbol": ["000001.SZ"] * len(DAYS),
            "open": [10.0] * len(DAYS),
            "close": [10.5] * len(DAYS),
        }
    ).to_parquet(daily, index=False)
    output = root / "output"
    output.mkdir(parents=True)
    (output / "main.py").write_text(strategy, encoding="utf-8")
    models = root / "models"
    models.mkdir()
    request = FoldSessionRequest(
        experiment_id="exp",
        epoch_id="epoch_001",
        fold=_fold(),
        run_id="run_x",
        parent=None,
        snapshot=SnapshotBundle("snap", str(daily), str(daily)),
        max_steps=10,
        max_backtests=15,
        max_llm_calls=200,
        deadline_seconds=1200.0,
    )
    return SmokeBacktestTool(
        request=request,
        output_dir=output,
        models_dir=models,
        modification_check=check or ModificationCheckTool(output, models_dir=models),
        evaluator=evaluator
        or LocalDailyEvaluationBackend(
            daily, root / "results", execution_mode="trusted"
        ),
        schedule=StrategySchedule("day", "09:00"),
        broker_profile=BrokerProfile(initial_cash=100_000),
        # Host-only in production; a sibling of the workspace here.
        scratch_root=root / "runtime" / "smoke",
    )


def test_a_lost_sandbox_aborts_the_session_instead_of_reading_as_a_bad_strategy(
    tmp_path: Path,
) -> None:
    """Every other smoke failure is an observation the Agent can act on. A lost
    session sandbox is not: no later call can run, so it must leave the tool as
    a session interrupt rather than a strategy the Agent could try to fix."""
    from autotrade.pipelines.local_backend import SandboxLost

    class LostEvaluator:
        def evaluate(self, _request, max_days=None):
            raise SandboxLost("the session sandbox is gone")

    tool = _tool(tmp_path, WORKING_STRATEGY, evaluator=LostEvaluator())
    with pytest.raises(SandboxLost):
        tool.invoke({"days": 1})
    # The rehearsal copy is still cleaned up on the way out.
    assert list((tmp_path / "runtime" / "smoke").iterdir()) == []


def test_the_rehearsal_replays_a_snapshot_the_agent_cannot_reach(
    tmp_path: Path,
) -> None:
    """The session is not frozen during a rehearsal, so what runs must be a copy
    outside the Agent's mounts: a write to output/ while the replay is running
    reaches neither the replayed bytes nor the next call's result."""
    replayed: list[str] = []

    class RecordingEvaluator:
        def __init__(self, inner) -> None:
            self.inner = inner

        def evaluate(self, request, max_days=None):
            main = Path(request.revision.output_path) / "main.py"
            replayed.append(main.read_text(encoding="utf-8"))
            # The Agent keeps working while the rehearsal runs.
            (tmp_path / "output" / "main.py").write_text(
                SUBSCRIPT_STRATEGY, encoding="utf-8"
            )
            return self.inner.evaluate(request, max_days=max_days)

    daily = tmp_path / "daily.parquet"
    tool = _tool(tmp_path, WORKING_STRATEGY)
    tool.evaluator = RecordingEvaluator(
        LocalDailyEvaluationBackend(daily, tmp_path / "results", execution_mode="trusted")
    )
    result = tool.invoke({"days": 2})

    assert result.value["status"] == "ok", result.value
    # The replay read the approved bytes, not the ones written underneath it.
    assert replayed == [WORKING_STRATEGY]
    assert (tmp_path / "output" / "main.py").read_text(encoding="utf-8") == (
        SUBSCRIPT_STRATEGY
    )


def test_smoke_runs_the_real_replay_over_a_short_window(tmp_path: Path) -> None:
    tool = _tool(tmp_path, WORKING_STRATEGY)
    result = tool.invoke({"days": 3})

    assert result.ok
    value = result.value
    assert value["status"] == "ok"
    # The replay really was truncated, not the whole 12-day window.
    assert value["replayed_trade_days"] == 3
    assert value["decision_calls"] == 3
    assert value["days_requested"] == 3
    assert value["order_count"] >= 1
    # Per-day timing is the number the 30 s per-decision cap is judged against.
    assert set(value["seconds_per_day"]) == {"strategy", "data_view"}
    assert all(seconds >= 0.0 for seconds in value["seconds_per_day"].values())
    assert "asof_dir" in value["hint"] and "snapshot_dir" in value["hint"]


def test_smoke_is_outside_every_official_accounting_surface(tmp_path: Path) -> None:
    tool = _tool(tmp_path, WORKING_STRATEGY)
    value = tool.invoke({}).value

    assert value["official"] is False
    assert value["counts_against_backtest_budget"] is False
    # No revision was committed and no result survived for a ledger or a freeze
    # to pick up: the tool owns no artifact store and no step tree at all.
    assert not hasattr(tool, "tree")
    assert not hasattr(tool, "artifact_store")
    results = tmp_path / "results"
    assert not results.exists() or not any(results.iterdir())


def test_a_failing_strategy_returns_the_exact_exception_text(tmp_path: Path) -> None:
    tool = _tool(tmp_path, SUBSCRIPT_STRATEGY)
    result = tool.invoke({"days": 2})

    # The point of the tool: the Agent reads the real failure instead of a
    # green hand-rolled script followed by a dead official backtest.
    assert result.ok, "a failed rehearsal is a reportable result, not a tool error"
    assert result.value["status"] == "failed"
    assert "AccountSnapshot" in str(result.value["error"])
    assert "not subscriptable" in str(result.value["error"])
    assert result.value["counts_against_backtest_budget"] is False


def test_days_argument_is_bounded(tmp_path: Path) -> None:
    tool = _tool(tmp_path, WORKING_STRATEGY)
    for days in (0, SMOKE_BACKTEST_MAX_DAYS + 1):
        with pytest.raises(ToolError, match="between 1 and"):
            tool.invoke({"days": days})
    with pytest.raises(ToolError, match="must be an integer"):
        tool.invoke({"days": "3"})


def test_smoke_enforces_the_same_static_gate_as_the_official_run(tmp_path: Path) -> None:
    class RejectingCheck:
        def invoke(self, _arguments):
            from autotrade.environment.tools import ToolResult

            return ToolResult(False, error="formal output exceeds 8 files")

    tool = _tool(tmp_path, WORKING_STRATEGY, check=RejectingCheck())
    with pytest.raises(ToolError, match="blocked by modification_check"):
        tool.invoke({})


def test_modification_check_rejects_reading_an_asof_domain_as_a_flat_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    check = ModificationCheckTool(output)

    (output / "main.py").write_text(
        'import pandas as pd\n\n'
        'def generate_orders(context):\n'
        '    daily = pd.read_parquet(context.asof_dir + "/daily.parquet")\n'
        '    return []\n',
        encoding="utf-8",
    )
    with pytest.raises(ToolError) as caught:
        check.invoke({})
    message = str(caught.value)
    assert "context.asof_dir/daily.parquet" in message
    # The message has to carry the correct spelling, or it only says "no".
    assert 'pd.read_parquet(context.asof_dir + "/daily")' in message
    assert "point-in-time violation" in message

    # The directory form, the flat SNAPSHOT form, and text_library shards are
    # all legitimate and must not be caught.
    (output / "main.py").write_text(
        'import pandas as pd\n\n'
        'def generate_orders(context):\n'
        '    daily = pd.read_parquet(context.asof_dir + "/daily")\n'
        '    frozen = pd.read_parquet(context.snapshot_dir + "/daily.parquet")\n'
        '    body = pd.read_parquet(context.asof_dir + "/text_library/news.parquet")\n'
        '    return []\n',
        encoding="utf-8",
    )
    assert check.invoke({}).ok


def test_smoke_backtest_is_registered_for_fold_sessions() -> None:
    from autotrade.agent.runner import _FOLD_TOOLS

    assert SmokeBacktestTool.spec.name == "smoke_backtest"
    assert SmokeBacktestTool.spec.name in _FOLD_TOOLS
    # mutating: it must dispatch in order with write/check/backtest, never in a
    # read-only parallel batch alongside an edit.
    assert SmokeBacktestTool.spec.mutating is True
    description = SmokeBacktestTool.spec.description
    assert "UNOFFICIAL" in description
    assert "DIRECTORY of parquet parts" in description
    assert "before daily_backtest" in description


def test_result_json_is_not_left_behind_for_a_ledger_to_find(tmp_path: Path) -> None:
    tool = _tool(tmp_path, WORKING_STRATEGY)
    tool.invoke({"days": 2})
    leftovers = list((tmp_path / "results").rglob("result.json"))
    assert leftovers == [], f"smoke run left {leftovers} behind"
    assert json.dumps(tool.invoke({"days": 2}).value)  # strict-JSON serialisable
