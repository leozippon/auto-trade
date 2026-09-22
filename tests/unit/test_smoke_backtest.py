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
from autotrade.environment.time_budget import InferenceTimeBudget
from autotrade.environment.tools import ToolError
from autotrade.environment.tools.modification_check import ModificationCheckTool
from autotrade.pipelines.config import (
    ReplaySpan,
    ResearchSessionRequest,
    SnapshotBundle,
)
from autotrade.pipelines.local_backend import (
    SMOKE_BACKTEST_MAX_DAYS,
    LocalDailyEvaluationBackend,
    SmokeBacktestTool,
)

from .test_inference_time_budget import FakeClock

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


def _tool(
    root: Path,
    strategy: str,
    *,
    check=None,
    evaluator=None,
    time_budget: InferenceTimeBudget | None = None,
) -> SmokeBacktestTool:
    daily = root / "daily.parquet"
    pd.DataFrame(
        {
            "trade_date": DAYS,
            "symbol": ["000001.SZ"] * len(DAYS),
            "open": [10.0] * len(DAYS),
            "close": [10.5] * len(DAYS),
            "pre_close": [10.5] * len(DAYS),
        }
    ).to_parquet(daily, index=False)
    output = root / "output"
    output.mkdir(parents=True)
    (output / "main.py").write_text(strategy, encoding="utf-8")
    models = root / "models"
    models.mkdir()
    snapshot = SnapshotBundle("snap", str(daily), str(daily))
    request = ResearchSessionRequest(
        experiment_id="exp",
        run_id="run_x",
        snapshot=snapshot,
        decision_time=datetime(2025, 9, 30, 23, 59, 59, tzinfo=UTC),
        research_years=(ReplaySpan("Y1", "valid", DAYS[0], DAYS[-1], snapshot),),
        input_window_start="20240101",
        max_replay_years=15,
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
        time_budget=time_budget or InferenceTimeBudget(duration_seconds=1200.0),
    )


def test_a_session_interrupt_aborts_the_session_instead_of_reading_as_a_bad_strategy(
    tmp_path: Path,
) -> None:
    """Every other smoke failure is an observation the Agent can act on. A
    session interrupt is not: the session is over, so it must leave the tool as
    an interrupt rather than a strategy the Agent could try to fix."""
    from autotrade.environment.tools.base import SessionInterrupt

    class InterruptedEvaluator:
        def evaluate(self, _request, max_days=None, start_day=None):
            raise SessionInterrupt("the session was interrupted")

    tool = _tool(tmp_path, WORKING_STRATEGY, evaluator=InterruptedEvaluator())
    with pytest.raises(SessionInterrupt):
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

        def evaluate(self, request, max_days=None, start_day=None):
            main = Path(request.revision.output_path) / "main.py"
            replayed.append(main.read_text(encoding="utf-8"))
            # The Agent keeps working while the rehearsal runs.
            (tmp_path / "output" / "main.py").write_text(
                SUBSCRIPT_STRATEGY, encoding="utf-8"
            )
            return self.inner.evaluate(request, max_days=max_days, start_day=start_day)

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
    # Per-day timing is the number the 30 s per-decision cap is judged against,
    # and only the strategy's own cost scales with the days a span replays.
    assert set(value["seconds_per_day"]) == {"strategy"}
    assert all(seconds >= 0.0 for seconds in value["seconds_per_day"].values())
    assert "asof_dir" in value["hint"] and "snapshot_dir" in value["hint"]


def test_smoke_is_outside_every_official_accounting_surface(tmp_path: Path) -> None:
    tool = _tool(tmp_path, WORKING_STRATEGY)
    value = tool.invoke({}).value

    assert value["official"] is False
    assert value["counts_against_replay_budget"] is False
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
    assert result.value["counts_against_replay_budget"] is False


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


def test_a_rehearsal_does_not_spend_the_session_thinking_clock(tmp_path: Path) -> None:
    """The negative path of the clock contract: a smoke costs no thinking time.

    A 5-day rehearsal is dominated by the same full ``fit`` a Validation runs,
    and the prompt requires one before every batch, so charging it to the
    session deadline priced the rehearsal in the one currency the session
    cannot refill while the formal verdict stayed free. The failure case is
    the one that hurt (an hour-long smoke dying at the fit cap), so it must
    both leave the clock alone and still hand the Agent the reason.
    """

    class FitCapEvaluator:
        """An hour of host wall clock, then the fit-cap failure."""

        def __init__(self, clock: FakeClock) -> None:
            self.clock = clock

        def evaluate(self, _request, max_days=None, start_day=None):
            self.clock.advance(3600.0)
            raise RuntimeError("strategy fit exceeded 3600s")

    clock = FakeClock()
    budget = InferenceTimeBudget(duration_seconds=1200.0, clock=clock)
    tool = _tool(
        tmp_path,
        WORKING_STRATEGY,
        evaluator=FitCapEvaluator(clock),
        time_budget=budget,
    )
    assert tool.session_time_budget is budget

    remaining_before = budget.remaining()
    result = tool.invoke({"days": 5})

    # The hour the host spent is not taken out of the session's thinking time,
    # and the deadline the Runner checks moved with it.
    assert budget.remaining() == pytest.approx(remaining_before)
    assert budget.remaining() > 0.0
    # The failure is still the Agent's to read and act on.
    assert result.ok
    assert result.value["status"] == "failed"
    assert "fit exceeded 3600s" in str(result.value["error"])
    assert result.value["counts_against_replay_budget"] is False


def test_smoke_backtest_is_registered_for_fold_sessions() -> None:
    from autotrade.agent.runner import _SESSION_TOOLS

    assert SmokeBacktestTool.spec.name == "smoke_backtest"
    assert SmokeBacktestTool.spec.name in _SESSION_TOOLS
    # mutating: it must dispatch in order with write/check/backtest, never in a
    # read-only parallel batch alongside an edit.
    assert SmokeBacktestTool.spec.mutating is True
    description = SmokeBacktestTool.spec.description
    assert "UNOFFICIAL" in description
    assert "DIRECTORY of parquet parts" in description
    assert "before batch_validate" in description


def test_result_json_is_not_left_behind_for_a_ledger_to_find(tmp_path: Path) -> None:
    tool = _tool(tmp_path, WORKING_STRATEGY)
    tool.invoke({"days": 2})
    leftovers = list((tmp_path / "results").rglob("result.json"))
    assert leftovers == [], f"smoke run left {leftovers} behind"
    assert json.dumps(tool.invoke({"days": 2}).value)  # strict-JSON serialisable


def test_start_probes_a_later_window_inside_the_research_period(tmp_path: Path) -> None:
    """A rehearsal at the start of the span does not size a fit deep inside it.

    alpha158_lgbm read 335.7 s off a 3-day smoke at the start of a four-year
    period and then lost 12 replay-years to fits that ran past the cap late in
    the same period. ``start`` is how that reading is taken where it matters, so
    the window really has to open there: asking for five days from the third-last
    trading day can only return the days that are actually left.
    """

    tool = _tool(tmp_path, WORKING_STRATEGY)
    value = tool.invoke({"days": 5, "start": DAYS[10]}).value

    assert value["status"] == "ok", value
    assert value["start"] == DAYS[10]
    assert value["replayed_trade_days"] == 2
    # The default window still opens at the start of the research period and
    # says nothing about a probe date it was not given.
    assert "start" not in tool.invoke({"days": 2}).value


def test_start_outside_the_research_period_is_refused(tmp_path: Path) -> None:
    """The session may measure a fit anywhere it researches, and nowhere else."""

    tool = _tool(tmp_path, WORKING_STRATEGY)
    for outside in ("20250101", "20260101"):
        with pytest.raises(ToolError, match="outside the research period"):
            tool.invoke({"start": outside})
    with pytest.raises(ToolError, match="must be a YYYYMMDD string"):
        tool.invoke({"start": 20251010})
    with pytest.raises(ToolError, match="is not a date"):
        tool.invoke({"start": "not-a-day"})


def test_the_one_off_window_build_is_never_reported_per_day(tmp_path: Path) -> None:
    """Opening a late window publishes the whole as-of prefix once, on the
    window's first day, and costs minutes of host wall clock. Divided by the
    probe's days it reads as a per-day rate, and an audited session sized a
    full span on that number: it is reported as the whole-run figure it is,
    and the per-day block carries only what scales with the days replayed."""

    class LateWindowEvaluator:
        def evaluate(self, _request, max_days=None, start_day=None):
            del max_days, start_day
            from autotrade.pipelines.config import EvaluationResult

            return EvaluationResult(
                {
                    "replayed_trade_days": 3,
                    "decision_calls": 3,
                    # A cold late window: 279 s of the 335 s is day one.
                    "phase_seconds": {
                        "strategy": 4.5,
                        "data_view": 335.7,
                        "timeview_refresh": 279.3,
                    },
                },
                str(tmp_path / "gone" / "result.json"),
            )

    value = _tool(tmp_path, WORKING_STRATEGY, evaluator=LateWindowEvaluator()).invoke(
        {"days": 3, "start": "20251015"}
    ).value
    assert value["seconds_per_day"] == {"strategy": 1.5}
    assert value["window_build_seconds"] == 335.7
    # The raw instrumentation is still there, and the sub-phase it wraps is
    # never added to it.
    assert value["phase_seconds"]["timeview_refresh"] == 279.3
    description = SmokeBacktestTool.spec.description
    assert "window_build_seconds" in description and "never multiply it by days" in description


def test_a_rehearsal_reports_what_it_cost_its_container(tmp_path: Path) -> None:
    """The rehearsal is where a batch is sized, so it carries the telemetry.

    An arm that profiled its fit under ``shell`` concluded the replay was
    swapping against a ceiling it had guessed at. Here the numbers come from
    the container that actually ran the replay.
    """

    usage = {
        "peak_memory_bytes": 7_883_149_312,
        "memory_limit_bytes": 17_179_869_184,
        "fit_seconds": [335.7],
        "fit_timeout_seconds": 3600.0,
        "decision_timeout_seconds": 360.0,
    }

    class MeasuredEvaluator:
        def evaluate(self, _request, max_days=None, start_day=None):
            del max_days, start_day
            from autotrade.pipelines.config import EvaluationResult

            return EvaluationResult(
                {
                    "replayed_trade_days": 2,
                    "decision_calls": 2,
                    "phase_seconds": {"strategy": 1.0, "data_view": 0.5},
                    "resources": dict(usage),
                },
                str(tmp_path / "gone" / "result.json"),
            )

    tool = _tool(tmp_path, WORKING_STRATEGY, evaluator=MeasuredEvaluator())
    assert tool.invoke({"days": 2}).value["resources"] == usage


def test_a_failed_rehearsal_still_reports_what_it_was_using(tmp_path: Path) -> None:
    """A fit that died on its clock is exactly the run worth measuring."""

    from types import SimpleNamespace

    from autotrade.environment.executor import attach_strategy_resources

    usage = {"peak_memory_bytes": 30_064_771_072, "fit_timeout_seconds": 3600.0}

    class FitCapEvaluator:
        def evaluate(self, _request, max_days=None, start_day=None):
            del max_days, start_day
            error = RuntimeError("strategy fit exceeded 3600s")
            attach_strategy_resources(
                error, SimpleNamespace(resource_usage=lambda: dict(usage))
            )
            raise error

    tool = _tool(tmp_path, WORKING_STRATEGY, evaluator=FitCapEvaluator())
    value = tool.invoke({"days": 1}).value
    assert value["status"] == "failed"
    assert value["resources"] == usage
    # A replay that reported nothing carries no empty block.
    plain = tmp_path / "plain"
    plain.mkdir()
    assert "resources" not in _tool(plain, WORKING_STRATEGY).invoke({"days": 1}).value
