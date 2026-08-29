"""The optional ``fit(context)`` entrypoint of the strategy ABI.

What must always hold: ``fit`` runs at the first decision of a replay and again
only when its declared ``REFIT_PERIOD`` rolls over; it sees exactly the context
``generate_orders`` sees that day; the state it writes is read-only for
``generate_orders``; a fit timeout or exception fails the backtest explicitly;
and every replay starts from an empty state directory.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.executor import (
    CONTAINER_STATE_DIR,
    DockerStrategyExecutor,
    StrategyExecutionError,
    TrustedStrategyExecutor,
    docker_available,
)
from autotrade.environment.replay import BacktestError, run_daily_replay
from autotrade.environment.runtime import chmod_tree
from autotrade.environment.sandbox import SandboxConfig, SandboxLimits
from autotrade.environment.strategy import (
    CN_TZ,
    AccountSnapshot,
    FitSchedule,
    StrategyContext,
    StrategySchedule,
)
from autotrade.environment.strategy_loader import (
    StrategyLoadError,
    load_strategy,
    load_strategy_module,
    validate_strategy_source,
)
from autotrade.environment.tools.modification_check import ModificationCheckTool
from autotrade.pipelines.config import ArtifactRevision, EvaluationRequest, SnapshotBundle
from autotrade.pipelines.pit_backend import PITDailyEvaluationBackend

from tests.unit.test_pit_daily_backend import _pit_slot_paths, _write_domains
from tests.unit.test_sandbox_runtime import _executor_for_process

TEMPLATE = Path(__file__).resolve().parents[2] / "configs" / "agent_output_template"

FIT_STRATEGY = '''import numpy as np

REFIT_PERIOD = "quarter"


def fit(context):
    np.save(context.state_dir + "/seen.npy", np.array([len(context.bars)]))


def generate_orders(context):
    seen = int(np.load(context.state_dir + "/seen.npy")[0])
    return [{
        "symbol": "000001.SZ",
        "action": "buy",
        "quantity": 100,
        "execute_at": "2099-01-01T09:30:00+08:00",
        "fitted_on": seen,
        "visible_now": len(context.bars),
    }]
'''


# --- static contract -------------------------------------------------------


def test_loader_accepts_fit_with_a_declared_refit_period_and_rooted_state_io():
    assert validate_strategy_source(FIT_STRATEGY) == FitSchedule("quarter")
    assert validate_strategy_source("def generate_orders(context): return []") is None
    once = "def fit(context): pass\ndef generate_orders(context): return []"
    assert validate_strategy_source(once) == FitSchedule(None)
    assert validate_strategy_source("REFIT_PERIOD = None\n" + once) == FitSchedule(None)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("REFIT_PERIOD = 'quarter'\ndef generate_orders(context): return []", "requires a fit"),
        (
            "REFIT_PERIOD = 'week'\ndef fit(context): pass\ndef generate_orders(context): return []",
            "REFIT_PERIOD",
        ),
        (
            "REFIT_PERIOD = PERIOD\ndef fit(context): pass\ndef generate_orders(context): return []",
            "assigned once",
        ),
        ("async def fit(context): pass\ndef generate_orders(context): return []", "synchronous fit"),
        ("def fit(a, b): pass\ndef generate_orders(context): return []", "exactly one context"),
        (
            "import numpy as np\ndef fit(context): np.save('/tmp/x.npy', [])\n"
            "def generate_orders(context): return []",
            "save only below context.state_dir",
        ),
        (
            "import numpy as np\ndef generate_orders(context): return np.load('x.npy')",
            "load only below",
        ),
        (
            "import numpy as np\ndef fit(context): np.save(context.models_dir + '/w.npy', [])\n"
            "def generate_orders(context): return []",
            "save only below context.state_dir",
        ),
    ],
)
def test_loader_rejects_malformed_fit_declarations_and_unrooted_io(source, message):
    with pytest.raises(StrategyLoadError, match=message):
        validate_strategy_source(source)


def test_loader_reads_are_rooted_at_state_or_models_but_writes_only_at_state():
    validate_strategy_source(
        "import numpy as np\nimport pandas as pd\n"
        "def fit(ctx):\n"
        "    pd.read_parquet(ctx.asof_dir + '/daily').to_parquet(ctx.state_dir + '/d.parquet')\n"
        "    np.savez(ctx.state_dir + '/w.npz', w=np.load(ctx.models_dir + '/prior.npy'))\n"
        "def generate_orders(context):\n"
        "    np.load(context.state_dir + '/w.npz')\n    return []\n"
    )


def test_bare_load_strategy_refuses_a_fit_strategy(tmp_path: Path):
    path = tmp_path / "main.py"
    path.write_text(FIT_STRATEGY, encoding="utf-8")
    with pytest.raises(StrategyLoadError, match="defines fit"):
        load_strategy(path)
    loaded = load_strategy_module(path)
    assert loaded.fit_schedule == FitSchedule("quarter") and callable(loaded.fit)


def test_fit_schedule_is_due_at_start_and_on_period_rollover_only():
    assert FitSchedule(None).is_due("20240102", None)
    assert not FitSchedule(None).is_due("20240401", "20240102")
    quarterly = FitSchedule("quarter")
    assert quarterly.is_due("20240102", None)
    assert not quarterly.is_due("20240328", "20240102")
    assert quarterly.is_due("20240401", "20240328")
    assert FitSchedule("day").is_due("20240103", "20240102")


# --- replay semantics (in-host executor) ------------------------------------


def _daily(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trade_date": day, "symbol": "000001.SZ", "open": 10.0, "close": 10.0 + index}
            for index, day in enumerate(dates)
        ]
    )


def _trusted(tmp_path: Path, source: str = FIT_STRATEGY) -> tuple[TrustedStrategyExecutor, Path]:
    path = tmp_path / "main.py"
    path.write_text(source, encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()
    return TrustedStrategyExecutor.from_path(path, state_dir=state), state


def test_fit_runs_at_start_and_on_schedule_with_the_same_pit_bound_as_orders(tmp_path: Path):
    executor, state = _trusted(tmp_path)
    fitted: list[str] = []
    original = executor.fit

    def spy(context):
        fitted.append(context.inference_at.strftime("%Y%m%d"))
        original(context)

    dates = ["20240328", "20240329", "20240401", "20240402"]
    with patch.object(executor, "fit", side_effect=spy):
        result = run_daily_replay(
            daily=_daily(dates), strategy=executor, schedule=StrategySchedule("day", "18:00")
        )
    assert fitted == ["20240328", "20240401"]
    assert "fit" in result.phase_seconds and result.phase_seconds["fit"] >= 0
    pending = result.pending_orders
    assert [order["visible_now"] for order in pending] == [1, 2, 3, 4]
    # fit on 03-28 saw one bar (that day's close, visible at 18:00); orders on
    # 03-29 still read that state; the 04-01 refit saw three bars.
    assert [order["fitted_on"] for order in pending] == [1, 1, 3, 3]
    assert not (state / "seen.npy").stat().st_mode & 0o222


def test_state_is_read_only_during_generate_orders_in_host(tmp_path: Path):
    source = FIT_STRATEGY.replace(
        "    seen = int(", "    np.save(context.state_dir + '/leak.npy', np.zeros(1))\n    seen = int("
    )
    executor, _state = _trusted(tmp_path, source)
    with pytest.raises(BacktestError, match="generate_orders failed.*Permission denied"):
        run_daily_replay(
            daily=_daily(["20240328"]), strategy=executor, schedule=StrategySchedule("day", "18:00")
        )


def test_a_fit_exception_fails_the_backtest_explicitly(tmp_path: Path):
    source = FIT_STRATEGY.replace("    np.save(", "    raise ValueError('singular')\n    np.save(")
    executor, _state = _trusted(tmp_path, source)
    with pytest.raises(BacktestError, match="fit failed at 2024-03-28T18:00:00\\+08:00: singular"):
        run_daily_replay(
            daily=_daily(["20240328"]), strategy=executor, schedule=StrategySchedule("day", "18:00")
        )


def test_a_fit_strategy_cannot_run_without_a_state_dir(tmp_path: Path):
    path = tmp_path / "main.py"
    path.write_text(FIT_STRATEGY, encoding="utf-8")
    with pytest.raises(StrategyExecutionError, match="has no state_dir"):
        TrustedStrategyExecutor.from_path(path)


# --- Docker executor and worker protocol -----------------------------------


def test_docker_command_binds_state_read_only_for_orders_and_read_write_for_fit(tmp_path: Path):
    path = tmp_path / "main.py"
    path.write_text(FIT_STRATEGY, encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    with pytest.raises(StrategyExecutionError, match="has no state_dir"):
        with patch.object(DockerStrategyExecutor, "_start"):
            DockerStrategyExecutor(path, SandboxConfig(), models_dir=models)
    with patch.object(DockerStrategyExecutor, "_start"):
        orders_worker = DockerStrategyExecutor(
            path, SandboxConfig(), models_dir=models, state_dir=state
        )
        fit_worker = DockerStrategyExecutor(
            path, SandboxConfig(), models_dir=models, state_dir=state, state_writable=True
        )
    mounts = [arg for arg in orders_worker.docker_command() if arg.startswith("type=bind")]
    assert f"type=bind,src={models.resolve()},dst=/strategy-data/models,readonly" in mounts
    assert f"type=bind,src={state.resolve()},dst={CONTAINER_STATE_DIR},readonly" in mounts
    fit_mounts = [arg for arg in fit_worker.docker_command() if arg.startswith("type=bind")]
    assert f"type=bind,src={state.resolve()},dst={CONTAINER_STATE_DIR}" in fit_mounts
    assert orders_worker.context_state_dir == CONTAINER_STATE_DIR
    context = StrategyContext(
        inference_at=datetime(2024, 3, 28, 8, 30, tzinfo=CN_TZ),
        bars=(),
        account=AccountSnapshot(cash=1.0, positions={}),
        state_dir=orders_worker.context_state_dir,
        models_dir=orders_worker.context_models_dir,
    )
    record = orders_worker._context_record(context)
    assert record["state_dir"] == CONTAINER_STATE_DIR
    assert record["models_dir"] == "/strategy-data/models"


def test_worker_answers_fit_with_fitted_and_writes_state(tmp_path: Path):
    strategy = tmp_path / "main.py"
    strategy.write_text(FIT_STRATEGY, encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()
    context = StrategyContext(
        inference_at=datetime(2024, 3, 28, 8, 30, tzinfo=CN_TZ),
        bars=(),
        account=AccountSnapshot(cash=1.0, positions={}),
        state_dir=str(state),
    ).to_record()
    context.pop("bars")
    process = subprocess.Popen(
        [sys.executable, "-m", "autotrade.environment.strategy_worker", str(strategy)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None and process.stdout is not None
    message = {
        "type": "fit",
        "sequence": 0,
        "reset": True,
        "base_count": 0,
        "total_count": 0,
        "context": context,
        "bars": [],
    }
    process.stdin.write(json.dumps(message) + "\n")
    process.stdin.flush()
    assert json.loads(process.stdout.readline()) == {"type": "fitted", "sequence": 0}
    assert np.load(state / "seen.npy").tolist() == [0]
    process.stdin.write('{"type":"close"}\n')
    process.stdin.flush()
    assert process.wait(timeout=5) == 0


def test_fit_timeout_fails_explicitly_and_aborts_the_worker():
    limits = SandboxLimits(timeout_seconds=30.0, fit_timeout_seconds=0.05)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    executor = _executor_for_process(process, limits=limits)
    executor.fit_schedule = FitSchedule(None)
    started = time.monotonic()
    with (
        patch.object(executor, "_remove_container") as remove,
        pytest.raises(StrategyExecutionError, match="strategy fit exceeded 0.05s"),
    ):
        executor._roundtrip(
            StrategyContext(
                inference_at=datetime(2024, 3, 28, 8, 30, tzinfo=CN_TZ),
                bars=(),
                account=AccountSnapshot(cash=1.0, positions={}),
            ),
            kind="fit",
            timeout_seconds=limits.fit_timeout_seconds,
        )
    assert time.monotonic() - started < 5
    assert process.poll() is not None and executor._closed is True
    remove.assert_called_once()


@pytest.mark.skipif(not docker_available(), reason="Docker is unavailable")
def test_real_sandbox_fit_writes_state_that_generate_orders_can_only_read(tmp_path: Path):
    strategy = tmp_path / "main.py"
    strategy.write_text(
        '''import numpy as np


def fit(context):
    np.save(context.state_dir + "/coef.npy", np.array([0.5, len(context.bars)]))


def generate_orders(context):
    try:
        np.save(context.state_dir + "/leak.npy", np.zeros(1))
        leak = "written"
    except OSError as exc:
        leak = str(exc)
    return [{
        "symbol": "000001.SZ",
        "action": "buy",
        "quantity": 100,
        "execute_at": "2099-01-01T09:30:00+08:00",
        "coef": np.load(context.state_dir + "/coef.npy").tolist(),
        "leak": leak,
    }]
''',
        encoding="utf-8",
    )
    state = tmp_path / "state"
    state.mkdir()
    state.chmod(0o777)
    executor = DockerStrategyExecutor(
        strategy, SandboxConfig(limits=SandboxLimits(fit_timeout_seconds=120.0)), state_dir=state
    )
    try:
        context = StrategyContext(
            inference_at=datetime(2024, 3, 28, 8, 30, tzinfo=CN_TZ),
            bars=(),
            account=AccountSnapshot(cash=1.0, positions={}),
            state_dir=executor.context_state_dir,
        )
        executor.fit(context)
        assert np.load(state / "coef.npy").tolist() == [0.5, 0.0]
        (order,) = executor.execute(context)
    finally:
        executor.close()
    assert order["coef"] == [0.5, 0.0]
    assert "Read-only file system" in order["leak"]
    assert not (state / "leak.npy").exists()


# --- the shipped template and the PIT evaluation path -----------------------


def _pit_bundle(tmp_path: Path, replay_days: list[str]) -> tuple[Path, Path]:
    snapshot, replay = _pit_slot_paths(
        tmp_path, decision="fit", replay="fit", generation_id="generation_fit"
    )
    (snapshot / "text_library").mkdir()
    (replay / "text_library").mkdir()
    _write_domains(snapshot, replay)
    rng = np.random.default_rng(7)
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    history = [
        stamp.strftime("%Y%m%d")
        for stamp in pd.bdate_range(end=pd.Timestamp(replay_days[0]) - pd.Timedelta(days=1), periods=80)
    ]

    def frame(days: list[str], start_price: float) -> pd.DataFrame:
        rows = []
        for symbol_index, symbol in enumerate(symbols):
            price = start_price + symbol_index
            for day in days:
                price *= float(np.exp(rng.normal(0.0, 0.02)))
                rows.append(
                    {
                        "ts_code": symbol,
                        "trade_date": day,
                        "open": round(price, 2),
                        "close": round(price, 2),
                        "up_limit": round(price * 1.1, 2),
                        "down_limit": round(price * 0.9, 2),
                        "adj_factor": 1.0,
                        "available_at": f"{day[:4]}-{day[4:6]}-{day[6:]}T17:30:00+08:00",
                    }
                )
        return pd.DataFrame(rows)

    frame(history, 10.0).to_parquet(snapshot / "daily.parquet", index=False)
    frame(replay_days, 12.0).to_parquet(replay / "daily.parquet", index=False)
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": "snap_fit",
                "kind": "decision_input",
                "raw_generation": {"generation_id": "generation_fit"},
            }
        ),
        encoding="utf-8",
    )
    (replay / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": "replay_fit",
                "kind": "replay_slot",
                "label": "valid",
                "period_start": replay_days[0],
                "period_end": replay_days[-1],
                "available_from": "2024-03-27T23:59:59+08:00",
                "raw_generation": {"generation_id": "generation_fit"},
            }
        ),
        encoding="utf-8",
    )
    chmod_tree(snapshot, file_mode=0o444, dir_mode=0o555)
    return snapshot, replay


def _evaluate(tmp_path: Path, revision: Path, snapshot: Path, replay: Path, days: list[str], *, max_days=None):
    backend = PITDailyEvaluationBackend(tmp_path / "results", execution_mode="trusted")
    result = backend.evaluate(
        EvaluationRequest(
            ArtifactRevision("revision_fit", revision, revision.parent / "models"),
            SnapshotBundle("snap_fit", str(snapshot), str(replay), generation_id="generation_fit"),
            "valid",
            days[0],
            days[-1],
            StrategySchedule("day", "08:30"),
            BrokerProfile(initial_cash=1_000_000),
        ),
        max_days=max_days,
    )
    return result, json.loads(Path(result.result_ref).read_text(encoding="utf-8"))


def test_template_passes_modification_check_and_a_short_pit_smoke_replay(tmp_path: Path):
    output = tmp_path / "revision" / "output"
    shutil.copytree(TEMPLATE, output, ignore=shutil.ignore_patterns("__pycache__"))
    models = tmp_path / "revision" / "models"
    models.mkdir()
    check = ModificationCheckTool(output, models_dir=models).invoke({})
    assert check.value["fit"] == {"refit_period": "quarter"}

    days = ["20240328", "20240329", "20240401", "20240402"]
    snapshot, replay = _pit_bundle(tmp_path, days)
    result, record = _evaluate(tmp_path, output, snapshot, replay, days, max_days=3)
    summary = result.summary
    assert summary["replayed_trade_days"] == 3
    assert summary["phase_seconds"]["fit"] > 0
    assert summary["order_count"] >= 1
    assert any(execution["status"] == "filled" for execution in record["executions"])
    result_dir = Path(result.result_ref).parent
    assert not (result_dir / "state").exists()


def test_every_replay_starts_from_an_empty_state_directory(tmp_path: Path):
    revision = tmp_path / "revision" / "output"
    revision.mkdir(parents=True)
    (tmp_path / "revision" / "models").mkdir()
    (revision / "main.py").write_text(
        '''import numpy as np

def fit(context):
    np.save(context.state_dir + "/marker.npy", np.array([1]))


def generate_orders(context):
    return [{
        "symbol": "000001.SZ",
        "action": "buy",
        "quantity": 100,
        "execute_at": "2099-01-01T09:30:00+08:00",
        "state_dir": context.state_dir,
    }]
''',
        encoding="utf-8",
    )
    days = ["20240328", "20240329"]
    snapshot, replay = _pit_bundle(tmp_path, days)
    _first, first_record = _evaluate(tmp_path, revision, snapshot, replay, days)
    _second, second_record = _evaluate(tmp_path, revision, snapshot, replay, days)
    first_state = Path(first_record["pending_orders"][0]["state_dir"])
    second_state = Path(second_record["pending_orders"][0]["state_dir"])
    assert first_state != second_state
    assert not first_state.exists() and not second_state.exists()


def test_a_fit_strategy_that_leaves_no_state_fails_the_first_decision(tmp_path: Path):
    revision = tmp_path / "revision" / "output"
    revision.mkdir(parents=True)
    (tmp_path / "revision" / "models").mkdir()
    (revision / "main.py").write_text(
        "import numpy as np\n"
        "def fit(context):\n    pass\n"
        "def generate_orders(context):\n"
        "    np.load(context.state_dir + '/missing.npy')\n    return []\n",
        encoding="utf-8",
    )
    days = ["20240328", "20240329"]
    snapshot, replay = _pit_bundle(tmp_path, days)
    with pytest.raises(BacktestError, match="generate_orders failed.*missing.npy"):
        _evaluate(tmp_path, revision, snapshot, replay, days)
