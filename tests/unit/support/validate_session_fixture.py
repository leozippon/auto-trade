"""A research session over a real step tree and a stub evaluator: the harness
of the validate and test tool tests."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from autotrade.common.records import write_json_atomic
from autotrade.environment.artifacts import (
    FilesystemArtifactStore,
    ModificationConstraints,
    readonly_baseline,
)
from autotrade.environment.data.contracts import DEFAULT_BENCHMARK_INDEX
from autotrade.environment.executor import (
    GpuMemoryContention,
    StrategyExecutionError,
    StrategyMemoryExceeded,
    StrategyRaised,
    attach_strategy_resources,
)
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.replay.engine import BacktestError
from autotrade.environment.runtime import RunManifest
from autotrade.environment.sandbox import SandboxConfig
from autotrade.environment.step_tree import StepTree
from autotrade.environment.time_budget import InferenceTimeBudget
from autotrade.environment.tools.modification_check import ModificationCheckTool
from autotrade.environment.tools.step_rollback import StepRollbackTool
from autotrade.environment.tools.workspace import SafeWorkspace
from autotrade.pipelines.config import (
    DEFAULT_BATCH_VALIDATE_MAX_CONCURRENCY,
    BrokerProfile,
    BudgetUsed,
    EvaluationResult,
    ReplaySpan,
    ResearchSessionRequest,
    SessionResume,
    SnapshotBundle,
    StepResult,
    StrategySchedule,
)
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.session_tools import SessionValidations, ValidateTool
from tests.unit.support.acceptance_fixture import REQUIRED_RULES
from tests.unit.support.agent_fixture import TerminalTool

PARENT_SOURCE = "def generate_orders(context):\n    return []\n"
# A four-year research period, one slot per July-June year.
YEARS = (
    ("Y1", "20210701", "20220630"),
    ("Y2", "20220701", "20230630"),
    ("Y3", "20230701", "20240630"),
    ("Y4", "20240701", "20250630"),
)
# The validation year: only test replays it, so no span may name it.
VALIDATION_YEAR = ("Y5", "20250701", "20260630")
TEMPLATE_README = "# Strategy output contract\n\nRead-only template text.\n"


def _strategy(marker: str) -> str:
    return f"def generate_orders(context):\n    _ = {marker}\n    return []\n"


def _summary(
    total_return: float, year_windows: tuple[tuple[str, str], ...] = (("20210701", "20220630"),)
) -> dict[str, object]:
    """A replay summary with one sub-window per research year the request
    names, as the real backend reports them, and the benchmark block."""

    return {
        "total_return": total_return,
        "sharpe": 1.0,
        "max_drawdown": 0.05,
        "turnover": 1.2,
        "trade_count": 3,
        "replayed_trade_days": 60,
        "per_stock": [{"symbol": "000001.SZ"} for _ in range(200)],
        "weekly_returns": [{"week_end": "20220107", "return": 0.01}],
        "benchmark": {
            "ts_code": DEFAULT_BENCHMARK_INDEX,
            "label": "index",
            "benchmark_return": 0.01,
            "excess_return": total_return - 0.01,
            "active_max_drawdown": 0.02,
            "size_tilt": 0.1,
            "top_industry_weight": 0.2,
        },
        "sub_windows": [
            {
                "kind": "research_year",
                "label": f"{start[:6]}-{end[:6]}",
                "start": start,
                "end": end,
                "trade_days": 60 // len(year_windows),
                "partial": False,
                "return": total_return,
                "benchmark_return": 0.01,
                "excess_return": total_return - 0.01,
                "active_excess_return": 0.1 * (index + 1),
                "baseline_return": 0.0,
                "sharpe": 1.0,
                "max_drawdown": 0.05,
                "turnover": 1.2,
                "trade_count": 3,
            }
            for index, (start, end) in enumerate(year_windows)
        ],
    }


def _replay_failure(
    message: str, cause: Exception, resources: Mapping[str, object] | None = None
) -> BacktestError:
    """What the replay engine hands back: its own error over the executor's.

    ``raised_by_strategy`` reads the chain, so a test failure has to carry one
    to be classified the way a real one is, and so does the container telemetry
    the pipeline attaches on the way out.
    """

    error = BacktestError(message)
    error.__cause__ = cause
    if resources is not None:
        attach_strategy_resources(
            error, SimpleNamespace(resource_usage=lambda: dict(resources))
        )
    return error


class _Evaluator:
    """Replays a revision by reading its ``main.py`` marker.

    The three marker tuples make one named candidate blow up the way a real
    replay does, one per failure the environment actually produces: a strategy
    clock that ran out and a shared card taken by another process are the host's
    failures, while the strategy's own exception is a measurement of the
    strategy. Which one it is decides whether the candidate keeps its
    replay-years; any of them proves one failure does not sink the batch.
    """

    def __init__(
        self,
        results_root: Path,
        *,
        fail_markers: tuple[str, ...] = (),
        contention_markers: tuple[str, ...] = (),
        strategy_fail_markers: tuple[str, ...] = (),
        memory_kill_markers: tuple[str, ...] = (),
        rendezvous: int = 0,
        resources: Mapping[str, object] | None = None,
        alpha_scale: float = 0.0005,
    ) -> None:
        self.results_root = results_root
        self.resources = resources
        # Daily alpha per character of main.py: the default is large enough
        # that no fixture series ever draws down.
        self.alpha_scale = alpha_scale
        self.fail_markers = fail_markers
        self.contention_markers = contention_markers
        self.strategy_fail_markers = strategy_fail_markers
        self.memory_kill_markers = memory_kill_markers
        self.calls = 0
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()
        # Only the first ``rendezvous`` replays meet here, so a batch larger
        # than the concurrency bound still completes.
        self._barrier = threading.Barrier(rendezvous, timeout=10) if rendezvous else None
        self._arrived = 0
        # The real evaluators carry the strategy container's boundary here and
        # read it at evaluate() time; batch_validate widens both strategy
        # clocks on it for the pool's lifetime, so each replay records what it
        # was handed.
        self.sandbox = SandboxConfig()
        self.fit_timeouts: list[float] = []
        self.inference_timeouts: list[float] = []
        self.requests: list[object] = []
        # Merged into every summary: what a test needs the replay to report
        # beyond the fixture's (a benchmark excess, a cost block).
        self.summary_extra: dict[str, object] = {}

    def evaluate(self, request, max_days=None, start_day=None, end_day=None):
        del max_days, start_day, end_day
        source = (Path(request.revision.output_path) / "main.py").read_text(
            encoding="utf-8"
        )
        with self._lock:
            self.calls += 1
            self.requests.append(request)
            self._arrived += 1
            self.fit_timeouts.append(self.sandbox.fit_timeout_seconds)
            self.inference_timeouts.append(self.sandbox.timeout_seconds)
            call_index = self.calls
            arrival = self._arrived
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self._barrier is not None and arrival <= self._barrier.parties:
                self._barrier.wait()
            for marker in self.fail_markers:
                if marker in source:
                    raise _replay_failure(
                        f"generate_orders exceeded 30s ({marker})",
                        StrategyExecutionError("Docker generate_orders failed: timeout"),
                        self.resources,
                    )
            for marker in self.contention_markers:
                if marker in source:
                    contention = GpuMemoryContention(
                        "CUDA out of memory with 6195 MiB available to this call: "
                        "other processes on the card held the rest"
                    )
                    raise _replay_failure(
                        f"fit failed ({marker}): {contention}",
                        contention,
                        self.resources,
                    )
            for marker in self.strategy_fail_markers:
                if marker in source:
                    raise _replay_failure(
                        f"generate_orders raised ({marker})",
                        StrategyRaised("KeyError: 'close'"),
                        self.resources,
                    )
            for marker in self.memory_kill_markers:
                if marker in source:
                    killed = StrategyMemoryExceeded(
                        "Docker strategy fit failed: strategy worker was killed before a "
                        "response (exit 137, SIGKILL): the container's 32g memory cap is "
                        "what kills a worker this way"
                    )
                    raise _replay_failure(f"fit failed ({marker}): {killed}", killed)
            summary = {
                **_summary(0.01 * len(source), request.year_windows or ((request.start, request.end),)),
                **self.summary_extra,
            }
            if self.resources is not None:
                summary["resources"] = dict(self.resources)
            target = self.results_root / f"valid_{call_index:03d}" / "result.json"
            write_json_atomic(target, {"stats": summary})
            _write_style_sidecar(
                target.parent, alpha=self.alpha_scale * len(source), seed=call_index
            )
            return EvaluationResult(summary=dict(summary), result_ref=str(target))
        finally:
            with self._lock:
                self.active -= 1


def _write_style_sidecar(directory: Path, *, alpha: float, seed: int) -> None:
    """The daily series the registration check reads: 60 days, fifteen in each of the
    four research years the gate counts, of a return that is ``alpha`` plus
    benchmark and size exposure plus noise, and the random-baseline composite
    every formal replay carries, which the gate subtracts."""

    rng = np.random.default_rng(seed)
    days = [f"{2021 + index // 15}09{index % 15 + 1:02d}" for index in range(60)]
    benchmark = rng.normal(0.0003, 0.01, len(days))
    size = rng.normal(0.0, 0.004, len(days))
    baselines = rng.normal(0.0, 0.002, len(days))
    strategy = (
        alpha + 0.9 * benchmark + 0.2 * size + baselines + rng.normal(0.0, 0.004, len(days))
    )
    sidecar = {
        "strategy_daily": [[day, float(value)] for day, value in zip(days, strategy)],
        "benchmark_daily": [[day, float(value)] for day, value in zip(days, benchmark)],
        "size_factor_daily": [[day, float(value)] for day, value in zip(days, size)],
        "value_factor_daily": [
            [day, float(value)]
            for day, value in zip(days, np.random.default_rng(seed + 10_000).normal(0.0, 0.003, len(days)))
        ],
        "turnover_factor_daily": [
            [day, float(value)]
            for day, value in zip(days, np.random.default_rng(seed + 20_000).normal(0.0, 0.003, len(days)))
        ],
        "baseline_daily": [[day, float(value)] for day, value in zip(days, baselines)],
    }
    write_json_atomic(directory / "style_analysis.json", sidecar)


class _Session:
    """One research session's real tools over a temporary workspace."""

    def __init__(
        self,
        root: Path,
        *,
        fail_markers: tuple[str, ...] = (),
        contention_markers: tuple[str, ...] = (),
        strategy_fail_markers: tuple[str, ...] = (),
        memory_kill_markers: tuple[str, ...] = (),
        rendezvous: int = 0,
        record_failed_attempts: bool = True,
        deadline_seconds: float = 600.0,
        readonly_template: bool = False,
        trace: list[tuple[str, dict[str, object]]] | None = None,
        experiment_dir: Path | None = None,
        budget_used: BudgetUsed | None = None,
        steps_before: tuple[StepResult, ...] = (),
        resume: SessionResume | None = None,
        resources: Mapping[str, object] | None = None,
        alpha_scale: float = 0.0005,
        acceptance_rules: Mapping[str, object] | None = None,
        max_concurrency: int = DEFAULT_BATCH_VALIDATE_MAX_CONCURRENCY,
    ) -> None:
        # Where the worker roots a session, as every session has one: the
        # registration check reads a nominee's strategy from its revision store.
        experiment_dir = experiment_dir or root / "experiment"
        self.root = root
        self.trace_events = trace
        self.workspace_root = root / "workspace"
        self.output = self.workspace_root / "output"
        self.models = self.workspace_root / "models"
        self.parent = root / "parent"
        for directory in (self.output, self.models, self.parent):
            directory.mkdir(parents=True)
        (self.parent / "main.py").write_text(PARENT_SOURCE, encoding="utf-8")
        (self.output / "main.py").write_text(PARENT_SOURCE, encoding="utf-8")
        if readonly_template:
            # The read-only contract README travels with every formal artifact.
            for directory in (self.parent, self.output):
                (directory / "README.md").write_text(TEMPLATE_README, encoding="utf-8")
        # As the host does at seeding: the read-only files are pinned to the
        # bytes the workspace received, not to whatever the parent holds later.
        self.seeded_readonly = readonly_baseline(self.output)
        moment = datetime(2025, 6, 30, 23, 59, 59, tzinfo=UTC)
        snapshot = SnapshotBundle("snapshot", "decision", "")
        request = ResearchSessionRequest(
            experiment_id="exp",
            run_id="run_batch",
            snapshot=snapshot,
            decision_time=moment,
            years=tuple(
                ReplaySpan(
                    label,
                    "valid",
                    start,
                    end,
                    SnapshotBundle(f"snap_{label}", f"decision_{label}", f"replay_{label}"),
                )
                for label, start, end in (*YEARS, VALIDATION_YEAR)
            ),
            validation=ReplaySpan(
                "validation", "valid", VALIDATION_YEAR[1], VALIDATION_YEAR[2], SnapshotBundle("snap_v", "decision_v", "replay_v")
            ),
            window_months=60,
            max_llm_calls=10,
            deadline_seconds=deadline_seconds,
            deadline_grace_seconds=60.0,
            record_failed_attempts=record_failed_attempts,
            # Every year of the fixture is a research year: ``full`` is all of them.
            acceptance_rules={
                **REQUIRED_RULES,
                "research_years": len(YEARS),
                **(acceptance_rules or {"max_drawdown": 0.25}),
            },
            budget_used=budget_used or BudgetUsed(),
            steps_before=steps_before,
            resume=resume,
            benchmark_index=DEFAULT_BENCHMARK_INDEX,
        )
        self.tree = StepTree(root / "steps")
        self.evaluator = _Evaluator(
            # Where the worker roots them: a recorded result is read under
            # the experiment it belongs to (``ledger.recorded_path``).
            experiment_dir / "artifacts" / "results",
            fail_markers=fail_markers,
            contention_markers=contention_markers,
            strategy_fail_markers=strategy_fail_markers,
            memory_kill_markers=memory_kill_markers,
            rendezvous=rendezvous,
            resources=resources,
            alpha_scale=alpha_scale,
        )
        self.backtest = SessionValidations(
            request=request,
            output_dir=self.output,
            models_dir=self.models,
            # Where the worker roots it: the Steps an experiment records are
            # read back with the fingerprints its revision manifests hold.
            artifact_store=FilesystemArtifactStore(experiment_dir / "artifacts" / "strategy"),
            evaluator=self.evaluator,
            tree=self.tree,
            schedule=StrategySchedule(),
            broker_profile=BrokerProfile(),
            time_budget=InferenceTimeBudget(duration_seconds=deadline_seconds),
            ref_store=AgentRefStore(root / "experiment"),
            ledger=ExperimentLedger(root / "ledger.jsonl"),
            manifest=RunManifest.create(
                root / "run_manifest.json", {}, ref_store=AgentRefStore(root / "experiment")
            ),
            experiment_dir=experiment_dir,
        )
        self.workspace = SafeWorkspace(self.workspace_root)
        self.batch = ValidateTool(
            backtest=self.backtest,
            workspace=self.workspace,
            modification_check_factory=self._check,
            trace_emit=(
                (lambda event, payload: trace.append((event, payload)))
                if trace is not None
                else None
            ),
            max_concurrency=max_concurrency,
        )
        self.rollback = StepRollbackTool(
            self.tree,
            self.output,
            self.models,
            session_ref=self.backtest.ref_store.get_or_create("session", "research"),
        )
        self.finish = TerminalTool(self.tree)

    def _check(self, directory: Path) -> ModificationCheckTool:
        return ModificationCheckTool(
            directory,
            parent_dir=self.parent,
            models_dir=self.models,
            constraints=ModificationConstraints(),
            readonly_baseline=self.seeded_readonly,
        )

    def candidate(self, name: str, source: str) -> None:
        directory = self.workspace_root / "candidates" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "main.py").write_text(source, encoding="utf-8")

    def validate_one(self, name: str, source: str, *, span: str = "full") -> dict[str, object]:
        """One candidate as its own round; returns its row."""

        self.candidate(name, source)
        result = self.call(name, span=span)
        assert result.ok, result.error
        return result.value["candidates"][0]

    def call(
        self,
        *names: str,
        span: str | None = None,
        controls: tuple[str, ...] = (),
    ) -> object:
        arguments: dict[str, object] = {
            "candidates": [
                {
                    "name": name,
                    "hypothesis": f"{name} earns a positive active excess",
                    "path": f"candidates/{name}",
                    # Left out for a candidate, as a session may.
                    **({"control": True} if name in controls else {}),
                }
                for name in names
            ],
        }
        if span is not None:
            arguments["span"] = span
        return self.batch.invoke(arguments)
