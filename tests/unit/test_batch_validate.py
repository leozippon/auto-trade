"""``batch_validate``: one formal step over a pre-registered candidate set.

This tool fans a pre-registered set out under one shared parent and one span
of the research period, so the numbers are comparable; it is the only
Validation tool, so one candidate is a round too: same static gate, same
immutable revision, same replay-year budget, same selection path through
``step_rollback`` plus ``test`` at every width.

These tests cover the refusals that must happen before anything runs, the
spans a batch may replay on a multi-year research period, the per-candidate
isolation of a failure, the lineage the tree ends up with, and that a batch
node can actually be registered through the gate.
"""

from __future__ import annotations

import json
import math
import threading
import unittest
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

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
from autotrade.environment.replay.null_control import NullControlSetupError
from autotrade.environment.replay.stats import TRADING_DAYS_PER_YEAR
from autotrade.environment.replay.style import active_analysis
from autotrade.environment.runtime import RunManifest
from autotrade.environment.sandbox import SandboxConfig, SandboxSpec
from autotrade.environment.step_tree import StepTree
from autotrade.environment.time_budget import InferenceTimeBudget
from autotrade.environment.tools.base import ToolError, ToolRegistry
from autotrade.environment.tools.modification_check import ModificationCheckTool
from autotrade.environment.tools.step_rollback import StepRollbackTool
from autotrade.environment.tools.workspace import (
    LOW_DISK_SPACE,
    WORKSPACE_MIN_FREE_BYTES,
    SafeWorkspace,
)
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
from autotrade.pipelines.experiment import (
    null_control_seed,
    research_step_record,
    validation_statistics,
)
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.session_resume import load_recorded_steps
from autotrade.pipelines.session_tools import (
    BATCH_HYPOTHESIS_MAX_CHARS,
    BATCH_REJECTION_ESCALATE_AT,
    BATCH_VALIDATE_MAX_CANDIDATES,
    EnvironmentBlocked,
    NullControlTool,
    SessionValidations,
    ValidateTool,
    leg_cards,
    year_readings,
)
from autotrade.pipelines.verdict import (
    ACTIVE_DAILY_COLUMNS,
    _max_slice_drawdown,
    graded_statistics,
    residual_reading,
    residual_statistics,
)
from tests.unit.acceptance_fixture import REQUIRED_RULES
from tests.unit.terminal_tool import TerminalTool

PARENT_SOURCE ="def generate_orders(context):\n    return []\n"
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


# What a replay's summary says of its costs: the slippage the registration check's
# cost stress charges again on the replay's turnover.
COSTED = {"cost_sensitivity": {"slippage_bps": 5.0, "cost_per_bp_per_side": 1.2e-4}}


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

    def evaluate(self, request, max_days=None):
        del max_days
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
        self.finish = TerminalTool(
            self.tree,
            session_ref=self.backtest.ref_store.get_or_create("session", "research"),
            registration=self.backtest.registration,
        )

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


class BatchValidateRefusalTest(unittest.TestCase):
    """Everything that can refuse a batch runs before a slot is spent."""

    def test_byte_identical_candidates_are_refused_whole(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("1"))
            with self.assertRaises(ToolError) as caught:
                session.call("a", "b")
            self.assertIn("byte-identical", str(caught.exception))
            self.assertEqual(session.evaluator.calls, 0)
            self.assertEqual(session.tree.nodes(), [])

    def test_a_batch_is_refused_below_the_free_space_floor_however_often(self) -> None:
        """A host condition, not the Agent's input: refused before anything is
        reserved, however often it repeats, and lifted once space is back."""

        def free(nbytes: int):
            usage = SimpleNamespace(total=4 * WORKSPACE_MIN_FREE_BYTES, used=0, free=nbytes)
            return patch("autotrade.environment.tools.workspace.shutil.disk_usage", return_value=usage)

        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("low_disk"))
            with free(WORKSPACE_MIN_FREE_BYTES - 1):
                for _ in range(8):
                    with self.assertRaises(ToolError) as caught:
                        session.call("a", span="Y1")
                    self.assertEqual(caught.exception.error_type, LOW_DISK_SPACE)
                    self.assertIn("GiB free, below the 50 GiB floor", str(caught.exception))
                    self.assertIn("delete intermediate files", str(caught.exception))
            self.assertEqual(session.evaluator.calls, 0)
            self.assertEqual(session.tree.nodes(), [])
            with free(WORKSPACE_MIN_FREE_BYTES):
                result = session.call("a", span="Y1")
            self.assertTrue(result.ok, result.error)

    def test_a_candidate_with_the_start_logic_is_a_legal_candidate(self) -> None:
        """A session may re-validate its start: on another span, or on the full
        period to count it toward the registration check. Nothing refuses it."""
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("start", "# the start as it stands\n" + PARENT_SOURCE)
            result = session.call("start", span="Y4")
            self.assertTrue(result.ok, result.error)

    def test_a_candidate_failing_modification_check_refuses_the_batch(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("3"))
            # No main.py: the static gate every candidate runs.
            (session.workspace_root / "candidates" / "b").mkdir(parents=True)
            (session.workspace_root / "candidates" / "b" / "notes.py").write_text(
                "x = 1\n", encoding="utf-8"
            )
            with self.assertRaises(ToolError) as caught:
                session.call("a", "b")
            self.assertIn("failed modification_check", str(caught.exception))

    def test_the_working_copy_is_a_candidate_but_other_reserved_roots_are_not(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            (session.output / "main.py").write_text(_strategy("4"), encoding="utf-8")
            result = session.batch.invoke(
                {"candidates": [{"name": "live", "hypothesis": "h", "path": "output", "control": False}]}
            )
            self.assertTrue(result.ok, result.error)
            row = result.value["candidates"][0]
            node = session.tree.get_node(str(row["node_id"]))
            self.assertEqual(node["metadata"]["source_path"], "output")
            with self.assertRaises(ToolError) as caught:
                session.batch.invoke(
                    {"candidates": [{"name": "m", "hypothesis": "h", "path": "models", "control": False}]}
                )
            self.assertIn("reserved workspace root", str(caught.exception))

    def test_the_batch_size_bounds_are_enforced(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            for index in range(BATCH_VALIDATE_MAX_CANDIDATES + 1):
                session.candidate(f"c{index}", _strategy(str(index)))
            with self.assertRaises(ToolError):
                session.call()
            with self.assertRaises(ToolError):
                session.call(*[f"c{index}" for index in range(BATCH_VALIDATE_MAX_CANDIDATES + 1)])

    def test_a_hypothesis_over_the_cap_is_refused_and_the_schema_states_it(self) -> None:
        field = ValidateTool.spec.input_schema["properties"]["candidates"]["items"][
            "properties"
        ]["hypothesis"]
        # The cap has to hold a pre-registration with its falsification clause.
        self.assertEqual(BATCH_HYPOTHESIS_MAX_CHARS, 1_000)
        self.assertEqual(field["maxLength"], BATCH_HYPOTHESIS_MAX_CHARS)
        self.assertIn(f"{BATCH_HYPOTHESIS_MAX_CHARS} characters", field["description"])
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("5"))
            with self.assertRaises(ToolError) as caught:
                session.batch.invoke(
                    {
                        "candidates": [
                            {
                                "name": "one",
                                "hypothesis": "h" * (BATCH_HYPOTHESIS_MAX_CHARS + 1),
                                "path": "candidates/a",
                                "control": False,
                            }
                        ],
                    }
                )
            self.assertIn(
                f"hypothesis exceeds {BATCH_HYPOTHESIS_MAX_CHARS} characters",
                str(caught.exception),
            )
            self.assertEqual(caught.exception.error_type, "schema_error")
            self.assertEqual(session.evaluator.calls, 0)

    def test_a_duplicate_path_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("5"))
            with self.assertRaises(ToolError) as caught:
                session.batch.invoke(
                    {
                        "candidates": [
                            {"name": "one", "hypothesis": "h", "path": "candidates/a", "control": False},
                            {"name": "two", "hypothesis": "h", "path": "candidates/a", "control": False},
                        ],
                    }
                )
            self.assertIn("duplicate candidate path", str(caught.exception))


class BatchValidateSpanTest(unittest.TestCase):
    """A batch replays one span of a four-year research period as one book."""

    def test_a_contiguous_span_replays_its_years_from_its_first_decision_view(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("22"))
            result = session.call("a", "b", span="Y2..Y3")
            self.assertTrue(result.ok, result.error)
            value = result.value
            self.assertEqual(
                value["span"],
                {"label": "Y2..Y3", "start": "20220701", "end": "20240630", "years": 2, "days": 60},
            )
            # One row per research year of the span, named as spans name it.
            for row in value["candidates"]:
                self.assertEqual([year["label"] for year in row["by_year"]], ["Y2", "Y3"])
            for request in session.evaluator.requests:
                self.assertEqual((request.start, request.end), ("20220701", "20240630"))
                self.assertEqual(request.snapshot.decision_ref, "decision_Y2")
                self.assertEqual(request.snapshot.replay_ref, "replay_Y2")
                self.assertEqual(request.continuation, ("replay_Y3",))
            # Two candidates on two years cost four replay-years.
            for row, step in zip(value["candidates"], session.backtest.steps, strict=True):
                self.assertEqual(step.span, "Y2..Y3")
                node = session.tree.get_node(str(row["node_id"]))
                self.assertEqual(node["metadata"]["span"], "Y2..Y3")

    def test_the_default_and_the_whole_range_are_the_full_span(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.validate_one("default", _strategy("1"))
            session.validate_one("named", _strategy("2"), span="Y1..Y4")
            self.assertEqual([step.span for step in session.backtest.steps], ["full", "full"])
            request = session.evaluator.requests[-1]
            self.assertEqual((request.start, request.end), ("20210701", "20250630"))
            self.assertEqual(request.snapshot.decision_ref, "decision_Y1")
            self.assertEqual(request.continuation, ("replay_Y2", "replay_Y3", "replay_Y4"))

    def test_a_span_outside_the_research_years_is_refused_before_anything_runs(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("1"))
            for label in ("Y0", "Y6", "Y3..Y2", "Y2..Y6", "F", "heldout"):
                with self.assertRaises(ToolError) as caught:
                    session.call("a", span=label)
                self.assertIn("span must be full", str(caught.exception))
                self.assertEqual(caught.exception.error_type, "schema_error")
            # The validation year is a year of the geometry, but validate stays
            # inside the research years: a span reaching it is refused by name.
            for label in ("Y5", "Y4..Y5", "Y1..Y5"):
                with self.assertRaises(ToolError) as caught:
                    session.call("a", span=label)
                self.assertIn("past the end of the research years 20250630", str(caught.exception))
                self.assertEqual(caught.exception.error_type, "schema_error")
            self.assertEqual(session.evaluator.calls, 0)

    def test_the_null_control_ranks_a_node_over_its_own_span(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            calls: list[dict[str, object]] = []
            session.evaluator.null_control = _canned_null(calls)
            node = session.validate_one("y3", _strategy("7"), span="Y3")["node_id"]
            NullControlTool(session.backtest, max_calls=1).invoke({"node_id": node})
            self.assertEqual((calls[0]["start"], calls[0]["end"]), ("20230701", "20240630"))


class RegistrationTest(unittest.TestCase):
    """``test`` reads the Pipeline's own registration over the session's Steps
    (its refusal is pinned in test_experiment_validation); here, what registers."""

    def test_a_sub_span_node_does_not_register_and_a_lone_full_span_one_does(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            year = session.validate_one("year", _strategy("1"), span="Y2")["node_id"]
            registration = session.backtest.registration(year)
            self.assertFalse(registration["passed"])
            self.assertIn("registration_needs_full_span_validation", registration["reasons"])
            # One full-span validation is enough: no count of validations, no
            # score and no other node's reading stands between it and a test.
            full = session.validate_one("full", _strategy("22"))["node_id"]
            registration = session.backtest.registration(full)
            self.assertEqual((registration["passed"], registration["reasons"]), (True, []))

    def test_a_full_span_winner_among_two_is_tested_without_restoring_the_working_copy(
        self,
    ) -> None:
        """The Pipeline tests the nominated revision, not output/, so a winner
        is tested as it is: the working copy still holds the start and no
        step_rollback comes first."""
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.evaluator.summary_extra = COSTED
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("2" * 60))
            value = session.call("a", "b").value
            winner = value["candidates"][1]["node_id"]
            registration = session.backtest.registration(str(winner))
            self.assertTrue(registration["passed"], registration)
            tested = session.finish.invoke({"candidate": winner})
            self.assertEqual(tested.value["candidate"]["node_id"], winner)
            self.assertEqual(
                (session.output / "main.py").read_text(encoding="utf-8"),
                PARENT_SOURCE,
            )


class ControlsTest(unittest.TestCase):
    """A control is recorded and never registers for a test; a candidate
    that leaves control out is recorded as no control."""

    def test_a_control_is_recorded_and_can_never_be_nominated(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.evaluator.summary_extra = COSTED
            session.candidate("cand", _strategy("2" * 60))
            session.candidate("c_base", _strategy("1"))
            value = session.call("cand", "c_base", controls=("c_base",)).value
            # Charged like any candidate: two candidates x four years.
            cand, control = value["candidates"]
            self.assertEqual((cand["control"], control["control"]), (False, True))
            for row, step in zip(value["candidates"], session.backtest.steps, strict=True):
                node = session.tree.get_node(str(row["node_id"]))
                self.assertEqual(node["metadata"]["control"], row["control"])
                self.assertNotIn("offline_trials", node["metadata"])
                ledger_row = research_step_record(step)
                self.assertEqual(
                    (ledger_row["control"], ledger_row["batch_id"]),
                    (row["control"], value["batch_id"]),
                )
            registration = session.backtest.registration(str(control["node_id"]))
            self.assertFalse(registration["passed"])
            self.assertIn("registration_nominee_is_control", registration["reasons"])
            # Rows name a node by the short handle the tools accept.
            self.assertEqual(
                [cand["handle"], control["handle"]], session.backtest.named([cand["node_id"], control["node_id"]])
            )
            self.assertTrue(cand["handle"].startswith("valid_"))

    def test_control_is_optional_and_a_non_boolean_one_is_refused_before_anything_runs(
        self,
    ) -> None:
        schema = ValidateTool.spec.input_schema
        # Offline screens are no longer declared: nothing in the schema asks.
        self.assertNotIn("offline_trials", schema["properties"])
        self.assertEqual(schema["required"], ["candidates"])
        self.assertEqual(
            schema["properties"]["candidates"]["items"]["required"], ["name", "hypothesis", "path"]
        )
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("5"))
            candidate = {"name": "a", "hypothesis": "h", "path": "candidates/a"}
            with self.assertRaises(ToolError) as caught:
                session.batch.invoke({"candidates": [{**candidate, "control": "no"}]})
            self.assertEqual(caught.exception.error_type, "schema_error")
            self.assertEqual(caught.exception.blocked_target, "control")
            self.assertEqual(session.evaluator.calls, 0)


class BatchValidateRunTest(unittest.TestCase):
    def test_every_candidate_becomes_a_sibling_node_under_one_parent(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            # A batch branches off wherever the session stands, so start from a
            # real node rather than the empty root.
            branch_point = str(session.validate_one("start", _strategy("9"))["node_id"])
            self.assertTrue(session.rollback.invoke({"node_id": branch_point}).ok)
            self.assertEqual(session.tree.current_node_id, branch_point)
            for index, name in enumerate(("alpha", "beta", "gamma")):
                session.candidate(name, _strategy(str(index)))
            result = session.call("alpha", "beta", "gamma")
            self.assertTrue(result.ok)
            value = result.value
            self.assertEqual(value["parent_node_id"], branch_point)
            self.assertEqual(value["complete_validations"], 3)
            self.assertEqual(value["failed"], 0)
            node_ids = [row["node_id"] for row in value["candidates"]]
            self.assertEqual(len(set(node_ids)), 3)
            nodes = {node["node_id"]: node for node in session.tree.nodes()}
            for row, node_id in zip(value["candidates"], node_ids, strict=True):
                node = nodes[node_id]
                # One shared parent: the node the batch branched from, not the
                # sibling recorded just before.
                self.assertEqual(node["parent_node_id"], branch_point)
                self.assertTrue(node["complete_validation"])
                self.assertEqual(node["metadata"]["batch_id"], value["batch_id"])
                self.assertEqual(node["metadata"]["candidate"], row["name"])
                # The hypothesis is recorded with the node, not echoed back.
                self.assertTrue(node["metadata"]["hypothesis"])
                self.assertNotIn("hypothesis", row)
            # The batch never touched the working copy, so the tree position
            # comes back to where it branched from.
            self.assertEqual(session.tree.current_node_id, branch_point)
            self.assertEqual(len(session.backtest.steps), 4)

    def test_candidates_replay_concurrently_up_to_the_experiments_width(self) -> None:
        """The bound is host capacity, not correctness: each replay owns its
        result directory, as-of view and container, and the bookkeeping around
        them stays on the calling thread. The width is the experiment's own
        ``batch_validate_max_concurrency``, and both strategy deadlines scale
        with exactly the width it produced."""
        fit_base = SandboxConfig().fit_timeout_seconds
        for width in (DEFAULT_BATCH_VALIDATE_MAX_CONCURRENCY, 5):
            with self.subTest(width=width), TemporaryDirectory() as tmp:
                self._replay_full_width(Path(tmp), width, fit_base)
        # Width 1 is the serial path: one replay at a time, nothing scaled.
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), max_concurrency=1)
            names = [f"s{index}" for index in range(3)]
            for index, name in enumerate(names):
                session.candidate(name, _strategy(f"9{index}"))
            self.assertTrue(session.call(*names).ok)
            self.assertEqual(session.evaluator.peak, 1)
            self.assertEqual(session.evaluator.fit_timeouts, [fit_base] * 3)

    def _replay_full_width(self, tmp: Path, width: int, fit_base: float) -> None:
        session = _Session(tmp, rendezvous=width, max_concurrency=width)
        names = [f"c{index}" for index in range(BATCH_VALIDATE_MAX_CANDIDATES)]
        for index, name in enumerate(names):
            session.candidate(name, _strategy(str(index)))
        # The rendezvous only completes if that many replays are in flight
        # at once; a serialized implementation would time out and fail.
        result = session.call(*names)
        self.assertTrue(result.ok)
        self.assertEqual(
            result.value["complete_validations"], BATCH_VALIDATE_MAX_CANDIDATES
        )
        self.assertEqual(session.evaluator.peak, width)
        self.assertEqual(
            session.evaluator.fit_timeouts,
            [fit_base * width] * BATCH_VALIDATE_MAX_CANDIDATES,
        )
        # Recording order follows the input, not whichever replay landed
        # first, so the lineage is reproducible.
        self.assertEqual(
            [row["name"] for row in result.value["candidates"]], names
        )
        self.assertEqual(
            [row["handle"] for row in result.value["candidates"]],
            [f"valid_{index + 1:03d}" for index in range(len(names))],
        )

    def test_both_strategy_deadlines_scale_with_the_batch_replay_width(self) -> None:
        """The contention a batch adds is the environment's, not the strategy's.

        Both caps are a fixed per-call host wall clock, so with a fixed value a
        candidate's verdict depended on how many siblings happened to share the
        host: two explore_github folds lost both batch candidates to `strategy
        fit exceeded 3600s` while solo reruns of the same code did the same four
        refits in 1,550-2,027 s, and three experiments lost slots to `strategy
        inference exceeded 180s` inside a batch -- one with a byte-identical
        serial control that replayed 243 decision days at ~2.0 s/day. Both caps
        therefore move with the batch's own width, and only for the pool's
        lifetime.
        """

        config = SandboxConfig()
        fit_base = config.fit_timeout_seconds
        inference_base = config.timeout_seconds
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            # A one-candidate round: nothing else is running, so nothing is scaled.
            session.validate_one("s0", _strategy("50"))
            self.assertEqual(session.evaluator.fit_timeouts, [fit_base])
            self.assertEqual(session.evaluator.inference_timeouts, [inference_base])

            for count in (2, 3):
                names = [f"w{count}{index}" for index in range(count)]
                for index, name in enumerate(names):
                    session.candidate(name, _strategy(f"{count}{index}"))
                session.evaluator.fit_timeouts.clear()
                session.evaluator.inference_timeouts.clear()
                self.assertTrue(session.call(*names).ok)
                self.assertEqual(
                    session.evaluator.fit_timeouts, [fit_base * count] * count
                )
                self.assertEqual(
                    session.evaluator.inference_timeouts,
                    [inference_base * count] * count,
                )
                # Restored the moment the pool is done, so the next serial
                # replay is not evaluated against a widened clock.
                self.assertEqual(
                    session.evaluator.sandbox.fit_timeout_seconds, fit_base
                )
                self.assertEqual(
                    session.evaluator.sandbox.timeout_seconds, inference_base
                )
            # Everything else about the boundary is untouched.
            self.assertEqual(session.evaluator.sandbox, SandboxConfig())

    def test_one_failing_candidate_does_not_hide_the_others(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), strategy_fail_markers=("999",))
            session.candidate("good", _strategy("1"))
            session.candidate("bad", _strategy("999"))
            session.candidate("also_good", _strategy("2"))
            result = session.call("good", "bad", "also_good")
            self.assertTrue(result.ok)
            rows = {row["name"]: row for row in result.value["candidates"]}
            self.assertEqual(rows["good"]["status"], "ok")
            self.assertEqual(rows["also_good"]["status"], "ok")
            self.assertEqual(rows["bad"]["status"], "failed")
            self.assertIn("generate_orders raised", rows["bad"]["error"])
            # No node, no result_ref and no Step: exactly what the tool
            # description promises a failed candidate leaves behind.
            self.assertNotIn("node_id", rows["bad"])
            self.assertNotIn("result_ref", rows["bad"])
            self.assertEqual(result.value["complete_validations"], 2)
            self.assertEqual(result.value["failed"], 1)
            # A failed attempt is recorded as a dead end and never becomes a
            # parent, and the strategy's own exception still cost its
            # replay-years.
            failed = [
                node
                for node in session.tree.nodes()
                if node.get("status") == "failed"
            ]
            self.assertEqual(len(failed), 1)
            # The dead end hangs off the batch's own parent, not off whichever
            # sibling happened to be recorded before it.
            self.assertIsNone(failed[0]["parent_node_id"])
            self.assertNotIn("parent_node_id", result.value)
            # The dead end carries the same batch id and hypothesis as its
            # recorded siblings: the round is readable from the tree alone.
            self.assertEqual(failed[0]["metadata"]["candidate"], "bad")
            self.assertEqual(failed[0]["metadata"]["batch_id"], result.value["batch_id"])
            self.assertEqual(failed[0]["metadata"]["span"], "full")
            self.assertEqual(len(session.backtest.steps), 2)

    def test_every_row_says_what_the_replay_cost_its_container(self) -> None:
        """Completed and failed rows alike carry the container telemetry, in
        the same place: the row's own top level, because a failed row has no
        readings at all. The limits it ran under are the budgets facts and
        are not repeated per row.

        Two experiments sized a batch from a 3-day smoke and lost 20 replay-years to
        fits that ran past the concurrency-scaled cap, reasoning about a
        ceiling neither of them had measured.

        The fit and generate_orders seconds ride as summaries: a daily refit
        over a full span is a list of ~1,941 readings, re-sent with every later
        model call. The completed row's result_ref keeps the lists.
        """

        fits = [335.7, 4.1, 3.9, 4.4, 1204.2]
        decisions = [22.4, 0.4, 0.3, 21.9, 0.4, 23.1]
        usage = {
            "peak_memory_bytes": 7_883_149_312,
            "memory_limit_bytes": 17_179_869_184,
            "fit_seconds": fits,
            "decision_seconds": decisions,
            "fit_timeout_seconds": 10_800.0,
            "decision_timeout_seconds": 1_080.0,
        }
        summarised = {
            "peak_memory_bytes": 7_883_149_312,
            "fit_seconds": {"count": 5, "p50": 4.4, "p95": 1204.2, "max": 1204.2},
            "decision_seconds": {"count": 6, "p50": 0.4, "p95": 23.1, "max": 23.1},
        }
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), fail_markers=("999",), resources=usage)
            session.candidate("good", _strategy("1"))
            session.candidate("slow", _strategy("999"))
            result = session.call("good", "slow", span="Y1")
            rows = {row["name"]: row for row in result.value["candidates"]}
            self.assertEqual(rows["good"]["resources"], summarised)
            record = json.loads(
                (session.tree.root / rows["good"]["result_ref"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(record["stats"]["resources"]["fit_seconds"], fits)
            self.assertEqual(record["stats"]["resources"]["decision_seconds"], decisions)
            self.assertEqual(rows["slow"]["status"], "failed")
            self.assertEqual(
                list(rows["slow"]), ["name", "control", "status", "cause", "error", "wall_seconds", "resources"]
            )
            self.assertEqual(rows["slow"]["resources"], summarised)

    def test_a_row_without_telemetry_carries_no_empty_resources_block(self) -> None:
        """A replay that ran no container reports nothing, not zeros."""

        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), strategy_fail_markers=("999",))
            session.candidate("good", _strategy("1"))
            session.candidate("bad", _strategy("999"))
            rows = {
                row["name"]: row
                for row in session.call("good", "bad", span="Y1").value["candidates"]
            }
            self.assertNotIn("resources", rows["good"])
            self.assertNotIn("resources", rows["bad"])

    def test_a_wholly_failed_batch_reports_the_failure_not_a_success(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), strategy_fail_markers=("1", "2"))
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("2"))
            with self.assertRaises(ToolError) as caught:
                session.call("a", "b")
            error = caught.exception
            self.assertIn("all 2 candidates failed", str(error))
            rows = error.details["candidates"]
            self.assertEqual([row["status"] for row in rows], ["failed", "failed"])
            # Honest accounting: both strategies raised, so the replays measured
            # them and the replay-years are gone. The counters ride back on the
            # refusal, which is the only place this batch reports them.
            self.assertEqual(session.backtest.steps, [])

    def test_a_leg_the_shared_card_failed_names_the_siblings_beside_it(self) -> None:
        """Legs of a batch that are not pinned to a card of their own (here the
        experiment pins none) attach the same cards, so siblings that replayed while a
        leg ran out of GPU memory held memory on the same device. The leg's
        error names them, wherever it is read, so the Agent
        can tell its own batch may have overcommitted the card. The CUDA
        report cannot say whose memory filled it, so the failure stays the
        environment's and is refunded. A failure of the strategy's own code
        beside them, and a leg that ran alone, name nobody."""

        note = "of this batch replayed on the same GPU while this candidate ran"
        with TemporaryDirectory() as tmp:
            session = _Session(
                Path(tmp),
                contention_markers=("999",),
                strategy_fail_markers=("777",),
                rendezvous=3,
                max_concurrency=3,
            )
            session.candidate("gpu", _strategy("999"))
            session.candidate("own", _strategy("777"))
            session.candidate("ok", _strategy("1"))
            # The rendezvous holds all three replays in flight at once.
            result = session.call("gpu", "own", "ok", span="Y2")
            gpu, own, ok = result.value["candidates"]
            self.assertEqual((gpu["cause"], own["cause"]), ("environment", "strategy"))
            self.assertIn(f"own, ok {note}", gpu["error"])
            self.assertNotIn(note, own["error"])
            self.assertEqual(ok["status"], "ok")
            # The contended leg is refunded; the strategy's own failure is not.
            [dead_end] = [
                node
                for node in session.tree.nodes()
                if node["result_name"] == "valid_001"
            ]
            self.assertEqual(dead_end["error"], gpu["error"])
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), contention_markers=("999",), max_concurrency=1)
            session.candidate("gpu", _strategy("999"))
            session.candidate("ok", _strategy("1"))
            gpu, _ok = session.call("gpu", "ok", span="Y2").value["candidates"]
            self.assertEqual(gpu["cause"], "environment")
            self.assertIn("other processes on the card held the rest", gpu["error"])
            self.assertNotIn(note, gpu["error"])

    def test_concurrent_legs_each_take_one_card_of_the_experiment(self) -> None:
        """Legs replaying side by side are pinned to the experiment's cards in turn, so
        with as many cards as legs no two share one (to-do 32's pile-on onto the
        first card); a replay running alone attaches every card, so one strategy
        can still spread a model across them. A leg the shared card failed names
        only the siblings that ran on its own card."""

        note = "of this batch replayed on the same GPU while this candidate ran"
        cards = SandboxConfig(spec=SandboxSpec(gpu=(1, 4), gpu_count=2))

        def card_of(session: _Session) -> dict[str, tuple[int, ...] | None]:
            return {
                (Path(request.revision.output_path) / "main.py").read_text(
                    encoding="utf-8"
                ): request.gpu_devices
                for request in session.evaluator.requests
            }

        with TemporaryDirectory() as tmp:
            session = _Session(
                Path(tmp), contention_markers=("999",), rendezvous=2, max_concurrency=2
            )
            session.evaluator.sandbox = cards
            session.candidate("gpu", _strategy("999"))
            session.candidate("ok", _strategy("2"))
            gpu, ok = session.call("gpu", "ok", span="Y2").value["candidates"]
            self.assertEqual(sorted(card_of(session).values()), [(1,), (4,)])
            self.assertEqual((gpu["cause"], ok["status"]), ("environment", "ok"))
            self.assertNotIn(note, gpu["error"])
            session.evaluator.requests.clear()
            session.candidate("alone", _strategy("3"))
            self.assertTrue(session.call("alone", span="Y2").ok)
            self.assertEqual(list(card_of(session).values()), [None])
        with TemporaryDirectory() as tmp:
            session = _Session(
                Path(tmp), contention_markers=("999",), rendezvous=3, max_concurrency=3
            )
            session.evaluator.sandbox = cards
            session.candidate("gpu", _strategy("999"))
            session.candidate("x", _strategy("21"))
            session.candidate("y", _strategy("22"))
            rows = session.call("gpu", "x", "y", span="Y2").value["candidates"]
            placed = card_of(session)
            self.assertEqual(sorted(placed.values()), [(1,), (1,), (4,)])
            beside = [
                row["name"]
                for row, marker in zip(rows[1:], ("21", "22"), strict=True)
                if placed[_strategy(marker)] == placed[_strategy("999")]
            ]
            if beside:
                self.assertIn(f"{beside[0]} {note}", rows[0]["error"])
            else:
                self.assertNotIn(note, rows[0]["error"])

    def test_leg_cards_turn_through_the_experiments_cards(self) -> None:
        self.assertEqual(leg_cards((1, 4), 1), [])
        self.assertEqual(leg_cards((1, 4), 2), [1, 4])
        self.assertEqual(leg_cards((1, 4), 3), [1, 4, 1])
        self.assertEqual(leg_cards((), 3), [])
        config = SandboxConfig(spec=SandboxSpec(gpu=(1, 4), gpu_count=2))
        self.assertIs(config.on_devices(None), config)
        leg = config.on_devices((4,)).spec
        self.assertEqual((leg.gpu_count, leg.gpu), (1, (4,)))

    def test_a_mixed_batch_records_each_candidates_cause(self) -> None:
        """One completed, one strategy exception, one environment failure."""

        with TemporaryDirectory() as tmp:
            session = _Session(
                Path(tmp),
                strategy_fail_markers=("777",),
                contention_markers=("999",),
            )
            session.candidate("good", _strategy("1"))
            session.candidate("raised", _strategy("777"))
            session.candidate("contended", _strategy("999"))
            result = session.call("good", "raised", "contended")
            self.assertTrue(result.ok)
            rows = {row["name"]: row for row in result.value["candidates"]}
            self.assertEqual(rows["good"]["status"], "ok")
            self.assertNotIn("cause", rows["good"])
            self.assertEqual(rows["raised"]["cause"], "strategy")
            self.assertEqual(rows["contended"]["cause"], "environment")
            # Reserved 12 for three full-span candidates, 4 refunded.
            # Both dead ends are still recorded, each carrying what it was.
            dead_ends = {
                node["metadata"]["candidate"]: node["metadata"]["cause"]
                for node in session.tree.nodes()
                if node.get("status") == "failed"
            }
            self.assertEqual(
                dead_ends, {"raised": "strategy", "contended": "environment"}
            )

    def test_environment_failures_are_recorded_when_recording_a_candidate_fails(
        self,
    ) -> None:
        """Host IO failing while one completed candidate is recorded aborts the
        batch, but the environment failures before and after it measured
        nothing either way: their replay-years still come back."""

        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), contention_markers=("998", "999"))
            session.candidate("contended", _strategy("998"))
            session.candidate("good", _strategy("1"))
            session.candidate("late_contended", _strategy("999"))

            def disk_full(*_args: object, **_kwargs: object) -> str:
                raise OSError("No space left on device")

            session.backtest.record_validation = disk_full  # type: ignore[method-assign]
            with self.assertRaises(OSError):
                session.call("contended", "good", "late_contended")
            # Reserved 12 for three full-span candidates; both contended ones
            # are refunded, the completed one stays charged.
            self.assertEqual(session.backtest.steps, [])

    def test_a_memory_cap_kill_is_the_strategys_own_failure(self) -> None:
        """The memory cap is a budget the session was told about, with swap
        pinned to it, so a candidate killed there measured itself: no refund."""

        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), memory_kill_markers=("888",))
            session.candidate("good", _strategy("1"))
            session.candidate("big", _strategy("888"))
            result = session.call("good", "big", span="Y1")
            self.assertTrue(result.ok)
            rows = {row["name"]: row for row in result.value["candidates"]}
            self.assertEqual(rows["big"]["cause"], "strategy")
            self.assertIn("32g memory cap", rows["big"]["error"])

    def test_a_completed_row_has_the_documented_keys_and_a_readable_result_ref(
        self,
    ) -> None:
        """A row is a projection: the documented blocks in order, nothing that
        echoes the call (the hypothesis, the path) and nothing the batch states
        once. The node's result.json keeps the whole record."""
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("22"))
            value = session.call("a", "b").value
            for row in value["candidates"]:
                self.assertEqual(
                    list(row),
                    [
                        "name", "control", "status", "handle", "node_id", "revision_id",
                        "wall_seconds", "active", "book", "by_year", "concentration", "result_ref",
                    ],
                )
                self.assertEqual(
                    list(row["active"]),
                    [
                        "excess", "information_ratio", "residual_information_ratio", "residual_excess",
                        "tracking_error", "max_drawdown", "market_beta", "size_beta", "value_beta",
                        "excess_at_cost_stress",
                    ],
                )
                self.assertEqual(
                    list(row["book"]),
                    [
                        "return", "annualized_return", "excess_vs_benchmark", "baseline_return",
                        "excess_vs_baseline", "max_drawdown", "sharpe", "turnover", "avg_gross",
                        "zero_position_days", "size_tilt", "top_industry_weight", "orders", "trades",
                        "win_rate", "top5_share_of_gains", "breakeven_extra_slippage_bps",
                    ],
                )
                self.assertEqual([year["label"] for year in row["by_year"]], ["Y1", "Y2", "Y3", "Y4"])
                self.assertEqual(
                    list(row["by_year"][0]),
                    ["label", "active_excess", "excess_vs_benchmark", "return", "benchmark_return", "sharpe"],
                )
                flat = json.dumps(row)
                for echoed in ("hypothesis", "earns a positive", "candidates/", "threshold", "ts_code", "note"):
                    self.assertNotIn(echoed, flat)
                record = json.loads((session.tree.root / row["result_ref"]).read_text(encoding="utf-8"))
                self.assertIn("sub_windows", record["stats"])
                self.assertIn("per_stock", record["stats"])
            # The batch's constants ride once beside the rows.
            self.assertEqual(
                value["benchmark"], {"ts_code": DEFAULT_BENCHMARK_INDEX, "label": "index", "return": 0.01}
            )
            self.assertEqual(value["span"]["days"], 60)

    def test_the_threshold_is_the_batchs_and_each_row_reads_the_statistic_on_its_span(
        self,
    ) -> None:
        """A row reads the sealed gate's statistic over its own span -- the
        residual IR and the three betas of the active series -- and the batch
        states the threshold for that span's length once; nothing passes or
        fails."""
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            for name, marker, span in (("full", "333", "full"), ("year", "4", "Y2")):
                session.candidate(name, _strategy(marker))
                value = session.call(name, span=span).value
                [row] = value["candidates"]
                sidecar = json.loads(
                    (Path(session.backtest.steps[-1].validation.result_ref).parent / "style_analysis.json")
                    .read_text(encoding="utf-8")
                )
                expected = residual_reading(sidecar, alpha=REQUIRED_RULES["alpha"])
                active = row["active"]
                for key in ("residual_information_ratio", "residual_excess", "market_beta", "size_beta", "value_beta"):
                    self.assertAlmostEqual(active[key], expected[key], places=6, msg=key)
                self.assertAlmostEqual(value["threshold"], expected["threshold"], places=6)
                graded = graded_statistics(sidecar)
                self.assertAlmostEqual(active["excess"], graded["plain_excess"], places=6)
                self.assertAlmostEqual(active["information_ratio"], graded["information_ratio"], places=6)
                flat = json.dumps(value)
                for word in ("passed", "registration", "trials", "selection_statistics", "deflated", "verdict"):
                    self.assertNotIn(word, flat)
                # A one-year span has no other year to concentrate against.
                self.assertEqual("concentration" in row, span == "full")

    def test_every_row_reads_the_holders_money_and_the_forward_readings_cost_stress(
        self,
    ) -> None:
        """Two sessions quoted 0.148 and 1.608 as one node's "cost stress": the
        active reading and the raw cumulative excess. Every row carries the one
        the forward reading reports, computed by the host: the active excess at
        the experiment's multiplier. The book beside it reads the holder's
        money as a forward slice does."""
        with TemporaryDirectory() as tmp:
            session = _Session(
                Path(tmp), acceptance_rules={"max_drawdown": 0.25, "cost_stress_multiplier": 3.0}
            )
            # The fixture's replay turns its opening capital over 1.2 times.
            session.evaluator.summary_extra = COSTED
            row = session.validate_one("a", _strategy("1"))
            sidecar = json.loads(
                (
                    Path(session.backtest.steps[0].validation.result_ref).parent
                    / "style_analysis.json"
                ).read_text(encoding="utf-8")
            )

            def compound(key: str) -> float:
                return math.prod(1.0 + value for _day, value in sidecar[key]) - 1.0

            book, benchmark, baselines = (
                compound(key) for key in ("strategy_daily", "benchmark_daily", "baseline_daily")
            )
            for name, expected in (
                ("excess_vs_benchmark", book - benchmark),
                ("baseline_return", baselines),
                ("excess_vs_baseline", book - baselines),
            ):
                self.assertAlmostEqual(row["book"][name], expected, places=6, msg=name)
            self.assertEqual(row["book"]["return"], round(0.01 * len(_strategy("1")), 6))
            statistics = graded_statistics(sidecar)
            years = statistics["days"] / TRADING_DAYS_PER_YEAR
            self.assertAlmostEqual(
                row["active"]["excess_at_cost_stress"],
                statistics["plain_excess"] - 2.0 * 5.0 * 1.2 * 1e-4 / years,
                places=6,
            )
            self.assertNotIn("cost_stress_multiplier", json.dumps(row))

    def test_the_cost_stress_is_read_on_every_row_and_never_refuses_a_registration(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), acceptance_rules={"max_drawdown": 0.25})
            # The same edge, traded a hundred thousand times as much: the
            # stressed slippage on that turnover erases it.
            session.evaluator.summary_extra = {**COSTED, "turnover": 1.2e5}
            below = session.validate_one("below", _strategy("1"))
            session.evaluator.summary_extra = COSTED
            above = session.validate_one("above", _strategy("2"))
            self.assertLess(below["active"]["excess_at_cost_stress"], 0)
            self.assertGreater(above["active"]["excess_at_cost_stress"], 0)
            # A reading, not a condition: the node registers and is registered.
            finished = session.finish.invoke({"node_id": below["node_id"]})
            self.assertTrue(finished.finish)

    def test_concentration_reads_the_best_year_against_the_rest_of_the_span(self) -> None:
        """On a known active series -- a constant daily active return per
        year, 0.1 %, 0.4 % and -0.05 % -- each year's active excess, the best
        year's share of the positive ones and the active excess of the days
        outside it are exact; a one-year span has nothing to concentrate on."""

        years = [
            ReplaySpan(f"Y{index}", "valid", f"20{20 + index}0701", f"20{21 + index}0630", SnapshotBundle("s", "d", "r"))
            for index in (1, 2, 3)
        ]
        per_day = (0.001, 0.004, -0.0005)
        days = [[f"20{20 + index}09{day:02d}" for day in range(1, 11)] for index in (1, 2, 3)]
        analysis = {
            "strategy_daily": [[day, 0.0002 + active] for block, active in zip(days, per_day) for day in block],
            "baseline_daily": [[day, 0.0002] for block in days for day in block],
        }
        summary = {
            "sub_windows": [
                {"start": block[0], "end": block[-1], "active_excess_return": active * TRADING_DAYS_PER_YEAR}
                for block, active in zip(days, per_day)
            ]
        }
        readings = year_readings(analysis, summary, years)
        self.assertEqual([row["label"] for row in readings["by_year"]], ["Y1", "Y2", "Y3"])
        concentration = readings["concentration"]
        self.assertEqual(concentration["best_year"], "Y2")
        self.assertAlmostEqual(concentration["best_year_share"], 0.004 / (0.001 + 0.004), places=6)
        self.assertAlmostEqual(
            concentration["active_excess_ex_best_year"], (0.001 - 0.0005) / 2 * TRADING_DAYS_PER_YEAR, places=6
        )
        alone = year_readings(analysis, {"sub_windows": summary["sub_windows"][:1]}, years)
        self.assertEqual([row["label"] for row in alone["by_year"]], ["Y1"])
        self.assertNotIn("concentration", alone)
        with self.assertRaisesRegex(ValueError, "lies in no research year"):
            year_readings(analysis, {"sub_windows": [{"start": "20300901", "end": "20300910"}]}, years)

    def test_a_row_that_repeats_earlier_bytes_on_the_same_span_names_them(self) -> None:
        """A session spent 8 replay-years re-running the bytes of a node it
        already had on the same span. The replay still runs -- packs re-run a
        same-seed baseline in every batch, and GPU training is not always
        bit-reproducible -- but the row says which node it repeats."""
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            first = session.validate_one("a", _strategy("1"))
            self.assertNotIn("same_bytes_as", first)
            # The same bytes on another span are a probe, not a repeat.
            probe = session.validate_one("a_probe", _strategy("1"), span="Y1")
            self.assertNotIn("same_bytes_as", probe)
            again = session.validate_one("a_again", _strategy("1"))
            self.assertEqual(again["same_bytes_as"], [first["handle"]])
            self.assertEqual(again["status"], "ok")

    def test_nothing_selects_a_winner_on_the_agent_s_behalf(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("22"))
            value = session.call("a", "b").value
            # The rows are the whole answer: no row is ranked, named or
            # picked, and nothing beside them restates the protocol.
            self.assertEqual(
                list(value),
                [
                    "batch_id",
                    "validates_in_step",
                    "validates_per_step",
                    "complete_validations",
                    "span",
                    "threshold",
                    "benchmark",
                    "candidates",
                    "failed",
                ],
            )
            # The working copy is untouched by the batch.
            self.assertEqual(
                (session.output / "main.py").read_text(encoding="utf-8"),
                PARENT_SOURCE,
            )


def _annualized_ir(active: pd.Series) -> float:
    """The IR a session reads off ``active_return``: its annualised mean over
    its annualised standard deviation."""

    return float(active.mean() / active.std(ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR)


class EnvironmentBlockedTest(unittest.TestCase):
    """A span the host fails in every candidate, twice, ends the session.

    The refund makes resending a batch the environment failed free, which is
    right once and wrong forever: in round 20261015 two experiments resent batches
    on spans the host could never replay until their sessions ran out.
    """

    def test_the_second_batch_failed_whole_on_one_span_ends_the_session(self) -> None:
        trace: list[tuple[str, dict[str, object]]] = []
        with TemporaryDirectory() as tmp:
            # Four batches in one interval: more than the default allowance of 3,
            # which is not what this test is about.
            session = _Session(
                Path(tmp), fail_markers=("999",), trace=trace, acceptance_rules={"validates_per_step": 6}
            )
            session.candidate("blocked", _strategy("999"))
            session.candidate("good", _strategy("1"))
            # The first is refunded and reported like any failed batch.
            with self.assertRaises(ToolError):
                session.call("blocked")
            # Neither another span nor a batch in which a candidate completed
            # on this span is a second one: that batch proved the span runs.
            with self.assertRaises(ToolError):
                session.call("blocked", span="Y1")
            self.assertTrue(session.call("good", "blocked").ok)
            with self.assertRaises(EnvironmentBlocked) as caught:
                session.call("blocked")
            reason = caught.exception.reason
            self.assertTrue(
                reason.startswith(
                    "span full: 2 batches failed on the environment in every candidate; "
                    "last error: daily Validation failed: "
                ),
                reason,
            )
            self.assertIn("generate_orders exceeded 30s (999)", reason)
            # The batch that ended it settled first: refunded, dead end kept.
            dead_ends = [node for node in session.tree.nodes() if node.get("status") == "failed"]
            self.assertEqual(
                [node["metadata"]["span"] for node in dead_ends], ["full", "Y1", "full", "full"]
            )
            self.assertEqual({node["metadata"]["cause"] for node in dead_ends}, {"environment"})
            self.assertEqual(
                [payload for event, payload in trace if event == "environment_blocked"],
                [{"tool": "validate", "span": "full", "batches": 2, "reason": reason}],
            )

    def test_a_batch_with_a_candidate_that_raised_in_its_own_code_never_counts(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), fail_markers=("999",), strategy_fail_markers=("777",))
            session.candidate("slow", _strategy("999"))
            session.candidate("raised", _strategy("777"))
            # The strategy's own exception means the host ran the span up to
            # the strategy's code: the span is not blocked, however often.
            for _ in range(3):
                with self.assertRaises(ToolError):
                    session.call("slow", "raised")


class DailySeriesTest(unittest.TestCase):
    """Every completed Validation leaves the series the gate grades, day by
    day, where the session can read it; a failed one leaves nothing."""

    def test_every_completed_row_reads_back_the_graded_series(self) -> None:
        """Candidates and controls, full-span and sub-span rows alike; the file
        reproduces the IR and the active drawdown the gate measures."""

        with TemporaryDirectory() as tmp:
            # A small edge, so the active series does draw down.
            session = _Session(Path(tmp), alpha_scale=0.00001)
            session.candidate("edge", _strategy("'edge'"))
            session.candidate("base", _strategy("'base'"))
            full = session.call("edge", "base", controls=("base",)).value
            session.candidate("probe", _strategy("'probe'"))
            sub = session.call("probe", span="Y2").value
            steps = {step.step_id: step for step in session.backtest.steps}
            rows = [*full["candidates"], *sub["candidates"]]
            self.assertEqual(len(rows), 3)
            for row in rows:
                # Beside the record result_ref names, as the description says.
                self.assertEqual(row["result_ref"], f"{row['node_id']}/validation/result.json")
                path = session.tree.root / Path(row["result_ref"]).parent / "active_daily.csv"
                frame = pd.read_csv(path, dtype={"trade_date": str})
                self.assertEqual(tuple(frame.columns), ACTIVE_DAILY_COLUMNS)
                result_ref = steps[row["node_id"]].validation.result_ref
                sidecar = json.loads(
                    (Path(result_ref).parent / "style_analysis.json").read_text(
                        encoding="utf-8"
                    )
                )
                # One row per replayed day, the random baseline as its composite only.
                self.assertEqual(
                    list(frame["trade_date"]),
                    [day for day, _value in sidecar["strategy_daily"]],
                )
                strategy = np.asarray([value for _day, value in sidecar["strategy_daily"]])
                baselines = np.asarray([value for _day, value in sidecar["baseline_daily"]])
                np.testing.assert_allclose(frame["strategy_return"], strategy, rtol=1e-9)
                np.testing.assert_allclose(frame["baseline_return"], baselines, rtol=1e-9)
                np.testing.assert_allclose(
                    frame["active_return"], strategy - baselines, rtol=1e-8, atol=1e-12
                )
                # The IR every row's ledger record carries.
                self.assertAlmostEqual(
                    _annualized_ir(frame["active_return"]),
                    validation_statistics(result_ref)["information_ratio"],
                    delta=1e-6,
                )
            # The full-span candidate: the file reproduces the active
            # drawdown the verdict holds to its limit, and the residual
            # reading off the series and the three legs beside it.
            [edge] = [row for row in full["candidates"] if row["name"] == "edge"]
            frame = pd.read_csv(session.tree.root / Path(edge["result_ref"]).parent / "active_daily.csv")
            sidecar = json.loads(
                (Path(steps[edge["node_id"]].validation.result_ref).parent / "style_analysis.json")
                .read_text(encoding="utf-8")
            )
            drawdown = _max_slice_drawdown(active_analysis(sidecar), "", "")
            self.assertGreater(drawdown, 0.0)
            self.assertAlmostEqual(frame["active_drawdown"].max(), drawdown, delta=1e-9)
            legs = frame[["benchmark_return", "size_factor_return", "value_factor_return"]].dropna()
            kept = frame.loc[legs.index, "active_return"]
            reading = residual_statistics(
                kept, legs["benchmark_return"], legs["size_factor_return"], legs["value_factor_return"],
                alpha=REQUIRED_RULES["alpha"],
            )
            self.assertAlmostEqual(
                reading["residual_information_ratio"],
                edge["active"]["residual_information_ratio"],
                places=5,
            )
            equity = 1.0 + frame["active_cumulative"]
            np.testing.assert_allclose(
                frame["active_drawdown"],
                1.0 - equity / np.maximum.accumulate(np.maximum(equity, 1.0)),
                atol=1e-9,
            )
            # The tree lists it beside the record, so an older node names it too.
            node = session.tree.get_node(edge["node_id"])
            self.assertIn("validation/active_daily.csv", node["attachments"])

    def test_a_failed_validation_leaves_no_series(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), strategy_fail_markers=("999",))
            session.candidate("good", _strategy("1"))
            session.candidate("bad", _strategy("999"))
            rows = {
                row["name"]: row
                for row in session.call("good", "bad", span="Y1").value["candidates"]
            }
            self.assertNotIn("result_ref", rows["bad"])
            # The good row's copy in its node, the experiment's published
            # copy of that node and the host's beside its result; nothing for
            # the failed one, anywhere.
            [step] = session.backtest.steps
            node = Path(rows["good"]["result_ref"]).parent / "active_daily.csv"
            self.assertEqual(
                sorted(Path(tmp).rglob("*active_daily*")),
                sorted(
                    [
                        session.tree.root / node,
                        session.backtest.experiment_dir / "steps" / node,
                        Path(step.validation.result_ref).parent / "active_daily.csv",
                    ]
                ),
            )

    def test_a_series_that_cannot_be_written_leaves_no_partial_file_and_no_node(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("1"))

            def disk_full(frame: pd.DataFrame, path: Path, **_kwargs: object) -> None:
                Path(path).write_text("trade_date,strategy_return\n2021", encoding="utf-8")
                raise OSError("No space left on device")

            with (
                patch.object(pd.DataFrame, "to_csv", disk_full),
                self.assertRaises(OSError),
            ):
                session.call("a", span="Y1")
            self.assertEqual(sorted(Path(tmp).rglob("*active_daily*")), [])
            self.assertEqual(session.tree.nodes(), [])
            self.assertEqual(session.backtest.steps, [])


class BudgetSpentNoteTest(unittest.TestCase):
    """A batch result says one thing beyond its rows, and only when it is
    true: the batch that spends the replay-year budget says no further batch
    can run."""


class BatchValidateContractTest(unittest.TestCase):
    def test_the_tool_is_sequential_and_in_the_session_set(self) -> None:
        from autotrade.agent.runner import (
            _SESSION_TOOLS,
            _VALIDATION_TOOLS,
        )
        from autotrade.environment.tools.base import is_sequential_tool

        name = ValidateTool.spec.name
        self.assertTrue(ValidateTool.spec.mutating)
        self.assertTrue(is_sequential_tool(ValidateTool.spec))
        self.assertIn(name, _VALIDATION_TOOLS)
        self.assertIn(name, _SESSION_TOOLS)

    def test_the_description_states_the_selection_path(self) -> None:
        description = ValidateTool.spec.description
        self.assertIn("the only way to make a node that test can take", description)
        self.assertIn("acceptance_rules.significance", description)
        properties = ValidateTool.spec.input_schema["properties"]
        self.assertIn("span", properties)
        self.assertIn("hypothesis", properties["candidates"]["items"]["properties"])
        for retired in ("Fold", "parent strategy", "backtest budget", "Step budget", "residual_statistic"):
            self.assertNotIn(retired, description)

    def test_the_replay_tool_descriptions_stay_within_their_size_targets(self) -> None:
        """Every description rides in every model call: each says its tool's
        rules and the fields a name cannot carry, within a fixed budget, so
        the prose does not creep back into what each call re-sends."""

        from autotrade.environment.tools.modification_check import ModificationCheckTool
        from autotrade.environment.tools.step_rollback import StepRollbackTool
        from autotrade.pipelines.session_tools import SmokeBacktestTool, TestTool

        limits = (
            (ValidateTool.spec, 2_250),
            (TestTool.spec, 1_500),
            (SmokeBacktestTool.spec, 900),
            (NullControlTool(SimpleNamespace(), max_calls=12).spec, 700),  # type: ignore[arg-type]
            (ModificationCheckTool.spec, 212),
            (StepRollbackTool.spec, 77),
        )
        for spec, limit in limits:
            self.assertLessEqual(len(spec.description), limit, spec.name)

    def test_a_schema_error_returns_the_example_call(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            registry = ToolRegistry([session.batch])
            result = registry.invoke("validate", {"candidate": []})
            self.assertFalse(result.ok)
            self.assertIn("unknown argument", result.error)
            self.assertIn("correct call example", result.error)

    def test_the_runner_registers_every_batch_node_as_a_candidate(self) -> None:
        from autotrade.agent.runner import AgentSessionRunner

        record = {
            "ok": True,
            "value": {
                "candidates": [
                    {"node_id": "n1", "revision_id": "r1", "stats": {"sharpe": 1.0}},
                    {"node_id": "n2", "revision_id": "r2", "stats": {"sharpe": 2.0}},
                    {"status": "failed", "error": "boom"},
                ]
            },
        }
        runner = AgentSessionRunner.__new__(AgentSessionRunner)
        runner._complete_validation_nodes = set()
        runner._record_complete_validations(record)
        self.assertEqual(runner._complete_validation_nodes, {"n1", "n2"})


class BatchTemplateFilesTest(unittest.TestCase):
    """A candidate is laid out from its strategy modules; the read-only
    template README the Agent may not edit is supplied, not demanded."""

    def test_a_candidate_without_the_readme_is_accepted_and_carries_it(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), readonly_template=True)
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("22"))
            value = session.call("a", "b").value
            self.assertEqual(value["complete_validations"], 2)
            for row in value["candidates"]:
                snapshot = session.tree.node_output_dir(row["node_id"])
                self.assertEqual(
                    (snapshot / "README.md").read_text(encoding="utf-8"), TEMPLATE_README
                )
            # Only the template name was supplied; the package modules are
            # exactly what the Agent wrote.
            self.assertEqual(
                sorted(p.name for p in (session.workspace_root / "candidates" / "a").iterdir()),
                ["README.md", "main.py"],
            )

    def test_a_template_edited_after_seeding_does_not_refuse_the_batch(self) -> None:
        """The host template is a live file; a session is judged on its own copy.

        Editing ``configs/agent_output_template/README.md`` while sessions run
        used to refuse every candidate of every session seeded from the older
        template, because the read-only comparison read that file at check time.
        """

        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), readonly_template=True)
            (session.parent / "README.md").write_text(
                TEMPLATE_README + "\nA new contract section.\n", encoding="utf-8"
            )
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("22"))
            value = session.call("a", "b").value
            self.assertEqual(value["complete_validations"], 2)
            # Candidates are supplied the seeded copy, not the edited template.
            self.assertEqual(
                (session.workspace_root / "candidates" / "a" / "README.md").read_text(
                    encoding="utf-8"
                ),
                TEMPLATE_README,
            )

    def test_a_candidate_with_an_edited_readme_is_still_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), readonly_template=True)
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("22"))
            (session.workspace_root / "candidates" / "a" / "README.md").write_text(
                "rewritten contract\n", encoding="utf-8"
            )
            with self.assertRaises(ToolError) as caught:
                session.call("a", "b")
            self.assertIn("readonly files modified", str(caught.exception))


class RepeatedRejectionTest(unittest.TestCase):
    """A refused batch is free, so nothing used to bound repeating one.

    One audited session spent 3.89 h on 128 consecutive rejections carrying the
    same error. The counter keys on the signature — error type plus the target
    the rejection names — so the escalation cannot be dodged by a message whose
    text moves, and cannot be triggered by a genuinely different mistake.
    """

    def _readonly_loop(self, tmp: str, trace=None) -> _Session:
        """A session whose next batch always fails on the read-only baseline."""

        session = _Session(Path(tmp), readonly_template=True, trace=trace)
        session.candidate("a", _strategy("1"))
        session.candidate("b", _strategy("22"))
        (session.workspace_root / "candidates" / "a" / "README.md").write_text(
            "rewritten contract\n", encoding="utf-8"
        )
        return session

    def _refuse(self, session: _Session) -> str:
        with self.assertRaises(ToolError) as caught:
            session.call("a", "b")
        return str(caught.exception)

    def test_the_third_identical_rejection_escalates_and_names_the_recovery(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            trace: list[tuple[str, dict[str, object]]] = []
            session = self._readonly_loop(tmp, trace=trace)
            for attempt in range(1, BATCH_REJECTION_ESCALATE_AT):
                message = self._refuse(session)
                self.assertIn("readonly files modified", message)
                self.assertNotIn("repeated rejection", message)
                self.assertEqual(trace, [], attempt)
            escalated = self._refuse(session)
            # Says how often, that retrying is pointless, and what to do.
            self.assertIn("readonly files modified", escalated)
            self.assertIn(
                f"refused this exact rejection {BATCH_REJECTION_ESCALATE_AT} times",
                escalated,
            )
            self.assertIn("calling it again unchanged", escalated)
            self.assertIn("modification_check", escalated)
            self.assertIn("details.readonly_violations", escalated)
            # An escalation is not a charge: the budget is still untouched.
            self.assertEqual([event for event, _ in trace], ["batch_rejection_escalated"])
            payload = trace[0][1]
            self.assertEqual(payload["error_type"], "readonly_baseline")
            self.assertEqual(payload["blocked_target"], "candidates/a")
            self.assertEqual(payload["repeat_count"], BATCH_REJECTION_ESCALATE_AT)

    def test_a_different_signature_is_counted_on_its_own(self) -> None:
        with TemporaryDirectory() as tmp:
            session = self._readonly_loop(tmp)
            reserved = {
                "candidates": [
                    {"name": "a", "hypothesis": "h", "path": "candidates/a", "control": False},
                    {"name": "live", "hypothesis": "h", "path": "models", "control": False},
                ],
            }
            for _ in range(BATCH_REJECTION_ESCALATE_AT - 1):
                self._refuse(session)
                with self.assertRaises(ToolError) as caught:
                    session.batch.invoke(reserved)
                self.assertNotIn("repeated rejection", str(caught.exception))
            # The interleaved rejections neither reset nor advance each other.
            self.assertIn("repeated rejection", self._refuse(session))
            with self.assertRaises(ToolError) as caught:
                session.batch.invoke(reserved)
            escalated = str(caught.exception)
            self.assertIn("reserved workspace root", escalated)
            self.assertIn("repeated rejection", escalated)
            self.assertIn("change the input it names", escalated)


def _canned_null(calls: list[dict[str, object]]):
    """A backend null control that records how the tool asked for it."""

    def null_control(result_ref, *, start, end, profile, schedule, seed, step=None):
        calls.append(
            {"result_ref": result_ref, "start": start, "end": end, "seed": seed, "step": step}
        )
        return {
            "k": 500,
            "seed": seed,
            "matched": "circ_mv_decile",
            "observed_excess": 0.01,
            "excess_percentile": 0.62,
            "rejects_mean": 1.0,
        }

    return null_control


class NullControlToolTest(unittest.TestCase):
    """The session's null control is the registration's null control: the same
    backend call with the same seed, cached per node and capped per session."""

    def test_a_validated_node_is_ranked_once_and_the_block_is_reused(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            calls: list[dict[str, object]] = []
            session.evaluator.null_control = _canned_null(calls)
            node = session.validate_one("n", _strategy("7"))["node_id"]
            tool = NullControlTool(session.backtest, max_calls=2)
            first = tool.invoke({"node_id": node}).value
            self.assertEqual(
                list(first), ["handle", "null_control", "cached", "null_controls_used", "null_controls_remaining"]
            )
            self.assertFalse(first["cached"])
            self.assertEqual(first["null_control"]["excess_percentile"], 0.62)
            # The Agent-visible block is the same whitelist the run facts use.
            self.assertNotIn("seed", first["null_control"])
            self.assertEqual((first["null_controls_used"], first["null_controls_remaining"]), (1, 1))
            # Drawn exactly as the registration would draw it: whole period, frozen role.
            self.assertIsNone(calls[0]["step"])
            self.assertEqual(calls[0]["seed"], null_control_seed("research", "frozen"))
            self.assertEqual((calls[0]["start"], calls[0]["end"]), ("20210701", "20250630"))
            self.assertEqual(calls[0]["result_ref"], session.backtest.steps[0].validation.result_ref)
            again = tool.invoke({"node_id": node}).value
            self.assertTrue(again["cached"])
            self.assertEqual((len(calls), again["null_controls_used"]), (1, 1))
            # The block handed to the Pipeline is the backend's own, whole.
            self.assertEqual(tool.blocks[node]["seed"], calls[0]["seed"])

    def test_the_cap_the_node_check_and_a_missing_backend_refuse_clearly(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.evaluator.null_control = _canned_null([])
            first = session.validate_one("n1", _strategy("7"))["node_id"]
            second = session.validate_one("n2", _strategy("2"))["node_id"]
            tool = NullControlTool(session.backtest, max_calls=1)
            with self.assertRaises(ToolError) as refused:
                tool.invoke({"node_id": "not_a_node"})
            self.assertIn("neither a node_id nor the short handle", str(refused.exception))
            self.assertEqual(refused.exception.details["nodes"], ["valid_001", "valid_002"])
            # The short handle names the node the full id does.
            ranked = tool.invoke({"node_id": "valid_001"}).value
            self.assertEqual(ranked["handle"], "valid_001")
            with self.assertRaises(ToolError) as exhausted:
                tool.invoke({"node_id": second})
            self.assertEqual(exhausted.exception.error_type, "null_control_budget_exhausted")
            # A node already ranked still answers after the cap.
            self.assertTrue(tool.invoke({"node_id": first}).value["cached"])
            del session.evaluator.null_control
            with self.assertRaises(ToolError) as missing:
                NullControlTool(session.backtest, max_calls=1).invoke({"node_id": second})
            self.assertIn("not available", str(missing.exception))

    def test_a_failed_null_control_is_an_error_that_still_costs_a_call(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))

            def boom(result_ref, **kwargs):
                raise RuntimeError("no replacement name for 000001.SZ entered 20220104")

            session.evaluator.null_control = boom
            node = session.validate_one("n", _strategy("7"))["node_id"]
            tool = NullControlTool(session.backtest, max_calls=1)
            with self.assertRaises(ToolError) as failed:
                tool.invoke({"node_id": node})
            self.assertEqual(failed.exception.error_type, "null_control_failed")
            self.assertEqual(failed.exception.details["null_controls_remaining"], 0)
            # A failure is not a result the registration may reuse.
            self.assertNotIn(node, tool.blocks)

    def test_a_null_control_that_drew_nothing_costs_no_call(self) -> None:
        """A host that cannot set the null control up has measured nothing.

        The charge stands for minutes of replay. A call refused before its
        first draw spent none of them, so the budget comes back and the node
        can still be ranked once the host can resolve what it was validated
        against -- the same rule batch_validate applies to an environment
        failure.
        """

        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))

            def unresolved(result_ref, **kwargs):
                raise NullControlSetupError(
                    f"null control setup failed for {result_ref}: no replay slots"
                )

            session.evaluator.null_control = unresolved
            node = session.validate_one("n", _strategy("7"))["node_id"]
            tool = NullControlTool(session.backtest, max_calls=1)
            with self.assertRaises(ToolError) as refused:
                tool.invoke({"node_id": node})
            self.assertEqual(refused.exception.error_type, "null_control_failed")
            self.assertEqual(
                refused.exception.details,
                {"null_controls_used": 0, "null_controls_remaining": 1},
            )
            self.assertEqual(tool.used, 0)
            self.assertNotIn(node, tool.blocks)
            # And the cap it did not spend is still there to be spent.
            calls: list[dict[str, object]] = []
            session.evaluator.null_control = _canned_null(calls)
            ranked = tool.invoke({"node_id": node}).value
            self.assertEqual(
                (ranked["null_controls_used"], ranked["null_controls_remaining"]),
                (1, 0),
            )
            self.assertEqual(len(calls), 1)

    def test_the_tool_is_sequential_and_behind_the_writer_barrier(self) -> None:
        from autotrade.agent.runner import (
            _SESSION_TOOLS,
            _VALIDATION_TOOLS,
            _WRITER_BARRIER_TOOLS,
        )
        from autotrade.environment.tools.base import is_sequential_tool

        name = NullControlTool.spec.name
        self.assertTrue(is_sequential_tool(NullControlTool.spec))
        self.assertIn(name, _SESSION_TOOLS)
        self.assertIn(name, _WRITER_BARRIER_TOOLS)
        # It is not a Validation: its result never registers a candidate node.
        self.assertNotIn(name, _VALIDATION_TOOLS)


class RecordedValidationDurabilityTest(unittest.TestCase):
    """Every recorded Validation is made durable at once: the experiment's
    tree carries the node, a host-only sidecar carries what the Pipeline needs
    to register it, so an attempt that dies keeps them for its resumed attempt."""

    def test_a_recorded_validation_is_published_with_its_host_sidecar(self) -> None:

        with TemporaryDirectory() as tmp:
            experiment = Path(tmp) / "experiment"
            session = _Session(Path(tmp), experiment_dir=experiment)
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("22"))
            result = session.call("a", "b")
            nodes = [row["node_id"] for row in result.value["candidates"]]
            published = json.loads((experiment / "steps" / "tree.json").read_text(encoding="utf-8"))
            self.assertEqual([node["node_id"] for node in published["nodes"]], nodes)
            self.assertTrue(all((experiment / "steps" / node / "output" / "main.py").is_file() for node in nodes))
            steps = load_recorded_steps(experiment)
            self.assertEqual([step.step_id for step in steps], nodes)
            self.assertEqual({step.span for step in steps}, {"full"})
            self.assertEqual([step.revision_id for step in steps], [step.revision_id for step in session.backtest.steps])
            self.assertTrue(all(Path(step.validation.result_ref).is_file() for step in steps))
            # Host-only: the sidecars sit beside the run markers, never in a mount.
            self.assertTrue((experiment / ".host" / "steps").is_dir())
