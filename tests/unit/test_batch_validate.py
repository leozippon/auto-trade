"""``batch_validate``: one formal step over a pre-registered candidate set.

This tool fans a pre-registered set out under one shared parent and one span
of the research period, so the numbers are comparable; it is the only
Validation tool, so one candidate is a round too: same static gate, same
immutable revision, same replay-year budget, same selection path through
``step_rollback`` plus ``finish_session`` at every width.

These tests cover the refusals that must happen before anything runs, the
spans a batch may replay on a multi-year research period, the per-candidate
isolation of a failure, the lineage the tree ends up with, and that a batch
node can actually be frozen through the gate.
"""

from __future__ import annotations

import json
import threading
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from autotrade.environment.artifacts import (
    FilesystemArtifactStore,
    ModificationConstraints,
    readonly_baseline,
)
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import write_json_atomic
from autotrade.environment.sandbox import SandboxConfig
from autotrade.environment.step_tree import StepTree
from autotrade.environment.time_budget import InferenceTimeBudget
from autotrade.environment.tools.base import ToolError, ToolRegistry
from autotrade.environment.tools.finish_session import FinishSessionTool
from autotrade.environment.tools.modification_check import ModificationCheckTool
from autotrade.environment.tools.step_rollback import StepRollbackTool
from autotrade.environment.tools.workspace import SafeWorkspace
from autotrade.pipelines.config import (
    BrokerProfile,
    EvaluationResult,
    ReplaySpan,
    ResearchSessionRequest,
    SnapshotBundle,
    StrategySchedule,
)
from autotrade.pipelines.experiment import null_control_seed
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.local_backend import (
    BATCH_REJECTION_CHARGE_AFTER,
    BATCH_REJECTION_ESCALATE_AT,
    BATCH_VALIDATE_MAX_CANDIDATES,
    BATCH_VALIDATE_MAX_CONCURRENCY,
    BatchValidateTool,
    NullControlTool,
    SessionValidations,
    another_batch_round_fits,
    batch_select_hint,
    session_budget_status,
)

PARENT_SOURCE = "def generate_orders(context):\n    return []\n"
# A four-year research period, one slot per July-June year.
YEARS = (
    ("Y1", "20210701", "20220630"),
    ("Y2", "20220701", "20230630"),
    ("Y3", "20230701", "20240630"),
    ("Y4", "20240701", "20250630"),
)
TEMPLATE_README = "# Strategy output contract\n\nRead-only template text.\n"


def _strategy(marker: str) -> str:
    return f"def generate_orders(context):\n    _ = {marker}\n    return []\n"


def _summary(total_return: float) -> dict[str, object]:
    return {
        "total_return": total_return,
        "sharpe": 1.0,
        "max_drawdown": 0.05,
        "turnover": 1.2,
        "trade_count": 3,
        "replayed_trade_days": 60,
        "per_stock": [{"symbol": "000001.SZ"} for _ in range(200)],
        "weekly_returns": [{"week_end": "20220107", "return": 0.01}],
        "sub_windows": [
            {
                "kind": "year",
                "label": "202107-202206",
                "start": "20210701",
                "end": "20220630",
                "trade_days": 60,
                "partial": False,
                "return": total_return,
                "benchmark_return": 0.01,
                "excess_return": total_return - 0.01,
                "sharpe": 1.0,
                "max_drawdown": 0.05,
                "turnover": 1.2,
                "trade_count": 3,
            }
        ],
    }


class _Evaluator:
    """Replays a revision by reading its ``main.py`` marker.

    ``fail_markers`` makes one named candidate blow up the way a real replay
    does (a per-day timeout), which is what proves one failure does not sink
    the batch.
    """

    def __init__(
        self,
        results_root: Path,
        *,
        fail_markers: tuple[str, ...] = (),
        rendezvous: int = 0,
    ) -> None:
        self.results_root = results_root
        self.fail_markers = fail_markers
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

    def evaluate(self, request, max_days=None):
        del max_days
        source = (Path(request.revision.output_path) / "main.py").read_text(
            encoding="utf-8"
        )
        with self._lock:
            self.calls += 1
            self.requests.append(request)
            self._arrived += 1
            self.fit_timeouts.append(self.sandbox.limits.fit_timeout_seconds)
            self.inference_timeouts.append(self.sandbox.limits.timeout_seconds)
            call_index = self.calls
            arrival = self._arrived
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self._barrier is not None and arrival <= self._barrier.parties:
                self._barrier.wait()
            for marker in self.fail_markers:
                if marker in source:
                    raise TimeoutError(f"generate_orders exceeded 30s ({marker})")
            summary = _summary(0.01 * len(source))
            target = self.results_root / f"valid_{call_index:03d}" / "result.json"
            write_json_atomic(target, {"stats": summary})
            _write_style_sidecar(target.parent, alpha=0.0005 * len(source), seed=call_index)
            return EvaluationResult(summary=dict(summary), result_ref=str(target))
        finally:
            with self._lock:
                self.active -= 1


def _write_style_sidecar(directory: Path, *, alpha: float, seed: int) -> None:
    """The daily series the freeze gate reads: 60 days of a return that is
    ``alpha`` plus benchmark and size exposure plus noise."""

    rng = np.random.default_rng(seed)
    days = [f"2022{index // 20 + 1:02d}{index % 20 + 1:02d}" for index in range(60)]
    benchmark = rng.normal(0.0003, 0.01, len(days))
    size = rng.normal(0.0, 0.004, len(days))
    strategy = alpha + 0.9 * benchmark + 0.2 * size + rng.normal(0.0, 0.004, len(days))
    write_json_atomic(
        directory / "style_analysis.json",
        {
            "strategy_daily": [[day, float(value)] for day, value in zip(days, strategy)],
            "benchmark_daily": [[day, float(value)] for day, value in zip(days, benchmark)],
            "size_factor_daily": [[day, float(value)] for day, value in zip(days, size)],
        },
    )


class _Session:
    """One research session's real tools over a temporary workspace."""

    def __init__(
        self,
        root: Path,
        *,
        max_replay_years: int = 48,
        fail_markers: tuple[str, ...] = (),
        rendezvous: int = 0,
        record_failed_attempts: bool = True,
        deadline_seconds: float = 600.0,
        readonly_template: bool = False,
        trace: list[tuple[str, dict[str, object]]] | None = None,
    ) -> None:
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
            research_years=tuple(
                ReplaySpan(
                    label,
                    "valid",
                    start,
                    end,
                    SnapshotBundle(f"snap_{label}", f"decision_{label}", f"replay_{label}"),
                )
                for label, start, end in YEARS
            ),
            input_window_start="20200701",
            max_replay_years=max_replay_years,
            max_llm_calls=10,
            deadline_seconds=deadline_seconds,
            deadline_grace_seconds=60.0,
            finalize_before_deadline_seconds=30,
            record_failed_attempts=record_failed_attempts,
            acceptance_rules={"max_drawdown": 0.25},
        )
        self.tree = StepTree(root / "steps")
        self.evaluator = _Evaluator(
            root / "results", fail_markers=fail_markers, rendezvous=rendezvous
        )
        self.backtest = SessionValidations(
            request=request,
            output_dir=self.output,
            models_dir=self.models,
            artifact_store=FilesystemArtifactStore(root / "revisions"),
            evaluator=self.evaluator,
            tree=self.tree,
            schedule=StrategySchedule(),
            broker_profile=BrokerProfile(),
            time_budget=InferenceTimeBudget(duration_seconds=deadline_seconds),
            ref_store=AgentRefStore(root / "experiment"),
            ledger=ExperimentLedger(root / "ledger.jsonl"),
        )
        self.workspace = SafeWorkspace(self.workspace_root)
        self.batch = BatchValidateTool(
            backtest=self.backtest,
            workspace=self.workspace,
            modification_check_factory=self._check,
            trace_emit=(
                (lambda event, payload: trace.append((event, payload)))
                if trace is not None
                else None
            ),
        )
        self.rollback = StepRollbackTool(
            self.tree,
            self.output,
            self.models,
            session_ref=self.backtest.ref_store.get_or_create("session", "research"),
            run_id=self.backtest.ref_store.get_or_create("run", "run_batch"),
        )
        self.finish = FinishSessionTool(
            self.tree,
            session_ref=self.backtest.ref_store.get_or_create("session", "research"),
            run_ref=self.backtest.ref_store.get_or_create("run", "run_batch"),
            freeze_gate=self.backtest.freeze_gate,
            another_round_fits=lambda: another_batch_round_fits(self.backtest),
            budget_status=lambda: session_budget_status(self.backtest),
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

    def call(self, *names: str, span: str | None = None) -> object:
        arguments: dict[str, object] = {
            "candidates": [
                {
                    "name": name,
                    "hypothesis": f"{name} earns a positive neutralized excess",
                    "path": f"candidates/{name}",
                }
                for name in names
            ]
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
            self.assertEqual(session.backtest.replay_years_used, 0)
            self.assertEqual(session.evaluator.calls, 0)
            self.assertEqual(session.tree.nodes(), [])

    def test_a_candidate_with_the_start_logic_is_a_legal_candidate(self) -> None:
        """A session may re-validate its start: on another span, or on the full
        period to count it toward the freeze gate. Nothing refuses it."""
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
            self.assertEqual(session.backtest.replay_years_used, 0)

    def test_a_batch_larger_than_the_remaining_budget_is_refused_whole(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), max_replay_years=8)
            for index, name in enumerate("abc"):
                session.candidate(name, _strategy(str(index)))
            # Three candidates on the full four-year period need twelve.
            with self.assertRaises(ToolError) as caught:
                session.call("a", "b", "c")
            self.assertIn("has 8 left and this batch needs 12", str(caught.exception))
            self.assertEqual(caught.exception.error_type, "budget_exhausted")
            self.assertEqual(session.backtest.replay_years_used, 0)
            self.assertEqual(session.evaluator.calls, 0)
            # The same three on one year fit.
            self.assertTrue(session.call("a", "b", "c", span="Y2").ok)
            self.assertEqual(session.backtest.replay_years_used, 3)

    def test_the_working_copy_is_a_candidate_but_other_reserved_roots_are_not(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            (session.output / "main.py").write_text(_strategy("4"), encoding="utf-8")
            result = session.batch.invoke(
                {"candidates": [{"name": "live", "hypothesis": "h", "path": "output"}]}
            )
            self.assertTrue(result.ok, result.error)
            row = result.value["candidates"][0]
            node = session.tree.get_node(str(row["node_id"]))
            self.assertEqual(node["metadata"]["source_path"], "output")
            with self.assertRaises(ToolError) as caught:
                session.batch.invoke(
                    {"candidates": [{"name": "m", "hypothesis": "h", "path": "models"}]}
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
            self.assertEqual(session.backtest.replay_years_used, 0)

    def test_a_duplicate_path_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("5"))
            with self.assertRaises(ToolError) as caught:
                session.batch.invoke(
                    {
                        "candidates": [
                            {"name": "one", "hypothesis": "h", "path": "candidates/a"},
                            {"name": "two", "hypothesis": "h", "path": "candidates/a"},
                        ]
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
                {"label": "Y2..Y3", "start": "20220701", "end": "20240630", "years": 2},
            )
            for request in session.evaluator.requests:
                self.assertEqual((request.start, request.end), ("20220701", "20240630"))
                self.assertEqual(request.snapshot.decision_ref, "decision_Y2")
                self.assertEqual(request.snapshot.replay_ref, "replay_Y2")
                self.assertEqual(request.continuation, ("replay_Y3",))
            # Two candidates on two years cost four replay-years.
            self.assertEqual(value["replay_years_used"], 4)
            self.assertEqual(value["replay_years_remaining"], 44)
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
            self.assertEqual(session.backtest.replay_years_used, 8)

    def test_a_span_outside_the_research_years_is_refused_before_anything_runs(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("1"))
            # Six refusals: the seventh identical one would start costing
            # replay-years (RepeatedRejectionTest).
            for label in ("Y0", "Y5", "Y3..Y2", "Y2..Y5", "F", "heldout"):
                with self.assertRaises(ToolError) as caught:
                    session.call("a", span=label)
                self.assertIn("span must be full", str(caught.exception))
                self.assertEqual(caught.exception.error_type, "schema_error")
            self.assertEqual(session.evaluator.calls, 0)
            self.assertEqual(session.backtest.replay_years_used, 0)

    def test_the_null_control_ranks_a_node_over_its_own_span(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            calls: list[dict[str, object]] = []
            session.evaluator.null_control = _canned_null(calls)
            node = session.validate_one("y3", _strategy("7"), span="Y3")["node_id"]
            NullControlTool(session.backtest, max_calls=1).invoke({"node_id": node})
            self.assertEqual((calls[0]["start"], calls[0]["end"]), ("20230701", "20240630"))


class FreezeThroughTheGateTest(unittest.TestCase):
    """The finish tool reads the Pipeline's own gate over the session's Steps."""

    def test_the_gate_refuses_a_sub_span_node_and_a_lone_full_span_validation(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            year = session.validate_one("year", _strategy("1"), span="Y2")["node_id"]
            with self.assertRaises(ToolError) as sub_span:
                session.finish.invoke({"outcome": "freeze", "node_id": year})
            self.assertIn("freeze_needs_full_span_validation", str(sub_span.exception))
            full = session.validate_one("full", _strategy("22"))["node_id"]
            with self.assertRaises(ToolError) as lone:
                session.finish.invoke({"outcome": "freeze", "node_id": full})
            self.assertIn("freeze_too_few_full_span_validations", str(lone.exception))
            self.assertIn("full_span_validations=1", str(lone.exception))
            self.assertEqual(lone.exception.details["passing_nodes"], [])

    def test_a_full_span_winner_among_two_freezes_without_restoring_the_working_copy(
        self,
    ) -> None:
        """The Pipeline freezes the nominated revision, not output/, so a
        winner is nominated as it is: the working copy still holds the start
        and no step_rollback comes first."""
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("2" * 60))
            value = session.call("a", "b").value
            winner = value["candidates"][1]["node_id"]
            gate = session.backtest.freeze_gate(str(winner))
            self.assertTrue(gate["passed"], gate)
            finished = session.finish.invoke(
                {
                    "outcome": "freeze",
                    "node_id": winner,
                    "reason": "both full-period candidates resolve the only hypothesis this test registers",
                }
            )
            self.assertTrue(finished.finish)
            self.assertEqual(finished.value["node_id"], winner)
            self.assertEqual(finished.value["freeze_gate"]["full_span_validations"], 2)
            self.assertEqual(
                finished.value["revision_id"],
                str(session.tree.get_node(winner)["revision_id"]),
            )
            self.assertEqual(
                (session.output / "main.py").read_text(encoding="utf-8"),
                PARENT_SOURCE,
            )


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
                self.assertEqual(node["metadata"]["hypothesis"], row["hypothesis"])
            # The batch never touched the working copy, so the tree position
            # comes back to where it branched from.
            self.assertEqual(session.tree.current_node_id, branch_point)
            self.assertEqual(session.backtest.replay_years_used, 16)
            self.assertEqual(len(session.backtest.steps), 4)

    def test_candidates_replay_concurrently_up_to_the_bound(self) -> None:
        """The bound is host capacity, not correctness: each replay owns its
        result directory, as-of view and container, and the bookkeeping around
        them stays on the calling thread."""
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), rendezvous=BATCH_VALIDATE_MAX_CONCURRENCY)
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
            self.assertEqual(session.evaluator.peak, BATCH_VALIDATE_MAX_CONCURRENCY)
            # Recording order follows the input, not whichever replay landed
            # first, so the lineage is reproducible.
            self.assertEqual(
                [row["name"] for row in result.value["candidates"]], names
            )
            self.assertEqual(
                [row["result_name"] for row in result.value["candidates"]],
                [f"valid_{index + 1:03d}" for index in range(len(names))],
            )

    def test_both_strategy_deadlines_scale_with_the_batch_replay_width(self) -> None:
        """The contention a batch adds is the environment's, not the strategy's.

        Both caps are a fixed per-call host wall clock, so with a fixed value a
        candidate's verdict depended on how many siblings happened to share the
        host: two explore_github folds lost both batch candidates to `strategy
        fit exceeded 3600s` while solo reruns of the same code did the same four
        refits in 1,550-2,027 s, and three arms lost slots to `strategy
        inference exceeded 180s` inside a batch -- one with a byte-identical
        serial control that replayed 243 decision days at ~2.0 s/day. Both caps
        therefore move with the batch's own width, and only for the pool's
        lifetime.
        """

        limits = SandboxConfig().limits
        fit_base = limits.fit_timeout_seconds
        inference_base = limits.timeout_seconds
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
                    session.evaluator.sandbox.limits.fit_timeout_seconds, fit_base
                )
                self.assertEqual(
                    session.evaluator.sandbox.limits.timeout_seconds, inference_base
                )
            # Everything else about the boundary is untouched.
            self.assertEqual(session.evaluator.sandbox, SandboxConfig())

    def test_each_candidate_costs_the_years_of_its_span(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), max_replay_years=10)
            for index, name in enumerate(("a", "b")):
                session.candidate(name, _strategy(str(index)))
            value = session.call("a", "b").value
            self.assertEqual(value["replay_years_used"], 8)
            self.assertEqual(value["replay_years_remaining"], 2)
            # The next round continues the same numbering.
            self.assertEqual(
                sorted(row["result_name"] for row in value["candidates"]),
                ["valid_001", "valid_002"],
            )
            span = session.backtest.span("Y1")
            self.assertEqual(session.backtest.reserve(1, span), ["valid_003"])
            self.assertEqual(session.backtest.replay_years_remaining, 1)

    def test_one_failing_candidate_does_not_hide_the_others(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), fail_markers=("999",))
            session.candidate("good", _strategy("1"))
            session.candidate("bad", _strategy("999"))
            session.candidate("also_good", _strategy("2"))
            result = session.call("good", "bad", "also_good")
            self.assertTrue(result.ok)
            rows = {row["name"]: row for row in result.value["candidates"]}
            self.assertEqual(rows["good"]["status"], "ok")
            self.assertEqual(rows["also_good"]["status"], "ok")
            self.assertEqual(rows["bad"]["status"], "failed")
            self.assertIn("generate_orders exceeded", rows["bad"]["error"])
            # No node, no result_ref and no Step: exactly what the tool
            # description promises a failed candidate leaves behind.
            self.assertNotIn("node_id", rows["bad"])
            self.assertNotIn("result_ref", rows["bad"])
            self.assertEqual(result.value["complete_validations"], 2)
            self.assertEqual(result.value["failed"], 1)
            # A failed attempt is recorded as a dead end and never becomes a
            # parent, but it still cost its replay-years.
            failed = [
                node
                for node in session.tree.nodes()
                if node.get("status") == "failed"
            ]
            self.assertEqual(len(failed), 1)
            # The dead end hangs off the batch's own parent, not off whichever
            # sibling happened to be recorded before it.
            self.assertEqual(
                failed[0]["parent_node_id"], result.value["parent_node_id"]
            )
            # The dead end carries the same batch id and hypothesis as its
            # recorded siblings: the round is readable from the tree alone.
            self.assertEqual(failed[0]["metadata"]["candidate"], "bad")
            self.assertEqual(failed[0]["metadata"]["batch_id"], result.value["batch_id"])
            self.assertEqual(failed[0]["metadata"]["span"], "full")
            self.assertEqual(session.backtest.replay_years_used, 12)
            self.assertEqual(len(session.backtest.steps), 2)

    def test_a_wholly_failed_batch_reports_the_failure_not_a_success(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), fail_markers=("1", "2"))
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("2"))
            with self.assertRaises(ToolError) as caught:
                session.call("a", "b")
            error = caught.exception
            self.assertIn("all 2 candidates failed", str(error))
            rows = error.details["candidates"]
            self.assertEqual([row["status"] for row in rows], ["failed", "failed"])
            # Honest accounting: the replays ran, so the replay-years are gone.
            self.assertEqual(session.backtest.replay_years_used, 8)
            self.assertEqual(session.backtest.steps, [])

    def test_each_row_carries_the_sub_window_table_and_a_readable_result_ref(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("22"))
            value = session.call("a", "b").value
            for row in value["candidates"]:
                stats = row["stats"]
                self.assertEqual(stats["sub_windows"][0]["label"], "202107-202206")
                self.assertIn("total_return", stats)
                # The blocks that scale with the replay stay behind the
                # reference, like every Validation keeps them.
                self.assertNotIn("per_stock", stats)
                self.assertNotIn("weekly_returns", stats)
                attachment = session.tree.root / row["result_ref"]
                self.assertTrue(attachment.is_file())
                record = json.loads(attachment.read_text(encoding="utf-8"))
                self.assertIn("sub_windows", record["stats"])

    def test_each_row_carries_the_provisional_freeze_gate_over_the_arm(
        self,
    ) -> None:
        """Every revision the arm has validated so far is the trial pool, the
        whole batch included, so the rows of one round are gated against the
        same N and say how many that was."""
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            first = session.validate_one("first", _strategy("333"))
            statistics = first["selection_statistics"]
            self.assertEqual(statistics["trials"], 1)
            self.assertEqual(statistics["full_span_validations"], 1)
            self.assertFalse(statistics["freeze_gate_passed"])
            self.assertIn(
                "freeze_too_few_full_span_validations", statistics["freeze_gate_reasons"]
            )
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("22"))
            value = session.call("a", "b").value
            for row in value["candidates"]:
                statistics = row["selection_statistics"]
                self.assertEqual(statistics["trials"], 3)
                self.assertEqual(statistics["full_span_validations"], 3)
                self.assertTrue(0.0 <= statistics["deflated_sharpe_probability"] <= 1.0)
                self.assertIn("the freeze recomputes it", statistics["note"])
                self.assertNotIn("vs_parent", row)

    def test_nothing_selects_a_winner_on_the_agent_s_behalf(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("22"))
            value = session.call("a", "b").value
            self.assertNotIn("winner", value)
            self.assertNotIn("selected", value)
            self.assertIn("step_rollback", value["select_hint"])
            # A winning round starts the next one; the hint states when
            # finishing is warranted and never instructs it.
            self.assertNotIn("finish_session(", value["select_hint"])
            # The working copy is untouched by the batch.
            self.assertEqual(
                (session.output / "main.py").read_text(encoding="utf-8"),
                PARENT_SOURCE,
            )


class BatchSelectHintTest(unittest.TestCase):
    """The hint names the row leading on the neutralized excess and selects nothing."""

    def test_names_the_leader_on_the_neutralized_excess(self) -> None:
        rows = [
            {"name": "a", "node_id": "n_a", "status": "ok",
             "stats": {"benchmark": {"neutralized_excess_return": 0.12}}},
            {"name": "b", "node_id": "n_b", "status": "ok",
             "stats": {"benchmark": {"neutralized_excess_return": float("nan")}}},
            {"name": "c", "node_id": "n_c", "status": "failed", "error": "boom"},
        ]
        hint = batch_select_hint(rows, replay_years_remaining=8)
        self.assertIn("leading on neutralized excess: a (node_id=n_a)", hint)
        self.assertIn("step_rollback(node_id=<chosen>)", hint)
        self.assertIn("a freeze needs a full-span validation", hint)
        self.assertNotIn("finish_session(", hint)
        self.assertIn("pre-registered hypotheses are resolved", hint)

    def test_names_nobody_when_no_row_carries_the_figure(self) -> None:
        rows = [{"name": "a", "node_id": "n_a", "status": "ok", "stats": {"total_return": 0.2}}]
        hint = batch_select_hint(rows, replay_years_remaining=8)
        self.assertIn("no row carries a neutralized excess figure", hint)
        self.assertNotIn("leading on", hint)

    def test_a_spent_budget_leaves_only_the_handoff_and_the_finish(self) -> None:
        rows = [{"name": "a", "node_id": "n_a", "status": "ok", "stats": {}}]
        hint = batch_select_hint(rows, replay_years_remaining=0)
        self.assertIn("replay-year budget is spent", hint)
        self.assertIn("write any skills and finish_session", hint)


class BatchValidateContractTest(unittest.TestCase):
    def test_the_tool_is_sequential_phase_gated_and_in_the_session_set(self) -> None:
        from autotrade.agent.runner import (
            _PHASE_GATE_TOOLS,
            _SESSION_TOOLS,
        )
        from autotrade.environment.tools.base import is_sequential_tool

        name = BatchValidateTool.spec.name
        self.assertTrue(BatchValidateTool.spec.mutating)
        self.assertTrue(is_sequential_tool(BatchValidateTool.spec))
        self.assertIn(name, _PHASE_GATE_TOOLS)
        self.assertIn(name, _SESSION_TOOLS)

    def test_the_description_states_the_selection_path(self) -> None:
        description = BatchValidateTool.spec.description
        self.assertIn("step_rollback", description)
        self.assertIn("finish_session", description)
        self.assertIn("hypothesis", description)
        self.assertIn("span", BatchValidateTool.spec.input_schema["properties"])
        for retired in ("Fold", "parent strategy", "backtest budget", "Step budget"):
            self.assertNotIn(retired, description)

    def test_a_schema_error_returns_the_example_call(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            registry = ToolRegistry([session.batch])
            result = registry.invoke("batch_validate", {"candidate": []})
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
        runner._complete_validation_nodes = []
        runner._record_complete_validations(record)
        self.assertEqual(
            [item["node_id"] for item in runner._complete_validation_nodes],
            ["n1", "n2"],
        )


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
            self.assertEqual(session.backtest.replay_years_used, 0)


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
            self.assertIn("delta.readonly_violations", escalated)
            # An escalation is not a charge: the budget is still untouched.
            self.assertEqual(session.backtest.replay_years_used, 0)
            self.assertEqual([event for event, _ in trace], ["batch_rejection_escalated"])
            payload = trace[0][1]
            self.assertEqual(payload["error_type"], "readonly_baseline")
            self.assertEqual(payload["blocked_target"], "candidates/a")
            self.assertEqual(payload["repeat_count"], BATCH_REJECTION_ESCALATE_AT)
            self.assertIs(payload["charged_replay_year"], False)

    def test_every_attempt_past_the_sixth_consumes_a_replay_year(self) -> None:
        with TemporaryDirectory() as tmp:
            session = self._readonly_loop(tmp)
            for _ in range(BATCH_REJECTION_CHARGE_AFTER):
                self._refuse(session)
            self.assertEqual(session.backtest.replay_years_used, 0)
            charged = self._refuse(session)
            self.assertEqual(session.backtest.replay_years_used, 1)
            self.assertIn("consumed one replay-year", charged)
            self._refuse(session)
            self.assertEqual(session.backtest.replay_years_used, 2)
            # The bound is the budget: once it is gone, so is any further batch.
            while session.backtest.replay_years_remaining > 0:
                self._refuse(session)
            self.assertIn("replay-year budget is already spent", self._refuse(session))
            self.assertEqual(
                session.backtest.replay_years_used,
                session.backtest.request.max_replay_years,
            )

    def test_a_different_signature_is_counted_on_its_own(self) -> None:
        with TemporaryDirectory() as tmp:
            session = self._readonly_loop(tmp)
            reserved = {
                "candidates": [
                    {"name": "a", "hypothesis": "h", "path": "candidates/a"},
                    {"name": "live", "hypothesis": "h", "path": "models"},
                ]
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
            self.assertEqual(session.backtest.replay_years_used, 0)


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
    """The session's null control is the freeze's null control: the same
    backend call with the same seed, cached per node and capped per session."""

    def test_a_validated_node_is_ranked_once_and_the_block_is_reused(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            calls: list[dict[str, object]] = []
            session.evaluator.null_control = _canned_null(calls)
            node = session.validate_one("n", _strategy("7"))["node_id"]
            tool = NullControlTool(session.backtest, max_calls=2)
            first = tool.invoke({"node_id": node}).value
            self.assertFalse(first["cached"])
            self.assertEqual(first["null_control"]["excess_percentile"], 0.62)
            # The Agent-visible block is the same whitelist the run facts use.
            self.assertNotIn("seed", first["null_control"])
            self.assertEqual((first["null_controls_used"], first["null_controls_remaining"]), (1, 1))
            # Drawn exactly as the freeze would draw it: whole period, frozen role.
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
            self.assertIn("complete Validation of this session", str(refused.exception))
            self.assertEqual(refused.exception.details["candidates"], [first, second])
            self.assertTrue(tool.invoke({"node_id": first}).ok)
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
            # A failure is not a result the freeze may reuse.
            self.assertNotIn(node, tool.blocks)

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


class AnotherRoundFitsTest(unittest.TestCase):
    """``another_batch_round_fits`` waives the early-freeze reason: true while
    the session could still run one more round, false once the replay-year
    budget or the deadline window rules one out."""

    def _no_edge(self, session: _Session) -> object:
        return session.finish.invoke(
            {
                "outcome": "no_edge",
                "reason": "one round resolved the pre-registered hypotheses of this session",
            }
        )

    def test_a_round_still_fits_while_budget_and_time_allow(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), max_replay_years=24)
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("22"))
            session.call("a", "b")
            self.assertTrue(another_batch_round_fits(session.backtest))
            # One completed round is a legal finish: no round floor remains.
            self.assertTrue(self._no_edge(session).finish)

    def test_no_round_fits_when_the_budget_cannot_hold_another(self) -> None:
        with TemporaryDirectory() as tmp:
            # Eight replay-years: one two-candidate full-period round leaves
            # nothing for even a one-candidate, one-year round.
            session = _Session(Path(tmp), max_replay_years=8)
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("22"))
            session.call("a", "b")
            self.assertFalse(another_batch_round_fits(session.backtest))
            self.assertTrue(self._no_edge(session).finish)

    def test_no_round_fits_inside_the_deadline_window(self) -> None:
        with TemporaryDirectory() as tmp:
            # 80 s of budget minus the 60 s grace leaves 20 s before the main
            # deadline, inside the 30 s finalize reserve.
            session = _Session(Path(tmp), deadline_seconds=80.0)
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("22"))
            session.call("a", "b")
            self.assertFalse(another_batch_round_fits(session.backtest))
            self.assertTrue(self._no_edge(session).finish)


if __name__ == "__main__":
    unittest.main()
