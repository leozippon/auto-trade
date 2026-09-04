"""``batch_validate``: one formal step over a pre-registered candidate set.

The audited folds reached at most two formal Validations each and carried the
parent forward whenever a single quarter could not separate a challenger from
it — one candidate per serial step, each branching off the previous one, made
the evidence per decision both thin and path-dependent. This tool fans a
pre-registered set out under one shared parent so the numbers are comparable,
while keeping every guarantee ``daily_backtest`` gives: same static gate, same
immutable revision, same budget, same selection path through ``step_rollback``
plus ``finish_fold``.

These tests cover the refusals that must happen before anything runs, the
per-candidate isolation of a failure, the lineage the tree ends up with, and
that a batch node can actually be selected.
"""

from __future__ import annotations

import json
import threading
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from autotrade.environment.artifacts import (
    FilesystemArtifactStore,
    ModificationConstraints,
)
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import write_json_atomic
from autotrade.environment.step_tree import StepTree
from autotrade.environment.time_budget import InferenceTimeBudget
from autotrade.environment.tools.base import ToolError, ToolRegistry
from autotrade.environment.tools.finish_fold import FinishFoldTool
from autotrade.environment.tools.modification_check import ModificationCheckTool
from autotrade.environment.tools.step_rollback import StepRollbackTool
from autotrade.environment.tools.workspace import SafeWorkspace
from autotrade.pipelines.config import (
    BrokerProfile,
    EvaluationResult,
    FoldSessionRequest,
    FoldSpec,
    SnapshotBundle,
    StrategySchedule,
)
from autotrade.pipelines.local_backend import (
    BATCH_VALIDATE_MAX_CANDIDATES,
    BATCH_VALIDATE_MAX_CONCURRENCY,
    BatchValidateTool,
    FoldBacktestTool,
    another_batch_round_fits,
)

PARENT_SOURCE = "def generate_orders(context):\n    return []\n"
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
                "kind": "quarter",
                "label": "2022Q1",
                "start": "20220104",
                "end": "20220331",
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

    def evaluate(self, request, max_days=None):
        del max_days
        source = (Path(request.revision.output_path) / "main.py").read_text(
            encoding="utf-8"
        )
        with self._lock:
            self.calls += 1
            self._arrived += 1
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
            return EvaluationResult(summary=dict(summary), result_ref=str(target))
        finally:
            with self._lock:
                self.active -= 1


class _Session:
    """One Fold session's real tools over a temporary workspace."""

    def __init__(
        self,
        root: Path,
        *,
        max_backtests: int = 6,
        max_steps: int = 6,
        fail_markers: tuple[str, ...] = (),
        rendezvous: int = 0,
        with_parent: bool = True,
        record_failed_attempts: bool = True,
        deadline_seconds: float = 600.0,
        readonly_template: bool = False,
    ) -> None:
        self.root = root
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
        moment = datetime(2021, 12, 31, 23, 59, 59, tzinfo=UTC)
        request = FoldSessionRequest(
            experiment_id="exp",
            epoch_id="epoch_001",
            fold=FoldSpec(
                fold_id="fold_2022Q1",
                input_window_start="20200101",
                input_window_end="20211231",
                validation_start="20220101",
                validation_end="20220331",
                test_start="20220401",
                test_end="20220630",
                valid_decision_time=moment,
                test_decision_time=moment,
            ),
            run_id="run_batch",
            parent=None,
            prior="",
            snapshot=SnapshotBundle("snapshot", "decision", "replay"),
            max_steps=max_steps,
            max_backtests=max_backtests,
            max_llm_calls=10,
            deadline_seconds=deadline_seconds,
            deadline_grace_seconds=60.0,
            finalize_before_deadline_seconds=30,
            record_failed_attempts=record_failed_attempts,
        )
        self.tree = StepTree(root / "steps")
        self.evaluator = _Evaluator(
            root / "results", fail_markers=fail_markers, rendezvous=rendezvous
        )
        self.backtest = FoldBacktestTool(
            request=request,
            output_dir=self.output,
            models_dir=self.models,
            modification_check=self._check(self.output),
            artifact_store=FilesystemArtifactStore(root / "revisions"),
            evaluator=self.evaluator,
            tree=self.tree,
            schedule=StrategySchedule(),
            broker_profile=BrokerProfile(),
            time_budget=InferenceTimeBudget(duration_seconds=deadline_seconds),
            ref_store=AgentRefStore(root / "experiment"),
        )
        self.workspace = SafeWorkspace(self.workspace_root)
        self.batch = BatchValidateTool(
            backtest=self.backtest,
            workspace=self.workspace,
            modification_check_factory=self._check,
            parent_main_py=(self.parent / "main.py") if with_parent else None,
        )
        self.rollback = StepRollbackTool(
            self.tree,
            self.output,
            self.models,
            fold_id=self.backtest.ref_store.get_or_create("fold", "fold_2022Q1"),
            run_id=self.backtest.ref_store.get_or_create("run", "run_batch"),
        )
        self.finish = FinishFoldTool(
            self.tree,
            fold_id=self.backtest.ref_store.get_or_create("fold", "fold_2022Q1"),
            run_id=self.backtest.ref_store.get_or_create("run", "run_batch"),
            parent_main_py=self.parent / "main.py",
            current_output=self.output,
            current_models=self.models,
            another_round_fits=lambda: another_batch_round_fits(self.backtest),
        )

    def _check(self, directory: Path) -> ModificationCheckTool:
        return ModificationCheckTool(
            directory,
            parent_dir=self.parent,
            models_dir=self.models,
            constraints=ModificationConstraints(),
        )

    def candidate(self, name: str, source: str) -> None:
        directory = self.workspace_root / "candidates" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "main.py").write_text(source, encoding="utf-8")

    def call(self, *names: str) -> object:
        return self.batch.invoke(
            {
                "candidates": [
                    {
                        "name": name,
                        "hypothesis": f"{name} beats the parent on this window",
                        "path": f"candidates/{name}",
                    }
                    for name in names
                ]
            }
        )


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
            self.assertEqual(session.backtest.backtests, 0)
            self.assertEqual(session.evaluator.calls, 0)
            self.assertEqual(session.tree.nodes(), [])

    def test_a_candidate_with_the_parent_logic_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            # Comment-only difference: same executable structure as the parent,
            # so finish_fold could never select the node it would produce.
            session.candidate("a", "# a new idea\n" + PARENT_SOURCE)
            session.candidate("b", _strategy("2"))
            with self.assertRaises(ToolError) as caught:
                session.call("a", "b")
            self.assertIn("parent strategy's executable logic", str(caught.exception))
            self.assertEqual(session.backtest.backtests, 0)

    def test_a_candidate_failing_modification_check_refuses_the_batch(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("3"))
            # No main.py: the same static gate daily_backtest runs.
            (session.workspace_root / "candidates" / "b").mkdir(parents=True)
            (session.workspace_root / "candidates" / "b" / "notes.py").write_text(
                "x = 1\n", encoding="utf-8"
            )
            with self.assertRaises(ToolError) as caught:
                session.call("a", "b")
            self.assertIn("failed modification_check", str(caught.exception))
            self.assertEqual(session.backtest.backtests, 0)

    def test_a_batch_larger_than_the_remaining_budget_is_refused_whole(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), max_backtests=2, max_steps=6)
            for index, name in enumerate("abc"):
                session.candidate(name, _strategy(str(index)))
            with self.assertRaises(ToolError) as caught:
                session.call("a", "b", "c")
            self.assertIn("backtest budget", str(caught.exception))
            self.assertEqual(session.backtest.backtests, 0)
            self.assertEqual(session.evaluator.calls, 0)

    def test_a_batch_larger_than_the_remaining_step_budget_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), max_backtests=9, max_steps=2)
            for index, name in enumerate("abc"):
                session.candidate(name, _strategy(str(index)))
            with self.assertRaises(ToolError) as caught:
                session.call("a", "b", "c")
            self.assertIn("Step budget", str(caught.exception))
            self.assertEqual(session.backtest.backtests, 0)

    def test_the_live_working_copy_is_not_a_candidate(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("4"))
            with self.assertRaises(ToolError) as caught:
                session.batch.invoke(
                    {
                        "candidates": [
                            {"name": "a", "hypothesis": "h", "path": "candidates/a"},
                            {"name": "live", "hypothesis": "h", "path": "output"},
                        ]
                    }
                )
            self.assertIn("reserved workspace root", str(caught.exception))

    def test_the_batch_size_bounds_are_enforced(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            for index in range(BATCH_VALIDATE_MAX_CANDIDATES + 1):
                session.candidate(f"c{index}", _strategy(str(index)))
            with self.assertRaises(ToolError):
                session.call("c0")
            with self.assertRaises(ToolError):
                session.call(*[f"c{index}" for index in range(BATCH_VALIDATE_MAX_CANDIDATES + 1)])
            self.assertEqual(session.backtest.backtests, 0)

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


class BatchValidateRunTest(unittest.TestCase):
    def test_every_candidate_becomes_a_sibling_node_under_one_parent(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            # A batch branches off wherever the session stands, so start from a
            # real node rather than the empty root.
            branch_point = session.backtest.invoke({}).value["node_id"]
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
            self.assertEqual(session.backtest.backtests, 4)
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

    def test_each_candidate_costs_one_backtest_and_one_step(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), max_backtests=4, max_steps=4)
            for index, name in enumerate(("a", "b")):
                session.candidate(name, _strategy(str(index)))
            value = session.call("a", "b").value
            self.assertEqual(value["backtests_used"], 2)
            self.assertEqual(value["backtests_remaining"], 2)
            self.assertEqual(value["steps_used"], 2)
            # The next daily_backtest continues the same numbering.
            self.assertEqual(
                sorted(row["result_name"] for row in value["candidates"]),
                ["valid_001", "valid_002"],
            )
            self.assertEqual(session.backtest.reserve_validations(1), ["valid_003"])

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
            self.assertNotIn("node_id", rows["bad"])
            self.assertEqual(result.value["complete_validations"], 2)
            self.assertEqual(result.value["failed"], 1)
            # A failed attempt is recorded as a dead end and never becomes a
            # parent, but it still cost its Validation slot.
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
            self.assertEqual(session.backtest.backtests, 3)
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
            # Honest accounting: the replays ran, so the slots are gone.
            self.assertEqual(session.backtest.backtests, 2)
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
                self.assertEqual(stats["sub_windows"][0]["label"], "2022Q1")
                self.assertIn("total_return", stats)
                # The blocks that scale with the replay stay behind the
                # reference, exactly as daily_backtest keeps them.
                self.assertNotIn("per_stock", stats)
                self.assertNotIn("weekly_returns", stats)
                attachment = session.tree.root / row["result_ref"]
                self.assertTrue(attachment.is_file())
                record = json.loads(attachment.read_text(encoding="utf-8"))
                self.assertIn("sub_windows", record["stats"])

    def test_a_batch_node_can_be_restored_and_selected(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("22"))
            value = session.call("a", "b").value
            winner = value["candidates"][1]["node_id"]
            # finish_fold refuses while the working copy is still the parent.
            with self.assertRaises(ToolError) as caught:
                session.finish.invoke({"node_id": winner})
            self.assertIn("match the selected", str(caught.exception))
            restored = session.rollback.invoke({"node_id": winner})
            self.assertTrue(restored.ok)
            self.assertEqual(session.tree.current_node_id, winner)
            finished = session.finish.invoke({"node_id": winner})
            self.assertTrue(finished.finish)
            self.assertEqual(finished.value["node_id"], winner)

    def test_nothing_selects_a_winner_on_the_agent_s_behalf(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp))
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("22"))
            value = session.call("a", "b").value
            self.assertNotIn("winner", value)
            self.assertNotIn("selected", value)
            self.assertIn("step_rollback", value["select_hint"])
            # The working copy is untouched by the batch.
            self.assertEqual(
                (session.output / "main.py").read_text(encoding="utf-8"),
                PARENT_SOURCE,
            )


class BatchValidateContractTest(unittest.TestCase):
    def test_the_tool_is_sequential_phase_gated_and_fold_only(self) -> None:
        from autotrade.agent.runner import (
            _FOLD_TOOLS,
            _META_TOOLS,
            _PHASE_GATE_TOOLS,
        )
        from autotrade.environment.tools.base import is_sequential_tool

        name = BatchValidateTool.spec.name
        self.assertTrue(BatchValidateTool.spec.mutating)
        self.assertTrue(is_sequential_tool(BatchValidateTool.spec))
        self.assertIn(name, _PHASE_GATE_TOOLS)
        self.assertIn(name, _FOLD_TOOLS)
        self.assertNotIn(name, _META_TOOLS)

    def test_the_description_states_the_selection_path(self) -> None:
        description = BatchValidateTool.spec.description
        self.assertIn("step_rollback", description)
        self.assertIn("finish_fold", description)
        self.assertIn("hypothesis", description)

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
        # A daily_backtest result still registers its single node.
        runner._record_complete_validations(
            {"ok": True, "value": {"node_id": "n3", "revision_id": "r3"}}
        )
        self.assertEqual(
            [item["node_id"] for item in runner._complete_validation_nodes],
            ["n1", "n2", "n3"],
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
            self.assertEqual(session.backtest.backtests, 0)


class AnotherRoundFitsTest(unittest.TestCase):
    """``another_batch_round_fits`` is the waiver every remaining ``finish_fold``
    gate shares: true while the session could still run one more round, false
    once the backtest/Step budget or the deadline window rules one out."""

    def _finish_winner(self, session: _Session, value: dict) -> object:
        winner = value["candidates"][0]["node_id"]
        session.rollback.invoke({"node_id": winner})
        return session.finish.invoke({"node_id": winner})

    def test_a_round_still_fits_while_budget_and_time_allow(self) -> None:
        with TemporaryDirectory() as tmp:
            session = _Session(Path(tmp), max_backtests=6, max_steps=6)
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("22"))
            value = session.call("a", "b").value
            self.assertTrue(another_batch_round_fits(session.backtest))
            # One completed round is a legal finish: no round floor remains.
            self.assertTrue(self._finish_winner(session, value).finish)

    def test_no_round_fits_when_the_budget_cannot_hold_another(self) -> None:
        with TemporaryDirectory() as tmp:
            # Three backtests: one two-candidate round leaves a single slot,
            # fewer than the smallest batch.
            session = _Session(Path(tmp), max_backtests=3, max_steps=6)
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("22"))
            value = session.call("a", "b").value
            self.assertFalse(another_batch_round_fits(session.backtest))
            self.assertTrue(self._finish_winner(session, value).finish)

    def test_no_round_fits_inside_the_deadline_window(self) -> None:
        with TemporaryDirectory() as tmp:
            # 80 s of budget minus the 60 s grace leaves 20 s before the main
            # deadline, inside the 30 s finalize reserve.
            session = _Session(Path(tmp), deadline_seconds=80.0)
            session.candidate("a", _strategy("1"))
            session.candidate("b", _strategy("22"))
            value = session.call("a", "b").value
            self.assertFalse(another_batch_round_fits(session.backtest))
            self.assertTrue(self._finish_winner(session, value).finish)


if __name__ == "__main__":
    unittest.main()
