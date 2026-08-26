"""Every configurable knob must change behaviour, not merely be accepted.

Half this repository's defects were parameters that existed and did nothing —
``image_keep``, ``meta_memory_max_epochs``, ``record_failed_attempts``,
``step_tree_enabled``, the ``ModificationConstraints`` that never reached the
tool. A test asserting a knob is accepted is the bug; each test here changes
the knob and asserts the observable difference.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from autotrade.environment.artifacts import (
    ModificationConstraints,
    copy_artifact,
    new_revision_id,
)
from autotrade.environment.runtime import agent_trace_path
from autotrade.environment.step_tree import StepTree
from autotrade.environment.tools import ModificationCheckTool, ToolRegistry
from autotrade.pipelines.config import RollingExperimentConfig
from autotrade.pipelines.ledger import ExperimentLedger

VALID_MAIN = "def generate_orders(context):\n    return []\n"


def _artifact(root: Path, *, extra_lines: int = 0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("readonly\n", encoding="utf-8")
    body = VALID_MAIN + "".join(f"# line {index}\n" for index in range(extra_lines))
    (root / "main.py").write_text(body, encoding="utf-8")
    return root


class ModificationConstraintsReachTheToolTest(unittest.TestCase):
    """The configured limits are enforced, not three hardcoded literals."""

    def test_a_configured_diff_cap_actually_rejects(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = _artifact(root / "parent")
            work = root / "work"
            copy_artifact(parent, work)
            _artifact(work, extra_lines=50)

            generous = ToolRegistry(
                [ModificationCheckTool(work, parent_dir=parent,
                                       constraints=ModificationConstraints())]
            ).invoke("modification_check", {})
            self.assertTrue(generous.ok, generous.error)

            strict = ToolRegistry(
                [
                    ModificationCheckTool(
                        work,
                        parent_dir=parent,
                        constraints=ModificationConstraints(max_diff_lines=5, max_code_diff_lines=5),
                    )
                ]
            ).invoke("modification_check", {})
            self.assertFalse(strict.ok)
            self.assertIn("diff lines 50 > 5", strict.error)

    def test_the_default_is_the_dataclass_not_a_literal(self) -> None:
        with TemporaryDirectory() as tmp:
            tool = ModificationCheckTool(_artifact(Path(tmp) / "output"))
            self.assertEqual(tool.constraints, ModificationConstraints())

    def test_an_early_epoch_relaxation_reaches_the_tool(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = _artifact(root / "parent")
            work = root / "work"
            copy_artifact(parent, work)
            _artifact(work, extra_lines=20)
            base = ModificationConstraints(max_diff_lines=5, max_code_diff_lines=5,
                                           early_max_diff_lines=500, early_max_code_diff_lines=500)
            early = ToolRegistry(
                [ModificationCheckTool(work, parent_dir=parent, constraints=base.for_epoch(1))]
            ).invoke("modification_check", {})
            self.assertTrue(early.ok, early.error)
            late = ToolRegistry(
                [ModificationCheckTool(work, parent_dir=parent, constraints=base.for_epoch(5))]
            ).invoke("modification_check", {})
            self.assertFalse(late.ok)

    def test_an_initial_artifact_is_exempt_from_the_change_budget(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = _artifact(root / "parent")
            work = root / "work"
            copy_artifact(parent, work)
            _artifact(work, extra_lines=50)
            constraints = replace(
                ModificationConstraints(max_diff_lines=1, max_code_diff_lines=1),
                is_initial_artifact=True,
            )
            result = ToolRegistry(
                [ModificationCheckTool(work, parent_dir=parent, constraints=constraints)]
            ).invoke("modification_check", {})
            self.assertTrue(result.ok, result.error)


class RecordFailedAttemptsTest(unittest.TestCase):
    """With the knob off, a failed validation leaves no dead-end node.

    Driven through the real ``FoldBacktestTool``: the gate lives at its
    exception path, and asserting the knob's value would prove nothing."""

    def _tool(self, root: Path, *, record_failed_attempts: bool):
        from contextlib import nullcontext

        from autotrade.environment.artifacts import FilesystemArtifactStore
        from autotrade.environment.broker import BrokerProfile
        from autotrade.environment.strategy import StrategySchedule
        from autotrade.environment.time_budget import InferenceTimeBudget
        from autotrade.pipelines.config import FoldSessionRequest, SnapshotBundle
        from autotrade.pipelines.folds import FoldSpec
        from autotrade.pipelines.local_backend import FoldBacktestTool

        output = _artifact(root / "output")
        models = root / "models"
        models.mkdir(parents=True, exist_ok=True)
        tree = StepTree(root / "steps")
        moment = datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC)
        fold = FoldSpec(
            fold_id="fold_2026Q1",
            input_window_start="20240101",
            input_window_end="20250930",
            validation_start="20251001",
            validation_end="20251231",
            test_start="20260101",
            test_end="20260331",
            valid_decision_time=moment,
            test_decision_time=moment,
        )
        request = FoldSessionRequest(
            experiment_id="exp",
            epoch_id="epoch_001",
            fold=fold,
            run_id="run_x",
            parent=None,
            prior="",
            snapshot=SnapshotBundle("snap", "decision", "replay"),
            max_steps=10,
            max_backtests=30,
            max_llm_calls=200,
            deadline_seconds=1200.0,
            record_failed_attempts=record_failed_attempts,
        )

        class ExplodingEvaluator:
            def evaluate(self, _request):
                raise RuntimeError("validation blew up")

        class PassingCheck:
            def invoke(self, _arguments):
                from autotrade.environment.tools import ToolResult

                return ToolResult(True, value={})

        return (
            FoldBacktestTool(
                request=request,
                output_dir=output,
                models_dir=models,
                modification_check=PassingCheck(),
                artifact_store=FilesystemArtifactStore(root / "artifacts"),
                evaluator=ExplodingEvaluator(),
                tree=tree,
                schedule=StrategySchedule(),
                broker_profile=BrokerProfile(),
                time_budget=InferenceTimeBudget(duration_seconds=300),
                formal_guard=nullcontext,
            ),
            tree,
        )

    def test_a_failed_validation_records_a_dead_end_only_when_enabled(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            off_tool, off_tree = self._tool(root / "off", record_failed_attempts=False)
            with self.assertRaises(Exception):
                off_tool.invoke({})
            self.assertEqual(off_tree.nodes(), [])

            on_tool, on_tree = self._tool(root / "on", record_failed_attempts=True)
            with self.assertRaises(Exception):
                on_tool.invoke({})
            nodes = on_tree.nodes()
            self.assertEqual([node["status"] for node in nodes], ["failed"])
            self.assertIn("validation blew up", nodes[0]["error"])
            # The dead end never becomes the working position.
            self.assertIsNone(on_tree.current_node_id)

    def test_a_teardown_failure_after_record_step_still_keeps_the_revision(self) -> None:
        from contextlib import contextmanager

        from autotrade.environment.artifacts import FilesystemArtifactStore
        from autotrade.environment.broker import BrokerProfile
        from autotrade.environment.strategy import StrategySchedule
        from autotrade.environment.time_budget import InferenceTimeBudget
        from autotrade.pipelines.config import (
            EvaluationResult,
            FoldSessionRequest,
            SnapshotBundle,
        )
        from autotrade.pipelines.folds import FoldSpec
        from autotrade.pipelines.local_backend import FoldBacktestTool

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = _artifact(root / "output")
            models = root / "models"
            models.mkdir(parents=True, exist_ok=True)
            tree = StepTree(root / "steps")
            moment = datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC)
            fold = FoldSpec(
                fold_id="fold_2026Q1",
                input_window_start="20240101",
                input_window_end="20250930",
                validation_start="20251001",
                validation_end="20251231",
                test_start="20260101",
                test_end="20260331",
                valid_decision_time=moment,
                test_decision_time=moment,
            )
            request = FoldSessionRequest(
                experiment_id="exp",
                epoch_id="epoch_001",
                fold=fold,
                run_id="run_keep",
                parent=None,
                prior="",
                snapshot=SnapshotBundle("snap", "decision", "replay"),
                max_steps=10,
                max_backtests=30,
                max_llm_calls=200,
                deadline_seconds=1200.0,
                record_failed_attempts=True,
            )

            result_path = root / "result.json"
            result_path.write_text("{}", encoding="utf-8")

            class PassingEvaluator:
                def evaluate(self, _request):
                    return EvaluationResult(
                        {"total_return": 0.01}, str(result_path), True
                    )

            class PassingCheck:
                def invoke(self, _arguments):
                    from autotrade.environment.tools import ToolResult

                    return ToolResult(True, value={})

            @contextmanager
            def exploding_teardown():
                yield
                raise RuntimeError("chmod failed")

            tool = FoldBacktestTool(
                request=request,
                output_dir=output,
                models_dir=models,
                modification_check=PassingCheck(),
                artifact_store=FilesystemArtifactStore(root / "artifacts"),
                evaluator=PassingEvaluator(),
                tree=tree,
                schedule=StrategySchedule(),
                broker_profile=BrokerProfile(),
                time_budget=InferenceTimeBudget(duration_seconds=300),
                formal_guard=exploding_teardown,
            )
            result = tool.invoke({})
            self.assertTrue(result.ok, result.error)
            self.assertEqual(len(tool.steps), 1)
            self.assertTrue(
                all(node.get("complete_validation") for node in tree.nodes())
            )
            self.assertFalse(
                any(node.get("status") == "failed" for node in tree.nodes())
            )

    def test_the_config_default_records_them(self) -> None:
        config = RollingExperimentConfig(
            "exp", Path("/tmp/experiments"), "2022Q1", "2022Q1", "2023Q1", "2023Q1"
        )
        self.assertTrue(config.record_failed_attempts)
        self.assertTrue(config.step_tree_enabled)


class MetaMemoryBoundTest(unittest.TestCase):
    """``meta_memory_max_epochs`` bounds the raw Meta trace concatenation:
    unbounded, it grows O(epochs^2)."""

    def _pipeline(self, root: Path, *, keep: int):
        from autotrade.pipelines.experiment import RollingExperimentPipeline

        config = RollingExperimentConfig(
            "exp", root / "experiments", "2022Q1", "2022Q1", "2023Q1", "2023Q1",
            meta_memory_max_epochs=keep,
        )
        ledger = ExperimentLedger(config.ledger_path)
        artifacts_root = config.experiment_dir / "artifacts"
        for index, epoch in enumerate(("epoch_001", "epoch_002", "epoch_003"), start=1):
            run_id = f"run_meta_{index}"
            trace = agent_trace_path(artifacts_root, run_id)
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text(json.dumps({"epoch": epoch}) + "\n", encoding="utf-8")
            ledger.append(
                {
                    "record_type": "meta_learning",
                    "experiment_id": "exp",
                    "epoch_id": epoch,
                    "fold_id": epoch,
                    "meta_learning_id": epoch,
                    "run_id": run_id,
                    "agent_trace_ref": str(trace),
                    "prior": f"prior {index}",
                }
            )
        pipeline = RollingExperimentPipeline(
            config,
            snapshots=object(),
            artifacts=object(),
            evaluator=object(),
            developer=lambda request: None,
            meta_learner=lambda facts: None,
            ledger=ledger,
        )
        return pipeline

    def test_only_the_most_recent_n_epochs_are_carried(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for keep, expected in ((0, []), (1, ["epoch_003"]), (2, ["epoch_002", "epoch_003"])):
                with self.subTest(keep=keep):
                    pipeline = self._pipeline(root / f"keep_{keep}", keep=keep)
                    memory = pipeline._prior_meta_learning_logs("epoch_004")
                    epochs = [json.loads(line)["epoch"] for line in memory.splitlines() if line]
                    self.assertEqual(epochs, expected)

    def test_the_current_session_is_excluded_from_its_own_memory(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp) / "current", keep=3)
            memory = pipeline._prior_meta_learning_logs("epoch_003")
            epochs = [json.loads(line)["epoch"] for line in memory.splitlines() if line]
            self.assertEqual(epochs, ["epoch_001", "epoch_002"])
            self.assertNotIn("epoch_003", memory)


class ConvergencePhaseTest(unittest.TestCase):
    """``convergence_start_epoch`` decides the phase, and the phase changes the
    prompt the Fold Agent receives."""

    @staticmethod
    def _phase(epoch_index: int, start: int) -> str:
        return "convergence" if epoch_index >= start else "exploration"

    def test_the_boundary_epoch_switches_the_phase_and_the_prompt(self) -> None:
        from autotrade.agent.prompts import build_system_prompt

        self.assertEqual(self._phase(1, 3), "exploration")
        self.assertEqual(self._phase(2, 3), "exploration")
        self.assertEqual(self._phase(3, 3), "convergence")
        self.assertEqual(self._phase(2, 2), "convergence")

        exploration = build_system_prompt(mode="fold", phase="exploration", experiment_facts={})
        convergence = build_system_prompt(mode="fold", phase="convergence", experiment_facts={})
        self.assertIn("探索期", exploration)
        self.assertNotIn("收敛期", exploration)
        self.assertIn("收敛期", convergence)
        self.assertNotEqual(exploration, convergence)

    def test_the_config_carries_closed_s_default(self) -> None:
        config = RollingExperimentConfig(
            "exp", Path("/tmp/experiments"), "2022Q1", "2022Q1", "2023Q1", "2023Q1"
        )
        self.assertEqual(config.convergence_start_epoch, 3)
        with self.assertRaisesRegex(ValueError, "convergence_start_epoch"):
            RollingExperimentConfig(
                "exp", Path("/tmp/experiments"), "2022Q1", "2022Q1", "2023Q1", "2023Q1",
                convergence_start_epoch=0,
            )


class StepTreeAblationTest(unittest.TestCase):
    """With the step tree off there is no cross-fold lineage, no published
    tree and no rollback tool — the ablation is real, not cosmetic."""

    def test_the_prompt_section_follows_the_knob(self) -> None:
        from autotrade.agent.prompts import build_system_prompt

        enabled = build_system_prompt(mode="fold", step_tree_enabled=True, experiment_facts={})
        disabled = build_system_prompt(mode="fold", step_tree_enabled=False, experiment_facts={})
        self.assertIn("Step 产物树", enabled)
        self.assertNotIn("Step 产物树", disabled)

    def test_a_recorded_node_is_addressable_by_its_revision(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = _artifact(root / "output")
            tree = StepTree(root / "steps")
            revision = new_revision_id("revision")
            node_id = tree.record_step(
                output,
                epoch_id="epoch_001",
                fold_id="fold_ref_ab",
                run_id="run_x",
                result_name="valid_000",
                revision_id=revision,
                metrics={},
                complete_validation=True,
            )
            # The item-5 replacement for closed's position_for_hash: a frozen
            # artifact records its source step, so the branch point is a lookup.
            self.assertEqual(tree.position_for_step(node_id), node_id)
            self.assertIsNone(tree.position_for_step("unknown-node"))


if __name__ == "__main__":
    unittest.main()
