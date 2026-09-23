"""Every configurable knob must change behaviour, not merely be accepted.

Half this repository's defects were parameters that existed and did nothing —
``image_keep``, ``record_failed_attempts``,
the ``ModificationConstraints`` that never reached the
tool. A test asserting a knob is accepted is the bug; each test here changes
the knob and asserts the observable difference.
"""

from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from autotrade.environment.artifacts import (
    ModificationConstraints,
    copy_artifact,
    new_revision_id,
)
from autotrade.environment.data.contracts import DEFAULT_BENCHMARK_INDEX
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.step_tree import StepTree
from autotrade.environment.tools import ModificationCheckTool, ToolRegistry
from autotrade.pipelines.config import RollingExperimentConfig
from autotrade.pipelines.ledger import ExperimentLedger

from .fixtures_sandbox import PassingModificationCheck

VALID_MAIN = "def generate_orders(context):\n    return []\n"
# A real host path: the redaction recognises the roots the host keeps its files
# under, and a pytest temp directory is not one of them.
HOST_RESULT_PATH = f"{Path(__file__).resolve().parents[2]}/experiments/arm/private/result.json"


def _artifact(root: Path, *, extra_lines: int = 0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("readonly\n", encoding="utf-8")
    body = VALID_MAIN + "".join(f"# line {index}\n" for index in range(extra_lines))
    (root / "main.py").write_text(body, encoding="utf-8")
    return root


class ModificationConstraintsReachTheToolTest(unittest.TestCase):
    """The configured limits are enforced, not hardcoded literals."""

    def test_a_configured_size_cap_actually_rejects(self) -> None:
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
            # The rewrite is reported, never refused: the delta is information.
            self.assertEqual(generous.value["delta"]["diff_lines"], 50)

            strict = ToolRegistry(
                [
                    ModificationCheckTool(
                        work,
                        parent_dir=parent,
                        constraints=ModificationConstraints(max_strategy_bytes=10),
                    )
                ]
            ).invoke("modification_check", {})
            self.assertFalse(strict.ok)
            self.assertIn("exceeds 10 bytes", strict.error)

    def test_the_default_is_the_dataclass_not_a_literal(self) -> None:
        with TemporaryDirectory() as tmp:
            tool = ModificationCheckTool(_artifact(Path(tmp) / "output"))
            self.assertEqual(tool.constraints, ModificationConstraints())


class RecordFailedAttemptsTest(unittest.TestCase):
    """With the knob off, a failed validation leaves no dead-end node.

    Driven through the real ``SessionValidations``: the gate lives at its
    exception path, and asserting the knob's value would prove nothing."""

    def _tool(
        self,
        root: Path,
        *,
        record_failed_attempts: bool,
        error: Exception | None = None,
    ):
        from autotrade.environment.artifacts import FilesystemArtifactStore
        from autotrade.environment.broker import BrokerProfile
        from autotrade.environment.strategy import StrategySchedule
        from autotrade.environment.time_budget import InferenceTimeBudget
        from autotrade.environment.tools import SafeWorkspace
        from autotrade.pipelines.config import (
            ReplaySpan,
            ResearchSessionRequest,
            SnapshotBundle,
        )
        from autotrade.pipelines.session_tools import (
            BatchValidateTool,
            SessionValidations,
        )

        output = _artifact(root / "output")
        models = root / "models"
        models.mkdir(parents=True, exist_ok=True)
        tree = StepTree(root / "steps")
        snapshot = SnapshotBundle("snap", "decision", "replay")
        request = ResearchSessionRequest(
            experiment_id="exp",
            run_id="run_x",
            snapshot=snapshot,
            decision_time=datetime(2025, 6, 30, 23, 59, 59, tzinfo=UTC),
            research_years=(ReplaySpan("Y1", "valid", "20210701", "20220630", snapshot),),
            window_months=24,
            max_replay_years=30,
            max_llm_calls=200,
            deadline_seconds=1200.0,
            record_failed_attempts=record_failed_attempts,
            benchmark_index=DEFAULT_BENCHMARK_INDEX,
        )

        raised = error or RuntimeError(f"validation blew up at {HOST_RESULT_PATH}")

        class ExplodingEvaluator:
            def evaluate(self, _request):
                raise raised

        backtest = SessionValidations(
            request=request,
            output_dir=output,
            models_dir=models,
            artifact_store=FilesystemArtifactStore(root / "artifacts"),
            evaluator=ExplodingEvaluator(),
            tree=tree,
            schedule=StrategySchedule(),
            broker_profile=BrokerProfile(),
            time_budget=InferenceTimeBudget(duration_seconds=300),
            ref_store=AgentRefStore(root / "experiment"),
            ledger=ExperimentLedger(root / "ledger.jsonl"),
        )
        batch = BatchValidateTool(
            backtest=backtest,
            workspace=SafeWorkspace(root),
            modification_check_factory=lambda directory: PassingModificationCheck(
                directory, models
            ),
        )
        return (
            lambda: batch.invoke(
                {
                    "offline_trials": 0,
                    "candidates": [
                        {"name": "wc", "hypothesis": "h", "path": "output", "control": False}
                    ],
                }
            ),
            tree,
        )

    def test_a_failed_validation_records_a_dead_end_only_when_enabled(self) -> None:
        from autotrade.environment.tools.base import ToolError

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            off_tool, off_tree = self._tool(root / "off", record_failed_attempts=False)
            with self.assertRaises(ToolError):
                off_tool()
            self.assertEqual(off_tree.nodes(), [])

            on_tool, on_tree = self._tool(root / "on", record_failed_attempts=True)
            with self.assertRaises(ToolError):
                on_tool()
            nodes = on_tree.nodes()
            self.assertEqual([node["status"] for node in nodes], ["failed"])
            self.assertEqual(
                nodes[0]["error"],
                "daily Validation failed: RuntimeError: validation blew up at [host_path]",
            )
            self.assertNotIn(str(root), json.dumps(nodes))
            self.assertNotIn("private/result.json", json.dumps(nodes))
            # The dead end never becomes the working position.
            self.assertIsNone(on_tree.current_node_id)

    def test_failed_validation_exposes_unsupported_os_import(self) -> None:
        from autotrade.environment.strategy_loader import StrategyLoadError
        from autotrade.environment.tools.base import ToolError

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tool, tree = self._tool(
                root,
                record_failed_attempts=True,
                error=StrategyLoadError("strategy imports unsupported module: os"),
            )
            with self.assertRaises(ToolError) as caught:
                tool()
            message = str(caught.exception.details["candidates"][0]["error"])
            self.assertEqual(
                message,
                "daily Validation failed: StrategyLoadError: strategy imports unsupported module: os",
            )
            recorded = str(tree.nodes()[0]["error"])
            self.assertIn("strategy imports unsupported module: os", recorded)
            self.assertNotIn(str(root), message)

    def test_failed_validation_exposes_replay_slot_mode_mismatch(self) -> None:
        from autotrade.environment.tools.base import ToolError

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tool, tree = self._tool(
                root,
                record_failed_attempts=True,
                error=ValueError(
                    "EvaluationRequest mode does not match its immutable replay slot"
                ),
            )
            with self.assertRaises(ToolError) as caught:
                tool()
            message = str(caught.exception.details["candidates"][0]["error"])
            self.assertEqual(
                message,
                "daily Validation failed: ValueError: EvaluationRequest mode does not match its immutable replay slot",
            )
            recorded = str(tree.nodes()[0]["error"])
            self.assertIn(
                "EvaluationRequest mode does not match its immutable replay slot",
                recorded,
            )
            self.assertNotIn(str(root), message)

    def test_failed_validation_redacts_host_paths(self) -> None:
        from autotrade.environment.tools.base import ToolError

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tool, tree = self._tool(
                root,
                record_failed_attempts=True,
                error=RuntimeError(f"replay failed at {HOST_RESULT_PATH}"),
            )
            with self.assertRaises(ToolError) as caught:
                tool()
            message = str(caught.exception.details["candidates"][0]["error"])
            recorded = str(tree.nodes()[0]["error"])
            self.assertEqual(message, recorded)
            self.assertEqual(
                message, "daily Validation failed: RuntimeError: replay failed at [host_path]"
            )

    def test_the_config_default_records_them(self) -> None:
        config = RollingExperimentConfig("exp", Path("/tmp/experiments"))
        self.assertTrue(config.record_failed_attempts)


class StepTreeNodeTest(unittest.TestCase):
    def test_a_recorded_node_is_addressable_by_its_revision(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = _artifact(root / "output")
            tree = StepTree(root / "steps")
            revision = new_revision_id("revision")
            node_id = tree.record_step(
                output,
                epoch_id="epoch_001",
                session_ref="session_ref_ab",
                run_id="run_x",
                result_name="valid_000",
                revision_id=revision,
                metrics={},
            )
            # The item-5 replacement for closed's position_for_hash: a frozen
            # artifact records its source step, so the branch point is a lookup.
            self.assertEqual(tree.position_for_step(node_id), node_id)
            self.assertIsNone(tree.position_for_step("unknown-node"))


if __name__ == "__main__":
    unittest.main()
