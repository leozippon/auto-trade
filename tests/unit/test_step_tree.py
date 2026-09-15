import json
import tempfile
import unittest
from pathlib import Path

from autotrade.agent.experiment_facts import build_experiment_facts
from autotrade.agent.prompts import build_system_prompt
from autotrade.environment.artifacts import new_revision_id
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import RunManifest
from autotrade.environment.step_tree import StepTree

from .test_artifacts import write_artifact

NODE = dict(epoch_id="epoch_001", run_id="run_x")


class StepTreeTest(unittest.TestCase):
    def test_records_nodes_with_parent_lineage_and_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            artifact = write_artifact(tmp / "artifact")
            revision = new_revision_id("revision")
            tree = StepTree(tmp / "steps")
            node1 = tree.record_step(
                artifact,
                session_ref="session_ref_ab",
                result_name="valid_000",
                revision_id=revision,
                metrics={"total_return": 0.01},
                **NODE,
            )
            node2 = tree.record_step(
                artifact,
                session_ref="session_ref_ab",
                result_name="valid_001",
                revision_id=revision,
                metrics={"total_return": 0.02},
                **NODE,
            )
            reloaded = StepTree(tmp / "steps")
            self.assertEqual(reloaded.current_node_id, node2)
            nodes = {node["node_id"]: node for node in reloaded.nodes()}
            self.assertIsNone(nodes[node1]["parent_node_id"])
            self.assertEqual(nodes[node2]["parent_node_id"], node1)
            self.assertTrue((tmp / "steps" / node1 / "output" / "main.py").exists())
            self.assertEqual(nodes[node2]["revision_id"], revision)
            rendered = reloaded.render_ascii()
            self.assertIn(node1, rendered)
            self.assertIn("<- current", rendered)
            with self.assertRaisesRegex(ValueError, "already exists"):
                reloaded.record_step(
                    artifact,
                    session_ref="session_ref_ab",
                    result_name="valid_000",
                    revision_id=revision,
                    metrics={},
                    **NODE,
                )

    def test_run_id_prevents_rerun_node_collisions(self):
        # result_name restarts at valid_000 in every run; a fold re-executed
        # (rerun_fold / post-rollback) must not collide with its earlier run.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            artifact = write_artifact(tmp / "artifact")
            tree = StepTree(tmp / "steps")
            kwargs = dict(
                epoch_id="epoch_001",
                session_ref="session_ref_ab",
                result_name="valid_000",
                revision_id=new_revision_id("revision"),
                metrics={},
            )
            node1 = tree.record_step(artifact, run_id="run_x", **kwargs)
            node2 = tree.record_step(artifact, run_id="run_y", **kwargs)
            self.assertNotEqual(node1, node2)
            self.assertEqual(node1, "epoch_001__session_ref_ab__run_x__valid_000")
            with self.assertRaisesRegex(ValueError, "already exists"):
                tree.record_step(artifact, run_id="run_y", **kwargs)

    def test_epoch_id_prevents_cross_epoch_node_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            artifact = write_artifact(tmp / "artifact")
            tree = StepTree(tmp / "steps")
            kwargs = dict(
                session_ref="session_ref_ab",
                run_id="run_x",
                result_name="valid_000",
                revision_id=new_revision_id("revision"),
                metrics={},
            )
            node1 = tree.record_step(artifact, epoch_id="epoch_001", **kwargs)
            node2 = tree.record_step(artifact, epoch_id="epoch_002", **kwargs)
            self.assertNotEqual(node1, node2)
            self.assertIn("epoch_001", node1)
            self.assertIn("epoch_002", node2)

    def test_set_position_validates_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = StepTree(Path(tmp) / "steps")
            with self.assertRaisesRegex(ValueError, "unknown"):
                tree.set_position("nope")
            tree.set_position(None)
            self.assertIsNone(tree.current_node_id)

    def test_failed_attempt_is_dead_end_without_moving_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            artifact = write_artifact(tmp / "artifact")
            tree = StepTree(tmp / "steps")
            good = tree.record_step(
                artifact,
                session_ref="session_ref_ab",
                result_name="valid_000",
                revision_id=new_revision_id("revision"),
                metrics={"total_return": 0.01},
                **NODE,
            )
            failed = tree.record_failed_attempt(
                epoch_id="epoch_001",
                session_ref="session_ref_ab",
                run_id="run_x",
                result_name="failed_abc",
                error="boom",
            )
            reloaded = StepTree(tmp / "steps")
            # A failed attempt never becomes the working position or a parent.
            self.assertEqual(reloaded.current_node_id, good)
            nodes = {node["node_id"]: node for node in reloaded.nodes()}
            self.assertEqual(nodes[failed]["parent_node_id"], good)
            self.assertFalse(nodes[failed]["complete_validation"])
            self.assertEqual(nodes[failed]["error"], "boom")
            self.assertIsNone(nodes[failed]["revision_id"])
            # No output snapshot is copied for a dead end.
            self.assertFalse((tmp / "steps" / failed).exists())

    def test_save_writes_readable_rendering_with_failed_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            artifact = write_artifact(tmp / "artifact")
            tree = StepTree(tmp / "steps")
            tree.record_step(
                artifact,
                session_ref="session_ref_ab",
                result_name="valid_000",
                revision_id=new_revision_id("revision"),
                metrics={"total_return": 0.01, "sharpe": 1.5},
                **NODE,
            )
            tree.record_failed_attempt(
                epoch_id="epoch_001",
                session_ref="session_ref_ab",
                run_id="run_x",
                result_name="failed_abc",
                error="boom",
            )
            rendered = (tmp / "steps" / "tree.txt").read_text(encoding="utf-8")
            self.assertIn("valid_000", rendered)
            self.assertIn("ret=0.0100", rendered)
            self.assertIn("sharpe=1.5000", rendered)
            self.assertIn("[failed]", rendered)
            self.assertIn("<- current", rendered)

    def test_attachments_must_not_shadow_snapshot_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            artifact = write_artifact(tmp / "artifact")
            payload = tmp / "detail.json"
            payload.write_text("{}", encoding="utf-8")
            tree = StepTree(tmp / "steps")
            with self.assertRaisesRegex(ValueError, "shadow"):
                tree.record_step(
                    artifact,
                    session_ref="session_ref_ab",
                    result_name="valid_000",
                    revision_id=new_revision_id("revision"),
                    metrics={},
                    attachments={"output/x.json": payload},
                    **NODE,
                )
            for relpath in ("/absolute.json", "../escape.json"):
                with self.subTest(relpath=relpath), self.assertRaisesRegex(ValueError, "invalid step attachment"):
                    tree.record_step(
                        artifact,
                        session_ref="session_ref_ab",
                        result_name="valid_escape",
                        revision_id=new_revision_id("revision"),
                        metrics={},
                        attachments={relpath: payload},
                        **NODE,
                    )
            node_id = tree.record_step(
                artifact,
                session_ref="session_ref_ab",
                result_name="valid_001",
                revision_id=new_revision_id("revision"),
                metrics={},
                attachments={"detail.json": payload},
                **NODE,
            )
            stored = tree.get_node(node_id)["attachments"]["detail.json"]
            self.assertFalse(Path(stored).is_absolute())
            self.assertEqual(stored, f"{node_id}/detail.json")

    def test_failed_attempt_error_is_redacted_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tree = StepTree(tmp / "steps")
            tree.record_failed_attempt(
                epoch_id="epoch_001",
                session_ref="session_ref_ab",
                run_id="run_x",
                result_name="failed_secret",
                error="failed Authorization: Bearer secret-token-abc",
            )
            raw = (tmp / "steps" / "tree.json").read_text(encoding="utf-8")
            self.assertNotIn("secret-token-abc", raw)
            payload = json.loads(raw)
            self.assertIn("redacted", payload["nodes"][0]["error"].lower())


RESEARCH_MANIFEST = {
    "experiment_id": "exp",
    "run_id": "run_x",
    "epoch_id": "research",
    "fold_id": "s2",
    "kind": "research",
    "session": {"index": 2, "of": 4, "last": False},
    "research": {
        "decision_time": "2025-06-30T23:59:59+08:00",
        "input_window": "20230701..20250630",
        "research_period": "20210701..20250630",
        "years": [
            {"label": "Y1", "start": "20210701", "end": "20220630"},
            {"label": "Y2", "start": "20220701", "end": "20230630"},
            {"label": "Y3", "start": "20230701", "end": "20240630"},
            {"label": "Y4", "start": "20240701", "end": "20250630"},
        ],
        "spans": "full (every year), one year such as Y1, or contiguous years such as Y1..Y4",
    },
    "start": {"kind": "step_node", "node_id": "research__session_ref_x__run_ref_y__valid_003"},
    "arm": {"frozen": False, "freezes_per_arm": 1, "trials_to_date": 5, "full_span_validations_to_date": 2},
    "snapshot_config": {
        "decision_windows": {
            "daily_months": 24,
            "fundamentals_months": 24,
            "events_months": 24,
            "macro_months": 24,
            "text_months": 24,
            "intraday_trade_days": 21,
        }
    },
    "acceptance_rules": {"min_return": 0.0},
    "budgets": {"max_replay_years": 24, "max_null_controls": 3, "deadline_seconds": 43800.0},
}


class SessionFactsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._refs_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._refs_tmp.cleanup)
        self.ref_store = AgentRefStore(Path(self._refs_tmp.name) / "experiment")

    def test_facts_carry_the_research_geometry_and_the_arm_but_no_later_date(self):
        facts = build_experiment_facts(
            manifest=RESEARCH_MANIFEST,
            ref_store=self.ref_store,
            runtime_env={"python": {"version": "3.11"}, "tools": {"rg": {"available": True}}},
            data_summary={"views": {"snapshot": {"mount_path": "/mnt/snapshot", "files": []}}},
            max_llm_calls=10,
            context_compaction={"enabled": True, "token_threshold": 200000, "max_calls": 8},
            model_artifacts_empty=True,
        )
        prompt = build_system_prompt(experiment_facts=facts)

        self.assertIn("当前实验事实", prompt)
        self.assertEqual(facts["research_geometry"], RESEARCH_MANIFEST["research"])
        self.assertEqual(facts["identity"]["session"], {"index": 2, "of": 4, "last": False})
        self.assertTrue(facts["identity"]["session_ref"].startswith("session_ref_"))
        self.assertNotIn('"s2"', prompt)
        self.assertEqual(facts["arm"]["trials_to_date"], 5)
        self.assertIs(facts["arm"]["frozen"], False)
        self.assertEqual(facts["budgets"]["max_replay_years"], 24)
        self.assertIn("replay-year", facts["budgets"]["max_replay_years_note"])
        self.assertEqual(
            facts["artifact_contract"]["start"],
            {
                "kind": "step_node",
                "node_id": "research__session_ref_x__run_ref_y__valid_003",
                "model_artifacts_empty": True,
            },
        )
        self.assertIn("freeze_gate", facts["artifact_contract"]["acceptance_rules"])
        self.assertIn("20210701..20250630", facts["research_scope"]["research"])
        self.assertIn("session 2 of 4", facts["research_scope"]["research"])
        # The periods after research end exist and are sealed: no date of them.
        for later in ("2025-07", "202507", "2026", "20260630", "20260930"):
            self.assertNotIn(later, prompt)
        execution = facts["visible_timeline"]["execution_policy"]
        self.assertEqual(execution["strategy_clock"], "configured_schedule_only")
        self.assertFalse(execution["historical_minutes_drive_strategy"])
        self.assertEqual(execution["missing_exact_price"], "reject")

    def test_no_fact_key_names_a_fold_a_parent_or_a_hidden_stage(self):
        facts = build_experiment_facts(manifest=RESEARCH_MANIFEST, ref_store=self.ref_store)

        def keys(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield str(key)
                    yield from keys(item)
            elif isinstance(value, list):
                for item in value:
                    yield from keys(item)

        names = set(keys(facts))
        for retired in (
            "fold",
            "fold_period",
            "validation_periods",
            "parent",
            "parent_control_available",
            "confirmation_fold",
            "test_visible",
            "heldout_visible",
            "development_window",
            "max_backtests_per_fold",
            "max_steps",
        ):
            self.assertNotIn(retired, names)
        self.assertNotIn("data_profile", facts)
        self.assertNotIn("paths", facts)

    def test_run_manifest_public_view_keeps_research_keys_and_drops_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            public_path = Path(tmp) / "artifacts" / "run_manifest.json"
            manifest = RunManifest.create(
                public_path,
                {
                    **RESEARCH_MANIFEST,
                    "test_decision_time": "2025-07-04T09:25:00+08:00",
                    "snapshots": {
                        "decision_input": {"snapshot_id": "decision"},
                        "heldout_replay": {"snapshot_id": "heldout_replay"},
                    },
                    "experiment_parameters": {"heldout_end": "20260930"},
                    "backtest_summaries": [
                        {"mode": "valid", "span": "Y2", "total_return": 0.1},
                        {"mode": "heldout", "total_return": 0.2},
                    ],
                },
                ref_store=AgentRefStore(Path(tmp) / "experiment"),
            )

            public = json.loads(public_path.read_text(encoding="utf-8"))
            host = json.loads(manifest.host_path.read_text(encoding="utf-8"))

            self.assertNotIn("test_decision_time", public)
            self.assertNotIn("experiment_parameters", public)
            self.assertNotIn("heldout_replay", public["snapshots"])
            self.assertEqual(public["backtest_summaries"], [{"mode": "valid", "span": "Y2", "total_return": 0.1}])
            for key in ("research", "session", "start", "arm", "budgets"):
                self.assertEqual(public[key], RESEARCH_MANIFEST[key], key)
            # The raw session id never crosses, only its opaque ref.
            self.assertTrue(str(public["session_ref"]).startswith("session_ref_"))
            self.assertEqual(host["fold_id"], "s2")
            self.assertEqual(host["experiment_parameters"]["heldout_end"], "20260930")


class PromptCompositionTest(unittest.TestCase):
    """The session prompt is assembled from a stable contract plus per-run context.

    The split matters for provider prompt caching and for the Agent's own
    reading of what is fixed versus what changed this session, so each
    injectable is asserted to appear, to be omitted when empty, and to sit in
    the right half of the prompt.
    """

    MARKER = "# 本会话动态上下文"

    def test_the_step_tree_section_is_a_per_experiment_knob(self) -> None:
        without = build_system_prompt()
        self.assertNotIn("# Step 产物树", without)
        with_tree = build_system_prompt(step_tree_enabled=True)
        self.assertIn("# Step 产物树", with_tree)
        self.assertIn("step_rollback", with_tree)
        self.assertIn("finish_session", with_tree)
        self.assertLess(with_tree.index("# Step 产物树"), with_tree.index(self.MARKER))

    def test_the_session_directive_is_optional_and_framed_as_a_hypothesis(self) -> None:
        without = build_system_prompt()
        self.assertNotIn("研究者本会话指令", without)
        with_directive = build_system_prompt(session_directive="优先检验行业中性化后的动量残差。")
        self.assertIn("研究者本会话指令", with_directive)
        self.assertIn("优先检验行业中性化后的动量残差。", with_directive)
        # A directive never relaxes the hard contract; it enters as a hypothesis
        # inside the dynamic half.
        self.assertGreater(with_directive.index("研究者本会话指令"), with_directive.index(self.MARKER))
        # Whitespace-only directives collapse to the no-section prompt.
        self.assertEqual(build_system_prompt(session_directive="  \n"), without)

    def test_the_experiment_level_exploration_direction_is_additive(self) -> None:
        prompt = build_system_prompt(
            exploration_directive="持续检验事件冲击的图传播。",
            session_directive="本会话先做行业边消融。",
        )
        # The standing experiment direction precedes the per-session hypothesis.
        self.assertLess(prompt.index("持续检验事件冲击的图传播。"), prompt.index("本会话先做行业边消融。"))
        self.assertEqual(build_system_prompt(exploration_directive="   "), build_system_prompt())

    def test_the_static_contract_is_byte_identical_across_two_different_sessions(self) -> None:
        first = build_system_prompt(
            experiment_facts={"identity": {"run_id": "run_1"}},
            prior_prompt="方向 A",
            exploration_directive="长期假设 A",
            session_directive="当前假设 A",
        )
        second = build_system_prompt(
            experiment_facts={"identity": {"run_id": "run_2"}},
            prior_prompt="方向 B",
            exploration_directive="长期假设 B",
            session_directive="当前假设 B",
        )
        first_prefix, first_context = first.split(self.MARKER, 1)
        second_prefix, second_context = second.split(self.MARKER, 1)
        self.assertEqual(first_prefix, second_prefix)
        self.assertNotEqual(first_context, second_context)
        # Purpose, protocol, decision contract, evidence, constraints, facts,
        # feedback -- then the per-run context.
        order = [
            first.index("# 身份与任务"),
            first.index("# 研究协议"),
            first.index("# 决策合同"),
            first.index("# 证据标准"),
            first.index("# 原则"),
            first.index("# 工具与工作方式"),
            first.index("# 角色与写权"),
            first.index("# 执行合同与边界"),
            first.index("# 禁止事项"),
            first.index("# 预算与事实"),
            first.index("# 反馈通道"),
            first.index(self.MARKER),
        ]
        self.assertEqual(order, sorted(order))

    def test_the_prior_rides_in_its_own_section_not_in_the_facts_blob(self) -> None:
        prompt = build_system_prompt(prior_prompt="偏好小步修改")
        prefix, dynamic = prompt.split(self.MARKER, 1)
        self.assertIn("偏好小步修改", dynamic)
        self.assertNotIn("偏好小步修改", prefix)
        self.assertIn("上一会话的交接", dynamic)
