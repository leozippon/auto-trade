import json
import tempfile
import unittest
from pathlib import Path

from autotrade.agent.experiment_facts import build_experiment_facts
from autotrade.agent.prompts import build_system_prompt
from autotrade.environment.artifacts import new_revision_id
from autotrade.environment.runtime import RunManifest
from autotrade.environment.step_tree import StepTree

from .test_artifacts import write_artifact

NODE = dict(epoch_id="epoch_001", run_id="run_x", complete_validation=True)


class StepTreeTest(unittest.TestCase):
    def test_records_nodes_with_parent_lineage_and_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            artifact = write_artifact(tmp / "artifact")
            revision = new_revision_id("revision")
            tree = StepTree(tmp / "steps")
            node1 = tree.record_step(
                artifact,
                fold_id="fold_ref_ab",
                result_name="valid_000",
                revision_id=revision,
                metrics={"total_return": 0.01},
                **NODE,
            )
            node2 = tree.record_step(
                artifact,
                fold_id="fold_ref_ab",
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
                    fold_id="fold_ref_ab",
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
                fold_id="fold_ref_ab",
                result_name="valid_000",
                revision_id=new_revision_id("revision"),
                metrics={},
                complete_validation=True,
            )
            node1 = tree.record_step(artifact, run_id="run_x", **kwargs)
            node2 = tree.record_step(artifact, run_id="run_y", **kwargs)
            self.assertNotEqual(node1, node2)
            self.assertEqual(node1, "epoch_001__fold_ref_ab__run_x__valid_000")
            with self.assertRaisesRegex(ValueError, "already exists"):
                tree.record_step(artifact, run_id="run_y", **kwargs)

    def test_epoch_id_prevents_cross_epoch_node_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            artifact = write_artifact(tmp / "artifact")
            tree = StepTree(tmp / "steps")
            kwargs = dict(
                fold_id="fold_ref_ab",
                run_id="run_x",
                result_name="valid_000",
                revision_id=new_revision_id("revision"),
                metrics={},
                complete_validation=True,
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
                fold_id="fold_ref_ab",
                result_name="valid_000",
                revision_id=new_revision_id("revision"),
                metrics={"total_return": 0.01},
                **NODE,
            )
            failed = tree.record_failed_attempt(
                epoch_id="epoch_001",
                fold_id="fold_ref_ab",
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
                fold_id="fold_ref_ab",
                result_name="valid_000",
                revision_id=new_revision_id("revision"),
                metrics={"total_return": 0.01, "sharpe": 1.5},
                **NODE,
            )
            tree.record_failed_attempt(
                epoch_id="epoch_001",
                fold_id="fold_ref_ab",
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
                    fold_id="fold_ref_ab",
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
                        fold_id="fold_ref_ab",
                        result_name="valid_escape",
                        revision_id=new_revision_id("revision"),
                        metrics={},
                        attachments={relpath: payload},
                        **NODE,
                    )
            node_id = tree.record_step(
                artifact,
                fold_id="fold_ref_ab",
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
                fold_id="fold_ref_ab",
                run_id="run_x",
                result_name="failed_secret",
                error="failed Authorization: Bearer secret-token-abc",
            )
            raw = (tmp / "steps" / "tree.json").read_text(encoding="utf-8")
            self.assertNotIn("secret-token-abc", raw)
            payload = json.loads(raw)
            self.assertIn("redacted", payload["nodes"][0]["error"].lower())


class PhasePromptTest(unittest.TestCase):
    def test_experiment_facts_replace_raw_fold_schedule(self):
        manifest = {
            "experiment_id": "exp",
            "run_id": "run_x",
            "epoch_id": "epoch_001",
            "fold_id": "fold_2022Q1",
            "kind": "fold",
            "fold": {
                "input_window": "20200101..20210930",
                "validation_period": "20211001..20211231",
                "test_period": "20220101..20220331",
                "test_decision_time": "2022-01-04T09:25:00+08:00",
            },
            "fold_period": "quarter",
            "valid_decision_time": "2021-10-08T09:25:00+08:00",
            "snapshot_config": {
                "decision_windows": {
                    "daily_months": 21,
                    "fundamentals_months": 21,
                    "events_months": 21,
                    "macro_months": 21,
                    "text_months": 21,
                    "intraday_trade_days": 21,
                }
            },
            "acceptance_rules": {"min_return": 0.0},
        }
        facts = build_experiment_facts(
            manifest=manifest,
            runtime_env={"python": {"version": "3.11"}, "tools": {"rg": {"available": True}}},
            data_summary={"views": {"snapshot": {"mount_path": "/mnt/snapshot", "files": []}}},
            max_llm_calls=10,
            context_compaction={"enabled": True, "token_threshold": 200000, "max_calls": 8},
            model_artifacts_empty=True,
        )

        prompt = build_system_prompt(
            fold_info=manifest["fold"],
            acceptance_rules={"min_return": 0.0},
            experiment_facts=facts,
        )

        self.assertIn("当前实验事实", prompt)
        self.assertIn("hidden_schedule_redacted", prompt)
        self.assertIn("fold_ref_", prompt)
        self.assertNotIn("fold_2022Q1", prompt)
        self.assertNotIn("test_period", prompt)
        self.assertNotIn("test_decision_time", prompt)
        self.assertNotIn("20220101..20220331", prompt)
        execution = facts["visible_timeline"]["execution_policy"]
        self.assertEqual(execution["strategy_clock"], "configured_schedule_only")
        self.assertFalse(execution["historical_minutes_drive_strategy"])
        self.assertEqual(execution["missing_exact_price"], "reject")

    def test_unit_contract_and_test_visibility_are_explicit_by_agent_kind(self):
        unit_contract = {
            "daily.parquet": {"pct_chg_turnover_dv": "decimal; 5%=0.05"},
            "events.parquet": {"moneyflow.*_amount": "10k_CNY; CNY 5m=500"},
        }
        fold_facts = build_experiment_facts(
            manifest={"kind": "fold", "fold_id": "fold_x"},
            data_summary={"unit_contract": unit_contract},
        )
        meta_facts = build_experiment_facts(
            manifest={
                "kind": "meta_learning",
                "epoch_id": "epoch_001",
                "meta_learning_id": "epoch_001_after_fold_003",
                "trigger_after_folds": 3,
                "fold_exploration_directive": "event graph",
            },
            data_summary={"unit_contract": unit_contract},
        )

        # The facts builder no longer produces the always-dropped data-profile
        # / paths sections (the unit contract reaches the Agent via
        # data_summary.json, not via prompt facts).
        self.assertNotIn("data_profile", fold_facts)
        self.assertNotIn("paths", fold_facts)
        self.assertFalse(fold_facts["visibility_policy"]["historical_frozen_test_metrics_visible"])
        self.assertTrue(meta_facts["visibility_policy"]["historical_frozen_test_metrics_visible"])
        self.assertFalse(meta_facts["visibility_policy"]["test_visible"])
        self.assertFalse(meta_facts["visibility_policy"]["heldout_visible"])
        self.assertEqual(meta_facts["identity"]["meta_learning_id"], "epoch_001_after_fold_003")
        self.assertEqual(meta_facts["identity"]["trigger_after_folds"], 3)
        self.assertTrue(meta_facts["meta_learning"]["fold_exploration_directive_present"])

    def test_fold_facts_opaque_parent_artifact_id(self):
        # Frozen artifact ids embed the raw fold label of the fold that produced
        # them (strategy_<epoch>_fold_<period>), so the facts must project them.
        facts = build_experiment_facts(
            manifest={
                "experiment_id": "exp",
                "run_id": "run_2",
                "epoch_id": "epoch_001",
                "fold_id": "fold_2022Q2",
                "kind": "fold",
                "is_initial_artifact": False,
                "parent_strategy_artifact_id": "strategy_epoch_001_fold_2022Q1",
            }
        )
        rendered = json.dumps(facts, ensure_ascii=False, sort_keys=True)
        parent = facts["artifact_contract"]["parent"]
        self.assertTrue(str(parent["id"]).startswith("strategy_ref_"))
        self.assertNotIn("fold_2022Q1", rendered)
        self.assertNotIn("fold_2022Q2", rendered)

    def test_meta_experiment_facts_do_not_inline_sample_dates(self):
        manifest = {
            "experiment_id": "exp",
            "run_id": "run_meta",
            "epoch_id": "epoch_001",
            "fold_id": "epoch_001_meta_learning",
            "kind": "meta_learning",
            "valid_decision_time": "2021-10-08T09:25:00+08:00",
            "experiment_parameters": {
                "fold_period": "quarter",
                "snapshot_config": {"decision_windows": {"daily_months": 21, "intraday_trade_days": 21}},
            },
            "development_inputs": {"development_history": "/mnt/agent/workspace/development_history.json"},
        }
        data_summary = {
            "views": {
                "snapshot": {
                    "mount_path": "/mnt/snapshot",
                    "decision_time": "2021-10-08T09:25:00+08:00",
                    "period_start": "20200101",
                    "period_end": "20210930",
                    "files": [
                        {
                            "path": "daily.parquet",
                            "mount_path": "/mnt/snapshot/daily.parquet",
                            "rows": 10,
                            "date_ranges": {"trade_date": {"min": "20200101", "max": "20210930"}},
                        }
                    ],
                }
            }
        }

        facts = build_experiment_facts(manifest=manifest, data_summary=data_summary)
        rendered = json.dumps(facts, ensure_ascii=False, sort_keys=True)

        self.assertIn("sample_window_only", rendered)
        self.assertNotIn("2021-10-08", rendered)
        self.assertNotIn("20200101", rendered)
        self.assertNotIn("20210930", rendered)


    def test_run_manifest_public_view_redacts_test_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            public_path = Path(tmp) / "artifacts" / "run_manifest.json"
            manifest = RunManifest.create(
                public_path,
                {
                    "kind": "fold",
                    "fold": {
                        "fold_id": "fold_2022Q1",
                        "input_window": "20200101..20210930",
                        "validation_period": "20211001..20211231",
                        "test_period": "20220101..20220331",
                        "test_decision_time": "2022-01-04T09:25:00+08:00",
                    },
                    "test_decision_time": "2022-01-04T09:25:00+08:00",
                    "max_backtests_per_fold": 30,
                    "snapshots": {
                        "valid_decision_input": {"snapshot_id": "valid"},
                        "valid_replay": {"snapshot_id": "valid_replay"},
                        "test_decision_input": {"snapshot_id": "test"},
                        "test_replay": {"snapshot_id": "test_replay"},
                    },
                    "experiment_parameters": {
                        "fold_period": "quarter",
                        "epochs": 3,
                        "periods": {"first_test_period": "2022Q1", "heldout_first_period": "2025Q1"},
                        "test_first_period": "2022Q1",
                        "heldout_periods": ["2025Q1", "2025Q2"],
                    },
                    "backtest_summaries": [
                        {"mode": "valid", "total_return": 0.1},
                        {"mode": "frozen_test", "total_return": 0.2},
                    ],
                },
            )

            public = json.loads(public_path.read_text(encoding="utf-8"))
            host = json.loads(manifest.host_path.read_text(encoding="utf-8"))

            self.assertNotIn("test_decision_time", public)
            self.assertNotIn("test_period", public["fold"])
            self.assertNotIn("test_decision_input", public["snapshots"])
            self.assertNotIn("test_replay", public["snapshots"])
            self.assertEqual([item["mode"] for item in public["backtest_summaries"]], ["valid"])
            # The raw fold label never crosses either, only its opaque ref.
            self.assertNotIn("fold_2022Q1", json.dumps(public, ensure_ascii=False))
            # experiment_parameters must be projected with the same test/held-out
            # stripping as snapshots: any test_* or heldout_* key is a schedule leak.
            self.assertNotIn("periods", public["experiment_parameters"])
            self.assertNotIn("test_first_period", public["experiment_parameters"])
            self.assertNotIn("heldout_periods", public["experiment_parameters"])
            self.assertEqual(public["experiment_parameters"]["fold_period"], "quarter")
            self.assertEqual(host["experiment_parameters"]["test_first_period"], "2022Q1")
            self.assertEqual(host["fold"]["test_period"], "20220101..20220331")
            self.assertIn("test_replay", host["snapshots"])
            # Budget config is pure (no test/held-out leak) and is asserted in the
            # prompt facts, so it must survive into the agent-visible manifest too.
            self.assertEqual(public["max_backtests_per_fold"], manifest.data["max_backtests_per_fold"])


if __name__ == "__main__":
    unittest.main()


class PromptCompositionTest(unittest.TestCase):
    """The Fold prompt is assembled from a stable contract plus per-run context.

    The split matters for provider prompt caching and for the Agent's own
    reading of what is fixed versus what changed this Fold, so each injectable
    is asserted to appear, to be omitted when empty, and to sit in the right
    half of the prompt.
    """

    BASE = dict(fold_info={"fold_id": "f"}, acceptance_rules={}, experiment_facts={})

    def test_phase_and_step_tree_sections(self) -> None:
        exploration = build_system_prompt(**self.BASE)
        self.assertIn("探索期", exploration)
        self.assertNotIn("收敛期", exploration)
        self.assertNotIn("Step 产物树", exploration)

        convergence = build_system_prompt(**self.BASE, phase="convergence", step_tree_enabled=True)
        self.assertIn("收敛期", convergence)
        self.assertNotIn("探索期", convergence)
        self.assertIn("Step 产物树", convergence)
        self.assertIn("step_rollback", convergence)
        self.assertIn("finish_fold", convergence)

    def test_fold_directive_is_optional_and_framed_as_a_hypothesis(self) -> None:
        without = build_system_prompt(**self.BASE)
        self.assertNotIn("研究者本 Fold 指令", without)
        with_directive = build_system_prompt(
            **self.BASE, fold_directive="优先检验行业中性化后的动量残差。"
        )
        self.assertIn("研究者本 Fold 指令", with_directive)
        self.assertIn("优先检验行业中性化后的动量残差。", with_directive)
        # A directive never relaxes the hard contract; it enters as a hypothesis
        # inside the dynamic half.
        dynamic_index = with_directive.index("# 本 Fold 动态上下文")
        self.assertGreater(with_directive.index("研究者本 Fold 指令"), dynamic_index)
        # Whitespace-only directives collapse to the no-section prompt.
        self.assertEqual(build_system_prompt(**self.BASE, fold_directive="  \n"), without)

    def test_the_experiment_level_exploration_direction_is_additive(self) -> None:
        prompt = build_system_prompt(
            **self.BASE,
            fold_exploration_directive="持续检验事件冲击的图传播。",
            fold_directive="本 Fold 先做行业边消融。",
        )
        self.assertIn("持续检验事件冲击的图传播。", prompt)
        self.assertIn("本 Fold 先做行业边消融。", prompt)
        # The standing experiment direction precedes the per-Fold hypothesis.
        self.assertLess(
            prompt.index("持续检验事件冲击的图传播。"), prompt.index("本 Fold 先做行业边消融。")
        )
        self.assertEqual(
            build_system_prompt(**self.BASE, fold_exploration_directive="   "),
            build_system_prompt(**self.BASE),
        )

    def test_the_static_contract_is_byte_identical_across_two_different_folds(self) -> None:
        marker = "# 本 Fold 动态上下文"
        first = build_system_prompt(
            fold_info={"fold_id": "first"},
            acceptance_rules={"min_return": 0.0},
            experiment_facts={"identity": {"run_id": "run_1"}},
            prior_prompt="方向 A",
            fold_exploration_directive="长期假设 A",
            fold_directive="当前假设 A",
        )
        second = build_system_prompt(
            fold_info={"fold_id": "second"},
            acceptance_rules={"min_return": 0.1},
            experiment_facts={"identity": {"run_id": "run_2"}},
            phase="convergence",
            prior_prompt="方向 B",
            fold_exploration_directive="长期假设 B",
            fold_directive="当前假设 B",
        )
        first_prefix, first_context = first.split(marker, 1)
        second_prefix, second_context = second.split(marker, 1)
        self.assertEqual(first_prefix, second_prefix)
        self.assertNotEqual(first_context, second_context)
        # And the fixed half really is the contract, in order.
        order = [
            first.index("# 角色与目标"),
            first.index("# 核心执行合同"),
            first.index("# 环境与配置"),
            first.index("# 动作与流程"),
            first.index("# 提交合同"),
            first.index("# 禁止事项"),
            first.index(marker),
        ]
        self.assertEqual(order, sorted(order))

    def test_the_prior_rides_in_its_own_section_not_in_the_facts_blob(self) -> None:
        prompt = build_system_prompt(**self.BASE, prior_prompt="偏好小步修改")
        self.assertIn("偏好小步修改", prompt)
        dynamic = prompt.split("# 本 Fold 动态上下文", 1)[1]
        self.assertIn("偏好小步修改", dynamic)
        self.assertNotIn("偏好小步修改", prompt.split("# 本 Fold 动态上下文", 1)[0])

    def test_an_unknown_mode_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode must be fold, meta, or meta_learning"):
            build_system_prompt(mode="authoring")
