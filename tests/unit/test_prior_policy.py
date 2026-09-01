"""The unified Meta PRIOR must remain transferable and non-empty."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from autotrade.environment.tools import FinishMetaTool, SafeWorkspace, ToolRegistry
from autotrade.environment.tools.finish_meta import FinishMetaTool as _FinishMetaToolModule
from autotrade.environment.tools.prior_policy import (
    PRIOR_MAX_CHARS,
    prior_content_violation,
    prior_policy_violation,
    strict_transferable_content_violation,
    visible_window_dates,
)

assert FinishMetaTool is _FinishMetaToolModule

MANIFEST_2021 = {
    "meta_learning_visible_fold": {
        "input_window": "20200101..20210930",
        "validation_period": "20211001..20211231",
        "valid_decision_time": "2021-10-08T09:25:00+08:00",
    },
    "valid_decision_time": "2021-10-08T09:25:00+08:00",
}
MANIFEST_2024 = {
    "meta_learning_visible_fold": {
        "input_window": "20240101..20240930",
        "validation_period": "20241001..20241231",
        "valid_decision_time": "2024-10-08T09:25:00+08:00",
    },
    "valid_decision_time": "2024-10-08T09:25:00+08:00",
}


def _prior(root: Path, text: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "PRIOR.md"
    path.write_text(text, encoding="utf-8")
    return path


class PriorPolicyTest(unittest.TestCase):
    def test_the_window_is_derived_from_the_manifest_not_hard_coded(self) -> None:
        dates = visible_window_dates(MANIFEST_2021)
        self.assertEqual(
            dates, {"2020", "2021", "20200101", "20210930", "20211001", "20211231"}
        )
        self.assertEqual(visible_window_dates(MANIFEST_2024) & {"2021"}, set())
        self.assertIn("2024", visible_window_dates(MANIFEST_2024))
        self.assertEqual(visible_window_dates({}), set())

    def test_a_calendar_date_blocks_prior(self) -> None:
        with TemporaryDirectory() as tmp:
            path = _prior(
                Path(tmp),
                "## 策略探索方向\n日内数据仅覆盖 21 个交易日（2021 年 8-9 月），样本不足。",
            )
            violation = prior_policy_violation(
                path, window_dates=visible_window_dates(MANIFEST_2021)
            )
            self.assertTrue(violation.startswith("PRIOR.md line "))
            self.assertIn("calendar date", violation)

    def test_a_bare_visible_window_year_tracks_the_window(self) -> None:
        with TemporaryDirectory() as tmp:
            path = _prior(Path(tmp), "## 策略探索方向\n对标 2024 的市场结构轮动。")
            self.assertIn(
                "calendar date",
                prior_policy_violation(
                    path, window_dates=visible_window_dates(MANIFEST_2024)
                ),
            )
            self.assertEqual(
                prior_policy_violation(
                    path, window_dates=visible_window_dates(MANIFEST_2021)
                ),
                "",
            )

    def test_transferable_counts_and_cadence_are_allowed(self) -> None:
        with TemporaryDirectory() as tmp:
            path = _prior(
                Path(tmp),
                "核心持仓按季度轮动；样本交易日不足（约 21 个），换手率 50%-80%。",
            )
            self.assertEqual(
                prior_policy_violation(
                    path, window_dates=visible_window_dates(MANIFEST_2021)
                ),
                "",
            )

    def test_the_resource_limit_is_enforced(self) -> None:
        with TemporaryDirectory() as tmp:
            path = _prior(Path(tmp), "可迁移方向。\n" * (PRIOR_MAX_CHARS // 6))
            violation = prior_policy_violation(path)
            self.assertIn("characters", violation)
            self.assertIn(str(PRIOR_MAX_CHARS), violation)

    def test_missing_and_empty_prior_are_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIn("write PRIOR.md", prior_policy_violation(root / "PRIOR.md"))
            self.assertIn("non-empty", prior_policy_violation(_prior(root, "   \n")))


class FinishMetaToolTest(unittest.TestCase):
    def _registry(self, root: Path, manifest: dict) -> ToolRegistry:
        return ToolRegistry(
            [
                FinishMetaTool(
                    SafeWorkspace(root), window_dates=visible_window_dates(manifest)
                )
            ]
        )

    def test_finish_meta_refuses_a_dated_prior_with_a_typed_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _prior(root, "2021 年四季度动量最好。")
            result = self._registry(root, MANIFEST_2021).invoke("finish_meta", {})
            self.assertFalse(result.ok)
            self.assertEqual(result.value["error_type"], "prior_policy")
            self.assertIn("calendar date", result.error)

    def test_finish_meta_accepts_prior_without_a_path_argument(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _prior(root, "样本交易日不足时降低换手；按季度轮动。")
            result = self._registry(root, MANIFEST_2021).invoke("finish_meta", {})
            self.assertTrue(result.ok, result.error)
            self.assertNotIn("path", result.value)
            self.assertEqual(result.value["status"], "meta_learning_done")

    def test_the_agent_can_rewrite_and_finish_after_a_refusal(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = self._registry(root, MANIFEST_2021)
            _prior(root, "在 20211001 之后减仓。")
            self.assertFalse(registry.invoke("finish_meta", {}).ok)
            _prior(root, "估值分位偏高时减仓，与具体日期无关。")
            self.assertTrue(registry.invoke("finish_meta", {}).ok)

    def test_finish_meta_rejects_path_arguments(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _prior(root, "可迁移方向。")
            result = self._registry(root, MANIFEST_2021).invoke(
                "finish_meta", {"prior_path": "PRIOR.md"}
            )
            self.assertFalse(result.ok)


class TestFigureLeakTest(unittest.TestCase):
    """A Test leak is a hidden-stage number, not the word "test" near a digit:
    the gate must keep catching reported figures while ordinary engineering and
    mechanism prose (unit tests, clock times, offsets) passes."""

    LEAKS = (
        "Fold1 Test 收益 0.31",
        "Test 夏普 1.35，回撤 12%",
        "测试集年化收益 12.3%",
        "test total_return 0.21",
        "在 test 上 IC 0.03",
        # English reporting is as much a leak as the Chinese wording.
        "test set return 0.21",
        "Test excess was +3.2%",
        "Test performance 1.4 IR",
        "测试段表现 1.35",
        "test 段 +8%",
    )
    ORDINARY = (
        "边界单测 test_b0_exit_due.py 全绿，覆盖 3 个用例",
        "H+1 开盘首卖，阈值 ≥3pp，09:30 之前不下单",
        "先跑一次冒烟测试，再做 2 轮完整验证",
        "unit tests: 12 passed",
        "写 3 个测试脚本覆盖边界",
        "test 3 candidates",
        "unit test 2",
        "测试了 4 个参数",
    )

    def test_shared_skill_content_rejects_figures_and_allows_prose(self) -> None:
        for line in self.LEAKS:
            violation = strict_transferable_content_violation(line)
            self.assertIn("Test figure", violation, line)
            # The matched span is named, so the fix does not need bisection.
            self.assertTrue(violation.endswith("'"), violation)
        for line in self.ORDINARY:
            self.assertEqual(strict_transferable_content_violation(line), "", line)

    def test_shared_skills_reject_a_figure_written_before_the_stage(self) -> None:
        self.assertIn(
            "Test figure", strict_transferable_content_violation("夏普 1.42（Test 窗口）")
        )

    def test_prior_rejects_the_same_figures_and_keeps_process_prose(self) -> None:
        for line in self.LEAKS:
            self.assertIn("Test figure", prior_content_violation(line), line)
        for line in self.ORDINARY:
            self.assertEqual(prior_content_violation(line), "", line)
        self.assertEqual(
            prior_content_violation("每个 fold 先做 2 轮预注册测试，再复盘"), ""
        )

    def test_a_boundary_word_does_not_carry_a_figure_past_the_gate(self) -> None:
        """The exemption exists for prohibition sentences, not for lines that
        quote the very result the prohibition forbids."""

        for line in (
            "不得泄露 Test：测试段 Sharpe 1.2",
            "排除法：test 收益 0.21 说明该方向可行",
        ):
            self.assertIn("Test figure", prior_content_violation(line), line)
        for line in (
            "不得使用 Test/Held-out。",
            "审查窗口排除 Held-out。",
        ):
            self.assertEqual(prior_content_violation(line), "", line)


class IdentifierDateTest(unittest.TestCase):
    def test_experiment_id_style_identifiers_are_not_dates(self) -> None:
        from autotrade.environment.tools.prior_policy import calendar_policy_violation

        self.assertEqual(calendar_policy_violation("# PRIOR — factor_cs_20260901"), "")
        self.assertEqual(calendar_policy_violation("实验 momentum_20250115_v2 的方向"), "")
        self.assertIn("calendar date", calendar_policy_violation("样本截至20210930 为止"))
        self.assertIn("calendar date", calendar_policy_violation("从 20210930 开始"))
        self.assertIn("calendar date", calendar_policy_violation("2021 年 8 月"))
        self.assertEqual(calendar_policy_violation("节点 fold_2022Q1 的父本"), "")
        self.assertIn("calendar date", calendar_policy_violation("在 2022Q1 回撤最大"))
        self.assertIn("calendar date", calendar_policy_violation("Q1 2022 的样本"))
