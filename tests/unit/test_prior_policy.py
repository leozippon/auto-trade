"""Shared skills carry no sealed-stage content; PRIOR has only a length bound."""

from __future__ import annotations

import unittest

from autotrade.environment.tools.prior_policy import strict_transferable_content_violation


class ForwardFigureLeakTest(unittest.TestCase):
    """A forward leak is a sealed-stage number, not the word "forward" near a
    digit: the gate must catch reported forward-period figures while ordinary
    research prose (forward returns, walk-forward, trailing windows, in-sample
    test sets) passes."""

    LEAKS = (
        "前推期收益 0.31",
        "前推期夏普 1.35，回撤 12%",
        "前推回放年化收益 12.3%",
        "forward period total_return 0.21",
        "在前推期上 IC 0.03",
        # English reporting is as much a leak as the Chinese wording.
        "forward-period return 0.21",
        "Forward test excess was +3.2%",
        "Forward period performance 1.4 IR",
        "前推段表现 1.35",
        "前推期 +8%",
    )
    ORDINARY = (
        # The label and methods of ordinary factor research.
        "5 日 forward return 的 rank IC 0.04",
        "forward returns over 20 days: IC 0.05, turnover 30%",
        "walk-forward test 夏普 1.2，全部在研究期内",
        "从研究期末前推 20 个交易日，动量收益 3%",
        # The retired Test stage is no longer a sealed period: an in-sample
        # train/test split is research-period evidence.
        "LightGBM 测试集 IC 0.05（研究期内的时间切分）",
        "Test 夏普 1.35，回撤 12%",
        "边界单测 test_b0_exit_due.py 全绿，覆盖 3 个用例",
        "H+1 开盘首卖，阈值 ≥3pp，09:30 之前不下单",
        # Naming the sealed stage without a figure is allowed.
        "冻结产物会在前推期被连续回放一次，所以滚动重拟合要写进 fit",
    )

    def test_shared_skill_content_rejects_figures_and_allows_prose(self) -> None:
        for line in self.LEAKS:
            violation = strict_transferable_content_violation(line)
            self.assertIn("forward-period figure", violation, line)
            # The matched span is named, so the fix does not need bisection.
            self.assertTrue(violation.endswith("'"), violation)
        for line in self.ORDINARY:
            self.assertEqual(strict_transferable_content_violation(line), "", line)

    def test_shared_skills_reject_a_figure_written_before_the_stage(self) -> None:
        self.assertIn(
            "forward-period figure",
            strict_transferable_content_violation("夏普 1.42（前推期）"),
        )

    def test_held_out_and_forward_selection_are_refused_but_a_boundary_rule_passes(self) -> None:
        self.assertIn("Held-out", strict_transferable_content_violation("Held-out sharpe 1.2"))
        self.assertIn(
            "uses the forward period",
            strict_transferable_content_violation("根据前推期选择动量因子"),
        )
        self.assertEqual(strict_transferable_content_violation("不得使用 前推期/Held-out。"), "")
        self.assertEqual(
            strict_transferable_content_violation("never use forward period/held-out data"), ""
        )
