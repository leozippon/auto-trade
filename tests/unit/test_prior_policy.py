"""Shared skills carry no hidden-stage content; PRIOR has only a length bound."""

from __future__ import annotations

import unittest

from autotrade.environment.tools.prior_policy import strict_transferable_content_violation


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
        # `test` inside a longer word never names the hidden stage, and
        # `backtest` is the domain's most common such word: a Validation figure
        # written beside it was the live half of the reported false positives.
        "该止损条件已在 backtest 中反复验证，回撤超过 3pp 立即平仓",
        "backtest 年化 12% 是 Validation 口径",
        "latest gate value at 09:30, drawdown >= 3pp",
        "H+1 的 latest 快照里 contest 名单更新 2 次",
    )
    # The stage reference is what carries the leak; keeping it a whole word must
    # not weaken any of the wordings the gate exists for.
    WHOLE_WORD_LEAKS = ("Test 年化 12%", "测试窗口 +8%", "test Sharpe 1.2")

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

    def test_the_stage_reference_must_be_a_whole_word(self) -> None:
        for line in self.WHOLE_WORD_LEAKS:
            self.assertIn("Test figure", strict_transferable_content_violation(line), line)

    def test_held_out_and_test_selection_are_refused_but_a_boundary_rule_passes(self) -> None:
        self.assertIn("Held-out", strict_transferable_content_violation("Held-out sharpe 1.2"))
        self.assertIn(
            "uses Test", strict_transferable_content_violation("根据Test选择动量因子")
        )
        self.assertEqual(strict_transferable_content_violation("不得使用 Test/Held-out。"), "")
