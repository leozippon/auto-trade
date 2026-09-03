"""Two things a return number alone hides: who paid for it, and who earned it.

The audited 2026 Held-out beat nothing net of cost and rested on a handful of
trades in one name while its headline return looked healthy. Every result
therefore carries what one more basis point of slippage per side costs
(``cost_sensitivity``) and how concentrated the realized gains are
(``pnl_concentration``). These tests pin the arithmetic of both, and that an
unstatable field stays ``None`` with a reason instead of becoming zero.
"""

from __future__ import annotations

import unittest

from autotrade.environment.replay.stats import (
    ReplayResult,
    attach_cost_sensitivity,
    compute_return_stats,
)

_CURVE = (
    {"trade_date": "20220104", "initial_equity": 1000.0, "equity": 1100.0, "cash": 0.0},
    {"trade_date": "20220331", "initial_equity": 1000.0, "equity": 1124.0, "cash": 0.0},
)
# Seven closed trades: one name (A) exits twice for +120 of the +144 gross gain,
# and one loser takes −20 back.
_REALIZED = tuple(
    {
        "status": "filled",
        "action": "sell",
        "symbol": symbol,
        "price": 10.0,
        "quantity": 100,
        "realized_pnl": pnl,
        "matched_at": "2022-01-04T09:30:00+08:00",
    }
    for symbol, pnl in (
        ("A", 100.0),
        ("A", 20.0),
        ("B", 10.0),
        ("C", 5.0),
        ("D", 5.0),
        ("E", 4.0),
        ("F", -20.0),
    )
)


class PnlConcentrationTest(unittest.TestCase):
    def block(self, executions=_REALIZED) -> dict[str, object]:
        stats = compute_return_stats(ReplayResult(_CURVE, executions, ("20220104",), ()))
        return stats["pnl_concentration"]

    def test_gains_losses_and_net_are_one_decomposition(self) -> None:
        block = self.block()
        self.assertAlmostEqual(block["gross_gains"], 144.0)
        self.assertAlmostEqual(block["gross_losses"], -20.0)
        self.assertAlmostEqual(block["net_realized"], 124.0)
        # Signed losses, so the three numbers cannot drift apart.
        self.assertAlmostEqual(
            block["gross_gains"] + block["gross_losses"], block["net_realized"]
        )

    def test_shares_are_measured_against_the_gross_gain_not_the_net(self) -> None:
        block = self.block()
        # Top five trades: 100 + 20 + 10 + 5 + 5 of the 144 gross gain.
        self.assertAlmostEqual(block["top5_share_of_gross_gains"], 140.0 / 144.0)
        # Top name sums both of A's exits, which a per-trade view would miss.
        self.assertAlmostEqual(block["top_name_share_of_gross_gains"], 120.0 / 144.0)

    def test_a_window_that_realized_no_gain_reports_no_share(self) -> None:
        for executions in ((), _REALIZED[-1:]):
            block = self.block(executions)
            self.assertEqual(block["gross_gains"], 0)
            self.assertIsNone(block["top5_share_of_gross_gains"])
            self.assertIsNone(block["top_name_share_of_gross_gains"])


class CostSensitivityTest(unittest.TestCase):
    def summary(self, *, turnover=20.0, excess=0.05, benchmark=True) -> dict[str, object]:
        summary: dict[str, object] = {"turnover": turnover}
        if benchmark:
            summary["benchmark"] = {"benchmark_return": 0.01, "excess_return": excess}
        return summary

    def test_break_even_and_the_stressed_excess_price_the_same_turnover(self) -> None:
        summary = self.summary()
        attach_cost_sensitivity(summary, 5.0)
        block = summary["cost_sensitivity"]
        self.assertEqual(block["slippage_bps"], 5.0)
        # 20x turnover: one extra bp per side costs 0.2% of equity.
        self.assertAlmostEqual(block["cost_per_bp_per_side"], 0.002)
        self.assertAlmostEqual(block["breakeven_extra_slippage_bps"], 25.0)
        self.assertAlmostEqual(block["excess_at_2x_slippage"], 0.04)
        self.assertNotIn("reason", block)

    def test_a_non_positive_excess_has_no_break_even_and_says_why(self) -> None:
        summary = self.summary(excess=-0.01)
        attach_cost_sensitivity(summary, 5.0)
        block = summary["cost_sensitivity"]
        self.assertIsNone(block["breakeven_extra_slippage_bps"])
        self.assertEqual(block["reason"], "excess_not_positive")
        # The stress is still stated: it is the excess that is already negative.
        self.assertAlmostEqual(block["excess_at_2x_slippage"], -0.02)

    def test_an_unmeasurable_excess_or_turnover_is_reported_not_zeroed(self) -> None:
        without_benchmark = self.summary(benchmark=False)
        attach_cost_sensitivity(without_benchmark, 5.0)
        block = without_benchmark["cost_sensitivity"]
        self.assertEqual(block["reason"], "no_benchmark_excess")
        self.assertIsNone(block["breakeven_extra_slippage_bps"])
        self.assertIsNone(block["excess_at_2x_slippage"])
        self.assertEqual(block["cost_per_bp_per_side"], 0.002)

        never_traded = self.summary(turnover=0.0)
        attach_cost_sensitivity(never_traded, 5.0)
        block = never_traded["cost_sensitivity"]
        self.assertEqual(block["reason"], "no_turnover")
        self.assertIsNone(block["breakeven_extra_slippage_bps"])
        # Nothing was traded, so no amount of slippage can touch the excess.
        self.assertAlmostEqual(block["excess_at_2x_slippage"], 0.05)


if __name__ == "__main__":
    unittest.main()
