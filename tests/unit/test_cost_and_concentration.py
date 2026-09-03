"""Two things a return number alone hides: who paid for it, and who earned it.

The audited 2026 Held-out beat nothing net of cost and rested on a handful of
trades in one name while its headline return looked healthy. Every result
therefore carries what one more basis point of slippage per side costs
(``cost_sensitivity``) and how concentrated the realized gains are
(``pnl_concentration``). These tests pin the arithmetic of both, and that an
unstatable field stays ``None`` with a reason instead of becoming zero.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.replay.stats import (
    ReplayResult,
    attach_cost_sensitivity,
    compute_return_stats,
)
from autotrade.environment.strategy import StrategySchedule
from autotrade.pipelines.config import (
    ArtifactRevision,
    EvaluationRequest,
    SnapshotBundle,
)
from autotrade.pipelines.local_backend import LocalDailyEvaluationBackend

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

    def test_fewer_than_five_winners_never_pulls_a_loser_into_the_top_five(
        self,
    ) -> None:
        """The metric flags a gain resting on a handful of trades, so the slice
        must be the largest gains. Signed sorting fills a short slice with
        losses and deflates the share exactly in the sparse windows the metric
        exists to catch."""
        executions = tuple(
            row
            for row in _REALIZED
            if row["symbol"] in ("A", "F")  # +100, +20, -20: two winners
        )
        block = self.block(executions)
        self.assertAlmostEqual(block["gross_gains"], 120.0)
        self.assertAlmostEqual(block["top5_share_of_gross_gains"], 1.0)

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


class EveryEvaluationBackendPricesTheBlockTest(unittest.TestCase):
    """The block is a property of a completed Validation, not of one backend.

    ``AcceptanceRules`` fails closed on a missing ``cost_sensitivity`` whenever
    ``cost_stress_multiplier > 1``, so a backend that skips it makes every
    stressed experiment unpassable and advertises a summary block it never
    produces.
    """

    STRATEGY = """def generate_orders(context):
    if context.account.cash <= 0 or dict(context.account.positions):
        return []
    return [{
        "symbol": "000001.SZ",
        "action": "buy",
        "quantity": 100,
        "execute_at": context.inference_at.replace(hour=15, minute=0).isoformat(),
    }]
"""

    def test_the_local_daily_backend_prices_the_block_like_the_pit_backend(
        self,
    ) -> None:
        days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2025-10-01", periods=6)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily = root / "daily.parquet"
            pd.DataFrame(
                {
                    "trade_date": days,
                    "symbol": ["000001.SZ"] * len(days),
                    "open": [10.0] * len(days),
                    "close": [10.5] * len(days),
                }
            ).to_parquet(daily, index=False)
            revision = root / "revision"
            revision.mkdir()
            (revision / "main.py").write_text(self.STRATEGY, encoding="utf-8")

            result = LocalDailyEvaluationBackend(
                daily, root / "results", execution_mode="trusted"
            ).evaluate(
                EvaluationRequest(
                    ArtifactRevision("revision_cost", revision),
                    SnapshotBundle("snap_cost", str(daily), str(daily)),
                    "valid",
                    days[0],
                    days[-1],
                    StrategySchedule("day", "09:00"),
                    BrokerProfile(initial_cash=100_000, slippage_bps=5.0),
                )
            )

        block = result.summary["cost_sensitivity"]
        self.assertEqual(block["slippage_bps"], 5.0)
        self.assertAlmostEqual(
            block["cost_per_bp_per_side"], float(result.summary["turnover"]) * 1e-4
        )


if __name__ == "__main__":
    unittest.main()
