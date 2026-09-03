import math
import statistics
import json
import tempfile
import unittest
from pathlib import Path

from autotrade.pipelines.config import AcceptanceRules
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.reporting import (
    _compound_active_return,
    _std,
    _tstat,
    build_experiment_report,
)


PERIODS = {
    "fold_2022Q1": "20220101..20220331",
    "fold_2022Q2": "20220401..20220630",
}
# Frozen per-window benchmark returns (as the replay-time style block records).
B_Q1 = 0.05
B_Q2 = 110.0 / 105.0 - 1.0
BENCHMARKS = {"fold_2022Q1": B_Q1, "fold_2022Q2": B_Q2}


def heldout_record(result, *, run_id="run_ho", epoch_id="epoch_002"):
    """A held-out row as the pipeline writes it: result plus its verdict."""
    return {
        "record_type": "heldout",
        "experiment_id": "e",
        "epoch_id": epoch_id,
        "fold_id": "heldout_2026Q1",
        "run_id": run_id,
        "period": {"start": "20260101", "end": "20260331"},
        "result": result,
        "verdict": AcceptanceRules().heldout_verdict(result),
    }


def fold_record(fold_id, valid_ret, test_ret, epoch_id="epoch_001", benchmark=True):
    test_result = {
        "total_return": test_ret,
        "sharpe": 0.8,
        "max_drawdown": 0.07,
        "order_count": 4,
    }
    if benchmark:
        test_result["benchmark"] = {
            "label": "沪深300",
            "benchmark_return": BENCHMARKS[fold_id],
            "beta": 0.5,
            "n_days": 20,
        }
    return {
        "record_type": "fold",
        "experiment_id": "e",
        "epoch_id": epoch_id,
        "fold_id": fold_id,
        "run_id": f"run_{fold_id}",
        "fold_status": "frozen",
        "test_period": PERIODS[fold_id],
        "validation_result": {"total_return": valid_ret, "sharpe": 1.1, "max_drawdown": 0.05},
        "test_result": test_result,
    }


class ReportingTest(unittest.TestCase):
    def test_builds_charts_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ledger = ExperimentLedger(tmp / "ledger.jsonl")
            ledger.append(fold_record("fold_2022Q1", 0.03, 0.02))
            ledger.append(fold_record("fold_2022Q2", 0.01, -0.01))
            ledger.append(fold_record("fold_2022Q1", 0.02, 0.03, epoch_id="epoch_002"))
            ledger.append(fold_record("fold_2022Q2", 0.04, 0.04, epoch_id="epoch_002"))
            ledger.append(
                heldout_record(
                    {
                        "total_return": 0.015, "sharpe": 0.5, "max_drawdown": 0.04,
                        "order_count": 3,
                        "benchmark": {"label": "沪深300", "benchmark_return": 121.0 / 110.0 - 1.0},
                    }
                )
            )
            summary = build_experiment_report(tmp / "ledger.jsonl", tmp / "reports")
            # Held-out trailed its benchmark (1.5% vs 10%): discarded, reason named.
            self.assertEqual(summary["verdict"]["status"], "discarded")
            self.assertEqual(summary["verdict"]["reasons"], ["excess_return_not_positive"])
            self.assertTrue((tmp / "reports" / "epoch_comparison_returns.png").exists())
            self.assertTrue((tmp / "reports" / "epoch_returns" / "epoch_001_returns.png").exists())
            self.assertTrue((tmp / "reports" / "epoch_returns" / "epoch_002_returns.png").exists())
            self.assertEqual(summary["folds"], 4)
            self.assertEqual(summary["heldout_periods"], 1)
            self.assertEqual(
                summary["epoch_return_charts"],
                [
                    str(tmp / "reports" / "epoch_returns" / "epoch_001_returns.png"),
                    str(tmp / "reports" / "epoch_returns" / "epoch_002_returns.png"),
                ],
            )
            self.assertEqual(summary["epoch_comparison_chart"], str(tmp / "reports" / "epoch_comparison_returns.png"))
            self.assertAlmostEqual(summary["development"]["positive_rate"], 0.75)
            # Every Fold here ran a Test stage, so that is what was scored.
            self.assertEqual(summary["development"]["result_sources"], {"frozen_test": 4})
            self.assertAlmostEqual(summary["heldout"]["mean_return"], 0.015)
            self.assertEqual(summary["benchmark"]["status"], "ok")
            self.assertEqual(summary["benchmark"]["source"], "ledger_frozen_style")
            self.assertEqual(summary["status"], "ok")
            self.assertAlmostEqual(summary["development"]["mean_benchmark_return"], (B_Q1 + B_Q2) / 2)
            b_q2 = B_Q2
            dev_tests = [0.02, -0.01, 0.03, 0.04]
            dev_active = [0.02 - B_Q1, -0.01 - b_q2, 0.03 - B_Q1, 0.04 - b_q2]
            self.assertAlmostEqual(summary["development"]["mean_active_return"], statistics.mean(dev_active))
            # compound_active_return is the equity ratio ∏(1+r)/∏(1+b)−1 (matches the
            # "Relative equity vs benchmark" chart), NOT the arithmetic-diff compound.
            strategy = 1.02 * 0.99 * 1.03 * 1.04
            benchmark = 1.05 * (1.0 + b_q2) * 1.05 * (1.0 + b_q2)
            self.assertAlmostEqual(summary["development"]["compound_active_return"], strategy / benchmark - 1.0)
            arithmetic_compound = 1.0
            for value in dev_active:
                arithmetic_compound *= 1.0 + value
            arithmetic_compound -= 1.0
            self.assertNotAlmostEqual(summary["development"]["compound_active_return"], arithmetic_compound, places=4)
            # Dispersion + significance stats over the per-fold development results.
            self.assertAlmostEqual(summary["development"]["std_return"], statistics.stdev(dev_tests))
            self.assertAlmostEqual(summary["development"]["std_active_return"], statistics.stdev(dev_active))
            self.assertAlmostEqual(
                summary["development"]["active_return_tstat"],
                statistics.mean(dev_active) / (statistics.stdev(dev_active) / math.sqrt(len(dev_active))),
            )

    def test_read_rejects_unversioned_and_unknown_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            record = fold_record("fold_2022Q1", 0.03, 0.02)
            # Missing version: legacy formats are not tolerated — migrate.
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                ExperimentLedger(path).read()
            # Unknown version: newer formats must not be silently misread.
            # true/1.0/"1" must also reject (bool subclasses int; 1.0 == 1).
            for bad in (2, True, 1.0, "1"):
                path.write_text(json.dumps({**record, "schema_version": bad}) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "schema_version"):
                    ExperimentLedger(path).read()

    def test_append_stamps_win_over_caller_supplied_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = ExperimentLedger(Path(tmp) / "ledger.jsonl")
            record = fold_record("fold_2022Q1", 0.03, 0.02)
            record["schema_version"] = 999
            record["recorded_at"] = "1999-01-01T00:00:00+00:00"
            ledger.append(record)
            stored = ledger.read()[0]
            self.assertEqual(stored["schema_version"], 1)
            self.assertNotEqual(stored["recorded_at"], "1999-01-01T00:00:00+00:00")

    def test_rerun_supersedes_earlier_fold_and_heldout_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ledger = ExperimentLedger(tmp / "ledger.jsonl")
            ledger.append(fold_record("fold_2022Q1", 0.03, 0.02))
            # Re-run of the SAME (epoch, fold): only the later record counts.
            ledger.append(fold_record("fold_2022Q1", 0.05, 0.06))
            for total in (0.010, 0.020):
                ledger.append(
                    heldout_record(
                        {
                            "total_return": total, "sharpe": 0.5, "max_drawdown": 0.04,
                            "order_count": 3,
                            "benchmark": {"label": "沪深300", "benchmark_return": 0.01},
                        },
                        run_id=f"run_ho_{total}",
                        epoch_id="epoch_001",
                    )
                )
            summary = build_experiment_report(tmp / "ledger.jsonl", tmp / "reports")
            self.assertEqual(summary["folds"], 1)
            self.assertEqual(summary["heldout_periods"], 1)
            self.assertAlmostEqual(summary["development"]["mean_return"], 0.06)
            self.assertAlmostEqual(summary["heldout"]["mean_return"], 0.020)
            # The verdict follows the superseding held-out record (2% beats 1%).
            self.assertEqual(summary["verdict"]["status"], "graduated")

    def test_warns_when_frozen_benchmark_blocks_missing(self):
        # A ledger record without the replay-time benchmark block must flag the
        # report status as "warning" (docs/pipeline-design.md §4.2) — the report
        # never falls back to the mutable raw lake.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ledger = ExperimentLedger(tmp / "ledger.jsonl")
            ledger.append(fold_record("fold_2022Q1", 0.03, 0.02, benchmark=False))
            summary = build_experiment_report(tmp / "ledger.jsonl", tmp / "reports")
            self.assertEqual(summary["benchmark"]["status"], "missing_frozen_benchmark")
            self.assertEqual(summary["status"], "warning")

            ledger.append(fold_record("fold_2022Q2", 0.02, 0.01))
            partial = build_experiment_report(tmp / "ledger.jsonl", tmp / "reports_p")
            self.assertEqual(partial["benchmark"]["status"], "partial_coverage")
            self.assertEqual(partial["status"], "warning")

    def test_default_folds_are_scored_on_their_validation_window(self):
        # The default design has no Test stage: each Fold is scored on its own
        # Validation replay over the validation window, so the development
        # summary and the charts carry real numbers with `test_result: null`.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ledger = ExperimentLedger(tmp / "ledger.jsonl")
            for fold_id, period, total, bench in (
                ("fold_2022", "20220101..20221231", 0.05, -0.21),
                ("fold_2023", "20230101..20231231", 0.15, -0.05),
            ):
                ledger.append(
                    {
                        "record_type": "fold",
                        "experiment_id": "e",
                        "epoch_id": "epoch_001",
                        "fold_id": fold_id,
                        "run_id": f"run_{fold_id}",
                        "fold_status": "frozen",
                        "validation_period": period,
                        "test_period": None,
                        "validation_result": {
                            "total_return": total,
                            "sharpe": 1.1,
                            "max_drawdown": 0.05,
                            "order_count": 7,
                            "benchmark": {"label": "CSI 300", "benchmark_return": bench},
                        },
                        "test_result": None,
                    }
                )
            summary = build_experiment_report(tmp / "ledger.jsonl", tmp / "reports")
            development = summary["development"]
            self.assertEqual(summary["folds"], 2)
            self.assertEqual(development["result_sources"], {"validation": 2})
            self.assertAlmostEqual(development["mean_return"], 0.10)
            self.assertAlmostEqual(development["median_return"], 0.10)
            self.assertAlmostEqual(development["worst_return"], 0.05)
            self.assertAlmostEqual(development["positive_rate"], 1.0)
            self.assertAlmostEqual(development["mean_benchmark_return"], (-0.21 - 0.05) / 2)
            self.assertAlmostEqual(
                development["mean_active_return"], ((0.05 + 0.21) + (0.15 + 0.05)) / 2
            )
            self.assertAlmostEqual(
                development["compound_active_return"],
                (1.05 * 1.15) / (0.79 * 0.95) - 1.0,
            )
            self.assertIsNotNone(development["std_return"])
            self.assertIsNotNone(development["active_return_tstat"])
            # Scored periods carry their frozen benchmark block: no coverage gap.
            self.assertEqual(summary["benchmark"]["status"], "ok")
            self.assertEqual(summary["status"], "ok")
            # Held-out is still the only graduation evidence.
            self.assertIsNone(summary["verdict"])
            self.assertTrue((tmp / "reports" / "epoch_comparison_returns.png").exists())
            self.assertTrue(
                (tmp / "reports" / "epoch_returns" / "epoch_001_returns.png").exists()
            )

    def test_summary_reports_the_selection_width_and_its_correction(self):
        """The report carries how many candidates each Fold searched and the
        deflated-Sharpe probability of the one it froze. A Fold whose
        probability is unavailable is left out of the mean, never counted as
        0, and a ledger written before the block reports nothing at all."""

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ledger = ExperimentLedger(tmp / "ledger.jsonl")
            for fold_id, period, candidates, probability in (
                ("fold_2022", "20220101..20221231", 6, 0.80),
                ("fold_2023", "20230101..20231231", 2, 0.40),
                ("fold_2024", "20240101..20241231", 1, None),
            ):
                ledger.append(
                    {
                        "record_type": "fold",
                        "experiment_id": "e",
                        "epoch_id": "epoch_001",
                        "fold_id": fold_id,
                        "run_id": f"run_{fold_id}",
                        "fold_status": "frozen",
                        "validation_period": period,
                        "test_period": None,
                        "validation_result": {
                            "total_return": 0.10,
                            "sharpe": 1.1,
                            "max_drawdown": 0.05,
                            "benchmark": {"label": "CSI 300", "benchmark_return": 0.02},
                        },
                        "test_result": None,
                        "selection_statistics": {
                            "candidates_evaluated": candidates,
                            "trials": candidates,
                            "deflated_sharpe_probability": probability,
                            "unavailable_reason": (
                                None if probability else "fewer_than_two_trials"
                            ),
                        },
                    }
                )
            development = build_experiment_report(
                tmp / "ledger.jsonl", tmp / "reports"
            )["development"]
            self.assertAlmostEqual(development["mean_candidates_evaluated"], 3.0)
            self.assertAlmostEqual(
                development["mean_deflated_sharpe_probability"], 0.60
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ledger = ExperimentLedger(tmp / "ledger.jsonl")
            ledger.append(
                {
                    "record_type": "fold",
                    "experiment_id": "e",
                    "epoch_id": "epoch_001",
                    "fold_id": "fold_2022",
                    "run_id": "run_2022",
                    "fold_status": "frozen",
                    "validation_period": "20220101..20221231",
                    "test_period": None,
                    "validation_result": {
                        "total_return": 0.10,
                        "sharpe": 1.1,
                        "max_drawdown": 0.05,
                        "benchmark": {"label": "CSI 300", "benchmark_return": 0.02},
                    },
                    "test_result": None,
                }
            )
            development = build_experiment_report(
                tmp / "ledger.jsonl", tmp / "reports"
            )["development"]
            self.assertIsNone(development["mean_candidates_evaluated"])
            self.assertIsNone(development["mean_deflated_sharpe_probability"])

    def test_a_failed_frozen_test_scores_nothing_and_never_falls_back(self):
        # With a Test stage the frozen Test is the scored result: when it failed
        # the Fold contributes no numbers rather than borrowing its Validation.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ledger = ExperimentLedger(tmp / "ledger.jsonl")
            record = fold_record("fold_2022Q1", 0.03, 0.02)
            record["test_result"] = {"status": "failed", "error": "boom"}
            ledger.append(record)
            summary = build_experiment_report(tmp / "ledger.jsonl", tmp / "reports")
            self.assertEqual(summary["development"]["result_sources"], {"frozen_test": 1})
            self.assertIsNone(summary["development"]["mean_return"])
            self.assertIsNone(summary["development"]["mean_active_return"])
            self.assertEqual(summary["benchmark"]["status"], "missing_frozen_benchmark")

    def test_the_walk_forward_record_lists_each_epochs_parent_controls(self):
        # The inherited strategy replayed on the next Fold's window, per Epoch;
        # a failed control stays visible as a transition without numbers.
        def fold(epoch_id, fold_id, period, control):
            return {
                "record_type": "fold",
                "experiment_id": "e",
                "epoch_id": epoch_id,
                "fold_id": fold_id,
                "run_id": f"run_{epoch_id}_{fold_id}",
                "fold_status": "frozen",
                "validation_period": period,
                "test_period": None,
                "validation_result": {"total_return": 0.1, "sharpe": 1.0, "max_drawdown": 0.05},
                "test_result": None,
                "parent_control": control,
            }

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ledger = ExperimentLedger(tmp / "ledger.jsonl")
            ledger.append(fold("epoch_001", "fold_2022", "20220101..20221231", None))
            ledger.append(
                fold(
                    "epoch_001",
                    "fold_2023",
                    "20230101..20231231",
                    {
                        "status": "ok",
                        "parent_strategy_artifact_id": "strategy_a",
                        "step_id": "node_control",
                        "validation_result": {
                            "total_return": 0.06,
                            "sharpe": 0.8,
                            "max_drawdown": 0.04,
                            "benchmark": {"label": "CSI 300", "benchmark_return": 0.02},
                        },
                        "validation_result_ref": "ref",
                    },
                )
            )
            ledger.append(
                fold(
                    "epoch_001",
                    "fold_2024",
                    "20240101..20241231",
                    {"status": "failed", "parent_strategy_artifact_id": "strategy_b", "error": "TimeoutError: slow"},
                )
            )
            summary = build_experiment_report(tmp / "ledger.jsonl", tmp / "reports")
        walk = summary["walk_forward"]
        self.assertEqual([row["epoch_id"] for row in walk], ["epoch_001"])
        transitions = walk[0]["transitions"]
        self.assertEqual([row["fold"] for row in transitions], ["2023", "2024"])
        self.assertEqual(transitions[0]["parent_strategy_artifact_id"], "strategy_a")
        self.assertAlmostEqual(transitions[0]["excess_return"], 0.04)
        self.assertEqual(transitions[0]["sharpe"], 0.8)
        self.assertEqual(transitions[1]["status"], "failed")
        self.assertIsNone(transitions[1]["excess_return"])
        self.assertAlmostEqual(walk[0]["mean_excess_return"], 0.04)
        # A failed control cannot be chained: it stays a transition that proved
        # nothing, visible as the gap to scored_transitions.
        chain = walk[0]["chain"]
        self.assertEqual(chain["transitions"], 2)
        self.assertEqual(chain["scored_transitions"], 1)
        self.assertEqual(chain["positive_transitions"], 1)
        self.assertAlmostEqual(chain["return"], 0.06)
        self.assertIsNone(chain["excess_at_2x_slippage_sum"])
        # The walk-forward record leads the summary; the fold table follows it.
        self.assertEqual(next(iter(summary)), "walk_forward")

    def test_the_walk_forward_chain_compounds_the_transitions_it_scored(self):
        # Three quarterly steps of a trailing window, each scored on its own new
        # quarter: the chain is what the process actually earned walking forward.
        def fold(label, period, step, step_return, benchmark, stressed):
            return {
                "record_type": "fold",
                "experiment_id": "e",
                "epoch_id": "epoch_001",
                "fold_id": f"fold_{label}",
                "run_id": f"run_{label}",
                "fold_status": "frozen",
                "validation_period": period,
                "test_period": None,
                "validation_result": {"total_return": 0.5, "sharpe": 1.0, "max_drawdown": 0.05},
                "test_result": None,
                "parent_control": {
                    "status": "ok",
                    "parent_strategy_artifact_id": "strategy_a",
                    # The whole four-quarter window looks far better than the
                    # quarter that was actually new; the chain must ignore it.
                    "validation_result": {
                        "total_return": 0.40,
                        "benchmark": {"benchmark_return": 0.01},
                    },
                    "step_result": {
                        "label": label,
                        "start": step[0],
                        "end": step[1],
                        "total_return": step_return,
                        "benchmark": {"benchmark_return": benchmark},
                        "sharpe": 0.5,
                        "max_drawdown": 0.02,
                        "cost_sensitivity": {"excess_at_2x_slippage": stressed},
                    },
                    # The window's null control looks decisive; only the step's
                    # own percentile describes the transition.
                    "null_control": {
                        "k": 500,
                        "excess_percentile": 1.0,
                        "step": {"excess_percentile": 0.5 + step_return},
                    },
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ledger = ExperimentLedger(tmp / "ledger.jsonl")
            ledger.append(
                fold("2023Q2", "20220701..20230630", ("20230403", "20230630"), 0.10, 0.02, 0.06)
            )
            ledger.append(
                fold("2023Q3", "20221001..20230930", ("20230703", "20230928"), -0.05, 0.01, -0.08)
            )
            ledger.append(
                fold("2023Q4", "20230101..20231231", ("20231009", "20231229"), 0.03, 0.01, -0.01)
            )
            summary = build_experiment_report(tmp / "ledger.jsonl", tmp / "reports")
        chain = summary["walk_forward"][0]["chain"]
        rows = summary["walk_forward"][0]["transitions"]
        # Every row is scored on its step, spans the step, and says so.
        self.assertEqual({row["source"] for row in rows}, {"step_result"})
        self.assertEqual((rows[0]["period_start"], rows[0]["period_end"]), ("20230403", "20230630"))
        self.assertEqual(chain["transitions"], 3)
        self.assertEqual(chain["scored_transitions"], 3)
        self.assertEqual(chain["positive_transitions"], 2)
        strategy = 1.10 * 0.95 * 1.03 - 1.0
        benchmark = 1.02 * 1.01 * 1.01 - 1.0
        self.assertAlmostEqual(chain["return"], strategy)
        self.assertAlmostEqual(chain["benchmark_return"], benchmark)
        self.assertAlmostEqual(
            chain["excess_return"], (1.0 + strategy) / (1.0 + benchmark) - 1.0
        )
        # Cost-stressed excess adds up over the steps, not over overlapping windows.
        self.assertAlmostEqual(chain["excess_at_2x_slippage_sum"], -0.03)
        # Each row is ranked against the null of the span it was scored on.
        self.assertAlmostEqual(rows[0]["excess_percentile"], 0.6)
        self.assertAlmostEqual(chain["mean_excess_percentile"], (0.6 + 0.45 + 0.53) / 3)

    def test_a_held_out_row_without_a_verdict_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ledger = ExperimentLedger(tmp / "ledger.jsonl")
            ledger.append(fold_record("fold_2022Q1", 0.03, 0.02))
            row = heldout_record({"total_return": 0.01, "sharpe": 0.5, "max_drawdown": 0.04})
            del row["verdict"]
            ledger.append(row)
            with self.assertRaisesRegex(ValueError, "no verdict"):
                build_experiment_report(tmp / "ledger.jsonl", tmp / "reports")

    def test_requires_fold_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            ledger_path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no fold records"):
                build_experiment_report(ledger_path, Path(tmp) / "reports")

    def test_small_or_degenerate_samples_omit_dispersion_stats(self):
        # A single development fold has no dispersion, so std/t-stat are null.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ledger = ExperimentLedger(tmp / "ledger.jsonl")
            ledger.append(fold_record("fold_2022Q1", 0.03, 0.02))
            summary = build_experiment_report(tmp / "ledger.jsonl", tmp / "reports")
            self.assertIsNone(summary["development"]["std_return"])
            self.assertIsNone(summary["development"]["std_active_return"])
            self.assertIsNone(summary["development"]["active_return_tstat"])


class ReportingStatsTest(unittest.TestCase):
    def test_compound_active_return_uses_equity_ratio_not_arithmetic_diff(self):
        rows = [
            {"return": 0.50, "benchmark_return": 0.20},
            {"return": -0.40, "benchmark_return": 0.20},
        ]
        ratio = (1.5 * 0.6) / (1.2 * 1.2) - 1.0  # -0.375
        self.assertAlmostEqual(_compound_active_return(rows), ratio)
        arithmetic_diff = (1.0 + (0.50 - 0.20)) * (1.0 + (-0.40 - 0.20)) - 1.0  # -0.48
        self.assertNotAlmostEqual(_compound_active_return(rows), arithmetic_diff, places=4)

    def test_compound_active_return_skips_folds_missing_a_leg(self):
        rows = [
            {"return": 0.10, "benchmark_return": None},
            {"return": None, "benchmark_return": 0.05},
        ]
        self.assertIsNone(_compound_active_return(rows))

    def test_std_and_tstat_edges(self):
        self.assertIsNone(_std([0.02]))
        self.assertIsNone(_tstat([0.02]))
        self.assertEqual(_std([0.03, 0.03]), 0.0)
        self.assertIsNone(_tstat([0.03, 0.03, 0.03]))  # zero dispersion
        values = [0.01, -0.02, 0.04, 0.03]
        self.assertAlmostEqual(_std(values), statistics.stdev(values))
        self.assertAlmostEqual(
            _tstat(values),
            statistics.mean(values) / (statistics.stdev(values) / math.sqrt(len(values))),
        )


if __name__ == "__main__":
    unittest.main()
