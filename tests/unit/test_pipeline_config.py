"""Experiment configuration: acceptance semantics, validation, defaults drift.

The three default surfaces — the domain dataclasses, ``hitl_state``'s console
defaults, and the ``run_experiment`` CLI — must agree; the dataclasses are the
source of truth. Nothing else prevents a knob from meaning one thing in the
console and another on the command line.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unittest
from dataclasses import MISSING, fields
from pathlib import Path
from unittest.mock import patch

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.strategy import StrategySchedule
from autotrade.environment.llm import LOCAL_QWEN_MODEL
from autotrade.pipelines.config import AcceptanceRules, RollingExperimentConfig
from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS

#: The console create form is seeded from the pinned explore profile so a
#: new experiment hardlinks the PIT view seed. Period, screen, compact, and
#: loop knobs that a researcher gets without touching the form live here.
_CONSOLE_CREATE_PRESET: dict[str, object] = {
    "compact_keep_recent_messages": 10,
    "compact_max_calls": 10,
    "compact_max_tokens": 10_000,
    # Empty: the worker derives it from the model window and output ceiling.
    "compact_token_threshold": None,
    "epochs": 3,
    "development_first_period": "2022",
    "fold_period": "year",
    "gpu_count": 1,
    "heldout_first_period": "20260101..20260630",
    "heldout_last_period": "20260101..20260630",
    "include_events": True,
    "include_intraday": False,
    "include_text": True,
    "inference_time": "08:30",
    "initial_cash": 1_000_000.0,
    "initial_control_mode": "auto",
    "analysis_enabled": False,
    "development_last_period": "2025",
    # Per-Fold budgets for a one-year Validation with batch_validate available.
    "max_backtests_per_fold": 30,
    "max_fold_minutes": 720,
    "max_llm_calls": 1600,
    "max_steps_per_fold": 30,
    # Meta between every two consecutive Folds.
    "meta_learning_fold_interval": 1,
    "meta_model": LOCAL_QWEN_MODEL,
    "model": LOCAL_QWEN_MODEL,
    # The universe reaches the agent unfiltered; the strategy filters itself.
    "screen_boards": (),
    "screen_exclude_new_listed_days": 0,
    "screen_exclude_st": False,
    "strategy_period": "day",
    # Regular yearly Folds, no Test stage: Held-out is the verdict.
    "test_stage": False,
    "window_months": 24,
}

PERIODS = {
    "fold_period": "quarter",
    "development_first_period": "2022Q1",
    "development_last_period": "2022Q2",
    "heldout_first_period": "2023Q1",
    "heldout_last_period": "2023Q1",
}


def make_config(root: Path, **overrides: object) -> RollingExperimentConfig:
    values: dict[str, object] = {
        "experiment_id": "exp",
        "experiments_root": root / "experiments",
        **PERIODS,
    }
    values.update(overrides)
    return RollingExperimentConfig(**values)


def test_screen_exclude_st_cli_alias_and_help() -> None:
    from scripts.experiments._cli import add_snapshot_window_arguments

    parser = argparse.ArgumentParser()
    add_snapshot_window_arguments(parser)

    assert parser.parse_args([]).screen_exclude_st is False
    assert parser.parse_args(["--screen-exclude-st"]).screen_exclude_st is True
    assert parser.parse_args(["--no-screen-exclude-st"]).screen_exclude_st is False
    assert not re.search(
        r"(?m)^\s*--screen-exclude-st(?:[ =]|$)", parser.format_help()
    )


class AcceptanceRulesTest(unittest.TestCase):
    def test_nan_metrics_are_hard_rejects(self) -> None:
        # NaN compares False against every threshold; without the finiteness
        # guard a NaN total_return would pass acceptance outright.
        rules = AcceptanceRules()
        summary = {"total_return": math.nan, "sharpe": 1.0, "max_drawdown": 0.1}
        hard, warnings = rules.evaluate(summary)
        self.assertIn("non_finite_total_return", hard)
        # The finite Sharpe is above target, so nothing may claim otherwise.
        self.assertNotIn("sharpe_below_target", warnings)
        for key in ("sharpe", "max_drawdown"):
            with self.subTest(key=key):
                broken = {**summary, "total_return": 0.02, key: math.inf}
                self.assertIn(f"non_finite_{key}", rules.evaluate(broken)[0])
        # A boolean is not a metric, however happily it compares.
        self.assertIn(
            "non_finite_total_return",
            rules.evaluate({**summary, "total_return": True})[0],
        )

    def test_finite_metrics_keep_threshold_semantics(self) -> None:
        rules = AcceptanceRules()
        ok = {"total_return": 0.02, "sharpe": 0.5, "max_drawdown": 0.1}
        self.assertEqual(rules.evaluate(ok), ([], []))
        # Drawdown breach stays a HARD reject (risk limit), sign-independent.
        for drawdown in (0.30, -0.30):
            with self.subTest(drawdown=drawdown):
                hard, _ = rules.evaluate({**ok, "max_drawdown": drawdown})
                self.assertIn("max_drawdown_exceeded", hard)
        # Return/Sharpe shortfalls only WARN: the fold freezes instead of resetting.
        hard, warnings = rules.evaluate(
            {"total_return": -0.01, "sharpe": -0.2, "max_drawdown": 0.1}
        )
        self.assertEqual(hard, [])
        self.assertEqual(warnings, ["return_below_target", "sharpe_below_target"])

    def test_a_zero_trade_replay_freezes_with_a_warning_not_in_silence(self) -> None:
        """The observed silent-success case: a strategy that submits no order
        scores 0.0 everywhere, so both soft targets pass (0.0 < 0.0 is False)
        and the fold used to freeze with an empty warning list."""

        rules = AcceptanceRules()
        flat = {
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "order_count": 0,
            "trade_count": 0,
        }
        hard, warnings = rules.evaluate(flat)
        # Still warn-only: the fold freezes what it honestly found.
        self.assertEqual(hard, [])
        self.assertEqual(warnings, ["no_trades"])
        # One realized trade is a result, however small; it must not warn.
        traded = {**flat, "trade_count": 1, "total_return": 0.01, "sharpe": 0.1}
        self.assertEqual(rules.evaluate(traded), ([], []))

    def test_absent_sharpe_is_not_an_integrity_failure(self) -> None:
        rules = AcceptanceRules()
        hard, warnings = rules.evaluate({"total_return": 0.02, "max_drawdown": 0.1})
        self.assertEqual(hard, [])
        self.assertEqual(warnings, [])

    def test_rule_values_must_be_finite_and_ranged(self) -> None:
        for kwargs in (
            {"min_return": math.nan},
            {"min_sharpe": math.inf},
            {"max_drawdown": math.nan},
            {"max_drawdown": 1.5},
            {"max_drawdown": -0.1},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                AcceptanceRules(**kwargs)

    def test_walk_forward_consistency_needs_two_thirds_rounded_up(self) -> None:
        """Graduation term (b): positive excess in >= ceil(2/3) of the transitions."""
        rules = AcceptanceRules()
        for transitions, positive, status in (
            (3, 2, "consistent"),
            (3, 1, "inconsistent"),
            (2, 2, "consistent"),
            (2, 1, "inconsistent"),
            (1, 1, "consistent"),
            (1, 0, "inconsistent"),
        ):
            with self.subTest(transitions=transitions, positive=positive):
                block = rules.walk_forward_consistency(
                    {"source": "parent_control", "transitions": transitions, "positive_excess": positive}
                )
                self.assertEqual(block["status"], status)
                self.assertEqual(block["required"], math.ceil(2 * transitions / 3))
        self.assertEqual(rules.walk_forward_consistency(None), {"status": "not_applicable", "transitions": 0})
        self.assertEqual(
            rules.walk_forward_consistency({"transitions": 0})["status"], "not_applicable"
        )

    def test_held_out_verdict_names_the_walk_forward_term_when_it_fails(self) -> None:
        rules = AcceptanceRules()
        passing = {
            "total_return": 0.10,
            "sharpe": 1.0,
            "max_drawdown": -0.05,
            "benchmark": {"benchmark_return": 0.02},
        }
        # Held-out passes but only 1 of 3 walk-forward transitions beat the
        # benchmark: term (b) fails and the reason carries the counts.
        verdict = rules.heldout_verdict(
            passing, {"source": "parent_control", "transitions": 3, "positive_excess": 1}
        )
        self.assertEqual(verdict["status"], "discarded")
        self.assertEqual(verdict["reasons"], ["walkforward_excess_inconsistent(1/3<2)"])
        self.assertEqual(verdict["walk_forward"]["status"], "inconsistent")
        # 2 of 3 suffice; both terms then hold.
        verdict = rules.heldout_verdict(
            passing, {"source": "parent_control", "transitions": 3, "positive_excess": 2}
        )
        self.assertEqual((verdict["status"], verdict["reasons"]), ("graduated", []))
        self.assertEqual(verdict["walk_forward"]["status"], "consistent")
        # No transitions: term (b) is not applicable and (a) alone decides.
        verdict = rules.heldout_verdict(passing, {"source": "parent_control", "transitions": 0})
        self.assertEqual(verdict["status"], "graduated")
        self.assertEqual(verdict["walk_forward"], {"status": "not_applicable", "transitions": 0})
        self.assertEqual(rules.heldout_verdict(passing)["walk_forward"]["status"], "not_applicable")
        # Term (a) failures and term (b) failures are both listed.
        verdict = rules.heldout_verdict(
            {**passing, "sharpe": -0.1},
            {"source": "frozen_test", "transitions": 2, "positive_excess": 1},
        )
        self.assertEqual(
            verdict["reasons"],
            ["sharpe_not_positive", "walkforward_excess_inconsistent(1/2<2)"],
        )

    def test_record_round_trips_the_three_thresholds(self) -> None:
        rules = AcceptanceRules(min_return=0.01, min_sharpe=0.2, max_drawdown=0.3)
        self.assertEqual(
            rules.to_record(),
            {"min_return": 0.01, "min_sharpe": 0.2, "max_drawdown": 0.3},
        )


class RollingExperimentConfigValidationTest(unittest.TestCase):
    def test_valid_defaults_pass(self) -> None:
        config = make_config(Path("/tmp"))
        self.assertEqual(config.development_first_period, "2022Q1")
        self.assertFalse(config.test_stage)
        self.assertEqual(config.epochs, 3)
        self.assertEqual(config.meta_learning_fold_interval, 1)
        self.assertEqual(config.fold_exploration_directive, "")
        self.assertEqual(config.max_fold_minutes, 720)
        self.assertEqual(
            (config.max_steps_per_fold, config.max_backtests_per_fold, config.max_llm_calls),
            (30, 30, 1600),
        )
        self.assertEqual(config.experiment_dir, Path("/tmp/experiments/exp"))
        self.assertEqual(
            config.ledger_path,
            Path("/tmp/experiments/exp/ledgers/experiment_ledger.jsonl"),
        )

    def test_positive_int_knobs_reject_zero_negatives_floats_and_booleans(self) -> None:
        for name in (
            "epochs",
            "window_months",
            "min_region_trade_days",
            "max_steps_per_fold",
            "max_backtests_per_fold",
            "max_llm_calls",
            "max_fold_minutes",
        ):
            for value in (0, -1, 1.5, True, math.nan):
                with self.subTest(field=name, value=value):
                    with self.assertRaisesRegex(
                        ValueError, f"{name} must be a positive integer"
                    ):
                        make_config(Path("/tmp"), **{name: value})

    def test_non_negative_int_knobs_accept_zero_but_not_negatives(self) -> None:
        for name in ("meta_learning_fold_interval", "meta_memory_max_epochs"):
            self.assertEqual(getattr(make_config(Path("/tmp"), **{name: 0}), name), 0)
            for value in (-1, 1.5, True, math.inf):
                with self.subTest(field=name, value=value):
                    with self.assertRaisesRegex(
                        ValueError, f"{name} must be a non-negative integer"
                    ):
                        make_config(Path("/tmp"), **{name: value})

    def test_experiment_id_must_be_a_safe_path_component(self) -> None:
        for experiment_id in ("../escape", "with space", "sub/dir", "", "dot.name"):
            with self.subTest(experiment_id=experiment_id):
                with self.assertRaisesRegex(ValueError, "experiment_id"):
                    make_config(Path("/tmp"), experiment_id=experiment_id)

    def test_development_and_heldout_windows_must_not_overlap(self) -> None:
        # The held-out window is pre-registered and must stay strictly after the
        # last development test period; an overlap silently trains on held-out.
        with self.assertRaises(ValueError):
            make_config(
                Path("/tmp"),
                development_last_period="2023Q1",
                heldout_first_period="2023Q1",
                heldout_last_period="2023Q2",
            )


class DefaultsDriftTest(unittest.TestCase):
    """The console defaults, the domain dataclasses and the CLI must agree."""

    def test_console_defaults_match_the_domain_dataclasses(self) -> None:
        for field_obj in fields(RollingExperimentConfig):
            if (
                field_obj.name not in WEB_CREATE_DEFAULTS
                or field_obj.default is MISSING
            ):
                continue
            self.assertEqual(
                WEB_CREATE_DEFAULTS[field_obj.name], field_obj.default, field_obj.name
            )
        profile = BrokerProfile()
        for key in ("initial_cash", "commission_bps", "slippage_bps"):
            if key not in WEB_CREATE_DEFAULTS:
                continue
            self.assertEqual(WEB_CREATE_DEFAULTS[key], getattr(profile, key), key)
        rules = AcceptanceRules()
        for key in ("min_return", "min_sharpe", "max_drawdown"):
            self.assertEqual(WEB_CREATE_DEFAULTS[key], getattr(rules, key), key)
        schedule = StrategySchedule()
        self.assertEqual(WEB_CREATE_DEFAULTS["strategy_period"], schedule.period)
        self.assertEqual(WEB_CREATE_DEFAULTS["inference_time"], schedule.inference_time)

    def test_the_params_loader_defaults_are_the_dataclass_defaults(self) -> None:
        """An absent knob in `params.json` resolves to the dataclass default.

        The loader used to carry its own fallbacks, so an experiment created
        before a knob existed ran with a different cadence, budget or meta
        interval than the console offered, and nothing reported the divergence.
        """
        import tempfile

        from autotrade.pipelines.worker import resolve_worker_options

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "experiments").mkdir()
            options = resolve_worker_options(
                {
                    "experiment_id": "defaults_demo",
                    "development_first_period": "2023",
                    "development_last_period": "2025",
                    "heldout_first_period": "20260101..20260630",
                    "heldout_last_period": "20260101..20260630",
                    "strategy_path": "configs/agent_output_template/main.py",
                    "data_backend": "pit",
                    "raw_dir": "data/raw",
                    "fundamental_events_root": "data/pit/fundamental_events",
                    "fundamental_events_status": (
                        "results/data_quality/fundamental_events_status.json"
                    ),
                },
                experiment_dir=repo_root / "experiments/defaults_demo",
                repo_root=repo_root,
                preflight=True,
            )
        for field_obj in fields(RollingExperimentConfig):
            if field_obj.default is MISSING:
                continue
            with self.subTest(field=field_obj.name):
                self.assertEqual(
                    getattr(options.rolling, field_obj.name),
                    field_obj.default,
                    field_obj.name,
                )

    def test_the_console_create_form_is_seeded_with_the_research_preset(self) -> None:
        """The create defaults the owner set from a real launch.

        Pinned individually because they are the values a researcher gets
        without touching the form; a silent revert to the library fallbacks
        would change every new experiment and nothing else would notice.
        """
        self.assertEqual(
            {key: WEB_CREATE_DEFAULTS[key] for key in sorted(_CONSOLE_CREATE_PRESET)},
            dict(sorted(_CONSOLE_CREATE_PRESET.items())),
        )
        # The identity of an experiment is never pre-filled from the preset.
        self.assertIsNone(WEB_CREATE_DEFAULTS["experiment_id"])

    def test_the_parameter_schema_holds_no_second_copy_of_the_defaults(self) -> None:
        """`params_schema` renders `WEB_CREATE_DEFAULTS`, it does not restate it.

        Two tables would drift: the form would offer one value and the worker
        would be configured with another. Proved by moving the console table
        and watching every rendered default move with it.
        """
        from autotrade.webui.params_schema import parameter_schema

        def rendered() -> dict[str, object]:
            schema = parameter_schema()
            return {
                field["key"]: field["default"]
                for group in schema["groups"]
                for field in group["fields"]
            }

        baseline = rendered()
        self.assertEqual(
            baseline,
            {
                key: (list(value) if isinstance(value, tuple) else value)
                for key, value in WEB_CREATE_DEFAULTS.items()
                if key in baseline
            },
        )
        moved = {
            "model": "deepseek-v4-pro",
            "max_steps_per_fold": 7,
            "screen_boards": ("gem", "star"),
            "development_first_period": "2019Q3",
        }
        with patch.dict(WEB_CREATE_DEFAULTS, moved):
            after = rendered()
        self.assertEqual(after["model"], "deepseek-v4-pro")
        self.assertEqual(after["max_steps_per_fold"], 7)
        self.assertEqual(after["screen_boards"], ["gem", "star"])
        self.assertEqual(after["development_first_period"], "2019Q3")
        self.assertEqual(rendered(), baseline, "the schema retained a mutated default")

    def test_removed_minute_replay_knobs_are_absent_from_every_surface(self) -> None:
        removed = {
            "auction_enabled",
            "auction_preopen_time",
            "auction_decision_time",
            "auction_close_time",
            "execution_lag_bars",
            "intraday_decision_minutes",
            "decision_max_sim_minutes",
            "offsession_tick_minutes",
            "timeview_enabled",
            "replay_granularity",
        }
        self.assertTrue(
            removed.isdisjoint(field.name for field in fields(RollingExperimentConfig))
        )
        self.assertTrue(removed.isdisjoint(WEB_CREATE_DEFAULTS))

    def test_console_defaults_match_the_parameter_schema(self) -> None:
        from autotrade.webui.params_schema import parameter_schema

        schema = parameter_schema()
        for group in schema["groups"]:
            for field in group["fields"]:
                key = field["key"]
                if key not in WEB_CREATE_DEFAULTS or "default" not in field:
                    continue
                expected = WEB_CREATE_DEFAULTS[key]
                actual = field["default"]
                if isinstance(expected, tuple):
                    expected = list(expected)
                if field.get("type") in {"string", "period", "time", "text"}:
                    # A text field with no console default renders as an empty box.
                    expected = expected or ""
                    actual = actual or ""
                self.assertEqual(actual, expected, key)

    def test_worker_accepts_exactly_the_console_parameters(self) -> None:
        from autotrade.pipelines.worker import _ALLOWED_PARAMS

        # Every knob the console can write must be one the worker accepts,
        # otherwise creating an experiment produces a worker that refuses it.
        unknown = sorted(set(WEB_CREATE_DEFAULTS) - set(_ALLOWED_PARAMS))
        self.assertEqual(unknown, [])

    def test_cli_defaults_match_the_console_defaults(self) -> None:
        from scripts.experiments.run_experiment import build_parser

        parser = build_parser()
        skip = {
            # Repo-root-resolved path defaults (the console keeps them
            # repo-relative by design).
            "raw_dir",
            "fundamental_events_root",
            "fundamental_events_status",
            "experiments_root",
            "work_root",
            "template_dir",
        }
        mismatches = {}
        for action in parser._actions:
            if action.dest not in WEB_CREATE_DEFAULTS or action.dest in skip:
                continue
            cli_default = (
                tuple(action.default)
                if isinstance(action.default, list)
                else action.default
            )
            expected = WEB_CREATE_DEFAULTS[action.dest]
            expected = tuple(expected) if isinstance(expected, list) else expected
            if cli_default != expected:
                mismatches[action.dest] = (cli_default, expected)
        self.assertEqual(mismatches, {})


RESTORED_CONSOLE_PARAMETERS = (
    "commission_bps",
    "slippage_bps",
    "gpu_count",
    "nl_failure_policy",
    "per_call_timeout_seconds",
    "record_failed_attempts",
    "finalize_before_deadline_seconds",
    "convergence_start_epoch",
    "disable_step_tree",
    "max_total_holdings",
    "max_single_name_weight",
    "disable_meta_sandbox_rebuild",
    "meta_sandbox_rebuild_timeout_seconds",
    "meta_sandbox_image_keep",
)


class ConsoleParameterSurfaceTest(unittest.TestCase):
    """A create-form field must be renderable, submittable and persisted.

    The failure mode is a knob that renders and then 400s on submit, or one the
    worker silently drops: the console, the worker's accepted set and the
    on-disk params.json are asserted together.
    """

    def _schema_fields(self) -> dict:
        from autotrade.webui.params_schema import parameter_schema

        return {
            field["key"]: field
            for group in parameter_schema()["groups"]
            for field in group["fields"]
        }

    def test_every_restored_parameter_is_rendered_and_accepted(self) -> None:
        from autotrade.pipelines.worker import _ALLOWED_PARAMS

        fields = self._schema_fields()
        for key in RESTORED_CONSOLE_PARAMETERS:
            with self.subTest(key=key):
                self.assertIn(key, fields, f"{key} is not rendered on the create form")
                self.assertIn(key, WEB_CREATE_DEFAULTS, f"{key} has no console default")
                self.assertIn(key, _ALLOWED_PARAMS, f"the worker would reject {key}")
                self.assertTrue(fields[key].get("label"))
                self.assertTrue(fields[key].get("help"), f"{key} has no help text")

    def test_every_rendered_field_is_a_parameter_the_worker_accepts(self) -> None:
        from autotrade.pipelines.worker import _ALLOWED_PARAMS

        fields = self._schema_fields()
        # Rendering a field the worker rejects is a control that 400s on submit.
        unaccepted = sorted(
            set(fields) - set(_ALLOWED_PARAMS) - {"experiment_id", "inherit_from"}
        )
        self.assertEqual(unaccepted, [])
        # And every rendered field has a default the form can seed from.
        self.assertEqual(
            sorted(set(fields) - set(WEB_CREATE_DEFAULTS) - {"experiment_id"}), []
        )

    def test_the_defaults_match_the_domain_objects_they_configure(self) -> None:
        from autotrade.environment.broker import BrokerProfile

        profile = BrokerProfile()
        self.assertEqual(WEB_CREATE_DEFAULTS["commission_bps"], profile.commission_bps)
        self.assertEqual(WEB_CREATE_DEFAULTS["slippage_bps"], profile.slippage_bps)
        self.assertEqual(
            WEB_CREATE_DEFAULTS["max_total_holdings"], profile.max_total_holdings
        )
        self.assertEqual(
            WEB_CREATE_DEFAULTS["max_single_name_weight"],
            profile.max_single_name_weight,
        )
        config = make_config(Path("/tmp"))
        for key in (
            "convergence_start_epoch",
            "record_failed_attempts",
            "meta_sandbox_rebuild_timeout_seconds",
            "meta_sandbox_image_keep",
            "finalize_before_deadline_seconds",
            "per_call_timeout_seconds",
        ):
            with self.subTest(key=key):
                self.assertEqual(WEB_CREATE_DEFAULTS[key], getattr(config, key), key)

    def test_a_create_request_setting_all_of_them_persists_every_value(self) -> None:
        import tempfile
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from autotrade.webui.manager import ExperimentManager
        from autotrade.webui.server import create_app

        overrides = {
            "commission_bps": 2.5,
            "slippage_bps": 7.5,
            "gpu_count": 2,
            "nl_failure_policy": "fail",
            "per_call_timeout_seconds": 120,
            "record_failed_attempts": False,
            "finalize_before_deadline_seconds": 60,
            "convergence_start_epoch": 2,
            "disable_step_tree": True,
            "max_total_holdings": 12,
            "max_single_name_weight": 0.15,
            "disable_meta_sandbox_rebuild": True,
            "meta_sandbox_rebuild_timeout_seconds": 600,
            "meta_sandbox_image_keep": 1,
        }
        self.assertEqual(sorted(overrides), sorted(RESTORED_CONSOLE_PARAMETERS))
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with patch.object(
                ExperimentManager, "start_worker", return_value={"spawned": False}
            ):
                response = TestClient(create_app(repo_root)).post(
                    "/api/experiments",
                    json={
                        "params": {
                            "experiment_id": "params_demo",
                            "fold_period": "quarter",
                            "development_first_period": "2024Q1",
                            "development_last_period": "2024Q1",
                            "heldout_first_period": "2024Q2",
                            "heldout_last_period": "2024Q2",
                            **overrides,
                        }
                    },
                )
            self.assertEqual(response.status_code, 200, response.text)
            params = json.loads(
                (repo_root / "experiments/params_demo/hitl/params.json").read_text(
                    encoding="utf-8"
                )
            )
        for key, value in overrides.items():
            with self.subTest(key=key):
                self.assertEqual(params[key], value, key)


class WorkerEntryPointTest(unittest.TestCase):
    def test_the_worker_accepts_and_threads_a_poll_interval(self) -> None:
        import inspect

        from autotrade.pipelines.worker import run_local_interactive_worker

        signature = inspect.signature(run_local_interactive_worker)
        self.assertIn("poll_seconds", signature.parameters)
        self.assertEqual(signature.parameters["poll_seconds"].default, 2.0)

    def test_the_interactive_entry_point_exposes_the_flag(self) -> None:
        import importlib.util

        repo_root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "run_interactive_experiment",
            repo_root / "scripts/experiments/run_interactive_experiment.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parser = module.build_parser()
        args = parser.parse_args(
            ["--experiment-dir", "/tmp/exp", "--poll-seconds", "0.5"]
        )
        self.assertEqual(args.poll_seconds, 0.5)
        # A knob that is accepted and never forwarded is the defect class this
        # covers: the flag must reach the worker call.
        source = (
            repo_root / "scripts/experiments/run_interactive_experiment.py"
        ).read_text(encoding="utf-8")
        self.assertIn("poll_seconds=", source)
