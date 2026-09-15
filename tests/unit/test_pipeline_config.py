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
from autotrade.environment.llm import LOCAL_QWEN_MODEL
from autotrade.environment.strategy import StrategySchedule
from autotrade.pipelines import verdict
from autotrade.pipelines.calendar import ResearchGeometry
from autotrade.pipelines.config import (
    DEFAULT_RESEARCH_GEOMETRY,
    AcceptanceRules,
    RollingExperimentConfig,
)
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
    # Four July-June research years, a forward year, a Held-out quarter.
    "research_start": "20210701",
    "research_end": "20250630",
    "forward_end": "20260630",
    "heldout_end": "20260930",
    "research_sessions": 4,
    "gpu_count": 1,
    "include_events": True,
    "include_intraday": False,
    "include_text": True,
    "inference_time": "08:30",
    "initial_cash": 1_000_000.0,
    "initial_control_mode": "auto",
    # Per-session budgets: replay-years of batch_validate, minutes and calls.
    "max_replay_years_per_session": 24,
    "max_session_minutes": 720,
    "max_llm_calls": 1600,
    "model": LOCAL_QWEN_MODEL,
    # The universe reaches the agent unfiltered; the strategy filters itself.
    "screen_boards": (),
    "screen_exclude_new_listed_days": 0,
    "screen_exclude_st": False,
    "strategy_period": "day",
    "window_months": 24,
}


def make_config(root: Path, **overrides: object) -> RollingExperimentConfig:
    values: dict[str, object] = {
        "experiment_id": "exp",
        "experiments_root": root / "experiments",
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
        # A drawdown breach WARNS, sign-independent: the fold still freezes its
        # validated work and the cap decides at graduation instead.
        for drawdown in (0.30, -0.30):
            with self.subTest(drawdown=drawdown):
                hard, warnings = rules.evaluate({**ok, "max_drawdown": drawdown})
                self.assertEqual(hard, [])
                self.assertIn("drawdown_above_target", warnings)
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
        self.assertEqual(warnings, ["no_orders"])
        # One filled order is a result, however small; it must not warn. A
        # buy-and-hold book has orders but no closed round trip, so
        # trade_count stays 0 and must not be the trigger.
        traded = {**flat, "order_count": 1, "total_return": 0.01, "sharpe": 0.1}
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

    def test_record_round_trips_every_threshold(self) -> None:
        rules = AcceptanceRules(
            min_return=0.01, min_sharpe=0.2, max_drawdown=0.3, cost_stress_multiplier=3.0
        )
        self.assertEqual(
            rules.to_record(),
            {
                "min_return": 0.01,
                "min_sharpe": 0.2,
                "max_drawdown": 0.3,
                "cost_stress_multiplier": 3.0,
            },
        )
        self.assertEqual(AcceptanceRules.from_record(rules.to_record()), rules)

    def test_the_agent_facts_state_the_rules_from_their_single_sources(self) -> None:
        """The ``acceptance_rules`` fact is derived, never retyped: the freeze
        gate and verdict thresholds come from ``verdict`` and this experiment's
        own rules, and no date of any period appears in it."""

        facts = AcceptanceRules(max_drawdown=0.2, cost_stress_multiplier=3.0).agent_facts()
        freeze = facts["freeze_gate"]
        self.assertIn("span=full", freeze["span"])
        self.assertIn("finite", freeze["finite_metrics"])
        self.assertIn(str(verdict.FREEZE_MIN_DSR_PROBABILITY), freeze["deflated_sharpe_probability"])
        self.assertIn(str(verdict.FREEZE_MIN_FULL_SPAN_VALIDATIONS), freeze["full_span_validations"])
        self.assertEqual(freeze["freezes_per_arm"], 1)
        targets = facts["targets"]
        self.assertEqual(targets["max_drawdown"], 0.2)
        self.assertIn("not selection criteria", targets["role"])
        graduation = facts["graduation"]
        self.assertIn("tracking error", graduation["forward"]["minimum_detectable_excess"])
        self.assertEqual(graduation["forward"]["max_drawdown"], "<= 0.2")
        self.assertIn("3.0", graduation["forward"]["excess_at_cost_stress"])
        self.assertEqual(graduation["heldout"]["max_drawdown"], "<= 0.2")
        rendered = json.dumps(facts)
        self.assertIsNone(re.search(r"20\d{6}", rendered))

    def test_a_record_with_a_retired_key_still_rebuilds_the_rules(self) -> None:
        rules = AcceptanceRules.from_record(
            {"max_drawdown": 0.2, "heldout_min_trades": 5, "confirmation_folds": 2}
        )
        self.assertEqual(rules, AcceptanceRules(max_drawdown=0.2))

    def test_the_cost_stress_multiplier_must_be_a_stress(self) -> None:
        with self.assertRaisesRegex(ValueError, "cost_stress_multiplier must be at least one"):
            AcceptanceRules(cost_stress_multiplier=0.5)
        with self.assertRaisesRegex(ValueError, "cost_stress_multiplier must be finite"):
            AcceptanceRules(cost_stress_multiplier=float("nan"))


class RollingExperimentConfigValidationTest(unittest.TestCase):
    def test_valid_defaults_pass(self) -> None:
        config = make_config(Path("/tmp"))
        self.assertEqual(config.geometry, DEFAULT_RESEARCH_GEOMETRY)
        self.assertEqual((config.research_sessions, config.session_max_attempts), (4, 3))
        self.assertEqual(config.fold_exploration_directive, "")
        self.assertEqual(config.max_session_minutes, 720)
        self.assertEqual(
            (config.max_replay_years_per_session, config.max_llm_calls),
            (24, 1600),
        )
        self.assertEqual(config.experiment_dir, Path("/tmp/experiments/exp"))
        self.assertEqual(
            config.ledger_path,
            Path("/tmp/experiments/exp/ledgers/experiment_ledger.jsonl"),
        )

    def test_positive_int_knobs_reject_zero_negatives_floats_and_booleans(self) -> None:
        for name in (
            "research_sessions",
            "session_max_attempts",
            "window_months",
            "max_replay_years_per_session",
            "max_llm_calls",
            "max_session_minutes",
        ):
            for value in (0, -1, 1.5, True, math.nan):
                with self.subTest(field=name, value=value):
                    with self.assertRaisesRegex(
                        ValueError, f"{name} must be a positive integer"
                    ):
                        make_config(Path("/tmp"), **{name: value})

    def test_non_negative_int_knobs_accept_zero_but_not_negatives(self) -> None:
        for name in (
            "max_null_controls_per_session",
            "deadline_grace_minutes",
            "finalize_before_deadline_seconds",
        ):
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

    def test_the_geometry_must_be_a_research_geometry(self) -> None:
        with self.assertRaisesRegex(TypeError, "geometry must be a ResearchGeometry"):
            make_config(Path("/tmp"), geometry=DEFAULT_RESEARCH_GEOMETRY.to_record())


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
        for key in (
            "min_return",
            "min_sharpe",
            "max_drawdown",
            "cost_stress_multiplier",
        ):
            self.assertEqual(WEB_CREATE_DEFAULTS[key], getattr(rules, key), key)
        for key, value in DEFAULT_RESEARCH_GEOMETRY.to_record().items():
            self.assertEqual(WEB_CREATE_DEFAULTS[key], value, key)
        schedule = StrategySchedule()
        self.assertEqual(WEB_CREATE_DEFAULTS["strategy_period"], schedule.period)
        self.assertEqual(WEB_CREATE_DEFAULTS["inference_time"], schedule.inference_time)

    def test_the_params_loader_defaults_are_the_dataclass_defaults(self) -> None:
        """An absent knob in `params.json` resolves to the dataclass default.

        The loader used to carry its own fallbacks, so an experiment created
        before a knob existed ran with a different cadence or budget than the
        console offered, and nothing reported the divergence.
        """
        import tempfile

        from autotrade.pipelines.worker import resolve_worker_options

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "experiments").mkdir()
            options = resolve_worker_options(
                {
                    "experiment_id": "defaults_demo",
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

    def test_the_geometry_and_gate_knobs_reach_the_configuration(self) -> None:
        """A knob accepted and never forwarded is the defect class here."""
        import tempfile

        from autotrade.pipelines.worker import resolve_worker_options

        geometry = ResearchGeometry(
            research_start="20210701",
            research_end="20240630",
            forward_end="20250630",
            heldout_end="20260630",
        )
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "experiments").mkdir()
            options = resolve_worker_options(
                {
                    "experiment_id": "geometry_demo",
                    **geometry.to_record(),
                    "research_sessions": 3,
                    "cost_stress_multiplier": 3.0,
                    "max_drawdown": 0.2,
                    "strategy_path": "configs/agent_output_template/main.py",
                    "data_backend": "pit",
                    "raw_dir": "data/raw",
                    "fundamental_events_root": "data/pit/fundamental_events",
                    "fundamental_events_status": (
                        "results/data_quality/fundamental_events_status.json"
                    ),
                },
                experiment_dir=repo_root / "experiments/geometry_demo",
                repo_root=repo_root,
                preflight=True,
            )
        self.assertEqual(options.rolling.geometry, geometry)
        self.assertEqual(options.rolling.research_sessions, 3)
        self.assertEqual(
            options.rolling.acceptance,
            AcceptanceRules(max_drawdown=0.2, cost_stress_multiplier=3.0),
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
            "max_replay_years_per_session": 7,
            "screen_boards": ("gem", "star"),
            "research_start": "20190701",
        }
        with patch.dict(WEB_CREATE_DEFAULTS, moved):
            after = rendered()
        self.assertEqual(after["model"], "deepseek-v4-pro")
        self.assertEqual(after["max_replay_years_per_session"], 7)
        self.assertEqual(after["screen_boards"], ["gem", "star"])
        self.assertEqual(after["research_start"], "20190701")
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
    "disable_step_tree",
    "max_total_holdings",
    "max_single_name_weight",
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
            set(fields) - set(_ALLOWED_PARAMS) - {"experiment_id"}
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
            "record_failed_attempts",
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
            "disable_step_tree": True,
            "max_total_holdings": 12,
            "max_single_name_weight": 0.15,
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
                            **DEFAULT_RESEARCH_GEOMETRY.to_record(),
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


PIT_SEED_BASE_PARAMS = {
    "experiment_id": "seed_demo",
    "strategy_path": "configs/agent_output_template/main.py",
    "data_backend": "pit",
    "raw_dir": "data/raw",
    "fundamental_events_root": "data/pit/fundamental_events",
    "fundamental_events_status": "results/data_quality/fundamental_events_status.json",
}


class PitViewsSeedParameterTest(unittest.TestCase):
    """Which prebuilt PIT views an experiment may reuse is its own parameter.

    The default tree is an optimisation and stays lenient; a tree named
    explicitly is the only way an arm with a non-default dataset selection gets
    prebuilt views at all, so it is checked at create time instead of turning
    into hours of silent cold building.
    """

    def _resolve(self, repo_root: Path, params: dict):
        from autotrade.pipelines.worker import resolve_worker_options

        return resolve_worker_options(
            {**PIT_SEED_BASE_PARAMS, **params},
            experiment_dir=repo_root / "experiments/seed_demo",
            repo_root=repo_root,
            preflight=True,
        )

    def _seed(self, repo_root: Path, name: str, params: dict) -> Path:
        """A prebuilt seed tree carrying the contract `params` resolve to.

        Built through the same `_snapshot_config` the prebuild script calls, so
        the tree here is the one that script would leave behind.
        """

        from autotrade.pipelines.pit_views_seed import pit_cache_provider_record
        from autotrade.pipelines.worker import _snapshot_config

        seed = repo_root / "data" / name
        seed.mkdir(parents=True)
        (seed / "provider.json").write_text(
            json.dumps(
                pit_cache_provider_record(
                    generation_id="generation_test",
                    release_raw_dir=repo_root / "raw",
                    snapshot_config=_snapshot_config({**PIT_SEED_BASE_PARAMS, **params}),
                )
            ),
            encoding="utf-8",
        )
        return seed

    def test_the_default_seed_is_optional_and_an_explicit_one_is_required(self) -> None:
        import tempfile

        from autotrade.pipelines.config import DEFAULT_PIT_VIEWS_SEED

        selection = {"macro_datasets": ["cn_gdp", "fut_daily"]}
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "experiments").mkdir()
            # Absent parameter: the default tree, which need not exist here.
            options = self._resolve(repo_root, {})
            self.assertEqual(
                options.pit_views_seed, repo_root / DEFAULT_PIT_VIEWS_SEED
            )
            self.assertFalse(options.pit_views_seed_required)
            # Naming the default explicitly must not turn it into a demand.
            options = self._resolve(
                repo_root, {"pit_views_seed": str(DEFAULT_PIT_VIEWS_SEED)}
            )
            self.assertEqual(
                options.pit_views_seed, repo_root / DEFAULT_PIT_VIEWS_SEED
            )
            self.assertFalse(options.pit_views_seed_required)
            # A matching tree named explicitly: resolved and mandatory.
            seed = self._seed(repo_root, "pit_views_seed_ext", selection)
            options = self._resolve(
                repo_root, {**selection, "pit_views_seed": "data/pit_views_seed_ext"}
            )
            self.assertEqual(options.pit_views_seed, seed)
            self.assertTrue(options.pit_views_seed_required)

    def test_a_seed_that_cannot_apply_fails_the_create(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "experiments").mkdir()
            with self.assertRaisesRegex(ValueError, "existing directory"):
                self._resolve(repo_root, {"pit_views_seed": "data/never_built"})
            with self.assertRaisesRegex(ValueError, "must stay inside the repository"):
                self._resolve(repo_root, {"pit_views_seed": "../elsewhere"})
            # Built for the default selection, asked for by an arm that adds a
            # dataset: the mismatch names both contracts instead of silently
            # cold-building every view.
            self._seed(repo_root, "pit_views_seed_default", {})
            with self.assertRaises(ValueError) as caught:
                self._resolve(
                    repo_root,
                    {
                        "macro_datasets": ["cn_gdp", "fut_daily"],
                        "pit_views_seed": "data/pit_views_seed_default",
                    },
                )
            message = str(caught.exception)
            self.assertIn("fut_daily", message)
            self.assertIn("sw_daily", message)

    def test_the_daily_backend_needs_no_pit_seed(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "experiments").mkdir()
            (repo_root / "configs" / "agent_output_template").mkdir(parents=True)
            template = repo_root / "configs/agent_output_template/main.py"
            template.write_text("def generate_orders(context):\n    return []\n")
            daily = repo_root / "daily.parquet"
            daily.write_text("not read at preflight", encoding="utf-8")
            options = self._resolve(
                repo_root,
                {
                    "data_backend": "daily",
                    "daily_path": "daily.parquet",
                    "developer_mode": "baseline",
                },
            )
            self.assertIsNone(options.pit_views_seed)
            self.assertFalse(options.pit_views_seed_required)
