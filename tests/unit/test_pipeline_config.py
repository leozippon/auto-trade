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
from dataclasses import MISSING, fields, replace
from pathlib import Path
from unittest.mock import patch

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.llm import LOCAL_QWEN_MODEL
from autotrade.environment.strategy import StrategySchedule
from autotrade.pipelines import verdict
from autotrade.pipelines.calendar import ResearchGeometry
from autotrade.pipelines.config import (
    DEFAULT_RESEARCH_GEOMETRY,
    MANDATED_DEFAULTS,
    AcceptanceRules,
    RollingExperimentConfig,
    acceptance_for,
)
from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS
from tests.unit.gpu_probe import stubbed_gpu_probe

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
    "gpu_count": 1,
    "include_events": True,
    "include_intraday": False,
    "include_text": True,
    "inference_time": "08:30",
    "initial_cash": 1_000_000.0,
    "initial_control_mode": "auto",
    # The research session's budgets: replay-years of batch_validate, minutes
    # and calls, spent across every attempt of the arm's one session.
    "max_replay_years": 96,
    "max_research_minutes": 2400,
    "max_llm_calls": 6400,
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
        self.assertEqual(rules.evaluate(summary), ["non_finite_total_return"])
        for key in ("sharpe", "max_drawdown"):
            with self.subTest(key=key):
                broken = {**summary, "total_return": 0.02, key: math.inf}
                self.assertIn(f"non_finite_{key}", rules.evaluate(broken))
        # A boolean is not a metric, however happily it compares.
        self.assertIn(
            "non_finite_total_return",
            rules.evaluate({**summary, "total_return": True}),
        )

    def test_finite_metrics_keep_threshold_semantics(self) -> None:
        rules = AcceptanceRules(max_drawdown=0.25)
        ok = {"total_return": 0.02, "sharpe": 0.5, "max_drawdown": 0.1}
        self.assertEqual(rules.evaluate(ok), [])
        # A drawdown breach is a HARD reject, sign-independent: F4/H3 enforce
        # the same limit forward, so freezing a book that already breached it
        # spends a forward test on a candidate the verdict must reject.
        for drawdown in (0.30, -0.30):
            with self.subTest(drawdown=drawdown):
                self.assertEqual(
                    rules.evaluate({**ok, "max_drawdown": drawdown}),
                    ["max_drawdown_above_limit"],
                )
        # It is the round's own limit, not a constant.
        self.assertEqual(
            AcceptanceRules(max_drawdown=0.35).evaluate({**ok, "max_drawdown": 0.30}), []
        )
        # A modest or negative result is not refused here: how much edge is
        # enough is the deflated Sharpe's question, asked by the gate itself.
        # A book that never traded is finite at 0.0 everywhere and passes the
        # same way -- it fails the gate, which has no IR to deflate.
        for summary in (
            {"total_return": -0.01, "sharpe": -0.2, "max_drawdown": 0.1},
            {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "order_count": 0},
        ):
            with self.subTest(summary=summary):
                self.assertEqual(rules.evaluate(summary), [])

    def test_absent_sharpe_is_not_an_integrity_failure(self) -> None:
        self.assertEqual(
            AcceptanceRules().evaluate({"total_return": 0.02, "max_drawdown": 0.1}), []
        )

    def test_rule_values_must_be_finite_and_ranged(self) -> None:
        band = {"tracking_error_cap": 0.08, "beta_min": 0.85, "beta_max": 1.15}
        for kwargs in (
            {"max_drawdown": math.nan},
            {"max_drawdown": 1.5},
            {"max_drawdown": -0.1},
            {"active_max_drawdown": 1.5},
            {"active_max_drawdown": math.nan},
            # The band is set exactly when the cap is, and is a band.
            {"tracking_error_cap": 0.08},
            {"beta_min": 0.85, "beta_max": 1.15},
            {**band, "beta_max": None},
            {**band, "beta_min": 1.15},
            {**band, "tracking_error_cap": 0.0},
            {**band, "tracking_error_cap": math.inf},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                AcceptanceRules(**kwargs)
        self.assertEqual(AcceptanceRules(**band).mandate, band)

    def test_record_round_trips_every_threshold(self) -> None:
        rules = AcceptanceRules(
            max_drawdown=0.3,
            cost_stress_multiplier=3.0,
            active_max_drawdown=0.12,
            tracking_error_cap=0.06,
            beta_min=0.9,
            beta_max=1.1,
        )
        self.assertEqual(
            rules.to_record(),
            {
                "max_drawdown": 0.3,
                "cost_stress_multiplier": 3.0,
                "active_max_drawdown": 0.12,
                "tracking_error_cap": 0.06,
                "beta_min": 0.9,
                "beta_max": 1.1,
            },
        )
        self.assertEqual(AcceptanceRules.from_record(rules.to_record()), rules)
        # Every params.json and run manifest written before the active series
        # was graded names the equity limit and the cost stress alone, and many
        # still carry the retired targets: such a record rebuilds the rules it
        # does name, with no tracking mandate.
        old = AcceptanceRules.from_record(
            {"min_return": 0.01, "min_sharpe": 0.2, "max_drawdown": 0.25, "cost_stress_multiplier": 2.0}
        )
        self.assertEqual(old, AcceptanceRules(max_drawdown=0.25))
        self.assertIsNone(old.tracking_error_cap)

    def test_the_tracking_mandate_is_set_by_hand_and_never_by_the_capital(self) -> None:
        """One switch, and it is the operator's: naming a ``tracking_error_cap``
        is what turns the mandate on. No account size appears in the rule, so
        the same request gives the same gates on any capital."""

        free = {
            "max_drawdown": 0.45,
            "cost_stress_multiplier": 2.0,
            "active_max_drawdown": 0.30,
            "tracking_error_cap": None,
            "beta_min": None,
            "beta_max": None,
        }
        tracking = {**free, "tracking_error_cap": 0.08, **MANDATED_DEFAULTS}
        # No cap: no mandate, whatever the request says about the account.
        self.assertEqual(acceptance_for({}).to_record(), free)
        self.assertEqual(AcceptanceRules().to_record(), free)
        for cash in (0, 100_000, 149_999, 150_000, 1_000_000, 10_000_000):
            with self.subTest(cash=cash):
                self.assertEqual(acceptance_for({"initial_cash": cash}).to_record(), free)
                self.assertEqual(
                    acceptance_for({"initial_cash": cash, "tracking_error_cap": 0.08}).to_record(),
                    tracking,
                )
        # A request that names the limits as nulls keeps the same defaults.
        self.assertEqual(acceptance_for(dict.fromkeys(free)).to_record(), free)
        # A cap brings the band and the tracker's drawdowns; each of them is
        # still overridable on its own.
        self.assertEqual(
            acceptance_for({"tracking_error_cap": 0.08, "max_drawdown": 0.3, "beta_min": 0.9}).to_record(),
            {**tracking, "max_drawdown": 0.3, "beta_min": 0.9},
        )
        self.assertEqual(
            acceptance_for({"max_drawdown": 0.3, "active_max_drawdown": 0.2}).to_record(),
            {**free, "max_drawdown": 0.3, "active_max_drawdown": 0.2},
        )
        # A band without a cap is refused, and so is a cap of zero: absence is
        # how the mandate is switched off, not a magic value.
        for overrides in (
            {"beta_min": 0.9, "beta_max": 1.1},
            {"beta_min": 0.9},
            {"tracking_error_cap": 0},
            {"tracking_error_cap": 0, "beta_min": 0.9},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                acceptance_for(overrides)

    def test_the_agent_facts_state_the_rules_from_their_single_sources(self) -> None:
        """The ``acceptance_rules`` fact is derived, never retyped: the freeze
        gate and verdict thresholds come from ``verdict`` and this experiment's
        own rules, and no date of any period appears in it."""

        rules = AcceptanceRules(
            max_drawdown=0.2, cost_stress_multiplier=3.0, active_max_drawdown=0.11
        )
        facts = rules.agent_facts()
        freeze = facts["freeze_gate"]
        self.assertIn(str(verdict.PANEL_DRAWS), facts["graded_series"])
        self.assertIn("benchmark.active_information_ratio", facts["graded_series"])
        self.assertIn(str(verdict.FREEZE_MIN_ACTIVE_IR), freeze["active_information_ratio"])
        self.assertIn("75%", freeze["positive_years"])
        self.assertIn("0.11", freeze["active_max_drawdown"])
        self.assertEqual(facts["graduation"]["forward"]["active_max_drawdown"], "<= 0.11")
        self.assertEqual(facts["graduation"]["heldout"]["active_max_drawdown"], "<= 0.11")
        # No mandate is said in words; a mandate states its two limits and that
        # meeting them is not evidence.
        self.assertIn("reported, not graded", freeze["tracking_mandate"])
        mandate = acceptance_for({"tracking_error_cap": 0.08}).agent_facts()["freeze_gate"][
            "tracking_mandate"
        ]
        self.assertIn("<= 0.08", mandate["tracking_error"])
        self.assertIn("0.85..1.15", mandate["market_beta"])
        self.assertIn("not evidence", mandate["role"])
        self.assertIn("span=full", freeze["span"])
        self.assertIn("finite", freeze["finite_metrics"])
        self.assertIn(str(verdict.FREEZE_MIN_DSR_PROBABILITY), freeze["deflated_sharpe_probability"])
        self.assertIn(str(verdict.FREEZE_MIN_FULL_SPAN_VALIDATIONS), freeze["full_span_validations"])
        self.assertEqual(freeze["freezes_per_arm"], 1)
        # The drawdown limit is the gate's, stated once and from the same field
        # the forward and Held-out conditions below quote.
        self.assertIn("0.2", freeze["max_drawdown"])
        self.assertEqual(facts["graduation"]["forward"]["max_drawdown"], "<= 0.2")
        # The gate states every rule it enforces and no rule it does not.
        self.assertEqual(sorted(facts), ["freeze_gate", "graded_series", "graduation"])
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
        self.assertEqual(config.session_max_attempts, 3)
        self.assertEqual(config.research_directive, "")
        self.assertEqual(config.max_research_minutes, 2400)
        self.assertEqual(
            (config.max_replay_years, config.max_llm_calls, config.max_null_controls),
            (96, 6400, 12),
        )
        self.assertEqual(config.experiment_dir, Path("/tmp/experiments/exp"))
        self.assertEqual(
            config.ledger_path,
            Path("/tmp/experiments/exp/ledgers/experiment_ledger.jsonl"),
        )

    def test_positive_int_knobs_reject_zero_negatives_floats_and_booleans(self) -> None:
        for name in (
            "session_max_attempts",
            "window_months",
            "max_replay_years",
            "max_llm_calls",
            "max_research_minutes",
        ):
            for value in (0, -1, 1.5, True, math.nan):
                with self.subTest(field=name, value=value):
                    with self.assertRaisesRegex(
                        ValueError, f"{name} must be a positive integer"
                    ):
                        make_config(Path("/tmp"), **{name: value})

    def test_non_negative_int_knobs_accept_zero_but_not_negatives(self) -> None:
        for name in (
            "max_null_controls",
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
        self.assertEqual(
            WEB_CREATE_DEFAULTS["cost_stress_multiplier"], rules.cost_stress_multiplier
        )
        # The limits the account's capital derives are offered empty, so the
        # form never pins one side of the capital switch onto every arm.
        for key in rules.to_record():
            if key != "cost_stress_multiplier":
                self.assertIsNone(WEB_CREATE_DEFAULTS[key], key)
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
            if field_obj.default is MISSING or field_obj.name == "acceptance":
                continue
            with self.subTest(field=field_obj.name):
                self.assertEqual(
                    getattr(options.rolling, field_obj.name),
                    field_obj.default,
                    field_obj.name,
                )
        # A request that names no limit gets the rules' own defaults, mandate
        # off, on whatever capital the Broker profile carries.
        self.assertEqual(options.rolling.acceptance, AcceptanceRules())
        self.assertEqual(
            RollingExperimentConfig("demo", Path("experiments")).acceptance,
            AcceptanceRules(),
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
                    "max_research_minutes": 300,
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
        self.assertEqual(options.rolling.max_research_minutes, 300)
        # The two the request names; it named no cap, so no tracking mandate.
        self.assertEqual(
            options.rolling.acceptance,
            replace(AcceptanceRules(), max_drawdown=0.2, cost_stress_multiplier=3.0),
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
            "max_replay_years": 7,
            "screen_boards": ("gem", "star"),
            "research_start": "20190701",
        }
        with patch.dict(WEB_CREATE_DEFAULTS, moved):
            after = rendered()
        self.assertEqual(after["model"], "deepseek-v4-pro")
        self.assertEqual(after["max_replay_years"], 7)
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

    def test_a_retired_parameter_is_accepted_but_offered_nowhere(self) -> None:
        """The acceptance targets set warnings no run ever recorded, so they are
        gone from the rules and from the create form. Every params.json on disk
        still names them, and rejecting a key the console itself wrote would
        make those arms unreadable to the listing and unresumable."""

        from autotrade.pipelines.worker import _ALLOWED_PARAMS, RETIRED_PARAMS

        fields = self._schema_fields()
        for key in RETIRED_PARAMS:
            with self.subTest(key=key):
                self.assertIn(key, _ALLOWED_PARAMS)
                self.assertNotIn(key, fields)
                self.assertNotIn(key, WEB_CREATE_DEFAULTS)

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
            "max_total_holdings": 12,
            "max_single_name_weight": 0.15,
        }
        self.assertEqual(sorted(overrides), sorted(RESTORED_CONSOLE_PARAMETERS))
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with (
                patch.object(ExperimentManager, "start_worker", return_value={"spawned": False}),
                stubbed_gpu_probe([0, 1]),
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

    def test_a_create_request_the_host_gpus_cannot_serve_is_refused(self) -> None:
        import tempfile
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from autotrade.environment.gpu import GpuUnavailableError
        from autotrade.webui.manager import ExperimentManager
        from autotrade.webui.server import create_app

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with (
                patch.object(ExperimentManager, "start_worker", return_value={"spawned": False}),
                patch(
                    "autotrade.environment.gpu.select_gpus",
                    side_effect=GpuUnavailableError("requested 2 GPU(s), 1 qualify"),
                ),
            ):
                response = TestClient(create_app(repo_root)).post(
                    "/api/experiments",
                    json={
                        "params": {
                            "experiment_id": "params_gpu",
                            **DEFAULT_RESEARCH_GEOMETRY.to_record(),
                            "gpu_count": 2,
                        }
                    },
                )
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn("当前 GPU 无法满足实验默认分配", response.json()["detail"])
            self.assertIn("1 qualify", response.json()["detail"])
            self.assertFalse((repo_root / "experiments/params_gpu").exists())


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

    def _resolve(self, repo_root: Path, params: dict, *, preflight: bool = True):
        from autotrade.pipelines.worker import resolve_worker_options

        merged = {**PIT_SEED_BASE_PARAMS, **params}
        return resolve_worker_options(
            merged,
            experiment_dir=repo_root / "experiments" / str(merged["experiment_id"]),
            repo_root=repo_root,
            preflight=preflight,
        )

    def _seed(
        self,
        repo_root: Path,
        name: str,
        params: dict,
        *,
        generation_id: str = "gen_seed",
        **release: object,
    ) -> Path:
        """A prebuilt seed tree carrying the contract `params` resolve to.

        Built through the same `_snapshot_config` the prebuild script calls,
        over a release published first (``release`` forwards to
        ``publish_release``), so the tree here is the one that script would
        leave behind.
        """

        from autotrade.pipelines.pit_backend import required_release_raw_datasets
        from autotrade.pipelines.pit_views_seed import pit_cache_provider_record
        from autotrade.pipelines.worker import _snapshot_config
        from tests.unit.research_release_fixture import publish_release

        config = _snapshot_config({**PIT_SEED_BASE_PARAMS, **params})
        built_from = publish_release(
            repo_root,
            generation_id,
            datasets=required_release_raw_datasets(config),
            **release,
        )
        seed = repo_root / "data" / name
        seed.mkdir(parents=True)
        (seed / "provider.json").write_text(
            json.dumps(
                pit_cache_provider_record(
                    generation_id=built_from.generation_id,
                    release_raw_dir=built_from.raw_dir,
                    snapshot_config=config,
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

    def test_a_named_seed_binds_the_release_it_was_built_from(self) -> None:
        """The nightly chain commits a new generation every night. An experiment
        naming a seed pins the release that seed was built from, so the worker
        still accepts the seed's views once the lake has moved on; an
        experiment naming none still pins the newest release."""
        import tempfile

        from autotrade.pipelines.pit_backend import (
            ResearchPITSnapshotProvider,
            required_release_raw_datasets,
        )
        from autotrade.pipelines.worker import _snapshot_config
        from tests.unit.research_release_fixture import publish_release

        selection = {"macro_datasets": ["cn_gdp", "fut_daily"]}
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            template = repo_root / "configs/agent_output_template/main.py"
            template.parent.mkdir(parents=True)
            template.write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
            seed = self._seed(repo_root, "pit_views_seed_ext", selection)
            view = seed / "decision" / "20250630T235959+0800"
            view.mkdir(parents=True)
            (view / "manifest.json").write_text("{}", encoding="utf-8")
            config = _snapshot_config({**PIT_SEED_BASE_PARAMS, **selection})
            publish_release(
                repo_root, "gen_newer", datasets=required_release_raw_datasets(config)
            )
            params = {**selection, "pit_views_seed": "data/pit_views_seed_ext"}

            self._resolve(repo_root, params)  # the console's create pre-flight
            options = self._resolve(repo_root, params, preflight=False)  # worker start
            snapshots = ResearchPITSnapshotProvider(
                experiment_dir=options.experiment_dir,
                raw_dir=options.raw_dir,
                fundamental_events_root=options.fundamental_events_root,
                fundamental_events_status=options.fundamental_events_status,
                config=options.snapshot_config,
                cache_root=options.pit_cache_root,
                pit_views_seed=options.pit_views_seed,
                pit_views_seed_required=options.pit_views_seed_required,
            )
            self.assertEqual(snapshots.release.generation_id, "gen_seed")
            self.assertTrue(
                (options.pit_cache_root / "decision" / view.name / "manifest.json").is_file()
            )

            unseeded = self._resolve(
                repo_root, {**selection, "experiment_id": "no_seed"}, preflight=False
            )
            pin = json.loads(
                (unseeded.experiment_dir / "research_release" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(pin["generation_id"], "gen_newer")

    def test_a_named_seed_whose_release_cannot_serve_fails_the_create(self) -> None:
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            template = repo_root / "configs/agent_output_template/main.py"
            template.parent.mkdir(parents=True)
            template.write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
            # The release the seed was built from is gone: refused by the
            # pre-flight and by the worker, which pins nothing else instead.
            self._seed(repo_root, "seed_gone", {})
            shutil.rmtree(repo_root / "data" / "research_releases" / "gen_seed")
            params = {"pit_views_seed": "data/seed_gone"}
            for preflight in (True, False):
                with self.assertRaisesRegex(ValueError, r"gen_seed.*missing or incomplete"):
                    self._resolve(repo_root, params, preflight=preflight)
            self.assertFalse((repo_root / "experiments/seed_demo/research_release").exists())
            # The release ends before Held-out: refused at create, not at start.
            self._seed(
                repo_root,
                "seed_short",
                {},
                generation_id="gen_short",
                trading_days=("20250630", "20260630"),
            )
            with self.assertRaisesRegex(ValueError, "Held-out"):
                self._resolve(repo_root, {"pit_views_seed": "data/seed_short"})

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
