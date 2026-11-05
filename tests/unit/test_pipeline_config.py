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

import pytest

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.gpu import host_min_free_gpu_memory_mib
from autotrade.environment.strategy import StrategySchedule
from autotrade.pipelines import verdict
from autotrade.pipelines.calendar import ResearchGeometry
from autotrade.pipelines.config import (
    ACCEPTANCE_KEYS,
    BATCH_VALIDATE_MAX_CANDIDATES,
    DEFAULT_RESEARCH_GEOMETRY,
    MANDATED_DEFAULTS,
    RETIRED_SWITCHES,
    AcceptanceRules,
    RollingExperimentConfig,
    acceptance_for,
)
from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS, WEB_REQUIRED_PARAMS
from tests.unit.acceptance_fixture import CREATE_RULES, REQUIRED_RULES
from tests.unit.gpu_probe import gpu_roster, stubbed_gpu_probe

#: The console create form is seeded from the pinned explore profile so a
#: new experiment hardlinks the PIT view seed. Period, screen, compact, and
#: loop knobs that a researcher gets without touching the form live here.
_CONSOLE_CREATE_PRESET: dict[str, object] = {
    "compact_keep_recent_messages": 10,
    "compact_max_calls": 10,
    "compact_max_tokens": 10_000,
    # Null: each conversation role compacts at its bound.
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
    "model": "gpt-6-astra",
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
        "acceptance": AcceptanceRules(**REQUIRED_RULES),
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
        rules = AcceptanceRules(**REQUIRED_RULES)
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
        rules = AcceptanceRules(**REQUIRED_RULES, max_drawdown=0.25)
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
            AcceptanceRules(**REQUIRED_RULES, max_drawdown=0.35).evaluate({**ok, "max_drawdown": 0.30}), []
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
            AcceptanceRules(**REQUIRED_RULES).evaluate({"total_return": 0.02, "max_drawdown": 0.1}), []
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
            {"alpha": math.nan},
            {"alpha": 0.0},
            {"alpha": 1.0},
            {"sealed_years": 0},
            {"sealed_years": 2.5},
            {"ablation_required": 1},
            {"ablation_required": "true"},
            {"forward_confidence": 0.0},
            {"forward_confidence": 1.0},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                AcceptanceRules(**{**REQUIRED_RULES, **kwargs})
        self.assertEqual(AcceptanceRules(**REQUIRED_RULES, **band).mandate, band)

    def test_record_round_trips_every_threshold(self) -> None:
        rules = AcceptanceRules(**REQUIRED_RULES,
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
                **AcceptanceRules(**REQUIRED_RULES).to_record(),
                "max_drawdown": 0.3,
                "cost_stress_multiplier": 3.0,
                "active_max_drawdown": 0.12,
                "tracking_error_cap": 0.06,
                "beta_min": 0.9,
                "beta_max": 1.1,
            },
        )
        self.assertEqual(AcceptanceRules.from_record(rules.to_record()), rules)
        # A params.json or run manifest written before the required rules
        # existed -- every arm created under the deflated-Sharpe gate -- is
        # refused, never read under a value it did not state.
        for missing in REQUIRED_RULES:
            stated = {key: value for key, value in REQUIRED_RULES.items() if key != missing}
            with self.assertRaisesRegex(ValueError, f"lack the required {missing}"):
                AcceptanceRules.from_record({"max_drawdown": 0.25, "min_active_ir": 0.75, **stated})
            with self.assertRaisesRegex(ValueError, f"lack the required {missing}"):
                acceptance_for({"max_drawdown": 0.25, **stated})
            with self.assertRaises(TypeError):
                AcceptanceRules(max_drawdown=0.25, **stated)  # type: ignore[arg-type]
        # A record that states them rebuilds the rules it names, retired bars
        # ignored, with no tracking mandate, and the forward bars it leaves out
        # resolve to today's defaults.
        old = AcceptanceRules.from_record(
            {
                "min_return": 0.01,
                "max_drawdown": 0.25,
                "min_dsr_probability": 0.975,
                "min_mean_gross": 0.5,
                "heldout_tolerance_z": 1.28,
                **REQUIRED_RULES,
            }
        )
        self.assertEqual(old, AcceptanceRules(**REQUIRED_RULES, max_drawdown=0.25))
        self.assertIsNone(old.tracking_error_cap)
        self.assertEqual(old.forward_confidence, verdict.FORWARD_CONFIDENCE)
        self.assertEqual(acceptance_for({**REQUIRED_RULES, "max_drawdown": 0.25}), AcceptanceRules(**REQUIRED_RULES, max_drawdown=0.25))

    def test_the_tracking_mandate_is_set_by_hand_and_never_by_the_capital(self) -> None:
        """One switch, and it is the operator's: naming a ``tracking_error_cap``
        is what turns the mandate on. No account size appears in the rule, so
        the same request gives the same gates on any capital."""

        free = AcceptanceRules(**REQUIRED_RULES).to_record()
        tracking = {**free, "tracking_error_cap": 0.08, **MANDATED_DEFAULTS}
        # No cap: no mandate, whatever the request says about the account.
        self.assertEqual(acceptance_for(REQUIRED_RULES).to_record(), free)
        self.assertEqual(AcceptanceRules(**REQUIRED_RULES).to_record(), free)
        for cash in (0, 100_000, 149_999, 150_000, 1_000_000, 10_000_000):
            with self.subTest(cash=cash):
                self.assertEqual(acceptance_for({**REQUIRED_RULES, "initial_cash": cash}).to_record(), free)
                self.assertEqual(
                    acceptance_for({**REQUIRED_RULES, "initial_cash": cash, "tracking_error_cap": 0.08}).to_record(),
                    tracking,
                )
                self.assertEqual(tracking["max_drawdown"], 0.45)
                self.assertEqual(tracking["active_max_drawdown"], 0.30)
        # A request that names the defaulted limits as nulls keeps the same
        # defaults; a required rule named as null is a rule not stated.
        self.assertEqual(
            acceptance_for({**dict.fromkeys(free), **REQUIRED_RULES}).to_record(), free
        )
        with self.assertRaisesRegex(ValueError, "lack the required alpha"):
            acceptance_for(dict.fromkeys(free))
        # A cap fills a blank beta band; drawdowns stay 0.45 / 0.30 unless named.
        self.assertEqual(
            acceptance_for({**REQUIRED_RULES, "tracking_error_cap": 0.08, "max_drawdown": 0.3, "beta_min": 0.9}).to_record(),
            {**tracking, "max_drawdown": 0.3, "beta_min": 0.9},
        )
        self.assertEqual(
            acceptance_for({**REQUIRED_RULES, "max_drawdown": 0.3, "active_max_drawdown": 0.2}).to_record(),
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
        """The ``acceptance_rules`` fact is derived, never retyped: the sealed
        gate's statistic and bar come from ``verdict`` and this experiment's
        own rules, registration lists only what it checks, and no date of any
        period appears in it."""

        rules = AcceptanceRules(**{**REQUIRED_RULES, "sealed_years": 5},
            max_drawdown=0.2, cost_stress_multiplier=3.0, active_max_drawdown=0.11
        )
        facts = rules.agent_facts()
        self.assertEqual(sorted(facts), ["freeze_registration", "graded_series", "graduation", "sealed_gate"])
        self.assertIn(str(verdict.PANEL_DRAWS), facts["graded_series"])
        self.assertIn("benchmark.active_information_ratio", facts["graded_series"])
        sealed = facts["sealed_gate"]
        self.assertIn("intercept times sqrt(244) over the residual standard deviation", sealed["statistic"])
        self.assertIn("the size leg and the value leg", sealed["statistic"])
        self.assertEqual(sealed["bar"], "z(1 - alpha) / sqrt(years)")
        self.assertEqual((sealed["alpha"], sealed["sealed_years"]), (0.05, 5))
        self.assertEqual(sealed["bar_over_sealed_years"], round(verdict.residual_bar(5.0, 0.05), 3))
        self.assertEqual(sealed["bar_over_one_year"], 1.645)
        self.assertIn("there is no second reading", sealed["read"])
        self.assertIn("residual_reading", sealed["on_visible_spans"])
        registration = facts["freeze_registration"]
        self.assertEqual(
            sorted(registration),
            [
                "ablation",
                "control",
                "finite_metrics",
                "freezes_per_arm",
                "max_drawdown",
                "seed_replicates",
                "span",
                "tracking_mandate",
            ],
        )
        # An arm that asks for no ablation says so; one that asks defines it.
        self.assertIn("takes no ablation", registration["ablation"])
        asked = AcceptanceRules(**{**REQUIRED_RULES, "ablation_required": True}).agent_facts()
        ablation = asked["freeze_registration"]["ablation"]
        self.assertTrue(ablation["required"])
        self.assertIn("exactly one line changed", ablation["definition"])
        self.assertIn("never gates", ablation["reading"])
        self.assertIn("span=full", registration["span"])
        self.assertIn("finite", registration["finite_metrics"])
        self.assertIn("0.2", registration["max_drawdown"])
        self.assertIn("ends the session", registration["freezes_per_arm"])
        self.assertIn(
            "declared deterministic in finish_session (its package uses no randomness) owes none",
            registration["seed_replicates"],
        )
        # No mandate is said in words; a mandate states its two limits and that
        # meeting them is not evidence.
        self.assertIn("reported, not graded", registration["tracking_mandate"])
        mandate = acceptance_for({**REQUIRED_RULES, "tracking_error_cap": 0.08}).agent_facts()[
            "freeze_registration"
        ]["tracking_mandate"]
        self.assertIn("<= 0.08", mandate["tracking_error"])
        self.assertIn("0.85..1.15", mandate["market_beta"])
        self.assertIn("not evidence", mandate["role"])
        # The forward readings are stated with what they are taken at, and no
        # bar: none is judged.
        self.assertIn("slippage multiplied by 3.0", facts["graduation"]["reported"])
        self.assertNotIn("<=", json.dumps(facts["graduation"]))
        rendered = json.dumps(facts)
        self.assertIsNone(re.search(r"20\d{6}", rendered))
        for retired in ("deflated", "trials", "offline", "positive_years", "information_ratio_bar"):
            self.assertNotIn(retired, rendered)

    def test_one_series_is_stated_and_the_sealed_gate_alone_admits(self) -> None:
        """Every reading is of the active series: the facts name it once and
        no neutralized figure. Graduation is the sealed gate's alone; the
        forward bound and the cost stress are stated as readings, over the
        frozen book and its seed replicates, that decide nothing."""

        facts = AcceptanceRules(**REQUIRED_RULES).agent_facts()
        graded = facts["graded_series"]
        for field in ("benchmark.active_excess", "sub_windows[].active_excess_return", "daily_series"):
            self.assertIn(field, graded)
        self.assertNotIn("neutral", json.dumps(facts).lower())
        graduation = facts["graduation"]
        self.assertEqual(sorted(graduation), ["admits", "reported"])
        self.assertIn("exactly when the sealed gate passes", graduation["admits"])
        self.assertIn("80% block-bootstrap lower bound", graduation["reported"])
        self.assertIn("frozen book and its seed replicates", graduation["reported"])
        self.assertIn("none decides the verdict", graduation["reported"])

    def test_a_record_with_a_retired_key_still_rebuilds_the_rules(self) -> None:
        rules = AcceptanceRules.from_record(
            {
                "max_drawdown": 0.2,
                "heldout_min_trades": 5,
                "confirmation_folds": 2,
                **dict.fromkeys(RETIRED_SWITCHES, False),
                **REQUIRED_RULES,
            }
        )
        self.assertEqual(rules, AcceptanceRules(**REQUIRED_RULES, max_drawdown=0.2))

    def test_the_cost_stress_multiplier_must_be_a_stress(self) -> None:
        with self.assertRaisesRegex(ValueError, "cost_stress_multiplier must be at least one"):
            AcceptanceRules(**REQUIRED_RULES, cost_stress_multiplier=0.5)
        with self.assertRaisesRegex(ValueError, "cost_stress_multiplier must be finite"):
            AcceptanceRules(**REQUIRED_RULES, cost_stress_multiplier=float("nan"))


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

    def test_batch_width_runs_from_one_to_the_candidate_cap(self) -> None:
        name = "batch_validate_max_concurrency"
        self.assertEqual(getattr(make_config(Path("/tmp")), name), 3)
        for value in (1, BATCH_VALIDATE_MAX_CANDIDATES):
            self.assertEqual(getattr(make_config(Path("/tmp"), **{name: value}), name), value)
        for value in (0, BATCH_VALIDATE_MAX_CANDIDATES + 1, 2.0, True):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, f"{name} must be an integer from 1 to {BATCH_VALIDATE_MAX_CANDIDATES}"
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
        rules = AcceptanceRules(**REQUIRED_RULES).to_record()
        # A retired switch is no parameter a request can name.
        self.assertFalse(RETIRED_SWITCHES & set(WEB_CREATE_DEFAULTS))
        # Drawdowns and the mandate switch stay empty so the form does not pin
        # a mandate onto every arm, and the sealed years and the ablation have
        # no console value: every request states them (WEB_REQUIRED_PARAMS).
        # Every other rule shows what an empty request would stamp.
        stated = {"sealed_years", "ablation_required"}
        blank = {"max_drawdown", "active_max_drawdown", "tracking_error_cap", "beta_min", "beta_max", *stated}
        self.assertEqual(set(rules), set(ACCEPTANCE_KEYS))
        self.assertLessEqual(stated, WEB_REQUIRED_PARAMS)
        for key, value in rules.items():
            if key in blank:
                self.assertIsNone(WEB_CREATE_DEFAULTS[key], key)
            else:
                self.assertEqual(WEB_CREATE_DEFAULTS[key], value, key)
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
                    **REQUIRED_RULES,
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
        self.assertEqual(options.rolling.acceptance, AcceptanceRules(**REQUIRED_RULES))

    def test_the_retired_switches_are_accepted_and_inert_and_any_other_key_refused(self) -> None:
        """Every params.json written before 2026-10-07 carries the retired
        switches. The worker loads such an arm as if they were absent, whatever
        their values, and runs it under the one rule set: its account pays the
        dividend tax, which the run facts state. A key it does not know is
        refused as ever."""
        import tempfile

        from autotrade.agent.experiment_facts import _broker_replay_facts
        from autotrade.pipelines.worker import resolve_worker_options

        base = {
            "experiment_id": "legacy_demo",
            "strategy_path": "configs/agent_output_template/main.py",
            **REQUIRED_RULES,
            "data_backend": "pit",
            "raw_dir": "data/raw",
            "fundamental_events_root": "data/pit/fundamental_events",
            "fundamental_events_status": "results/data_quality/fundamental_events_status.json",
        }
        self.assertEqual(
            RETIRED_SWITCHES,
            {
                "require_raw_excess_at_cost_stress",
                "require_forward_plain_selection",
                "require_seed_replicates",
                "independent_offline_trials",
                "dividend_tax",
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "experiments").mkdir()

            def options(params: dict[str, object]):
                return resolve_worker_options(
                    params,
                    experiment_dir=repo_root / "experiments/legacy_demo",
                    repo_root=repo_root,
                    preflight=True,
                )

            current = options(base)
            for value in (True, False, "not even a boolean"):
                with self.subTest(value=value):
                    legacy = options({**base, **dict.fromkeys(RETIRED_SWITCHES, value)})
                    self.assertEqual(legacy, current)
            with self.assertRaisesRegex(ValueError, r"unknown experiment parameters: \['require_heldout_magic'\]"):
                options({**base, "require_heldout_magic": True})
        profile = current.rolling.broker_profile
        self.assertTrue(profile.dividend_tax)
        self.assertEqual(current.rolling.acceptance, AcceptanceRules(**REQUIRED_RULES))
        policy = _broker_replay_facts({"broker_profile": profile.to_record()})["dividend_tax_policy"]
        self.assertTrue(policy["charged"])
        self.assertEqual(policy["record_date_cutover"], "20150908")
        old, new = policy["rules"]["2012_85"], policy["rules"]["2015_101"]
        self.assertEqual(old["withholding_rate"], 0.05)
        self.assertEqual(new["withholding_rate"], 0.0)
        self.assertEqual(old["final_rate_by_months_held_at_most"], {"1": 0.2, "12": 0.1})
        self.assertEqual((old["final_rate_beyond"], new["final_rate_beyond"]), (0.05, 0.0))

    def test_the_permitted_boards_are_pinned_per_arm_and_stated_only_where_pinned(self) -> None:
        """An arm without the key buys on every board and its facts say nothing
        new; a stamped value reaches the Broker profile, an invalid one is
        refused by the pre-flight the console runs before creating."""
        import tempfile

        from autotrade.agent.experiment_facts import (
            PERMITTED_BOARDS_RULE,
            _broker_replay_facts,
        )
        from autotrade.pipelines.worker import resolve_worker_options

        base = {
            "experiment_id": "boards_demo",
            "strategy_path": "configs/agent_output_template/main.py",
            **REQUIRED_RULES,
            "data_backend": "pit",
            "raw_dir": "data/raw",
            "fundamental_events_root": "data/pit/fundamental_events",
            "fundamental_events_status": "results/data_quality/fundamental_events_status.json",
        }
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "experiments").mkdir()

            def profile(params: dict[str, object]) -> BrokerProfile:
                return resolve_worker_options(
                    params,
                    experiment_dir=repo_root / "experiments/boards_demo",
                    repo_root=repo_root,
                    preflight=True,
                ).rolling.broker_profile

            recorded = profile(base)
            pinned = profile({**base, "permitted_boards": ["gem", "main"]})
            for value, message in (
                ([], "must include the main board"),
                (["gem"], "must include the main board"),
                (["main", "nasdaq"], "unknown permitted_boards"),
                ("main", "permitted_boards must be a list of strings"),
            ):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                    profile({**base, "permitted_boards": value})
        self.assertIsNone(recorded.permitted_boards)
        self.assertIsNone(recorded.to_record()["permitted_boards"])
        self.assertEqual(pinned.permitted_boards, ("main", "gem"))
        self.assertEqual(pinned.to_record()["permitted_boards"], ["main", "gem"])
        # A profile read back from the JSON a Paper book keeps is the same profile.
        self.assertEqual(BrokerProfile.from_record(json.loads(json.dumps(pinned.to_record()))), pinned)
        # The run facts state the boards only where the profile pins them.
        self.assertNotIn("permitted_boards", _broker_replay_facts({"broker_profile": recorded.to_record()}))
        self.assertEqual(
            _broker_replay_facts({"broker_profile": pinned.to_record()})["permitted_boards"],
            {"boards": ["main", "gem"], "rule": PERMITTED_BOARDS_RULE},
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
                    **REQUIRED_RULES,
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
            replace(AcceptanceRules(**REQUIRED_RULES), max_drawdown=0.2, cost_stress_multiplier=3.0),
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
            "model": "deepseek-flash",
            "max_replay_years": 7,
            "screen_boards": ("gem", "star"),
            "research_start": "20190701",
        }
        with patch.dict(WEB_CREATE_DEFAULTS, moved):
            after = rendered()
        self.assertEqual(after["model"], "deepseek-flash")
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
                if field.get("type") in {"string", "time", "text"}:
                    # A text field with no console default renders as an empty box.
                    expected = expected or ""
                    actual = actual or ""
                self.assertEqual(actual, expected, key)

    def test_worker_accepts_exactly_the_console_parameters(self) -> None:
        from autotrade.pipelines.arm_params import PARAMETERS

        # Every knob the console can write must be one the worker accepts,
        # otherwise creating an experiment produces a worker that refuses it.
        unknown = sorted(set(WEB_CREATE_DEFAULTS) - set(PARAMETERS))
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
        from autotrade.pipelines.arm_params import PARAMETERS

        fields = self._schema_fields()
        for key in RESTORED_CONSOLE_PARAMETERS:
            with self.subTest(key=key):
                self.assertIn(key, fields, f"{key} is not rendered on the create form")
                self.assertIn(key, WEB_CREATE_DEFAULTS, f"{key} has no console default")
                self.assertIn(key, set(PARAMETERS), f"the worker would reject {key}")
                self.assertTrue(fields[key].get("label"))
                self.assertTrue(fields[key].get("help"), f"{key} has no help text")

    def test_every_rendered_field_is_a_parameter_the_worker_accepts(self) -> None:
        from autotrade.pipelines.arm_params import PARAMETERS

        fields = self._schema_fields()
        # Rendering a field the worker rejects is a control that 400s on submit.
        unaccepted = sorted(
            set(fields) - set(PARAMETERS) - {"experiment_id"}
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
                            **DEFAULT_RESEARCH_GEOMETRY.to_record(), **CREATE_RULES,
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

        from autotrade.webui.manager import ExperimentManager
        from autotrade.webui.server import create_app

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            # A busy card is still eligible unless its free memory is below the margin.
            kept = host_min_free_gpu_memory_mib() - 1
            roster = gpu_roster([0], busy=[1])
            short = next(row for row in roster if row["index"] == 1)
            short["memory_used_mib"] = int(short["memory_total_mib"]) - kept
            short["memory_free_mib"] = kept
            with (
                patch.object(ExperimentManager, "start_worker", return_value={"spawned": False}),
                patch("autotrade.environment.gpu.list_gpus", return_value=roster),
            ):
                response = TestClient(create_app(repo_root)).post(
                    "/api/experiments",
                    json={
                        "params": {
                            "experiment_id": "params_gpu",
                            **DEFAULT_RESEARCH_GEOMETRY.to_record(), **CREATE_RULES,
                            "gpu_count": 2,
                        }
                    },
                )
            self.assertEqual(response.status_code, 400, response.text)
            detail = response.json()["detail"]
            self.assertIn("当前 GPU 无法满足实验默认分配", detail)
            self.assertIn("requested 2 GPU(s), 1 free (0)", detail)
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
    **REQUIRED_RULES,
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

    def test_a_seed_and_its_release_deployed_without_a_live_lake_are_accepted(self) -> None:
        """A deployment that carries only a seed and the release it was built
        from (the A100 instance) creates the arm as the workstation does: the
        seed's recorded release path names the host that built it, the
        release manifest's sources are read relative to the repository, and
        no live ``data/raw`` or PIT root is required at pre-flight or create."""
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            origin = Path(tmp) / "auto-trade"
            (origin / "experiments").mkdir(parents=True)
            selection = {"macro_datasets": ["cn_gdp", "fut_daily"]}
            self._seed(origin, "pit_views_seed_ext", selection)
            # What the deploy carries: the checkout, the seed and its release.
            instance = Path(tmp) / "personal_fast_storage" / "auto-trade"
            shutil.move(str(origin), str(instance))
            shutil.rmtree(instance / "data" / "raw")
            shutil.rmtree(instance / "data" / "pit")
            strategy = instance / PIT_SEED_BASE_PARAMS["strategy_path"]
            strategy.parent.mkdir(parents=True)
            strategy.write_text("", encoding="utf-8")
            named = {**selection, "pit_views_seed": "data/pit_views_seed_ext"}
            for preflight in (True, False):
                options = self._resolve(instance, named, preflight=preflight)
                self.assertEqual(options.pit_views_seed, instance / "data" / "pit_views_seed_ext")
                self.assertTrue(options.pit_views_seed_required)
            self.assertTrue((instance / "experiments" / "seed_demo" / "research_release").is_dir())

    def test_a_benchmark_index_the_release_cannot_key_fails_the_create(self) -> None:
        """An arm is refused where it names its benchmark, not at replay end.

        The parameter selects the benchmark's own bars and its constituent
        table out of the release the arm pins; a release published before an
        index carries neither, and a replay would only report an unmeasured
        benchmark hours later. An index the lake does not carry at all is
        refused before any release is read.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "experiments").mkdir()
            with self.assertRaisesRegex(ValueError, "399905.SZ"):
                self._resolve(repo_root, {"benchmark_index": "399905.SZ"})
            # A release published before CSI 1000 was downloaded: the code is a
            # valid index, its partitions are not there.
            self._seed(
                repo_root,
                "pit_views_seed_partial",
                {"macro_datasets": ["index_daily", "index_weight"]},
                benchmark_indexes=("000300.SH", "000905.SH"),
            )
            named = {
                "macro_datasets": ["index_daily", "index_weight"],
                "pit_views_seed": "data/pit_views_seed_partial",
            }
            with self.assertRaises(ValueError) as caught:
                self._resolve(repo_root, {**named, "benchmark_index": "000852.SH"})
            message = str(caught.exception)
            self.assertIn("000852.SH", message)
            self.assertIn("index_daily/ts_code=000852.SH", message)
            self.assertIn("index_weight/index_code=000852.SH", message)
            # An index the same release does carry is accepted and forwarded.
            options = self._resolve(repo_root, {**named, "benchmark_index": "000905.SH"})
            self.assertEqual(options.rolling.benchmark_index, "000905.SH")

    def test_a_benchmark_history_that_starts_inside_research_fails_the_create(self) -> None:
        """An eight-year arm is refused where it names a release whose
        benchmark history starts after its research start, not graded with
        unmeasured first years; the four-year geometry on the same release is
        accepted."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "experiments").mkdir()
            selection = {"macro_datasets": ["index_daily", "index_weight"]}
            self._seed(repo_root, "pit_views_seed_bench", selection)  # history from 20200102
            named = {
                **selection,
                "pit_views_seed": "data/pit_views_seed_bench",
                "benchmark_index": "000905.SH",
            }
            with self.assertRaises(ValueError) as caught:
                self._resolve(repo_root, {**named, "research_start": "20170701"})
            message = str(caught.exception)
            self.assertIn("no history before research_start 20170701", message)
            self.assertIn("index_daily/ts_code=000905.SH", message)
            self.assertIn("index_weight/index_code=000905.SH", message)
            options = self._resolve(repo_root, named)
            self.assertEqual(options.rolling.geometry.research_start, "20210701")

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


def test_a_session_request_takes_every_arm_field_from_its_config(tmp_path: Path) -> None:
    """The research session's request is derived from the arm's config in one
    place: a value the arm sets reaches the session, and the budgets are the
    ones handed over after the HITL override, not the config's."""

    from datetime import UTC, datetime

    from autotrade.environment.artifacts import ModificationConstraints
    from autotrade.pipelines.config import ResearchSessionRequest, SnapshotBundle

    config = make_config(
        tmp_path,
        experiment_id="arm_request",
        window_months=36,
        benchmark_index="000905.SH",
        record_failed_attempts=False,
        finalize_before_deadline_seconds=120,
        max_null_controls=4,
        batch_validate_max_concurrency=2,
        step_constraints=ModificationConstraints(max_strategy_files=3),
    )
    budgets = {
        "max_replay_years": 7,
        "max_llm_calls": 11,
        "max_api_spend_usd": 2.5,
        "deadline_seconds": 600.0,
        "deadline_grace_seconds": 60.0,
    }
    request = ResearchSessionRequest.for_arm(
        config,
        budgets=budgets,
        run_id="run_1",
        snapshot=SnapshotBundle("snap", "decision", "replay"),
        decision_time=datetime(2025, 6, 30, tzinfo=UTC),
        research_years=(),
        sealed_period=("20250701", "20260630"),
    )

    assert (
        request.experiment_id,
        request.window_months,
        request.benchmark_index,
        request.record_failed_attempts,
        request.finalize_before_deadline_seconds,
        request.max_null_controls,
        request.batch_validate_max_concurrency,
        request.modification_constraints,
    ) == ("arm_request", 36, "000905.SH", False, 120, 4, 2, config.step_constraints)
    assert request.acceptance_rules == config.acceptance.to_record()
    assert {name: getattr(request, name) for name in budgets} == budgets
    with pytest.raises(TypeError):
        ResearchSessionRequest.for_arm(config, budgets=budgets, window_months=12)
