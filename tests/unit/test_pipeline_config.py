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
    AcceptanceRules,
    ExperimentConfig,
    acceptance_for,
)
from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS, WEB_REQUIRED_PARAMS
from tests.unit.acceptance_fixture import (
    CREATE_RULES,
    NO_SEED,
    REQUIRED_RULES,
    TEST_GEOMETRY,
)
from tests.unit.gpu_probe import gpu_roster, stubbed_gpu_probe

#: The console create form is seeded from the pinned explore profile so a
#: new experiment hardlinks the prebuilt view cache. Period, screen, compact, and
#: loop knobs that a researcher gets without touching the form live here.
_CONSOLE_CREATE_PRESET: dict[str, object] = {
    "compact_keep_recent_messages": 10,
    "compact_max_calls": 10,
    "compact_max_tokens": 10_000,
    # Null: each conversation role compacts at its bound.
    "compact_token_threshold": None,
    # Ten August-July research years, a forward year, a Held-out quarter, and
    # the dataset selection and window the default prebuilt view cache was
    # built for: the current rounds' (scripts/experiments/_profiles.py).
    "research_start": "20150801",
    "research_end": "20250731",
    "forward_end": "20260731",
    "heldout_end": "20261031",
    "pit_views_seed": "data/pit_views_seed_research_10y_fundanns_20261106",
    "include_fundamentals": True,
    "fundamental_datasets": (),
    "include_macro": True,
    "macro_datasets": ("index_daily", "index_dailybasic", "sw_daily", "index_weight"),
    "include_events": False,
    "include_text": True,
    "text_datasets": ("anns_d",),
    "gpu_count": 1,
    "include_intraday": False,
    "inference_time": "08:00",
    "initial_cash": 1_000_000.0,
    "initial_control_mode": "auto",
    # The research session's budgets: minutes and calls, spent across every
    # attempt of the experiment's one session.
    "max_research_minutes": 10_080,
    "max_llm_calls": 6400,
    "model": "gpt-6-astra",
    # The universe reaches the agent unfiltered; the strategy filters itself.
    "screen_boards": (),
    "screen_exclude_new_listed_days": 0,
    "screen_exclude_st": False,
    "strategy_period": "day",
    "window_months": 132,
}


def make_config(root: Path, **overrides: object) -> ExperimentConfig:
    values: dict[str, object] = {
        "experiment_id": "exp",
        "experiments_root": root / "experiments",
        "acceptance": AcceptanceRules(**REQUIRED_RULES),
    }
    values.update(overrides)
    return ExperimentConfig(**values)


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

    def test_the_sealed_gates_rules_and_the_replay_year_budget_have_no_place_in_the_rules(self) -> None:
        """The validation record's rules carry no sealed years, no ablation and
        no replay-year budget: a rules object built with one is refused, and
        the budget record has no replay-year counter, so neither can come back
        through a round file or a create request unnoticed."""

        from dataclasses import fields

        from autotrade.pipelines.config import BudgetUsed

        retired = {"sealed_years", "ablation_required", "max_replay_years", "finalize_before_deadline_seconds"}
        for key in sorted(retired):
            with self.subTest(key=key), self.assertRaises(TypeError):
                AcceptanceRules(**REQUIRED_RULES, **{key: 1})
        names = {field.name for field in fields(AcceptanceRules)}
        self.assertFalse(names & retired)
        self.assertEqual(
            {"alpha", "research_years", "test_step_months", "validates_per_step"} - names, set()
        )
        self.assertNotIn("replay_years", {field.name for field in fields(BudgetUsed)})
        self.assertIn("test_replays", {field.name for field in fields(BudgetUsed)})
        for name in ("max_replay_years", "finalize_before_deadline_seconds"):
            self.assertFalse(hasattr(ExperimentConfig, name), name)

    def test_finite_metrics_keep_threshold_semantics(self) -> None:
        rules = AcceptanceRules(**REQUIRED_RULES, max_drawdown=0.25)
        ok = {"total_return": 0.02, "sharpe": 0.5, "max_drawdown": 0.1}
        self.assertEqual(rules.evaluate(ok), [])
        # A drawdown breach is a HARD reject, sign-independent: F4/H3 enforce
        # the same limit forward, so registering a book that already breached it
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
            {"research_years": 0},
            {"research_years": 2.5},
            {"test_step_months": 2},
            {"test_step_months": True},
            {"validates_per_step": 0},
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
        # A params.json or run manifest that leaves a required rule out is
        # refused, never read under a value it did not state.
        for missing in REQUIRED_RULES:
            stated = {key: value for key, value in REQUIRED_RULES.items() if key != missing}
            with self.assertRaisesRegex(ValueError, f"lack the required {missing}"):
                AcceptanceRules.from_record({"max_drawdown": 0.25, **stated})
            with self.assertRaisesRegex(ValueError, f"lack the required {missing}"):
                acceptance_for({"max_drawdown": 0.25, **stated})
            with self.assertRaises(TypeError):
                AcceptanceRules(max_drawdown=0.25, **stated)  # type: ignore[arg-type]
        # A record that states them rebuilds the rules it names, with no tracking
        # mandate, and the forward thresholds it leaves out resolve to today's
        # defaults; a key that names no rule is refused.
        old = AcceptanceRules.from_record({"max_drawdown": 0.25, **REQUIRED_RULES})
        with self.assertRaisesRegex(ValueError, "name no rule min_dsr_probability"):
            AcceptanceRules.from_record({"min_dsr_probability": 0.975, **REQUIRED_RULES})
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
        """The ``acceptance_rules`` fact is derived, never retyped: the
        residual statistic and its threshold come from ``verdict`` and this
        experiment's own rules, registration lists only what it checks, the
        validation block states the experiment's steps, and no date of any period
        appears in it."""

        rules = AcceptanceRules(
            **{**REQUIRED_RULES, "research_years": 5, "test_step_months": 1, "validates_per_step": 4},
            max_drawdown=0.2, cost_stress_multiplier=3.0, active_max_drawdown=0.11
        )
        facts = rules.agent_facts(validation_years=5)
        self.assertEqual(
            sorted(facts), ["forward", "graded_series", "registration", "significance", "validation"]
        )
        self.assertIn(str(verdict.BASELINE_DRAWS), facts["graded_series"])
        self.assertIn("benchmark.active_information_ratio", facts["graded_series"])
        statistic = facts["significance"]
        self.assertIn("intercept times sqrt(244) over the residual standard deviation", statistic["statistic"])
        self.assertIn("the size leg and the value leg", statistic["statistic"])
        self.assertIn("z(1 - alpha) / sqrt(years)", statistic["threshold"])
        self.assertEqual(statistic["alpha"], 0.05)
        self.assertEqual(statistic["threshold_over_one_year"], 1.645)
        self.assertEqual(statistic["threshold_over_validation_period"], round(verdict.residual_threshold(5.0, 0.05), 3))
        with self.assertRaises(ValueError):
            rules.agent_facts(validation_years=0)
        validation = facts["validation"]
        self.assertEqual(
            (validation["research_years"], validation["test_step_months"], validation["validates_per_step"]),
            (5, 1, 4),
        )
        self.assertIn("at most 4 validate calls", validation["validates_per_step_rule"])
        self.assertIn("no action that ends the session earlier", validation["end"])
        registration = facts["registration"]
        self.assertEqual(
            sorted(registration),
            ["checked", "control", "finite_metrics", "max_drawdown", "seed_replicates", "span", "tracking_mandate"],
        )
        self.assertIn("span=full", registration["span"])
        self.assertIn("<= 0.2", registration["max_drawdown"])
        self.assertIn("at every test", registration["checked"])
        self.assertIn("reported, not graded", registration["tracking_mandate"])
        self.assertIn("multiplied by 3.0", facts["forward"]["reported"])
        self.assertIn("deciding nothing", facts["forward"]["reported"])
        text = json.dumps(facts)
        self.assertNotRegex(text, r"20\d{6}")

    def test_one_series_is_stated_and_nothing_gates(self) -> None:
        """Every reading is of the active series: the facts name it once and
        no neutralized figure. The forward bound and the cost stress are
        stated as readings, over the registered book and its seed replicates, that
        decide nothing; no fact names a gate."""

        facts = AcceptanceRules(**REQUIRED_RULES).agent_facts(validation_years=1)
        graded = facts["graded_series"]
        for field in ("benchmark.active_excess", "sub_windows[].active_excess_return", "daily_series"):
            self.assertIn(field, graded)
        text = json.dumps(facts).lower()
        self.assertNotIn("neutral", text)
        self.assertNotIn("graduat", text)
        self.assertNotIn("sealed", text)
        forward = facts["forward"]
        self.assertEqual(sorted(forward), ["reported"])
        self.assertIn("80% block-bootstrap lower bound", forward["reported"])
        self.assertIn("registered book and its seed replicates", forward["reported"])
        self.assertIn("deciding nothing", forward["reported"])

    def test_a_record_naming_a_retired_key_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "name no rule dividend_tax, heldout_min_trades"):
            AcceptanceRules.from_record(
                {"max_drawdown": 0.2, "heldout_min_trades": 5, "dividend_tax": False, **REQUIRED_RULES}
            )

    def test_the_cost_stress_multiplier_must_be_a_stress(self) -> None:
        with self.assertRaisesRegex(ValueError, "cost_stress_multiplier must be at least one"):
            AcceptanceRules(**REQUIRED_RULES, cost_stress_multiplier=0.5)
        with self.assertRaisesRegex(ValueError, "cost_stress_multiplier must be finite"):
            AcceptanceRules(**REQUIRED_RULES, cost_stress_multiplier=float("nan"))


class ExperimentConfigValidationTest(unittest.TestCase):
    def test_valid_defaults_pass(self) -> None:
        config = make_config(Path("/tmp"))
        self.assertEqual(config.geometry, DEFAULT_RESEARCH_GEOMETRY)
        self.assertEqual(config.session_max_attempts, 3)
        self.assertEqual(config.research_directive, "")
        self.assertEqual(config.max_research_minutes, 10_080)
        self.assertEqual(
            (config.max_llm_calls, config.max_null_controls),
            (6400, 12),
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
        for field_obj in fields(ExperimentConfig):
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
        # Drawdowns and the mandate switch stay empty so the form does not pin
        # a mandate onto every experiment, and the research years have no console
        # value: every request states them (WEB_REQUIRED_PARAMS). Every other
        # rule shows what an empty request would stamp.
        stated = {"research_years"}
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
                    **NO_SEED,
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
        for field_obj in fields(ExperimentConfig):
            if field_obj.default is MISSING or field_obj.name == "acceptance":
                continue
            with self.subTest(field=field_obj.name):
                self.assertEqual(
                    getattr(options.experiment, field_obj.name),
                    field_obj.default,
                    field_obj.name,
                )
        # A request that names no limit gets the rules' own defaults, mandate
        # off, on whatever capital the Broker profile carries.
        self.assertEqual(options.experiment.acceptance, AcceptanceRules(**REQUIRED_RULES))

    def test_the_account_pays_the_dividend_tax_and_an_unknown_key_is_refused(self) -> None:
        """The worker runs every experiment under the one rule set: its account pays
        the dividend tax, which the run facts state. A key it does not know --
        a retired switch among them -- is refused."""
        import tempfile

        from autotrade.agent.experiment_facts import _broker_replay_facts
        from autotrade.pipelines.worker import resolve_worker_options

        base = {
            "experiment_id": "legacy_demo",
            "strategy_path": "configs/agent_output_template/main.py",
            **REQUIRED_RULES,
            "data_backend": "pit",
            **NO_SEED,
            "raw_dir": "data/raw",
            "fundamental_events_root": "data/pit/fundamental_events",
            "fundamental_events_status": "results/data_quality/fundamental_events_status.json",
        }
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
            for key in ("require_heldout_magic", "dividend_tax"):
                with self.assertRaisesRegex(ValueError, rf"unknown experiment parameters: \['{key}'\]"):
                    options({**base, key: True})
        profile = current.experiment.broker_profile
        self.assertTrue(profile.dividend_tax)
        self.assertEqual(current.experiment.acceptance, AcceptanceRules(**REQUIRED_RULES))
        policy = _broker_replay_facts({"broker_profile": profile.to_record()})["dividend_tax_policy"]
        self.assertTrue(policy["charged"])
        self.assertEqual(policy["record_date_cutover"], "20150908")
        old, new = policy["rules"]["2012_85"], policy["rules"]["2015_101"]
        self.assertEqual(old["withholding_rate"], 0.05)
        self.assertEqual(new["withholding_rate"], 0.0)
        self.assertEqual(old["final_rate_by_months_held_at_most"], {"1": 0.2, "12": 0.1})
        self.assertEqual((old["final_rate_beyond"], new["final_rate_beyond"]), (0.05, 0.0))

    def test_the_retired_ablation_rule_is_no_parameter_anywhere(self) -> None:
        """``ablation_required`` names no parameter and no rule: the worker, the
        rules and a create request each refuse it like any other unknown key."""
        import tempfile

        from autotrade.pipelines.experiment_params import console_request
        from autotrade.pipelines.worker import resolve_worker_options

        self.assertNotIn("ablation_required", ACCEPTANCE_KEYS)
        base = {
            "experiment_id": "sealed_demo",
            "strategy_path": "configs/agent_output_template/main.py",
            **REQUIRED_RULES,
            "data_backend": "pit",
            **NO_SEED,
            "raw_dir": "data/raw",
            "fundamental_events_root": "data/pit/fundamental_events",
            "fundamental_events_status": "results/data_quality/fundamental_events_status.json",
        }
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "experiments").mkdir()

            def options(params: dict[str, object]):
                return resolve_worker_options(
                    params,
                    experiment_dir=repo_root / "experiments/sealed_demo",
                    repo_root=repo_root,
                    preflight=True,
                )

            options(base)
            with self.assertRaisesRegex(ValueError, r"unknown experiment parameters: \['ablation_required'\]"):
                options({**base, "ablation_required": True})
            with self.assertRaisesRegex(ValueError, "name no rule ablation_required"):
                AcceptanceRules.from_record({**REQUIRED_RULES, "ablation_required": True})
            with self.assertRaisesRegex(ValueError, r"unknown experiment parameters: \['ablation_magic'\]"):
                options({**base, "ablation_magic": True})
        request = {"experiment_id": "sealed_demo", **DEFAULT_RESEARCH_GEOMETRY.to_record(), **CREATE_RULES}
        console_request(request, defaults=WEB_CREATE_DEFAULTS, experiments_root="experiments", work_root="work")
        with self.assertRaisesRegex(ValueError, "unknown experiment parameters: ablation_required"):
            console_request(
                {**request, "ablation_required": False},
                defaults=WEB_CREATE_DEFAULTS,
                experiments_root="experiments",
                work_root="work",
            )

    def test_the_permitted_boards_are_pinned_per_experiment_and_stated_only_where_pinned(self) -> None:
        """An experiment without the key buys on every board and its facts say nothing
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
            **NO_SEED,
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
                ).experiment.broker_profile

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
        # A profile read back from the JSON a Paper account keeps is the same profile.
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
                    **NO_SEED,
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
        self.assertEqual(options.experiment.geometry, geometry)
        self.assertEqual(options.experiment.max_research_minutes, 300)
        # The two the request names; it named no cap, so no tracking mandate.
        self.assertEqual(
            options.experiment.acceptance,
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
            "max_null_controls": 7,
            "screen_boards": ("gem", "star"),
            "research_start": "20190701",
        }
        with patch.dict(WEB_CREATE_DEFAULTS, moved):
            after = rendered()
        self.assertEqual(after["model"], "deepseek-flash")
        self.assertEqual(after["max_null_controls"], 7)
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
            removed.isdisjoint(field.name for field in fields(ExperimentConfig))
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
        from autotrade.pipelines.experiment_params import PARAMETERS

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
        from autotrade.pipelines.experiment_params import PARAMETERS

        fields = self._schema_fields()
        for key in RESTORED_CONSOLE_PARAMETERS:
            with self.subTest(key=key):
                self.assertIn(key, fields, f"{key} is not rendered on the create form")
                self.assertIn(key, WEB_CREATE_DEFAULTS, f"{key} has no console default")
                self.assertIn(key, set(PARAMETERS), f"the worker would reject {key}")
                self.assertTrue(fields[key].get("label"))
                self.assertTrue(fields[key].get("help"), f"{key} has no help text")

    def test_every_rendered_field_is_a_parameter_the_worker_accepts(self) -> None:
        from autotrade.pipelines.experiment_params import PARAMETERS

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
                            **DEFAULT_RESEARCH_GEOMETRY.to_record(), **CREATE_RULES, **NO_SEED,
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
                            **DEFAULT_RESEARCH_GEOMETRY.to_record(), **CREATE_RULES, **NO_SEED,
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
    # The span the synthetic release covers.
    **TEST_GEOMETRY.to_record(),
    "strategy_path": "configs/agent_output_template/main.py",
    "data_backend": "pit",
    "raw_dir": "data/raw",
    "fundamental_events_root": "data/pit/fundamental_events",
    "fundamental_events_status": "results/data_quality/fundamental_events_status.json",
    **REQUIRED_RULES,
}


class PitViewsSeedParameterTest(unittest.TestCase):
    """Which prebuilt PIT views an experiment may reuse is its own parameter.

    Every named tree, the console's default included, is checked at create time
    and binds the release it was built from, instead of turning into hours of
    silent cold building; ``""`` names none.
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

    def test_no_seed_is_optional_and_every_named_one_the_default_included_is_required(self) -> None:
        import tempfile

        from autotrade.pipelines.config import DEFAULT_PIT_VIEWS_SEED

        selection = {"macro_datasets": ["cn_gdp", "fut_daily"]}
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "experiments").mkdir()
            # No seed: every view is built cold.
            options = self._resolve(repo_root, NO_SEED)
            self.assertIsNone(options.pit_views_seed)
            self.assertFalse(options.pit_views_seed_required)
            # An absent parameter is the console's default, which is named, so a
            # repository without it refuses the create.
            for params in ({}, {"pit_views_seed": str(DEFAULT_PIT_VIEWS_SEED)}):
                with self.assertRaisesRegex(ValueError, "existing directory"):
                    self._resolve(repo_root, params)
            default = self._seed(repo_root, DEFAULT_PIT_VIEWS_SEED.name, {}, generation_id="gen_default")
            options = self._resolve(repo_root, {})
            self.assertEqual(options.pit_views_seed, default)
            self.assertTrue(options.pit_views_seed_required)
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
            # Built for the default selection, asked for by an experiment that adds a
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
        from (the A100 instance) creates the experiment as the workstation does: the
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
        """An experiment is refused where it names its benchmark, not at replay end.

        The parameter selects the benchmark's own bars and its constituent
        table out of the release the experiment pins; a release published before an
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
            self.assertEqual(options.experiment.benchmark_index, "000905.SH")

    def test_a_benchmark_history_that_starts_inside_research_fails_the_create(self) -> None:
        """An eight-year experiment is refused where it names a release whose
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
            self.assertEqual(options.experiment.geometry.research_start, "20210701")

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
                repo_root, {**selection, **NO_SEED, "experiment_id": "no_seed"}, preflight=False
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


def test_a_session_request_takes_every_experiment_field_from_its_config(tmp_path: Path) -> None:
    """The research session's request is derived from the experiment's config in one
    place: a value the experiment sets reaches the session, and the budgets are the
    ones handed over after the HITL override, not the config's."""

    from datetime import UTC, datetime

    from autotrade.environment.artifacts import ModificationConstraints
    from autotrade.pipelines.config import (
        ReplaySpan,
        ResearchSessionRequest,
        SnapshotBundle,
    )

    config = make_config(
        tmp_path,
        experiment_id="experiment_request",
        window_months=36,
        benchmark_index="000905.SH",
        record_failed_attempts=False,
        max_null_controls=4,
        batch_validate_max_concurrency=2,
        step_constraints=ModificationConstraints(max_strategy_files=3),
    )
    budgets = {
        "max_llm_calls": 11,
        "max_api_spend_usd": 2.5,
        "deadline_seconds": 600.0,
        "deadline_grace_seconds": 60.0,
    }
    request = ResearchSessionRequest.for_experiment(
        config,
        budgets=budgets,
        run_id="run_1",
        snapshot=SnapshotBundle("snap", "decision", "replay"),
        decision_time=datetime(2025, 6, 30, tzinfo=UTC),
        years=(),
        validation=ReplaySpan("validation", "valid", "20250701", "20260630", SnapshotBundle("v", "d", "r")),
    )

    assert (
        request.experiment_id,
        request.window_months,
        request.benchmark_index,
        request.record_failed_attempts,
        request.max_null_controls,
        request.batch_validate_max_concurrency,
        request.modification_constraints,
    ) == ("experiment_request", 36, "000905.SH", False, 4, 2, config.step_constraints)
    assert request.acceptance_rules == config.acceptance.to_record()
    assert {name: getattr(request, name) for name in budgets} == budgets
    with pytest.raises(TypeError):
        ResearchSessionRequest.for_experiment(config, budgets=budgets, window_months=12)
