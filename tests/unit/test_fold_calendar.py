"""The research calendar: one development window, an optional Test stage, and
the explicitly ranged Held-out.

The schedule decides which data every session may see, so these assert the
regions and the decision anchors themselves, not merely that a schedule was
produced. The failure paths matter as much as the happy one: a Held-out range
that silently widened to a whole calendar year would score the strategy on
months the data lake does not cover, and a Test stage over a one-period window
would have nothing to test.
"""

from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from autotrade.environment.data.contracts import CN_TZ
from autotrade.pipelines.config import RollingExperimentConfig
from autotrade.pipelines.folds import (
    FoldSpec,
    build_fold_schedule,
    heldout_periods,
    period_range,
    previous_period,
)
from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS, build_session_plan

# Weekday calendar: every yearly boundary below falls on the same day the SSE
# calendar puts it, so the anchors are the real ones.
TRADING_DAYS = [
    stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2019-01-01", "2026-06-30")
]


def anchor(day: str) -> datetime:
    return datetime.strptime(day, "%Y%m%d").replace(
        hour=23, minute=59, second=59, tzinfo=CN_TZ
    )


def default_config(**overrides: object) -> RollingExperimentConfig:
    values: dict[str, object] = {
        "experiment_id": "calendar_demo",
        "experiments_root": Path("/tmp/experiments"),
        "development_first_period": str(WEB_CREATE_DEFAULTS["development_first_period"]),
        "development_last_period": str(WEB_CREATE_DEFAULTS["development_last_period"]),
        "heldout_first_period": str(WEB_CREATE_DEFAULTS["heldout_first_period"]),
        "heldout_last_period": str(WEB_CREATE_DEFAULTS["heldout_last_period"]),
        "fold_period": str(WEB_CREATE_DEFAULTS["fold_period"]),
        "test_stage": bool(WEB_CREATE_DEFAULTS["test_stage"]),
        "window_months": int(WEB_CREATE_DEFAULTS["window_months"]),  # type: ignore[call-overload]
    }
    values.update(overrides)
    return RollingExperimentConfig(**values)  # type: ignore[arg-type]


def schedule(config: RollingExperimentConfig) -> list[FoldSpec]:
    return build_fold_schedule(
        config.development_first_period,
        config.development_last_period,
        TRADING_DAYS,
        window_months=config.window_months,
        period=config.fold_period,
        min_region_trade_days=config.min_region_trade_days,
        test_stage=config.test_stage,
    )


class SingleWindowDevelopmentTest(unittest.TestCase):
    def test_the_console_defaults_build_one_development_fold_without_a_test(self) -> None:
        config = default_config()
        self.assertFalse(config.test_stage)
        folds = schedule(config)
        self.assertEqual(len(folds), 1)
        fold = folds[0]
        # The whole 2022..2025 window is the validation region; the input
        # window is the 24 months before it (the macro data floor is 2020-01).
        self.assertEqual(fold.fold_id, "fold_20220101..20251231")
        self.assertEqual((fold.input_window_start, fold.input_window_end), ("20200101", "20211231"))
        self.assertEqual((fold.validation_start, fold.validation_end), ("20220101", "20251231"))
        self.assertEqual(fold.valid_decision_time, anchor("20211231"))
        self.assertFalse(fold.has_test)
        self.assertIsNone(fold.test_start)
        self.assertIsNone(fold.test_end)
        self.assertIsNone(fold.test_decision_time)
        # The ledger record says "no Test" explicitly, never a placeholder.
        record = fold.to_record()
        self.assertIsNone(record["test_period"])
        self.assertIsNone(record["test_decision_time"])
        self.assertEqual(record["validation_period"], "20220101..20251231")

    def test_an_explicit_range_can_name_the_whole_development_window(self) -> None:
        folds = build_fold_schedule(
            "20220301..20251231",
            "20220301..20251231",
            TRADING_DAYS,
            window_months=24,
            period="year",
        )
        self.assertEqual([fold.fold_id for fold in folds], ["fold_20220301..20251231"])
        self.assertEqual(folds[0].input_window_start, "20200301")
        self.assertEqual(folds[0].valid_decision_time, anchor("20220228"))

    def test_the_session_plan_has_one_fold_then_held_out(self) -> None:
        config = default_config()
        folds = schedule(config)
        heldout = heldout_periods(
            config.heldout_first_period,
            config.heldout_last_period,
            TRADING_DAYS,
            period=config.fold_period,
        )
        plan = build_session_plan(
            config.epochs,
            folds,
            heldout,
            meta_enabled=True,
            meta_learning_fold_interval=config.meta_learning_fold_interval,
        )
        # Epoch-start Meta, the single development Fold, then Held-out: the
        # fold interval has no second Fold to trigger on.
        self.assertEqual(
            [(row.get("kind"), row.get("fold_id")) for row in plan["sessions"]],
            [
                ("meta", "fold_20220101..20251231"),
                ("fold", "fold_20220101..20251231"),
                ("heldout", None),
            ],
        )
        self.assertEqual(plan["sessions"][-1]["periods"][0]["label"], "20260101..20260630")

    def test_a_fold_test_region_is_all_or_nothing(self) -> None:
        with self.assertRaisesRegex(ValueError, "together"):
            FoldSpec(
                fold_id="fold_x",
                input_window_start="20200101",
                input_window_end="20211231",
                validation_start="20220101",
                validation_end="20221231",
                valid_decision_time=anchor("20211231"),
                test_start="20230101",
            )


class TestStageScheduleTest(unittest.TestCase):
    def test_folds_roll_inside_the_development_window(self) -> None:
        config = default_config(test_stage=True)
        folds = schedule(config)
        # The first period is validation only; every later period is a test
        # period validated on the period before it. Nothing precedes 2022.
        self.assertEqual(
            [
                (
                    fold.fold_id,
                    fold.input_window_start,
                    fold.input_window_end,
                    fold.validation_start,
                    fold.validation_end,
                    fold.test_start,
                    fold.test_end,
                )
                for fold in folds
            ],
            [
                ("fold_2023", "20200101", "20211231", "20220101", "20221231", "20230101", "20231231"),
                ("fold_2024", "20210101", "20221231", "20230101", "20231231", "20240101", "20241231"),
                ("fold_2025", "20220101", "20231231", "20240101", "20241231", "20250101", "20251231"),
            ],
        )
        # Each region's research baseline is frozen at the close of the last
        # trading day before it starts; the test anchor of one fold is the
        # validation anchor of the next.
        self.assertEqual(
            [(fold.valid_decision_time, fold.test_decision_time) for fold in folds],
            [
                (anchor("20211231"), anchor("20221230")),
                (anchor("20221230"), anchor("20231229")),
                (anchor("20231229"), anchor("20241231")),
            ],
        )
        self.assertTrue(all(fold.has_test for fold in folds))
        self.assertEqual(folds[0].to_record()["test_period"], "20230101..20231231")

    def test_a_one_period_window_cannot_have_a_test_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two development periods"):
            build_fold_schedule(
                "2025", "2025", TRADING_DAYS, window_months=24, period="year", test_stage=True
            )

    def test_an_explicit_range_cannot_roll(self) -> None:
        # A bare range has no cadence neighbours, so there is nothing to roll
        # over; the same label also has no preceding period.
        with self.assertRaisesRegex(ValueError, "at least two development periods"):
            build_fold_schedule(
                "20230101..20230630",
                "20230101..20230630",
                TRADING_DAYS,
                window_months=24,
                period="year",
                test_stage=True,
            )
        with self.assertRaisesRegex(ValueError, "no preceding year"):
            previous_period("20230101..20230630", period="year")

    def test_test_stage_must_be_boolean(self) -> None:
        with self.assertRaisesRegex(ValueError, "test_stage must be boolean"):
            default_config(test_stage=1)


class HeldOutRangeTest(unittest.TestCase):
    def test_the_default_held_out_replays_exactly_its_range(self) -> None:
        config = default_config()  # constructing it asserts the no-overlap rule
        periods = heldout_periods(
            config.heldout_first_period,
            config.heldout_last_period,
            TRADING_DAYS,
            period=config.fold_period,
            min_region_trade_days=config.min_region_trade_days,
        )
        self.assertEqual(
            periods,
            [
                {
                    "label": "20260101..20260630",
                    "start": "20260101",
                    "end": "20260630",
                    "decision_time": anchor("20251231"),
                }
            ],
        )
        # The regression this guards: re-deriving a cadence label from the
        # range's start would replay the whole of 2026.
        self.assertNotEqual(periods[0]["end"], "20261231")

    def test_a_held_out_range_that_overlaps_development_is_refused(self) -> None:
        for test_stage in (False, True):
            with self.subTest(test_stage=test_stage):
                with self.assertRaisesRegex(ValueError, "must not overlap"):
                    default_config(
                        heldout_first_period="20251201..20260630",
                        heldout_last_period="20251201..20260630",
                        test_stage=test_stage,
                    )

    def test_two_different_explicit_ranges_cannot_be_enumerated(self) -> None:
        # "From one range to another" has no meaning at any cadence, and
        # answering it with cadence arithmetic would drop one of the two.
        with self.assertRaisesRegex(ValueError, "cannot be enumerated"):
            period_range("20260101..20260331", "20260401..20260630", period="year")

    def test_a_region_without_two_trading_days_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            heldout_periods(
                "20260103..20260104", "20260103..20260104", TRADING_DAYS, period="year"
            )
        with self.assertRaisesRegex(ValueError, "at least 2"):
            build_fold_schedule(
                "20260103..20260104",
                "20260103..20260104",
                TRADING_DAYS,
                window_months=24,
                period="year",
            )


if __name__ == "__main__":
    unittest.main()
