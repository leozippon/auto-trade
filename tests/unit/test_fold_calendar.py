"""The research calendar: one regular Fold per period of the development
window with Meta between them, an optional Test stage, and the explicitly
ranged Held-out.

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
        validation_periods=config.validation_periods,
    )


class RegularFoldScheduleTest(unittest.TestCase):
    def test_the_console_defaults_build_one_yearly_fold_per_year_without_a_test(self) -> None:
        config = default_config()
        self.assertFalse(config.test_stage)
        folds = schedule(config)
        # One regular Fold per year of 2022..2025, in order; each is validated
        # on its own year and takes the 24 months before it as input (the
        # macro data floor is 2020-01, so 2022 gets the most history there is).
        self.assertEqual(
            [
                (
                    fold.fold_id,
                    fold.input_window_start,
                    fold.input_window_end,
                    fold.validation_start,
                    fold.validation_end,
                )
                for fold in folds
            ],
            [
                ("fold_2022", "20200101", "20211231", "20220101", "20221231"),
                ("fold_2023", "20210101", "20221231", "20230101", "20231231"),
                ("fold_2024", "20220101", "20231231", "20240101", "20241231"),
                ("fold_2025", "20230101", "20241231", "20250101", "20251231"),
            ],
        )
        # Each decision anchor is the close of the last trading day before the
        # year: the SSE calendar puts those on the same weekdays.
        self.assertEqual(
            [fold.valid_decision_time for fold in folds],
            [anchor("20211231"), anchor("20221230"), anchor("20231229"), anchor("20241231")],
        )
        for fold in folds:
            self.assertFalse(fold.has_test)
            self.assertIsNone(fold.test_start)
            self.assertIsNone(fold.test_end)
            self.assertIsNone(fold.test_decision_time)
            # The ledger record says "no Test" explicitly, never a placeholder.
            record = fold.to_record()
            self.assertIsNone(record["test_period"])
            self.assertIsNone(record["test_decision_time"])
        self.assertEqual(folds[0].to_record()["validation_period"], "20220101..20221231")

    def test_an_explicit_range_is_one_period_and_therefore_one_fold(self) -> None:
        folds = build_fold_schedule(
            "20220101..20251231",
            "20220101..20251231",
            TRADING_DAYS,
            window_months=24,
            period="year",
        )
        self.assertEqual([fold.fold_id for fold in folds], ["fold_20220101..20251231"])
        self.assertEqual(
            (folds[0].validation_start, folds[0].validation_end), ("20220101", "20251231")
        )
        self.assertEqual(folds[0].input_window_start, "20200101")
        self.assertEqual(folds[0].valid_decision_time, anchor("20211231"))
        self.assertFalse(folds[0].has_test)

    def test_the_session_plan_interleaves_meta_between_every_two_folds(self) -> None:
        config = default_config()
        self.assertEqual((config.epochs, config.meta_learning_fold_interval), (3, 1))
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
        sessions = [(row.get("kind"), row.get("epoch_id"), row.get("fold_id")) for row in plan["sessions"]]
        def epoch(epoch_id: str) -> list[tuple[str, str, str | None]]:
            return [
                ("meta", epoch_id, "fold_2022"),
                ("fold", epoch_id, "fold_2022"),
                ("meta", epoch_id, "fold_2023"),
                ("fold", epoch_id, "fold_2023"),
                ("meta", epoch_id, "fold_2024"),
                ("fold", epoch_id, "fold_2024"),
                ("meta", epoch_id, "fold_2025"),
                ("fold", epoch_id, "fold_2025"),
            ]
        # 4 Folds x 3 Epochs = 12 Fold sessions, a Meta before every one of
        # them (the Epoch-start Meta covers the Epoch boundary), then one
        # Held-out after the last Epoch: no Meta ever follows the final Fold.
        self.assertEqual(
            sessions,
            [*epoch("epoch_001"), *epoch("epoch_002"), *epoch("epoch_003"), ("heldout", "epoch_003", None)],
        )
        kinds = [kind for kind, _epoch, _fold in sessions[:-1]]
        self.assertEqual(kinds.count("fold"), 12)
        self.assertEqual(kinds.count("meta"), 12)
        for index in range(1, len(kinds)):
            if kinds[index] == "fold":
                self.assertEqual(kinds[index - 1], "meta")
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


class TrailingValidationWindowTest(unittest.TestCase):
    """`validation_periods > 1` turns the Folds into walk-forward steps."""

    def test_one_period_reproduces_the_schedule_this_repository_runs_today(self) -> None:
        # The pinned yearly snapshot lives in RegularFoldScheduleTest; what
        # matters here is that asking for it explicitly changes nothing, so the
        # running arms keep their calendar when the knob appears.
        config = default_config()
        self.assertEqual(config.validation_periods, 1)
        explicit = build_fold_schedule(
            config.development_first_period,
            config.development_last_period,
            TRADING_DAYS,
            window_months=config.window_months,
            period=config.fold_period,
            validation_periods=1,
        )
        self.assertEqual(explicit, schedule(config))
        self.assertEqual([fold.fold_id for fold in explicit], ["fold_2022", "fold_2023", "fold_2024", "fold_2025"])

    def test_quarterly_folds_validate_the_trailing_four_quarters(self) -> None:
        folds = build_fold_schedule(
            "2022Q1", "2025Q4", TRADING_DAYS, window_months=24, period="quarter", validation_periods=4
        )
        # The first three quarters only ever serve as history: a Fold exists
        # for every label whose trailing four-quarter window fits inside the
        # development window, so 16 quarters give 13 Folds (12 transitions).
        self.assertEqual(len(folds), 13)
        self.assertEqual(folds[0].fold_id, "fold_2022Q4")
        self.assertEqual(folds[-1].fold_id, "fold_2025Q4")
        self.assertEqual(
            (folds[0].validation_start, folds[0].validation_end), ("20220101", "20221231")
        )
        # Input window and decision anchor still hang off validation_start, so
        # a 24-month window reaches the macro data floor exactly as before.
        self.assertEqual(
            (folds[0].input_window_start, folds[0].input_window_end), ("20200101", "20211231")
        )
        self.assertEqual(folds[0].valid_decision_time, anchor("20211231"))
        self.assertEqual(
            (folds[-1].validation_start, folds[-1].validation_end), ("20250101", "20251231")
        )
        self.assertEqual(folds[-1].valid_decision_time, anchor("20241231"))
        # Every step moves the whole window forward by exactly one quarter, and
        # only the last quarter of a window is new.
        self.assertEqual(
            [(fold.validation_start, fold.validation_end) for fold in folds[:3]],
            [
                ("20220101", "20221231"),
                ("20220401", "20230331"),
                ("20220701", "20230630"),
            ],
        )
        self.assertTrue(all(not fold.has_test for fold in folds))

    def test_each_fold_carries_its_own_quarter_as_the_step_region(self) -> None:
        # The step is the only part of the window the inherited parent has not
        # been developed on, so the walk-forward record is read from it.
        folds = build_fold_schedule(
            "2022Q1", "2025Q4", TRADING_DAYS, window_months=24, period="quarter", validation_periods=4
        )
        self.assertEqual(
            [(fold.step_start, fold.step_end) for fold in folds[:2]],
            [("20221001", "20221231"), ("20230101", "20230331")],
        )
        self.assertEqual((folds[-1].step_start, folds[-1].step_end), ("20251001", "20251231"))
        for fold in folds:
            self.assertTrue(fold.has_step)
            self.assertEqual(fold.step_end, fold.validation_end)

    def test_a_single_period_window_has_no_separate_step(self) -> None:
        # The whole validation region is the step; a copy of the same bounds
        # would be a second source for it.
        for fold in schedule(default_config()):
            self.assertFalse(fold.has_step)
            self.assertIsNone(fold.step_start)
            self.assertIsNone(fold.step_end)
        for fold in schedule(default_config(test_stage=True)):
            self.assertFalse(fold.has_step)

    def test_a_step_region_needs_both_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "step region needs start and end together"):
            FoldSpec(
                fold_id="fold_x",
                input_window_start="20200101",
                input_window_end="20211231",
                validation_start="20220101",
                validation_end="20221231",
                valid_decision_time=anchor("20211231"),
                step_start="20221001",
            )

    def test_a_trailing_window_is_quarterly_only(self) -> None:
        for period, first, last in (
            ("year", "2022", "2025"),
            ("month", "202201", "202512"),
        ):
            with self.subTest(period=period):
                with self.assertRaisesRegex(ValueError, "only supported at quarterly cadence"):
                    build_fold_schedule(
                        first, last, TRADING_DAYS, window_months=24, period=period, validation_periods=4
                    )
        with self.assertRaisesRegex(ValueError, "only supported at quarterly cadence"):
            default_config(validation_periods=4)  # the console default is yearly
        # The config accepts it at quarterly cadence and refuses the Test stage.
        config = default_config(
            fold_period="quarter",
            development_first_period="2022Q1",
            development_last_period="2025Q4",
            validation_periods=4,
        )
        self.assertEqual(len(schedule(config)), 13)
        with self.assertRaisesRegex(ValueError, "does not support a multi-period"):
            default_config(
                fold_period="quarter",
                development_first_period="2022Q1",
                development_last_period="2025Q4",
                validation_periods=4,
                test_stage=True,
            )

    def test_a_window_shorter_than_the_trailing_window_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation_periods=4 needs at least"):
            build_fold_schedule(
                "2022Q1", "2022Q3", TRADING_DAYS, window_months=24, period="quarter", validation_periods=4
            )

    def test_a_test_stage_cannot_take_a_multi_period_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support a multi-period"):
            build_fold_schedule(
                "2022Q1",
                "2025Q4",
                TRADING_DAYS,
                window_months=24,
                period="quarter",
                test_stage=True,
                validation_periods=4,
            )

    def test_validation_periods_must_be_a_positive_integer(self) -> None:
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "validation_periods must be a positive integer"):
                    build_fold_schedule(
                        "2022Q1",
                        "2025Q4",
                        TRADING_DAYS,
                        window_months=24,
                        period="quarter",
                        validation_periods=value,  # type: ignore[arg-type]
                    )
        with self.assertRaisesRegex(ValueError, "validation_periods must be a positive integer"):
            default_config(validation_periods=0)


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
