"""Timeview refresh-node table: cron drift guard + visibility-cutoff helpers."""

import argparse
import json
import re
import unittest
from datetime import date, datetime, time
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from autotrade.data_sources.tushare import cron_update
from autotrade.data_sources.tushare.common import (
    BOARD_TRADING_DEFAULT_DATASETS,
    DAILY_DOWNLOAD_DATASETS,
    DAILY_REQUIRED_DATASETS,
    EVENT_FLOW_DATASETS,
    FUNDAMENTAL_DATASETS,
    GLOBAL_CONTEXT_DEFAULT_DATASETS,
    INTRADAY_DATASETS,
    MACRO_REGIME_DEFAULT_DATASETS,
    REFERENCE_DATASETS,
    TEXT_DATASETS,
)

from autotrade.environment.data.contracts import (
    DOMAIN_REFRESH_NODES,
    EVENT_DATASET_REFRESH_NODES,
    REFRESH_NODES,
    TEXT_DATASET_REFRESH_NODES,
    domain_visible_cutoff,
    event_dataset_visible_cutoff,
    next_visible_boundary,
    text_dataset_visible_cutoff,
    visible_cutoff,
)

CN_TZ = ZoneInfo("Asia/Shanghai")
REPO_ROOT = Path(__file__).resolve().parents[2]
CRON_SCHEDULE = REPO_ROOT / "configs" / "tushare_update_schedule.json"

# Jobs that only audit/compare existing data and land nothing new — never nodes.
AUDIT_ONLY_JOBS = {
    "cn_nightly_full_audit",
    "cn_nightly_text_audit",
    "cn_daily_revision_sentinel",
    "cn_weekly_deep_audit",
    "cn_preopen_event_flow_audit_0920",
}

# Repair sweeps over already-node-governed data: they heal historical rows
# whose visibility stays governed by the existing nodes and row-level
# available_at stamps (the margin family's pre-open nodes, the evening / PIT-
# event-build nodes), and their launches precede no trading session — so they
# define no visibility boundary of their own.
WEEKLY_DEEP_SWEEP_JOBS = {
    "cn_weekly_reference_deep",
    "cn_weekly_fundamental_deep",
    "cn_evening_margin_backfill",
    "cn_evening_auction_backfill",
}


def _cron_jobs() -> set[str]:
    schedule = json.loads(CRON_SCHEDULE.read_text(encoding="utf-8"))
    return set(schedule["jobs"])


# Operator-invoked repair jobs (coverage re-requests).
# The "manual_" prefix in the schedule config declares them: they have no
# crontab line by design, repair history already governed by the existing
# nodes and row-level available_at stamps, and precede no trading session —
# so they define no visibility boundary of their own.
MANUAL_REPAIR_JOBS = {name for name in _cron_jobs() if name.startswith("manual_")}


CRONTAB = REPO_ROOT / "ops" / "cron" / "tushare_update.cron"

# Full dataset list a whole-tier download job lands, from the tushare
# package's canonical tier definitions (single source with the downloader).
# Daily here is the generic download default; stk_auction stays in the
# required/audit gate and is written only by dedicated capture/recheck jobs.
TIER_DATASETS = {
    "reference": REFERENCE_DATASETS,
    "daily": DAILY_DOWNLOAD_DATASETS,
    "fundamental": FUNDAMENTAL_DATASETS,
    "intraday": INTRADAY_DATASETS,
    "event_flow": EVENT_FLOW_DATASETS,
    "board_trading": BOARD_TRADING_DEFAULT_DATASETS,
    "text_evidence": TEXT_DATASETS,
    "macro": MACRO_REGIME_DEFAULT_DATASETS,
    "global": GLOBAL_CONTEXT_DEFAULT_DATASETS,
}

# A managed crontab line: "MM HH * * <*|dow-expr> ... --job <name> ...".
_CRON_LINE = re.compile(r"^\s*(\d{1,2})\s+(\d{1,2})\s+\*\s+\*\s+[\d*/,-]+\s+.*--job\s+(\S+)")


def _crontab_job_times() -> dict[str, time]:
    times: dict[str, time] = {}
    for line in CRONTAB.read_text(encoding="utf-8").splitlines():
        match = _CRON_LINE.match(line)
        if match:
            minute, hour, name = int(match.group(1)), int(match.group(2)), match.group(3)
            launch = time(hour, minute)
            times[name] = min(times.get(name, launch), launch)
    return times


class RefreshNodeDriftGuardTest(unittest.TestCase):
    @staticmethod
    def _job_datasets(job: dict) -> set[str] | None:
        """Datasets the job lands: an explicit --datasets list, else the full
        tier for download_tier jobs, else None (no dataset set derivable)."""
        args = list(job.get("extra_args", []))
        if "--datasets" in args:
            datasets: list[str] = []
            for value in args[args.index("--datasets") + 1:]:
                if str(value).startswith("--"):
                    break
                datasets.append(str(value))
            return set(datasets)
        if job.get("operation") == "download_tier":
            return set(TIER_DATASETS[job["tier"]])
        return None

    def test_every_node_is_a_real_cron_job(self) -> None:
        jobs = _cron_jobs()
        for name in REFRESH_NODES:
            self.assertIn(name, jobs, f"REFRESH_NODES[{name!r}] is not a cron job in the schedule")

    def test_audit_only_jobs_are_not_nodes(self) -> None:
        for job in AUDIT_ONLY_JOBS:
            self.assertNotIn(job, REFRESH_NODES, f"audit-only job {job!r} must not be a refresh node")

    def test_nightly_audit_does_not_expect_next_morning_event_data(self) -> None:
        schedule = json.loads(CRON_SCHEDULE.read_text(encoding="utf-8"))
        job = schedule["jobs"]["cn_nightly_full_audit"]
        self.assertEqual(job["end_date_offset_days"], 1)
        self.assertEqual(job["event_flow_end_extra_offset_days"], 1)
        self.assertNotIn("event_flow_end_date_mode", job)

    def test_nightly_audit_event_boundary_covers_weekdays_and_weekends(self) -> None:
        schedule = json.loads(CRON_SCHEDULE.read_text(encoding="utf-8"))
        job = schedule["jobs"]["cn_nightly_full_audit"]
        cases = (
            # 02:30 launch, generic end_date (D-1), expected event-flow end (D-2).
            ("Tuesday", "20260713", "20260712"),
            ("Wednesday", "20260714", "20260713"),
            ("Saturday", "20260717", "20260716"),
            ("Sunday", "20260718", "20260717"),
        )
        for launch_day, end_date, expected in cases:
            with self.subTest(launch_day=launch_day):
                ctx = cron_update.RunContext(
                    config=schedule,
                    repo_root=REPO_ROOT,
                    python="python",
                    job_name="cn_nightly_full_audit",
                    job=job,
                    start_date="20200101",
                    end_date=end_date,
                    timezone_name="Asia/Shanghai",
                )
                self.assertEqual(
                    cron_update.resolve_event_flow_audit_end_date(ctx),
                    expected,
                )

    def test_node_start_times_match_crontab(self) -> None:
        # Node ``start`` must equal the real installed crontab launch time, so the
        # Timeview ``ready_at`` cadence cannot silently drift from ingestion.
        cron_times = _crontab_job_times()
        for name, node in REFRESH_NODES.items():
            self.assertIn(name, cron_times, f"REFRESH_NODES[{name!r}] has no managed crontab line")
            self.assertEqual(
                node.start,
                cron_times[name],
                f"REFRESH_NODES[{name!r}].start {node.start} != crontab launch {cron_times[name]}",
            )

    def test_every_landing_job_has_a_node(self) -> None:
        # The crontab and the JSON schedule must list the same jobs, and every job
        # that lands data (not audit-only) must have a Timeview refresh node.
        cron_jobs = set(_crontab_job_times())
        schedule_jobs = _cron_jobs()
        self.assertEqual(
            cron_jobs,
            schedule_jobs - MANUAL_REPAIR_JOBS,
            "ops/cron/tushare_update.cron jobs differ from configs/tushare_update_schedule.json scheduled jobs",
        )
        for job in schedule_jobs - AUDIT_ONLY_JOBS - WEEKLY_DEEP_SWEEP_JOBS - MANUAL_REPAIR_JOBS:
            self.assertIn(job, REFRESH_NODES, f"data-landing job {job!r} has no Timeview refresh node")
        for job in MANUAL_REPAIR_JOBS:
            self.assertNotIn(job, REFRESH_NODES, f"manual repair job {job!r} must not be a refresh node")

    def test_managed_jobs_log_through_the_runner_only(self) -> None:
        # Every run writes its own log via the runner; cron lines carry no
        # shell redirects and no legacy dispatch-log plumbing.
        job_lines = [
            line for line in CRONTAB.read_text(encoding="utf-8").splitlines()
            if _CRON_LINE.match(line)
        ]
        self.assertTrue(job_lines)
        self.assertTrue(all("--dispatch-log" not in line for line in job_lines))
        self.assertTrue(all(">>" not in line and "2>&1" not in line for line in job_lines))

    def test_daily_download_tier_defaults_exclude_stk_auction(self) -> None:
        self.assertEqual(TIER_DATASETS["daily"], DAILY_DOWNLOAD_DATASETS)
        self.assertNotIn("stk_auction", TIER_DATASETS["daily"])
        self.assertIn("stk_auction", DAILY_REQUIRED_DATASETS)

    def test_evening_daily_set_cannot_bypass_dedicated_auction_capture(self) -> None:
        schedule = json.loads(CRON_SCHEDULE.read_text(encoding="utf-8"))
        args = schedule["jobs"]["cn_evening_full"]["extra_args"]

        def values_after(flag: str) -> list[str]:
            start = args.index(flag) + 1
            values: list[str] = []
            for value in args[start:]:
                if str(value).startswith("--"):
                    break
                values.append(str(value))
            return values

        daily = values_after("--daily-datasets")
        refreshed = values_after("--refresh-daily-datasets")
        self.assertTrue(daily)
        self.assertNotIn("stk_auction", daily)
        self.assertNotIn("stk_auction", refreshed)

    def test_auction_capture_runs_pre_open_only_with_evening_recheck(self) -> None:
        # The dedicated capture job runs only in the pre-open window (09:27 +
        # the 09:31 retry); the late-day catch-up/reconciliation is the
        # dedicated cn_evening_auction_backfill job, mirroring the margin
        # family's evening self-heal.
        auction_lines = [
            line
            for line in CRONTAB.read_text(encoding="utf-8").splitlines()
            if "--job cn_open_auction_capture_0927" in line and _CRON_LINE.match(line)
        ]
        self.assertEqual(len(auction_lines), 2)
        self.assertTrue(all(re.match(r"^(27|31)\s+9\s+\*\s+\*\s+\*", line) for line in auction_lines))
        self.assertTrue(all("--force-run" not in line for line in auction_lines))

    def test_evening_node_ready_at_matches_duration_fixture(self) -> None:
        # Conservative fallback: 23:35 launch + 210 min -> 03:05 next day.
        node = REFRESH_NODES["cn_evening_full"]
        self.assertEqual(
            node.ready_at(date(2022, 1, 5)),
            datetime(2022, 1, 6, 3, 5, tzinfo=CN_TZ),
        )

    def test_dataset_overrides_reference_real_nodes(self) -> None:
        for mapping in (DOMAIN_REFRESH_NODES, EVENT_DATASET_REFRESH_NODES, TEXT_DATASET_REFRESH_NODES):
            for key, node_names in mapping.items():
                for name in node_names:
                    self.assertIn(name, REFRESH_NODES, f"{key!r} maps to unknown node {name!r}")

    def test_auction_has_no_conflicting_fixed_refresh_cutoff(self) -> None:
        self.assertEqual(DOMAIN_REFRESH_NODES["auction"], ())
        self.assertIsNone(
            domain_visible_cutoff("auction", datetime(2022, 1, 5, 20, 0, tzinfo=CN_TZ))
        )

    def test_preopen_text_backfill_resolves_a_same_day_window(self) -> None:
        # Timeview grants replay visibility of news with available_at <= 08:55
        # of day D once the 08:55 node completes, so live ingestion must land
        # the same morning's news before the open: the job window ends at D
        # itself (natural-day, no trading-calendar clamp), looking back 2 days.
        schedule = json.loads(CRON_SCHEDULE.read_text(encoding="utf-8"))
        job = schedule["jobs"]["cn_preopen_text_backfill_0855"]
        self.assertEqual(job["end_date_offset_days"], 0)
        self.assertNotIn("end_date_mode", job)

        class FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 8, 7, 8, 55, tzinfo=tz)

        args = argparse.Namespace(
            config=str(CRON_SCHEDULE),
            job="cn_preopen_text_backfill_0855",
            start_date=None,
            end_date=None,
        )
        with (
            patch.object(cron_update, "datetime", FrozenDatetime),
            patch.dict(cron_update.os.environ, {"TUSHARE_UPDATE_START_DATE": ""}),
        ):
            ctx = cron_update.build_context(args)
        self.assertEqual(ctx.end_date, "20260807")
        self.assertEqual(ctx.start_date, "20260805")

    def test_dataset_refresh_overrides_are_landed_by_their_jobs(self) -> None:
        jobs = json.loads(CRON_SCHEDULE.read_text(encoding="utf-8"))["jobs"]
        for mapping in (EVENT_DATASET_REFRESH_NODES, TEXT_DATASET_REFRESH_NODES):
            for dataset, node_names in mapping.items():
                for node_name in node_names:
                    selected = self._job_datasets(jobs[node_name])
                    if selected is None:
                        continue  # full multi-tier update: no dataset set derivable
                    self.assertIn(
                        dataset,
                        selected,
                        f"{dataset!r} claims {node_name!r}, but that job does not download it",
                    )


class VisibilityCutoffTest(unittest.TestCase):
    def test_daily_domain_visible_only_through_prior_day_during_session(self) -> None:
        # During day D's session the evening node that lands D's daily core has not
        # finished (fallback D 23:35 -> D+1 03:05), so the cutoff is D-1's evening
        # start: daily for D-1 is visible, daily for D is not.
        when = datetime(2022, 1, 5, 9, 31, tzinfo=CN_TZ)
        cutoff = domain_visible_cutoff("daily", when)
        self.assertEqual(cutoff, datetime(2022, 1, 4, 23, 35, tzinfo=CN_TZ))

    def test_daily_domain_rolls_after_evening_completes(self) -> None:
        # After 03:05 on D+1 the evening node that ran D 23:35 has completed.
        when = datetime(2022, 1, 6, 3, 30, tzinfo=CN_TZ)
        cutoff = domain_visible_cutoff("daily", when)
        self.assertEqual(cutoff, datetime(2022, 1, 5, 23, 35, tzinfo=CN_TZ))

    def test_margin_secs_visible_same_day_after_preopen_node(self) -> None:
        # By 09:31 both the 09:03 backfill and 09:13 retry have completed, so the
        # same-day shortable universe (available ~09:00) is visible.
        when = datetime(2022, 1, 5, 9, 31, tzinfo=CN_TZ)
        cutoff = event_dataset_visible_cutoff("margin_secs", when)
        self.assertEqual(cutoff, datetime(2022, 1, 5, 9, 13, tzinfo=CN_TZ))

    def test_margin_secs_not_yet_visible_before_preopen_node(self) -> None:
        # At 08:00 no same-day margin_secs node has completed; the cutoff falls back
        # to the prior day's retry instant (yesterday's universe only).
        when = datetime(2022, 1, 5, 8, 0, tzinfo=CN_TZ)
        cutoff = event_dataset_visible_cutoff("margin_secs", when)
        self.assertEqual(cutoff, datetime(2022, 1, 4, 9, 13, tzinfo=CN_TZ))

    def test_fundamentals_visible_after_pit_build_completes(self) -> None:
        # Full-window rebuild boundary: ready ~04:05 (03:35 + 30 min).
        when = datetime(2022, 1, 5, 4, 10, tzinfo=CN_TZ)
        cutoff = domain_visible_cutoff("fundamentals", when)
        self.assertEqual(cutoff, datetime(2022, 1, 5, 3, 35, tzinfo=CN_TZ))
        before_ready = domain_visible_cutoff("fundamentals", datetime(2022, 1, 5, 4, 0, tzinfo=CN_TZ))
        self.assertLess(before_ready, datetime(2022, 1, 5, 3, 35, tzinfo=CN_TZ))

    def test_fundamental_pit_node_does_not_advance_on_sunday_or_monday(self) -> None:
        # The previous-trading-day range is unchanged on both launches, so the
        # runner skips them. Friday's data was last rebuilt on Saturday.
        saturday = datetime(2022, 1, 8, 3, 35, tzinfo=CN_TZ)
        for when in (
            datetime(2022, 1, 9, 4, 10, tzinfo=CN_TZ),
            datetime(2022, 1, 10, 4, 10, tzinfo=CN_TZ),
        ):
            with self.subTest(when=when):
                self.assertEqual(domain_visible_cutoff("fundamentals", when), saturday)

        self.assertEqual(
            next_visible_boundary(
                DOMAIN_REFRESH_NODES["fundamentals"],
                datetime(2022, 1, 10, 4, 10, tzinfo=CN_TZ),
            ),
            datetime(2022, 1, 11, 4, 5, tzinfo=CN_TZ),
        )

    def test_text_lands_on_weekends_unlike_the_trading_day_evening_node(self) -> None:
        # Sunday 23:35: the 23:15 text node has completed (ready 23:30), so
        # Sunday's news is visible. The evening node is weekday-only, so daily
        # market data is still Friday's.
        sunday_night = datetime(2022, 1, 9, 23, 35, tzinfo=CN_TZ)
        self.assertEqual(
            text_dataset_visible_cutoff("news", sunday_night),
            datetime(2022, 1, 9, 23, 15, tzinfo=CN_TZ),
        )
        self.assertEqual(
            domain_visible_cutoff("daily", sunday_night),
            datetime(2022, 1, 7, 23, 35, tzinfo=CN_TZ),
        )

    def test_text_datasets_without_an_override_use_the_text_node(self) -> None:
        # Announcements, policy documents and research reports have no pre-open
        # refinement; they roll on the daily text node alone.
        saturday_night = datetime(2022, 1, 8, 23, 40, tzinfo=CN_TZ)
        for dataset in ("anns_d", "major_news", "npr", "research_report", "report_rc"):
            self.assertNotIn(dataset, TEXT_DATASET_REFRESH_NODES)
            self.assertEqual(
                text_dataset_visible_cutoff(dataset, saturday_night),
                datetime(2022, 1, 8, 23, 15, tzinfo=CN_TZ),
                dataset,
            )

    def test_cctv_news_refined_by_preopen_text_node(self) -> None:
        # The evening node lands the bulk; the 08:55 pre-open backfill refines the
        # same-day short text, so by 09:00 the later (08:55) cutoff wins.
        when = datetime(2022, 1, 5, 9, 0, tzinfo=CN_TZ)
        cutoff = text_dataset_visible_cutoff("cctv_news", when)
        self.assertEqual(cutoff, datetime(2022, 1, 5, 8, 55, tzinfo=CN_TZ))

    def test_unknown_dataset_defaults_to_evening_node(self) -> None:
        when = datetime(2022, 1, 5, 9, 31, tzinfo=CN_TZ)
        self.assertEqual(
            event_dataset_visible_cutoff("anns_d", when),
            domain_visible_cutoff("daily", when),
        )

    def test_board_datasets_visible_from_preopen_backfill(self) -> None:
        # kpl_list/limit_step/limit_cpt_list publish next-day ~08:30 and land in
        # the 08:50 pre-open backfill: visible from 08:55, not the prior evening.
        for dataset in ("kpl_list", "limit_step", "limit_cpt_list"):
            after = event_dataset_visible_cutoff(dataset, datetime(2022, 1, 5, 8, 56, tzinfo=CN_TZ))
            self.assertEqual(after, datetime(2022, 1, 5, 8, 50, tzinfo=CN_TZ), dataset)
            before = event_dataset_visible_cutoff(dataset, datetime(2022, 1, 5, 8, 40, tzinfo=CN_TZ))
            self.assertEqual(before, datetime(2022, 1, 4, 23, 35, tzinfo=CN_TZ), dataset)
        # Hot lists land in the evening window only: default node applies.
        self.assertEqual(
            event_dataset_visible_cutoff("dc_hot", datetime(2022, 1, 5, 8, 56, tzinfo=CN_TZ)),
            datetime(2022, 1, 4, 23, 35, tzinfo=CN_TZ),
        )

    def test_visible_cutoff_none_before_any_node_completes(self) -> None:
        # Just after midnight on the very first day, no evening node has finished.
        when = datetime(2022, 1, 1, 0, 5, tzinfo=CN_TZ)
        # The prior day's evening node (2021-12-31 23:35 -> 2022-01-01 03:05) is not
        # done at 00:05, and the day-before-that completed 2021-12-31 03:05.
        cutoff = visible_cutoff(("cn_evening_full",), when)
        self.assertEqual(cutoff, datetime(2021, 12, 30, 23, 35, tzinfo=CN_TZ))

    def test_evening_node_does_not_advance_on_weekend_skip(self) -> None:
        monday_morning = datetime(2022, 1, 10, 9, 0, tzinfo=CN_TZ)
        self.assertEqual(
            visible_cutoff(("cn_evening_full",), monday_morning),
            datetime(2022, 1, 7, 23, 35, tzinfo=CN_TZ),
        )
        saturday = datetime(2022, 1, 8, 10, 0, tzinfo=CN_TZ)
        self.assertEqual(
            next_visible_boundary(("cn_evening_full",), saturday),
            datetime(2022, 1, 11, 3, 5, tzinfo=CN_TZ),
        )


if __name__ == "__main__":
    unittest.main()
