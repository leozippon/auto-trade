"""Paper refuses a pinned release unless every dataset it mounts was pulled
whole since the landing run due before the decision, with no open issue."""

from __future__ import annotations

import io
import os
from contextlib import chdir, redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from autotrade.data_sources.tushare import common, cron_update, download, versions
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.strategy import CN_TZ
from autotrade.paper.engine import PaperDataNotReady
from autotrade.paper.freshness import require_fresh_release, unready_datasets

SCHEDULE = cron_update.load_config(Path(__file__).resolve().parents[2] / cron_update.DEFAULT_CONFIG)
CORE = ("daily", "daily_basic", "adj_factor", "stk_limit", "suspend_d")
SESSIONS = ("20261021", "20261022", "20261023", "20261026")
MONDAY = datetime(2026, 10, 26, 8, 30, tzinfo=CN_TZ)
CONFIG = SnapshotConfig(
    events_datasets=("moneyflow", "margin", "stk_holdernumber"),
    macro_datasets=("index_daily",),
    text_datasets=("anns_d", "research_report"),
    fundamental_datasets=("income_vip",),
    include_intraday=False,
    include_industry=False,
)
THURSDAY_EVENING = "2026-10-22T15:50:00+00:00"
FRIDAY_EVENING = "2026-10-23T15:50:00+00:00"  # 23:50, after the 23:35 evening launch
FRESH = {
    **dict.fromkeys((*CORE, "moneyflow", "index_daily", "income_vip"), FRIDAY_EVENING),
    "margin": "2026-10-23T01:20:00+00:00",  # Friday's 09:15 pre-open retry
    "stk_holdernumber": "2026-10-25T01:50:00+00:00",  # Sunday's 09:45 disclosure run
    "anns_d": "2026-10-25T15:20:00+00:00",  # Sunday's 23:15 titles run
}


def record(refreshed: dict[str, str], issues: list[dict] | None = None) -> dict:
    return {
        "schema_version": 3,
        "state": "committed",
        "generation_id": "g",
        "datasets": {name: {"refreshed_at": at, "vintage": "v"} for name, at in refreshed.items()},
        "issues": issues or [],
    }


def test_a_release_every_landing_run_refreshed_is_ready():
    assert unready_datasets(record(FRESH), CONFIG, SCHEDULE, SESSIONS, MONDAY) == []
    # The 09:45 disclosure run of the decision's own morning lands after it:
    # Sunday's run is the one due, Saturday's is not enough.
    stale = record({**FRESH, "stk_holdernumber": "2026-10-24T01:50:00+00:00"})
    assert unready_datasets(stale, CONFIG, SCHEDULE, SESSIONS, MONDAY) == [
        "stk_holdernumber last refreshed 2026-10-24 09:50, before the cn_nightly_disclosure_full run of 2026-10-25 09:45"
    ]
    # Session-bound jobs skip the days their window does not move: after a
    # holiday the evening run due is the last session's, not a holiday's.
    holiday = dict(FRESH)
    holiday.update(dict.fromkeys((*CORE, "moneyflow", "index_daily", "income_vip"), "2026-09-30T15:50:00+00:00"))
    holiday.update({"margin": "2026-09-30T01:20:00+00:00", "stk_holdernumber": "2026-10-07T01:50:00+00:00", "anns_d": "2026-10-07T15:20:00+00:00"})
    after_holiday = datetime(2026, 10, 8, 8, 30, tzinfo=CN_TZ)
    assert unready_datasets(record(holiday), CONFIG, SCHEDULE, ("20260929", "20260930", "20261008"), after_holiday) == []


def test_a_record_without_refresh_times_asks_for_the_migration():
    with pytest.raises(PaperDataNotReady, match=r"predates per-dataset refresh times; run `python scripts/data/tushare_lake\.py migrate --apply`"):
        require_fresh_release({"schema_version": 2, "state": "committed", "generation_id": "g"}, CONFIG, SCHEDULE, SESSIONS, MONDAY)


class _DailyClient:
    """The vendor for the daily tier; ``defective`` names the (api, trade
    date) whose answer is defective, as a missing field or a repeated page
    would make the real client raise."""

    def __init__(self, defective: tuple[str, str] | None = None) -> None:
        self.defective = defective

    def query(self, api_name, params=None, fields="", retries=5):
        params = params or {}
        if (api_name, params.get("trade_date")) == self.defective:
            raise common.DefectiveAnswer(f"{api_name} answered without requested fields ['adj_factor']")
        columns = fields.split(",")
        row = [params.get("trade_date") if c == "trade_date" else "000001.SZ" if c == "ts_code" else 1.0 for c in columns]
        return common.ApiResult(columns, [row])


class _FailingClient:
    def query(self, api_name, params=None, fields="", retries=5):
        raise RuntimeError(f"{api_name} failed after 5 attempts: answered HTTP 502")


def _evening(tmp_path: Path, tiers: list[tuple[str, list[str], object]]) -> dict:
    """Run tiers of Friday's evening job in order, as its update steps do
    (the first failure ends the job), and fold their report, as the runner
    does, onto a record where the tiers' datasets were last refreshed
    Thursday evening."""

    raw = tmp_path / "data" / "raw"
    calendar = raw / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
    calendar.parent.mkdir(parents=True, exist_ok=True)
    days = pd.date_range("2026-09-01", "2026-12-31").strftime("%Y%m%d")
    pd.DataFrame({"cal_date": days, "is_open": [str(int(day in SESSIONS or pd.Timestamp(day).weekday() < 5)) for day in days]}).to_parquet(calendar, index=False)
    report = tmp_path / "run.jsonl"
    pulled = [dataset for _tier, datasets, _client in tiers for dataset in datasets]
    generation = record({**FRESH, **dict.fromkeys(pulled, THURSDAY_EVENING)})
    with chdir(tmp_path), patch.dict(os.environ, {versions.RUN_REPORT_ENV: str(report)}), redirect_stdout(io.StringIO()):
        for tier, datasets, client in tiers:
            args = download.parse_args([
                "download", "--tier", tier, "--raw-dir", str(raw), "--start-date", "20261023", "--end-date", "20261023",
                "--datasets", *datasets, "--revision-ledger", str(tmp_path / "ledger.jsonl"), "--min-interval-seconds", "0",
            ])
            with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
                try:
                    download.download_selected_tier(args)
                except RuntimeError:
                    break
    cron_update.fold_outcomes(generation["datasets"], generation["issues"], cron_update.read_run_report(report), at=FRIDAY_EVENING)
    return generation


def test_a_failed_evening_job_leaves_its_unfinished_steps_stale(tmp_path: Path):
    # The daily step lands Friday; the event step fails on the vendor, and
    # the fundamental step after it never runs.
    generation = _evening(tmp_path, [
        ("daily", list(CORE), _DailyClient()),
        ("event_flow", ["moneyflow"], _FailingClient()),
        ("fundamental", ["income_vip"], _FailingClient()),
    ])
    assert {name: generation["datasets"][name]["refreshed_at"] for name in CORE} == dict.fromkeys(CORE, FRIDAY_EVENING)
    with pytest.raises(PaperDataNotReady) as refused:
        require_fresh_release(generation, CONFIG, SCHEDULE, SESSIONS, MONDAY)
    assert str(refused.value) == (
        "generation g is not ready for the decision at 2026-10-26 08:30: "
        "income_vip last refreshed 2026-10-22 23:50, before the cn_evening_full run of 2026-10-23 23:35; "
        "moneyflow last refreshed 2026-10-22 23:50, before the cn_evening_full run of 2026-10-23 23:35"
    )


def test_a_defective_new_partition_keeps_its_dataset_unrefreshed_and_open(tmp_path: Path):
    # The daily step completes (exit 0) while adj_factor's answer for Friday
    # was defective: the partition is skipped, so the snapshot would join a
    # Friday without factors.
    generation = _evening(tmp_path, [("daily", list(CORE), _DailyClient(defective=("adj_factor", "20261023")))])
    assert not (tmp_path / "data" / "raw" / "adj_factor" / "trade_date=20261023.parquet").exists()
    assert generation["datasets"]["daily"]["refreshed_at"] == FRIDAY_EVENING
    assert generation["datasets"]["adj_factor"]["refreshed_at"] == THURSDAY_EVENING
    assert unready_datasets(generation, CONFIG, SCHEDULE, SESSIONS, MONDAY) == [
        "adj_factor last refreshed 2026-10-22 23:50, before the cn_evening_full run of 2026-10-23 23:35",
        "adj_factor has an open defect in trade_date=20261023 (adj_factor answered without requested fields ['adj_factor'])",
    ]
    # The next complete pull of the partition closes the defect and refreshes.
    cron_update.fold_outcomes(generation["datasets"], generation["issues"], [
        {"dataset": "adj_factor", "partition": "trade_date=20261023", "outcome": "appended", "at": "t"},
        {"dataset": "adj_factor", "partition": "", "outcome": "complete", "at": "t"},
    ], at="2026-10-24T01:00:00+00:00")
    assert unready_datasets(generation, CONFIG, SCHEDULE, SESSIONS, MONDAY) == []
