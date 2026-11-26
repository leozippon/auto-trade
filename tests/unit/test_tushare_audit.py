"""The data-quality audits (data_sources/tushare/audit.py)."""

import argparse
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from autotrade.data_sources.tushare import audit, common, specs, stamps, versions
from autotrade.data_sources.tushare import client as tushare_client
from autotrade.data_sources.tushare import io as tushare_io

from .support.tushare_fixture import (
    DAILY_FIELDS,
    EmptyClient,
    FieldClient,
    LakeTestCase,
)


class ErrorTradeDateClient:
    def query(self, api_name, params=None, fields="", retries=5):
        raise RuntimeError("mock source failure")


class AuditTest(LakeTestCase):
    def test_audit_window_never_expects_future_trading_days(self):
        """The calendar is published ahead; the audit must not demand data for
        days that have not happened, or it can never reach ok and real gaps
        hide in the permanent noise."""
        path = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2126.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        # A calendar reaching a century into the future, as the real one does
        # for the rest of the current year.
        pd.DataFrame([{"cal_date": "21260605", "is_open": "1"}]).to_parquet(path, index=False)
        self.assertEqual(common.latest_sse_calendar_date(self.raw_dir), "21260605")

        # The window ends at the last SETTLED day: today's session has not
        # closed and is not ingested until the evening cron, so expecting it
        # would keep the audit red from midnight onward -- the same
        # desensitization as expecting future days.
        today = datetime.now(audit.CN_TZ).strftime("%Y%m%d")
        resolved = audit.resolve_audit_end_date(self.raw_dir, None)
        self.assertLess(resolved, today)
        self.assertEqual(resolved, (datetime.now(audit.CN_TZ) - timedelta(days=1)).strftime("%Y%m%d"))
        # An explicit window is the operator's business and is never clamped:
        # re-auditing a historical window must stay possible.
        self.assertEqual(audit.resolve_audit_end_date(self.raw_dir, "20200101"), "20200101")

    def test_bak_basic_audit_ignores_trade_cal_lookahead_after_end_date(self):
        path = self.raw_dir / "bak_basic" / "trade_date=20260618.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"trade_date": "20260618", "ts_code": "000001.SZ"}]).to_parquet(path, index=False)
        findings = []

        audit.audit_bak_basic(self.raw_dir, {"20260618", "20260622"}, "20260618", lambda *item: findings.append(item))

        partition_finding = next(item for item in findings if item[1] == "bak_basic_partitions")
        self.assertEqual(partition_finding[0], "info")
        self.assertEqual(partition_finding[3]["missing_expected_files"], 0)
        self.assertEqual(partition_finding[3]["missing_sample"], [])

    def test_by_ts_code_universe_is_one_rule_for_download_and_audit(self):
        # The audit used to expect fx codes for any by-ts_code dataset it did
        # not know, where the downloader refuses one.
        args = argparse.Namespace(index_code=None, cn_index_code=None, fx_code=["USDCNH.FXCM"])
        self.assertEqual(
            audit.expected_macro_paths(self.raw_dir, specs.MACRO_SPECS["fx_daily"], "20260101", "20260131", args),
            {self.raw_dir / "fx_daily" / "ts_code=USDCNH.FXCM" / "year=2026.parquet"},
        )
        unknown = specs.DatasetSpec(api_name="new_by_code", strategy=specs.FetchStrategy.DATE_YEAR_BY_TS_CODE)
        with self.assertRaisesRegex(RuntimeError, "no ts_code universe defined for by-ts_code macro dataset 'new_by_code'"):
            audit.expected_macro_paths(self.raw_dir, unknown, "20260101", "20260131", args)

    def test_daily_audit_warns_on_exact_limit_without_pagination_probe(self):
        path = self.raw_dir / "daily" / "trade_date=20200102.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "trade_date": ["20200102"] * 5000,
            "ts_code": [f"{index:06d}.SZ" for index in range(5000)],
        }).to_parquet(path, index=False)
        findings = []
        audit.audit_trade_date_dataset(self.raw_dir, specs.DAILY_SPECS["daily"], {"20200102"}, lambda *item: findings.append(item))
        self.assertEqual(findings[0][0], "warning")
        self.assertEqual(findings[0][3]["exact_common_limit_row_count_dates"], ["20200102"])

    def test_trailing_zero_run_warns_for_zero_tolerant_dataset(self):
        # zero_rows_ok suppresses per-partition zero warnings, so a feed that
        # stops publishing (slb_len_mm: one year of empty partitions) never
        # surfaced. A long trailing run of zero-row partitions must warn.
        spec = specs.EVENT_FLOW_SPECS["block_trade"]
        columns = spec.fields.split(",") + ["available_at", "available_at_rule"]
        dates = ["20260701", "20260702", "20260703", "20260706", "20260707"]
        for index, trade_date in enumerate(dates):
            if index < 2:
                rows = pd.DataFrame([{
                    **{column: "1.0" for column in spec.fields.split(",")},
                    "trade_date": trade_date,
                    "available_at": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}T21:00:00+08:00",
                    "available_at_rule": "official_21_from:trade_date",
                }])
            else:
                rows = pd.DataFrame(columns=columns)
            tushare_io.write_parquet(
                self.raw_dir / "block_trade" / f"trade_date={trade_date}.parquet",
                rows, api_name="block_trade", params={"trade_date": trade_date},
                fields=columns,
            )
        expected = {self.raw_dir / "block_trade" / f"trade_date={d}.parquet" for d in dates}

        def run_audit():
            findings = []
            audit.audit_event_dataset(self.raw_dir, spec, expected, lambda *item: findings.append(item))
            return next(item for item in findings if item[1] == "block_trade_event_partitions")

        with patch.object(audit, "TRAILING_ZERO_RUN_WARN_PARTITIONS", 3):
            severity, _, _, details = run_audit()
            self.assertEqual(severity, "warning")
            self.assertEqual(details["trailing_zero_row_partitions"], 3)
        with patch.object(audit, "TRAILING_ZERO_RUN_WARN_PARTITIONS", 4):
            severity, _, _, details = run_audit()
            self.assertEqual(severity, "info")
            self.assertEqual(details["trailing_zero_row_partitions"], 3)

    def test_trailing_zero_run_warns_on_the_generic_board_path(self):
        # limit_list_d moved to the board tier; the stale-feed detector must
        # survive the move (the dedicated board audit used to enable it).
        spec = specs.BOARD_TRADING_SPECS["limit_list_d"]
        columns = spec.fields.split(",") + ["available_at", "available_at_rule"]
        dates = ["20260701", "20260702", "20260703", "20260706", "20260707"]
        for index, trade_date in enumerate(dates):
            if index < 2:
                rows = stamps.augment_board_frame(
                    pd.DataFrame([{
                        **{column: "1.0" for column in spec.fields.split(",")},
                        "trade_date": trade_date,
                    }]),
                    spec,
                    {"trade_date": trade_date},
                )
            else:
                rows = pd.DataFrame(columns=columns)
            tushare_io.write_parquet(
                self.raw_dir / "limit_list_d" / f"trade_date={trade_date}.parquet",
                rows, api_name="limit_list_d", params={"trade_date": trade_date},
                fields=columns,
            )
        expected = {self.raw_dir / "limit_list_d" / f"trade_date={d}.parquet" for d in dates}

        findings = []
        with patch.object(audit, "TRAILING_ZERO_RUN_WARN_PARTITIONS", 3):
            audit.audit_board_dataset(self.raw_dir, spec, expected, lambda *item: findings.append(item))
        severity, _, _, details = next(item for item in findings if item[1] == "limit_list_d_board_partitions")
        self.assertEqual(severity, "warning")
        self.assertEqual(details["trailing_zero_row_partitions"], 3)

    def test_text_trailing_zero_run_is_counted_per_source_subdirectory(self):
        # news partitions sort by source first, so a whole-listing trailing run
        # could only ever see the last-sorted source. A dead feed that sorts
        # earlier (src=deadfeed < src=livefeed) must still be detected, and the
        # live sibling must not be flagged.
        spec = specs.TEXT_SPECS["news"]
        columns = spec.fields.split(",") + ["available_at", "available_at_rule"]
        days = ["20260701", "20260702", "20260703", "20260704"]
        files = set()
        for source, dead in (("deadfeed", True), ("livefeed", False)):
            for index, day in enumerate(days):
                if dead and index > 0:
                    rows = pd.DataFrame(columns=columns)
                else:
                    rows = stamps.augment_text_frame(
                        pd.DataFrame([{
                            "datetime": f"{day[:4]}-{day[4:6]}-{day[6:]} 09:00:00",
                            "content": "内容", "title": f"标题{day}", "channels": "",
                        }]),
                        spec,
                    )
                path = self.raw_dir / "news" / f"src={source}" / f"date={day}.parquet"
                tushare_io.write_parquet(
                    path, rows, api_name="news", params={"src": source, "date": day},
                    fields=columns,
                )
                files.add(path)

        findings = []
        with patch.object(audit, "TRAILING_ZERO_RUN_WARN_PARTITIONS", 3):
            audit.audit_text_dataset(self.raw_dir, spec, files, lambda *item: findings.append(item))
        severity, _, _, details = next(item for item in findings if item[1] == "news_text_partitions")
        self.assertEqual(severity, "warning")
        self.assertEqual(details["trailing_zero_row_partitions"], 3)
        self.assertEqual(details["stale_feed_sources"], ["src=deadfeed"])

    def test_fundamental_cap_plateau_flags_periods_pinned_at_dataset_max(self):
        # A vendor per-call cap pins many periods at exactly the dataset
        # maximum (cashflow_vip held 23 periods at 6,400); organic counts vary,
        # so several periods sharing the exact maximum is the cap signature
        # regardless of the cap value.
        findings = []
        audit.audit_fundamental_cap_plateau({
            "income_vip": {"20231231": 11771, "20240331": 5300, "20240630": 7707},
            "balancesheet_vip": {"20231231": 11771, "20240331": 5310, "20240630": 6915},
            "cashflow_vip": {"20231231": 6400, "20240331": 6400, "20240630": 6400},
        }, lambda *item: findings.append(item))
        severity, check, message, details = findings[0]
        self.assertEqual(check, "fundamental_statement_cap_plateau")
        self.assertEqual(severity, "warning")
        self.assertEqual(message, "statement datasets have multiple periods pinned at their maximum row count")
        self.assertEqual(details["plateaus"], [
            {"dataset": "cashflow_vip", "max_rows": 6400, "periods_at_max": 3,
             "period_sample": ["20231231", "20240331", "20240630"]},
        ])

    def test_fundamental_cap_plateau_stays_silent_on_organic_counts(self):
        # The repaired lake diverges across statements by more than 2x per
        # period; only exact repeated maxima may warn, never ratios.
        findings = []
        audit.audit_fundamental_cap_plateau({
            "income_vip": {"20231231": 11771, "20240331": 5300},
            "balancesheet_vip": {"20231231": 11771, "20240331": 5310},
            "cashflow_vip": {"20231231": 6400, "20240331": 12047},
        }, lambda *item: findings.append(item))
        severity, _, message, details = findings[0]
        self.assertEqual(severity, "info")
        self.assertEqual(message, "no statement dataset has multiple periods pinned at exactly its maximum row count")
        self.assertEqual(details["plateaus"], [])

        # Tiny early-history partitions below the row floor never warn, and a
        # dataset subset is checked per dataset instead of being skipped.
        small = []
        audit.audit_fundamental_cap_plateau(
            {"cashflow_vip": {"20100331": 900, "20100630": 900, "20100930": 900}},
            lambda *item: small.append(item),
        )
        severity, _, _, details = small[0]
        self.assertEqual(severity, "info")
        self.assertEqual(details["datasets_checked"], ["cashflow_vip"])
        self.assertEqual(details["plateaus"], [])

    def test_fundamental_audit_wires_cap_plateau_check_over_period_partitions(self):
        stock_basic = self.raw_dir / "stock_basic" / "list_status=L.parquet"
        stock_basic.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"ts_code": "000001.SZ"}]).to_parquet(stock_basic, index=False)
        periods = ("20200331", "20200630", "20200930")
        rows_by_dataset = {
            "income_vip": (1200, 1100, 1300),
            "balancesheet_vip": (1250, 1150, 1350),
            "cashflow_vip": (1000, 1000, 1000),  # pinned at its maximum: cap signature
        }
        for dataset, row_counts in rows_by_dataset.items():
            for period, rows in zip(periods, row_counts):
                frame = pd.DataFrame([
                    {"ts_code": f"{index:06d}.SZ", "ann_date": f"{period[:6]}30", "f_ann_date": f"{period[:6]}30",
                     "end_date": period, "report_type": "1", "comp_type": "1", "end_type": "1"}
                    for index in range(rows)
                ])
                tushare_io.write_parquet(
                    self.raw_dir / dataset / f"period={period}.parquet",
                    frame, api_name=dataset, params={"period": period},
                    fields=list(frame.columns),
                )
        args = argparse.Namespace(
            fundamental_start_date="20200101",
            fundamental_end_date=None,
            fundamental_datasets=["income_vip", "balancesheet_vip", "cashflow_vip"],
        )
        findings = []
        audit.audit_fundamental_completeness(self.raw_dir, args, lambda *item: findings.append(item))
        severity, _, _, details = next(item for item in findings if item[1] == "fundamental_statement_cap_plateau")
        self.assertEqual(severity, "warning")
        self.assertEqual(details["plateaus"], [
            {"dataset": "cashflow_vip", "max_rows": 1000, "periods_at_max": 3, "period_sample": list(periods)},
        ])

    def test_index_weight_audit_checks_year_partitions_and_monthly_coverage(self):
        def write_year(code, year, months):
            path = self.raw_dir / "index_weight" / f"index_code={code}" / f"year={year}.parquet"
            rows = pd.DataFrame([
                {"index_code": code, "con_code": f"{m:06d}.SZ", "trade_date": f"{year}{m:02d}28", "weight": 1.0}
                for m in months
            ])
            path.parent.mkdir(parents=True, exist_ok=True)
            rows.to_parquet(path, index=False)

        def run_audit():
            findings = []
            audit.audit_index_weight(self.raw_dir, "20221231", lambda *item: findings.append(item))
            return findings[0]

        with patch.object(audit, "DEFAULT_CN_INDEX_CODES", ["000300.SH"]):
            write_year("000300.SH", 2020, range(1, 13))
            write_year("000300.SH", 2021, range(1, 12))  # closed year, 11 months
            severity, _, _, details = run_audit()
            self.assertEqual(severity, "error")  # year=2022 missing
            self.assertEqual(details["missing_year_partitions"], 1)

            write_year("000300.SH", 2022, range(1, 8))  # open year: no month check
            severity, _, _, details = run_audit()
            self.assertEqual(severity, "warning")
            self.assertEqual(details["closed_years_with_missing_months"], {"000300.SH/2021": 11})

            write_year("000300.SH", 2021, range(1, 13))
            severity, _, _, details = run_audit()
            self.assertEqual(severity, "info")

    def test_index_weight_audit_follows_a_backfill_below_the_floor(self):
        # The expected years start at the earliest backfilled year, per code
        # never before the index listed in the vendor catalog: CSI 1000 listed
        # 2014-10, STAR 50 2020-07, whose pre-launch years hold zero rows.
        def write_year(code, year, months):
            path = self.raw_dir / "index_weight" / f"index_code={code}" / f"year={year}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [{"index_code": code, "con_code": "000001.SZ", "trade_date": f"{year}{m:02d}28", "weight": 1.0} for m in months],
                columns=["index_code", "con_code", "trade_date", "weight"],
            ).to_parquet(path, index=False)

        def run_audit():
            findings = []
            audit.audit_index_weight(self.raw_dir, "20221231", lambda *item: findings.append(item))
            return findings[0]

        catalog = self.raw_dir / "index_basic" / "catalog.parquet"
        catalog.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"ts_code": ["000300.SH", "000852.SH", "000688.SH"], "list_date": ["20050408", "20141017", "20200723"]}
        ).to_parquet(catalog, index=False)
        with patch.object(audit, "DEFAULT_CN_INDEX_CODES", ["000300.SH", "000852.SH", "000688.SH"]):
            for year in range(2020, 2023):
                write_year("000300.SH", year, range(1, 13))
                write_year("000852.SH", year, range(1, 13))
            write_year("000688.SH", 2020, range(7, 13))
            for year in (2021, 2022):
                write_year("000688.SH", year, range(1, 13))
            severity, _, _, details = run_audit()
            self.assertEqual((severity, details["first_year"]), ("info", 2020))

            # Backfilled from 2014-07: the first year is short and exempt.
            write_year("000300.SH", 2014, range(7, 13))
            for year in range(2015, 2020):
                write_year("000300.SH", year, range(1, 13))
            write_year("000852.SH", 2014, range(10, 13))
            for year in range(2015, 2019):
                write_year("000852.SH", year, range(1, 13))
            for year in range(2014, 2020):
                write_year("000688.SH", year, ())
            severity, _, _, details = run_audit()
            self.assertEqual(severity, "error")  # 000852.SH year=2019 never landed
            self.assertEqual(details["first_year"], 2014)
            self.assertEqual(details["missing_year_partitions"], 1)
            self.assertIn("index_code=000852.SH/year=2019.parquet", details["missing_sample"][0])
            self.assertEqual(details["zero_row_year_partitions"], 0)

            write_year("000852.SH", 2019, range(1, 13))
            severity, _, _, details = run_audit()
            self.assertEqual(severity, "info")
            self.assertEqual(details["expected_files"], 9 + 9 + 3)

    def test_revision_sentinel_compares_without_overwriting_raw(self):
        self._write_trade_cal("20200102")
        path = self.raw_dir / "adj_factor" / "trade_date=20200102.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        original = pd.DataFrame([{"trade_date": "20200102", "ts_code": "999999.SZ", "adj_factor": 9.9}])
        original.to_parquet(path, index=False)
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["adj_factor"],
            sample_size=0,
            seed=None,
            page_limit=10000,
            revision_ledger=str(self.root / "sentinel_events.jsonl"),
            output=str(self.root / "sentinel_summary.json"),
            fail_on_revision=False,
            min_interval_seconds=0,
            timeout_seconds=1,
        )

        with patch.object(audit, "load_token", return_value="token"), patch.object(audit, "TuShareClient", return_value=FieldClient(DAILY_FIELDS, default=0)):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(audit.audit_revision_sentinel(args), 0)

        self.assertIn("REVISION_ALERT", output.getvalue())
        self.assertTrue((self.root / "sentinel_events.jsonl").exists())
        summary = json.loads((self.root / "sentinel_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "warning")
        self.assertEqual(summary["metadata"]["totals"]["revision_events"], 1)
        self.assertTrue(pd.read_parquet(path).equals(original))

    def test_weekly_sentinel_probes_event_datasets_and_flags_massive_revision(self):
        # The weekly deep audit reuses the sentinel over the flat event tier:
        # a vendor rewriting a large share of history (the fina_mainbz class)
        # must escalate to error, not drown as routine one-off warnings.
        cal_path = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2020.parquet"
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        days = ["20200102", "20200103", "20200106", "20200107", "20200108", "20200109"]
        pd.DataFrame({
            "exchange": ["SSE"] * 8,
            "cal_date": ["20200101"] + days + ["20200110"],
            "is_open": ["0"] + ["1"] * 6 + ["0"],
        }).to_parquet(cal_path, index=False)
        flow_dir = self.raw_dir / "moneyflow"
        flow_dir.mkdir(parents=True, exist_ok=True)
        for day in days:
            pd.DataFrame([{"trade_date": day, "ts_code": "000001.SZ", "net_mf_amount": 1.0}]).to_parquet(
                flow_dir / f"trade_date={day}.parquet", index=False
            )

        class RevisedFlowClient(EmptyClient):
            # A STRUCTURAL rewrite: the vendor replaced every stored key with a
            # different universe (value-only recalculation must stay warning).
            def query(self, api_name, params=None, fields="", retries=5):
                params = params or {}
                self.calls.append((api_name, dict(params)))
                columns = fields.split(",") if fields else ["trade_date", "ts_code", "net_mf_amount"]
                rows = [
                    [
                        params.get("trade_date", "") if col == "trade_date"
                        else f"{300000 + i:06d}.SZ" if col == "ts_code" else "2.0"
                        for col in columns
                    ]
                    for i in range(40)
                ]
                return tushare_client.ApiResult(columns, rows)

        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200101",
            end_date="20200110",
            datasets=["moneyflow"],
            sample_size=0,
            seed=None,
            page_limit=10000,
            revision_ledger=str(self.root / "sentinel_events.jsonl"),
            output=str(self.root / "sentinel_summary.json"),
            fail_on_revision=False,
            min_interval_seconds=0,
            timeout_seconds=1,
        )
        with patch.object(audit, "load_token", return_value="token"), patch.object(audit, "TuShareClient", return_value=RevisedFlowClient()):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(audit.audit_revision_sentinel(args), 1)
        summary = json.loads((self.root / "sentinel_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "error")
        finding = next(f for f in summary["findings"] if f["check"] == "moneyflow_revision_sentinel")
        self.assertEqual(finding["severity"], "error")
        self.assertTrue(finding["details"]["massive_revision"])
        self.assertEqual(finding["details"]["revision_events"], 6)
        # Nested-strategy datasets stay out: a probe would not be like-for-like.
        with self.assertRaises(RuntimeError):
            audit.revision_sentinel_spec("ths_hot")

    def test_top_list_name_is_a_revisable_value_not_part_of_the_key(self):
        # The vendor restates top_list.name to a later name on historical rows.
        # As a key field every such restatement read as one removed plus one
        # added key, which the sentinel escalated as a structural rewrite; as a
        # value it is a ledgered field revision on an unchanged key.
        spec = specs.BOARD_TRADING_SPECS["top_list"]
        self.assertNotIn("name", spec.key_columns)
        self.assertEqual(set(spec.key_columns), set(spec.fields.split(",")) - {"name"})
        keys = list(spec.key_columns)

        def rows(names: list[str]) -> pd.DataFrame:
            # One stock listed under two reasons on one day, plus 24 others.
            records = []
            for index, name in enumerate(names):
                records.append({
                    "trade_date": "20240611", "ts_code": f"{index:06d}.SZ", "name": name,
                    "close": 10.0 + index, "pct_change": 10.0, "turnover_rate": 5.0,
                    "amount": 1.0e8, "l_sell": 2.0e7, "l_buy": 3.0e7, "l_amount": 5.0e7,
                    "net_amount": 1.0e7, "net_rate": 10.0, "amount_rate": 50.0,
                    "float_values": 2.0e9, "reason": "日涨幅偏离值达到7%的前五只证券",
                })
            records.append({**records[0], "reason": "连续三个交易日内，涨幅偏离值累计达到20%的证券",
                            "l_buy": 4.0e7, "l_amount": 6.0e7, "net_amount": 2.0e7})
            return pd.DataFrame(records)

        old_df = rows([f"旧名{index}" for index in range(25)])
        new_df = rows([f"新名{index}" for index in range(25)])
        self.assertFalse(old_df.duplicated(keys).any())
        self.assertEqual(int(old_df.duplicated(["trade_date", "ts_code"]).sum()), 1)
        event = versions.revision_event(
            "probe", dataset="top_list", partition="trade_date=20240611", source="unit",
            key_columns=keys, diff=versions.diff_frames(old_df, new_df, keys), old_rows=len(old_df), new_rows=len(new_df),
        )
        self.assertEqual((event["changed_keys"], event["added_keys"], event["removed_keys"]), (26, 0, 0))
        self.assertEqual(event["changed_columns"], {"name": 26})
        self.assertFalse(audit.sentinel_event_is_structural(event))

    def test_business_payload_hollow_partition_is_an_error(self):
        d = self.raw_dir / "moneyflow"
        d.mkdir(parents=True, exist_ok=True)
        hollow = d / "trade_date=20260601.parquet"
        pd.DataFrame({
            "trade_date": ["20260601"], "ts_code": ["000001.SZ"],
            "buy_sm_vol": [None], "net_mf_amount": [None],
            "available_at": ["2026-06-02 09:00:00+08:00"], "available_at_rule": ["r"],
        }).to_parquet(hollow, index=False)

        findings: list[tuple[str, str, str, dict]] = []
        def add(sev, code, msg, details):
            findings.append((sev, code, msg, details))

        audit.audit_business_payload([hollow], "moneyflow", "moneyflow_payload", add, key_columns=("trade_date", "ts_code"))
        self.assertEqual(findings[0][0], "error")
        self.assertTrue(findings[0][3]["business_payload_empty"])

        # A healthy partition with one sparse column stays info.
        healthy = d / "trade_date=20260602.parquet"
        pd.DataFrame({
            "trade_date": ["20260602"], "ts_code": ["000001.SZ"],
            "buy_sm_vol": [1.0], "net_mf_amount": [None],
            "available_at": ["2026-06-03 09:00:00+08:00"], "available_at_rule": ["r"],
        }).to_parquet(healthy, index=False)
        findings.clear()
        audit.audit_business_payload([hollow, healthy], "moneyflow", "moneyflow_payload", add, key_columns=("trade_date", "ts_code"))
        self.assertEqual(findings[0][0], "info")  # newest non-empty partition wins

        # A requested source field that never arrived is a warning.
        findings.clear()
        audit.audit_business_payload(
            [healthy], "moneyflow", "moneyflow_payload", add,
            key_columns=("trade_date", "ts_code"), expected_fields=("buy_sm_vol", "sell_sm_vol"),
        )
        self.assertEqual(findings[0][0], "warning")
        self.assertEqual(findings[0][3]["missing_expected_fields"], ["sell_sm_vol"])

        # A file stripped down to keys is hollow, not merely field-incomplete:
        # every expected business field absent = error for spec'd families,
        # and require_business_columns covers fundamentals (empty spec fields).
        keys_only = d / "trade_date=20260603.parquet"
        pd.DataFrame({
            "trade_date": ["20260603"], "ts_code": ["000001.SZ"],
            "available_at": ["2026-06-04 09:00:00+08:00"], "available_at_rule": ["r"],
        }).to_parquet(keys_only, index=False)
        findings.clear()
        audit.audit_business_payload(
            [keys_only], "moneyflow", "moneyflow_payload", add,
            key_columns=("trade_date", "ts_code"), expected_fields=("trade_date", "ts_code", "buy_sm_vol", "sell_sm_vol"),
        )
        self.assertEqual(findings[0][0], "error")
        self.assertTrue(findings[0][3]["business_payload_empty"])
        findings.clear()
        audit.audit_business_payload(
            [keys_only], "income_vip", "income_vip_payload", add,
            key_columns=("trade_date", "ts_code"), require_business_columns=True,
        )
        self.assertEqual(findings[0][0], "error")
        # A genuinely key-only contract (suspend_d/top_list class) stays info.
        findings.clear()
        audit.audit_business_payload(
            [keys_only], "suspend_d", "suspend_d_payload", add,
            key_columns=("trade_date", "ts_code"), expected_fields=("trade_date", "ts_code"),
        )
        self.assertEqual(findings[0][0], "info")

    def test_full_market_coverage_flags_dropped_batch(self):
        codes = [f"{i:06d}.SZ" for i in range(100)]
        daily_dir = self.raw_dir / "daily"
        flow_dir = self.raw_dir / "moneyflow"
        daily_dir.mkdir(parents=True, exist_ok=True)
        flow_dir.mkdir(parents=True, exist_ok=True)
        for day, kept in (("20260601", 95), ("20260602", 80)):
            pd.DataFrame({"trade_date": [day] * 100, "ts_code": codes}).to_parquet(
                daily_dir / f"trade_date={day}.parquet", index=False
            )
            pd.DataFrame({"trade_date": [day] * kept, "ts_code": codes[:kept]}).to_parquet(
                flow_dir / f"trade_date={day}.parquet", index=False
            )

        findings: list[tuple[str, str, str, dict]] = []
        audit.audit_full_market_coverage(self.raw_dir, ["moneyflow", "block_trade"], lambda *f: findings.append(f))
        # block_trade is sparse by design and must not be checked at all.
        self.assertEqual([f[1] for f in findings], ["moneyflow_daily_coverage"])
        severity, _, _, details = findings[0]
        self.assertEqual(severity, "warning")
        self.assertEqual(details["days_checked"], 2)
        self.assertEqual(details["days_below_threshold"], 1)
        self.assertEqual(details["low_coverage_sample"][0]["trade_date"], "20260602")
        self.assertAlmostEqual(details["low_coverage_sample"][0]["coverage"], 0.8)

    def test_stock_basic_missing_list_status_slice_is_an_error(self):
        d = self.raw_dir / "stock_basic"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "ts_code": ["000001.SZ"], "symbol": ["000001"], "name": ["平安银行"],
            "list_status": ["L"], "list_date": ["19910403"],
        }).to_parquet(d / "list_status=L.parquet", index=False)
        findings: list[tuple[str, str, str, dict]] = []
        audit.audit_stock_basic(self.raw_dir, lambda *f: findings.append(f))
        severity, _, _, details = findings[-1]
        self.assertEqual(severity, "error")
        self.assertEqual(details["missing_list_status_slices"], ["D", "P"])

    def test_namechange_missing_code_batch_warns(self):
        d = self.raw_dir / "namechange"
        d.mkdir(parents=True, exist_ok=True)
        codes = [f"{i:06d}.SZ" for i in range(200)]
        pd.DataFrame({
            "ts_code": codes[:190], "name": ["x"] * 190,
            "start_date": ["20200101"] * 190, "end_date": [""] * 190,
            "ann_date": ["20200101"] * 190, "change_reason": ["上市"] * 190,
        }).to_parquet(d / "namechange.parquet", index=False)
        stock_basic = pd.DataFrame({"ts_code": codes})
        findings: list[tuple[str, str, str, dict]] = []
        audit.audit_namechange(self.raw_dir, stock_basic, lambda *f: findings.append(f))
        severity, _, _, details = findings[0]
        self.assertEqual(severity, "warning")
        self.assertEqual(details["stock_basic_codes_without_namechange"], 10)

    def test_expected_macro_paths_cover_new_strategies(self):
        import argparse

        from autotrade.data_sources.tushare import audit

        args = argparse.Namespace(datasets=None)
        static_paths = audit.expected_macro_paths(self.raw_dir, specs.MACRO_SPECS["fut_basic"], "20240101", "20240110", args)
        self.assertIn(self.raw_dir / "fut_basic" / "exchange=CFFEX.parquet", static_paths)
        self.assertEqual(len(static_paths), len(specs.MACRO_SPECS["fut_basic"].loop_values))
        self._write_trade_cal("20240104")
        # Loop venues that listed later must not be expected before their start.
        self._write_trade_cal("20180104")
        opt_paths = audit.expected_macro_paths(self.raw_dir, specs.MACRO_SPECS["opt_daily"], "20180104", "20180104", args)
        self.assertEqual({p.parent.name for p in opt_paths}, {"exchange=SSE"})

    def test_macro_completeness_month_expectation_clamps_to_open_dates(self):
        # The producing job ends on the last SSE open date; a weekend month
        # boundary must not expect the new month's partition yet, while a
        # missing month containing an elapsed trading day still errors.
        import argparse

        from autotrade.data_sources.tushare import audit

        cal_path = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"cal_date": "20260701", "is_open": "1"},
            {"cal_date": "20260731", "is_open": "1"},
            {"cal_date": "20260801", "is_open": "0"},
            {"cal_date": "20260802", "is_open": "0"},
            {"cal_date": "20260803", "is_open": "1"},
        ]).to_parquet(cal_path, index=False)
        month_path = self.raw_dir / "cn_schedule" / "month=202607.parquet"
        month_frame = pd.DataFrame([{
            "month": "202607", "publish_date": "20260715", "title": "t",
            "issuing_org": "o", "data_api": "a",
            "available_at": "2026-07-15 23:59:59+08:00",
            "available_at_rule": "conservative_date_eod",
        }])
        tushare_io.write_parquet(month_path, month_frame, api_name="cn_schedule", params={}, fields=list(month_frame.columns))

        def partition_finding(end_date):
            findings = []
            args = argparse.Namespace(datasets=["cn_schedule"], start_date="20260701", end_date=end_date)
            audit.audit_macro_completeness(
                self.raw_dir, args,
                lambda sev, check, msg, details=None: findings.append((sev, check, details or {})),
            )
            found = [f for f in findings if f[1] == "cn_schedule_macro_partitions"]
            self.assertEqual(len(found), 1)
            return found[0]

        sev, _, details = partition_finding("20260802")
        self.assertEqual(details.get("missing_expected_files"), 0)
        self.assertNotEqual(sev, "error")
        sev, _, details = partition_finding("20260803")
        self.assertEqual(details.get("missing_expected_files"), 1)
        self.assertEqual(sev, "error")

    def test_revision_sentinel_marks_source_failures_as_error(self):
        self._write_trade_cal("20200102")
        path = self.raw_dir / "daily" / "trade_date=20200102.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ"}]).to_parquet(path, index=False)
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["daily"],
            sample_size=0,
            seed=None,
            page_limit=10000,
            revision_ledger=str(self.root / "sentinel_error_events.jsonl"),
            output=str(self.root / "sentinel_error_summary.json"),
            fail_on_revision=False,
            min_interval_seconds=0,
            timeout_seconds=1,
        )

        with patch.object(audit, "load_token", return_value="token"), patch.object(audit, "TuShareClient", return_value=ErrorTradeDateClient()):
            self.assertEqual(audit.audit_revision_sentinel(args), 1)

        summary = json.loads((self.root / "sentinel_error_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "error")
        self.assertEqual(summary["metadata"]["totals"]["api_errors"], 1)

    def test_revision_sentinel_warns_on_missing_local_partition(self):
        self._write_trade_cal("20200102")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["adj_factor"],
            sample_size=0,
            seed=None,
            page_limit=10000,
            revision_ledger=str(self.root / "sentinel_missing_events.jsonl"),
            output=str(self.root / "sentinel_missing_summary.json"),
            fail_on_revision=False,
            min_interval_seconds=0,
            timeout_seconds=1,
        )

        with patch.object(audit, "load_token", return_value="token"), patch.object(audit, "TuShareClient", return_value=FieldClient(DAILY_FIELDS, default=0)):
            self.assertEqual(audit.audit_revision_sentinel(args), 0)

        summary = json.loads((self.root / "sentinel_missing_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "warning")
        self.assertEqual(summary["metadata"]["totals"]["missing_local_dates"], 1)

    def test_revision_sentinel_marks_required_remote_zero_as_error(self):
        self._write_trade_cal("20200102")
        path = self.raw_dir / "daily" / "trade_date=20200102.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ"}]).to_parquet(path, index=False)
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["daily"],
            sample_size=0,
            seed=None,
            page_limit=10000,
            revision_ledger=str(self.root / "sentinel_zero_events.jsonl"),
            output=str(self.root / "sentinel_zero_summary.json"),
            fail_on_revision=False,
            min_interval_seconds=0,
            timeout_seconds=1,
        )

        with patch.object(audit, "load_token", return_value="token"), patch.object(audit, "TuShareClient", return_value=EmptyClient()):
            self.assertEqual(audit.audit_revision_sentinel(args), 1)

        summary = json.loads((self.root / "sentinel_zero_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "error")
        self.assertEqual(summary["metadata"]["totals"]["remote_zero_dates"], 1)

    def test_revision_sentinel_default_ledger_for_temp_raw_stays_local(self):
        self._write_trade_cal("20200102")
        path = self.raw_dir / "daily" / "trade_date=20200102.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ", "open": 99.0}]).to_parquet(path, index=False)
        args = argparse.Namespace(
            raw_dir="raw",
            start_date="20200102",
            end_date="20200102",
            datasets=["daily"],
            sample_size=0,
            seed=None,
            page_limit=10000,
            revision_ledger=common.REVISION_EVENTS_PATH,
            output=str(self.root / "sentinel_default_ledger_summary.json"),
            fail_on_revision=False,
            min_interval_seconds=0,
            timeout_seconds=1,
        )

        with (
            patch.object(audit.Path, "cwd", return_value=self.root),
            patch.object(audit, "load_token", return_value="token"),
            patch.object(audit, "TuShareClient", return_value=FieldClient(DAILY_FIELDS, default=0)),
        ):
            self.assertEqual(audit.audit_revision_sentinel(args), 0)

        local_ledger = self.root / "revision_events.jsonl"
        formal_ledger = self.root / common.REVISION_EVENTS_PATH
        self.assertTrue(local_ledger.exists())
        self.assertFalse(formal_ledger.exists())
        event = json.loads(local_ledger.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(event["dataset"], "daily")
        self.assertEqual(event["changed_columns"]["open"], 1)

    def test_intraday_by_date_audit_errors_on_zero_row_partition(self):
        self._write_trade_cal()
        path = self.raw_dir / specs.STK_MINS_BY_DATE_DATASET / "trade_date=20200102.parquet"
        empty = pd.DataFrame(columns=specs.STK_MINS_REQUIRED_COLUMNS)
        tushare_io.write_parquet(path, empty, api_name=specs.STK_MINS_API_NAME, params={}, fields=list(empty.columns))
        status_path = self.root / "status.json"
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            output_dataset=specs.STK_MINS_BY_DATE_DATASET,
            codes=None,
            max_codes=None,
            expected_codes_source="none",
            min_rows_per_day=0,
            allow_missing_codes=0,
            full_scan=False,
            sample_limit=0,
            output=str(status_path),
        )

        self.assertEqual(audit.audit_intraday_by_date(args), 1)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        inventory = next(item for item in status["findings"] if item["check"] == f"{specs.STK_MINS_BY_DATE_DATASET}_inventory")
        self.assertEqual(inventory["severity"], "error")
        self.assertEqual(inventory["details"]["zero_row_files"], 1)

    def _write_by_date_minutes(self, trade_date: str, *, duplicate_first_bar: bool = False) -> None:
        rows = []
        for bar_time in ("09:30", "15:00"):
            rows.append({
                "ts_code": "000001.SZ",
                "trade_time": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} {bar_time}:00",
                "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "vol": 100.0, "amount": 100.0,
                "trade_date": trade_date,
                "available_at": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}T{bar_time}:00+08:00",
                "available_at_rule": "bar_close",
            })
        if duplicate_first_bar:
            rows.append(dict(rows[0]))
        path = self.raw_dir / specs.STK_MINS_BY_DATE_DATASET / f"trade_date={trade_date}.parquet"
        df = pd.DataFrame(rows)
        tushare_io.write_parquet(path, df, api_name=specs.STK_MINS_API_NAME, params={}, fields=list(df.columns))

    def test_intraday_by_date_deep_checks_cover_the_newest_dates(self):
        # The head slice re-validated the window's FIRST trading days forever;
        # deep checks must target what the pipeline just wrote.
        cal = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2020.parquet"
        cal.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"cal_date": "20200102", "is_open": "1"},
            {"cal_date": "20200103", "is_open": "1"},
        ]).to_parquet(cal, index=False)
        self._write_by_date_minutes("20200102")
        self._write_by_date_minutes("20200103", duplicate_first_bar=True)
        status_path = self.root / "status.json"
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200103",
            output_dataset=specs.STK_MINS_BY_DATE_DATASET,
            codes=None,
            max_codes=None,
            expected_codes_source="none",
            min_rows_per_day=0,
            allow_missing_codes=0,
            full_scan=False,
            sample_limit=1,
            output=str(status_path),
        )

        self.assertEqual(audit.audit_intraday_by_date(args), 0)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        deep = next(item for item in status["findings"] if item["check"] == f"{specs.STK_MINS_BY_DATE_DATASET}_deep_checks")
        self.assertEqual(deep["details"]["files_checked"], 1)
        self.assertEqual(deep["details"]["bad_days"], 1)
        self.assertEqual(deep["details"]["bad_day_sample"][0]["trade_date"], "20200103")

    def test_intraday_by_date_audit_warns_on_a_stale_tail_and_errors_on_a_hole(self):
        # The minute layer lands only through a manual job: days after its
        # newest partition are staleness, a day missing before it is a defect.
        cal = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2020.parquet"
        cal.parent.mkdir(parents=True, exist_ok=True)
        days = ["20200102", "20200103", "20200106", "20200107"]
        pd.DataFrame([{"cal_date": day, "is_open": "1"} for day in days]).to_parquet(cal, index=False)
        status_path = self.root / "status.json"
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date=days[0],
            end_date=days[-1],
            output_dataset=specs.STK_MINS_BY_DATE_DATASET,
            codes=None,
            max_codes=None,
            expected_codes_source="none",
            min_rows_per_day=0,
            allow_missing_codes=0,
            full_scan=False,
            sample_limit=0,
            output=str(status_path),
        )

        def findings() -> dict[str, dict]:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            prefix = specs.STK_MINS_BY_DATE_DATASET
            return {item["check"].removeprefix(f"{prefix}_"): item for item in status["findings"]}

        (self.raw_dir / specs.STK_MINS_BY_DATE_DATASET).mkdir(parents=True)
        self.assertEqual(audit.audit_intraday_by_date(args), 1)
        self.assertEqual(findings()["inventory"]["details"]["missing_files"], 4)
        self.assertEqual(findings()["staleness"]["severity"], "info")

        self._write_by_date_minutes("20200102")
        self._write_by_date_minutes("20200103")
        self.assertEqual(audit.audit_intraday_by_date(args), 0)
        found = findings()
        self.assertEqual(found["inventory"]["severity"], "info")
        self.assertEqual(found["staleness"]["severity"], "warning")
        self.assertEqual(
            found["staleness"]["details"],
            {"newest_trade_date": "20200103", "stale_trade_dates": 2, "stale_first": "20200106", "stale_last": "20200107"},
        )

        self._write_by_date_minutes("20200107")
        self.assertEqual(audit.audit_intraday_by_date(args), 1)
        found = findings()
        self.assertEqual(found["inventory"]["severity"], "error")
        self.assertEqual(found["inventory"]["details"]["missing_files"], 1)
        self.assertEqual(found["staleness"]["severity"], "info")

    def test_design_fact_checks_report_as_info(self):
        # Unconditional design-fact notes must not occupy the warning severity:
        # a healthy day should be able to read status=ok (round 18).
        daily = self.raw_dir / "daily"
        daily.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"ts_code": "000001.SZ"}]).to_parquet(daily / "trade_date=20200102.parquet", index=False)
        findings = []
        def add(severity, check, message, details=None):
            findings.append({"severity": severity, "check": check})

        audit.audit_pit_availability(self.raw_dir, add)
        audit.audit_fundamental_unit_and_pit_semantics(self.raw_dir, add)

        by_check = {item["check"]: item["severity"] for item in findings}
        self.assertEqual(by_check["pit_available_at"], "info")
        self.assertEqual(by_check["fundamental_unit_and_pit_semantics"], "info")

    def test_latest_parquet_schema_reads_the_newest_partition(self):
        dataset_dir = self.raw_dir / "daily"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"old_only": 1}]).to_parquet(dataset_dir / "trade_date=20200102.parquet", index=False)
        pd.DataFrame([{"old_only": 1, "added_later": 2}]).to_parquet(dataset_dir / "trade_date=20260101.parquet", index=False)

        self.assertEqual(audit.latest_parquet_schema(self.raw_dir, "daily"), ["old_only", "added_later"])

    def test_event_expectations_clamp_to_last_sse_open_date(self):
        # A weekend calendar end date must not expect the new month's event
        # partition before the next producing run (the month-boundary weekend
        # false-error class; macro/fundamental got the same clamp earlier).
        # Text is deliberately NOT clamped -- see the natural-day test below.
        cal = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        cal.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"cal_date": "20260731", "is_open": "1"},
            {"cal_date": "20260801", "is_open": "0"},
            {"cal_date": "20260802", "is_open": "0"},
        ]).to_parquet(cal, index=False)
        holders = pd.DataFrame([{
            "ts_code": "000001.SZ", "ann_date": "20260710", "end_date": "20260630",
            "holder_num": 10000, "available_at": "2026-07-10T23:59:59+08:00", "available_at_rule": "ann_date_eod",
        }])
        tushare_io.write_parquet(
            self.raw_dir / "stk_holdernumber" / "month=202607.parquet",
            holders, api_name="stk_holdernumber", params={}, fields=list(holders.columns),
        )
        status_path = self.root / "event_status.json"
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20260731",
            end_date="20260801",
            datasets=["stk_holdernumber"],
            output=str(status_path),
        )
        with patch.object(audit.Path, "cwd", return_value=self.root):
            audit.audit_event_flow_only(args)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        scope_finding = next(f for f in status["findings"] if f["check"] == "event_flow_expected_scope")
        self.assertEqual(scope_finding["details"]["covered_end_date"], "20260731")
        partitions = next(f for f in status["findings"] if f["check"] == "stk_holdernumber_event_partitions")
        self.assertEqual(partitions["details"]["missing_expected_files"], 0)

    def test_margin_is_expected_only_once_published_on_the_next_open_day(self):
        # Margin for T publishes at 09:00 on the next OPEN day: the 02:30 audit
        # after a weekend or holiday must not expect it yet, an older missing
        # day is still an error, and so is T itself once its morning has passed.
        cal = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        cal.parent.mkdir(parents=True, exist_ok=True)
        open_days = {"20260924", "20260928", "20260929", "20260930", "20261008", "20261009"}
        days = [f"202609{d:02d}" for d in range(24, 31)] + [f"202610{d:02d}" for d in range(1, 10)]
        pd.DataFrame([{"cal_date": d, "is_open": "1" if d in open_days else "0"} for d in days]).to_parquet(cal, index=False)
        spec = specs.EVENT_FLOW_SPECS["margin"]
        trading_dates = sorted(open_days)

        def land(trade_date: str) -> None:
            rows = pd.DataFrame([
                {col: (trade_date if col == "trade_date" else exchange if col == "exchange_id" else 1.0) for col in spec.fields.split(",")}
                for exchange in ("SSE", "SZSE", "BSE")
            ])
            df = stamps.augment_event_frame(rows, spec, trading_dates=trading_dates)
            tushare_io.write_parquet(
                self.raw_dir / "margin" / f"trade_date={trade_date}.parquet",
                df, api_name="margin", params={}, fields=list(df.columns),
            )

        def partitions(end_date: str, as_of: str) -> dict:
            status_path = self.root / "margin_status.json"
            args = argparse.Namespace(
                raw_dir=str(self.raw_dir), start_date="20260924", end_date=end_date,
                datasets=["margin"], as_of=as_of, output=str(status_path),
            )
            with patch.object(audit.Path, "cwd", return_value=self.root):
                audit.audit_event_flow_only(args)
            findings = json.loads(status_path.read_text(encoding="utf-8"))["findings"]
            return next(f for f in findings if f["check"] == "margin_event_partitions")

        # An empty folder stays an error even when nothing is published yet.
        self.assertEqual(partitions("20260927", "2026-09-27T02:30:00")["severity"], "error")

        # Holiday stretch: 0924's margin publishes 0928 09:00, not on the holiday 0925.
        land("20260928")
        before = partitions("20260927", "2026-09-28T02:30:00")
        self.assertEqual(before["details"]["expected_files"], 0)
        self.assertNotEqual(before["severity"], "error")
        after = partitions("20260927", "2026-09-28T09:20:00")
        self.assertEqual(after["severity"], "error")
        self.assertEqual(after["details"]["missing_sample"], [str((self.raw_dir / "margin" / "trade_date=20260924.parquet").resolve())])

        # Day T (0930) not yet published at the 02:30 audit after the holiday: pass.
        land("20260924")
        land("20260929")
        passing = partitions("20261002", "2026-10-03T02:30:00")
        self.assertEqual(passing["details"]["missing_expected_files"], 0)
        self.assertNotEqual(passing["severity"], "error")
        # ... and an error once its publication morning has passed.
        self.assertEqual(partitions("20261008", "2026-10-08T09:20:00")["severity"], "error")

        # Day T-1 (0929) missing after its publication time: error even while T is unpublished.
        (self.raw_dir / "margin" / "trade_date=20260929.parquet").unlink()
        (self.raw_dir / "margin" / "trade_date=20260929.parquet.meta.json").unlink()
        gap = partitions("20261002", "2026-10-03T02:30:00")
        self.assertEqual(gap["severity"], "error")
        self.assertEqual(gap["details"]["missing_expected_files"], 1)

    def _audit_text_window(self, start_date: str, end_date: str, output_name: str) -> dict:
        status_path = self.root / output_name
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            text_start_date=start_date,
            text_end_date=end_date,
            text_datasets=["irm_qa_sh"],
            news_src=[],
            major_news_src=[],
            output=str(status_path),
        )
        with patch.object(audit.Path, "cwd", return_value=self.root):
            audit.audit_text_only(args)
        return json.loads(status_path.read_text(encoding="utf-8"))

    def test_text_expectations_follow_natural_days_across_a_weekend(self):
        # Text publishes on weekends, and its producing job runs every calendar
        # evening, so a Saturday end date must expect Saturday's partition.
        # Clamping to the last SSE open day left weekend text unchecked until
        # the next session (data docs §4).
        cal = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        cal.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"cal_date": "20260731", "is_open": "1"},
            {"cal_date": "20260801", "is_open": "0"},
        ]).to_parquet(cal, index=False)
        qa = pd.DataFrame([{
            "trade_date": "20260731", "ts_code": "000001.SZ", "q": "问题", "a": "回答",
            "pub_time": "2026-07-31 18:00:00",
        }])
        tushare_io.write_parquet(
            self.raw_dir / "irm_qa_sh" / "date=20260731.parquet",
            qa, api_name="irm_qa_sh", params={}, fields=list(qa.columns),
        )

        missing = self._audit_text_window("20260731", "20260801", "text_status_missing.json")
        scope = next(f for f in missing["findings"] if f["check"] == "text_expected_scope")
        self.assertEqual(scope["details"]["end_date"], "20260801")
        self.assertNotIn("covered_end_date", scope["details"])
        partitions = next(f for f in missing["findings"] if f["check"] == "irm_qa_sh_text_partitions")
        self.assertEqual(partitions["details"]["missing_expected_files"], 1)
        self.assertEqual(missing["status"], "error")

        # The same window is clean once the calendar-day job has landed Saturday
        # (an empty vendor response still writes the partition).
        tushare_io.write_parquet(
            self.raw_dir / "irm_qa_sh" / "date=20260801.parquet",
            qa.iloc[0:0], api_name="irm_qa_sh", params={}, fields=list(qa.columns),
        )
        landed = self._audit_text_window("20260731", "20260801", "text_status_landed.json")
        partitions = next(f for f in landed["findings"] if f["check"] == "irm_qa_sh_text_partitions")
        self.assertEqual(partitions["details"]["missing_expected_files"], 0)
        self.assertNotIn(
            "irm_qa_sh_text_partitions",
            [f["check"] for f in landed["findings"] if f["severity"] == "error"],
        )


class AuditSubcommandExecutionTest(unittest.TestCase):
    """Every formal audit subcommand must EXECUTE, not just render as a cron
    command string: the 2026-07-25 selector-consolidation merge dropped two
    imports and the macro/text audits crashed with NameError for five nights
    while their status files stayed frozen at a pre-outage state."""

    def _run_audit(self, argv: list[str]) -> int:
        with patch.object(sys, "argv", ["tushare_audit.py", *argv]), redirect_stdout(io.StringIO()):
            return audit.main()

    def test_text_subcommand_executes_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            raw_dir.mkdir()
            output = Path(tmp) / "text_status.json"
            code = self._run_audit([
                "text", "--raw-dir", str(raw_dir), "--output", str(output),
                "--start-date", "20240101", "--end-date", "20240102",
                "--text-datasets", "cctv_news",
            ])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["report_type"], "text_evidence")
            # An empty lake is a completed audit with error findings, never a crash.
            self.assertEqual(report["status"], "error")
            self.assertEqual(code, 1)

    def test_macro_subcommand_executes_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            raw_dir.mkdir()
            output = Path(tmp) / "macro_status.json"
            code = self._run_audit([
                "macro", "--raw-dir", str(raw_dir), "--output", str(output),
                "--start-date", "20240101", "--end-date", "20240102",
                "--datasets", "cn_gdp",
            ])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["report_type"], "macro_context")
            self.assertEqual(report["status"], "error")
            self.assertEqual(code, 1)


class FundamentalAuditTest(unittest.TestCase):
    """``disclosure_date``'s whole contract IS its key set, so the business
    payload gate must exempt it instead of erroring on correct data."""

    @staticmethod
    def _run_dataset(raw: Path, dataset: str, partition: str) -> list[dict]:
        findings: list[dict] = []

        def add(severity, check, message, details=None):
            findings.append(
                {
                    "severity": severity,
                    "check": check,
                    "message": message,
                    "details": details or {},
                }
            )

        audit.audit_fundamental_dataset(
            raw,
            specs.FUNDAMENTAL_SPECS[dataset],
            {partition},
            "period" if dataset == "disclosure_date" else "ts_code",
            add,
        )
        return findings

    def test_disclosure_date_exact_key_schema_is_complete_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            path = raw / "disclosure_date" / "period=20240630.parquet"
            path.parent.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "end_date": "20240630",
                        "ann_date": "20240401",
                        "pre_date": "20240420",
                        "actual_date": "20240419",
                    }
                ]
            ).to_parquet(path, index=False)
            findings = self._run_dataset(raw, "disclosure_date", "20240630")

        payload = next(item for item in findings if item["check"] == "disclosure_date_payload")
        self.assertEqual(payload["severity"], "info")
        self.assertFalse(payload["details"]["business_payload_empty"])

    def test_other_fundamental_key_only_schema_remains_hollow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            path = raw / "fina_audit" / "ts_code=000001.SZ.parquet"
            path.parent.mkdir(parents=True)
            pd.DataFrame(
                [{"ts_code": "000001.SZ", "ann_date": "20240401", "end_date": "20231231"}]
            ).to_parquet(path, index=False)
            findings = self._run_dataset(raw, "fina_audit", "000001.SZ")

        payload = next(item for item in findings if item["check"] == "fina_audit_payload")
        self.assertEqual(payload["severity"], "error")
        self.assertTrue(payload["details"]["business_payload_empty"])

    def test_disclosure_date_missing_key_column_remains_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            path = raw / "disclosure_date" / "period=20240630.parquet"
            path.parent.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "end_date": "20240630",
                        "ann_date": "20240401",
                        "pre_date": "20240420",
                    }
                ]
            ).to_parquet(path, index=False)
            findings = self._run_dataset(raw, "disclosure_date", "20240630")

        keys = next(item for item in findings if item["check"] == "disclosure_date_keys")
        self.assertEqual(keys["severity"], "error")
        self.assertEqual(keys["details"]["missing_key_column_files"], 1)


class SnapshotUnitCoverageAuditTest(unittest.TestCase):
    """The evening audit, not the next morning's Paper run, is where a column
    the vendor added is supposed to surface."""

    @staticmethod
    def _inventory(file: str) -> dict[str, list[str]]:
        root = Path(__file__).resolve().parents[2]
        return json.loads(
            (root / "configs/data/snapshot_columns.json").read_text(encoding="utf-8")
        )["files"][file]

    def _run(self, raw: Path, file: str) -> tuple[int, dict]:
        findings: list[dict] = []

        def add(severity, check, message, details=None):
            findings.append({"severity": severity, "check": check, "message": message, "details": details or {}})

        count = audit.audit_snapshot_unit_coverage(raw, file, add)
        self.assertEqual(len(findings), 1)
        return count, findings[0]

    def test_current_vendor_columns_resolve_and_an_added_one_is_an_error(self):
        inventory = self._inventory("fundamentals.parquet")
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            for dataset, columns in inventory.items():
                directory = raw / dataset
                directory.mkdir(parents=True)
                pd.DataFrame([{column: "" for column in columns}]).to_parquet(
                    directory / "ts_code=000001.SZ.parquet", index=False
                )

            # The real committed column lists of every selectable fundamentals
            # dataset must classify: this is the condition the snapshot build
            # enforces, checked here from schemas alone.
            count, finding = self._run(raw, "fundamentals.parquet")
            self.assertEqual(count, 0)
            self.assertEqual(finding["severity"], "info")
            self.assertEqual(finding["check"], "snapshot_unit_coverage")
            self.assertEqual(finding["details"]["datasets"], 10)
            self.assertGreater(finding["details"]["columns_checked"], 300)
            self.assertEqual(finding["details"]["unresolved_columns"], [])
            # base_share is the 2026-09-21 vendor addition; it must resolve now.
            self.assertIn("base_share", inventory["dividend"])

            # Negative path: the next unregistered vendor column is an error
            # naming itself, the evening its partition lands.
            pd.DataFrame(
                [{column: "" for column in [*inventory["dividend"], "base_ratio"]}]
            ).to_parquet(raw / "dividend" / "ts_code=000002.SZ.parquet", index=False)
            count, finding = self._run(raw, "fundamentals.parquet")
            self.assertEqual(count, 1)
            self.assertEqual(finding["severity"], "error")
            self.assertIn("fundamentals.parquet:dividend:base_ratio", finding["message"])
            self.assertEqual(
                finding["details"]["unresolved_columns"],
                ["fundamentals.parquet:dividend:base_ratio"],
            )

    def test_summary_line_carries_the_count_the_job_state_reads_back(self):
        # The cron runner keeps the head and the integer fields of the summary
        # line, so the failing job state has to distinguish this from the
        # standing partition errors of the same audit.
        line = audit.audit_summary_line(
            "fundamental_raw", "error", errors=1, warnings=0, unresolved_columns=1, output="x.json"
        )
        self.assertEqual(
            audit.AUDIT_SUMMARY_RE.findall(line),
            ["fundamental_raw audit status=error errors=1 warnings=0 unresolved_columns=1"],
        )


if __name__ == "__main__":
    unittest.main()
