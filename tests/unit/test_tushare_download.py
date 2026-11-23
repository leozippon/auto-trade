"""The tier downloads, the landing loop and its skip rules, the update steps and
the download CLI (data_sources/tushare/download.py).
"""

import argparse
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import chdir, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, patch

import pandas as pd
import pytest

from autotrade.data_sources.tushare import (
    audit,
    common,
    download,
    specs,
    stamps,
    versions,
)
from autotrade.data_sources.tushare import client as tushare_client
from autotrade.data_sources.tushare import io as tushare_io
from autotrade.data_sources.tushare import lock as tushare_lock

from .support.tushare_fixture import (
    DAILY_FIELDS,
    EmptyClient,
    FieldClient,
    LakeTestCase,
    param,
    write_trade_cal,
)

# A macro row answering its request's month and end date.
MACRO_FIELDS = {
    "month": param("m", "202605"),
    "date": param("end_date", "20260529"),
    "trade_date": param("end_date", "20260529"),
    "publish_date": "20260529",
    "title": "sample",
    "issuing_org": "sample_org",
    "data_api": lambda api_name, params: api_name,
}

# A board row answering its request's trade date and dimension values.
BOARD_FIELDS = {
    "trade_date": param("trade_date", "20200102"),
    "ts_code": "000001.SZ",
    "name": "sample",
    "ts_name": "sample",
    "tag": param("tag", "涨停"),
    "limit_type": param("limit_type", "涨停池"),
    "data_type": param("market", "热股"),
    "rank_time": "2020-01-02 10:00:00",
    "hm_name": "sample_hot_money",
    "hm_orgs": "sample_org",
    "exalter": "sample_broker",
    "side": "0",
    "reason": "sample_reason",
    "nums": "2",
    "rank": 1,
    "desc": "sample_desc",
    "orgs": "[]",
}


class NoQueryClient:
    def query(self, api_name, params=None, fields="", retries=5):
        raise AssertionError(f"unexpected TuShare query: {api_name}")


class ReferenceClient:
    def __init__(self):
        self.calls = []

    def query(self, api_name, params=None, fields="", retries=5):
        params = params or {}
        self.calls.append((api_name, dict(params)))
        result_fields = fields.split(",") if fields else []
        rows = []
        if api_name == "stock_basic" and params.get("list_status") == "L":
            rows = [["000001.SZ" if field == "ts_code" else params.get("list_status", "") if field == "list_status" else "" for field in result_fields]]
        elif api_name == "namechange":
            rows = [[params.get("ts_code", ""), "sample", "20200101", "", "20200101", "name"]]
        elif api_name == "index_classify":
            rows = [["801010.SI" if field == "index_code" else "L1" if field == "level" else "sample" for field in result_fields]]
        elif api_name == "index_member_all":
            rows = [["801010.SI" if field == "l1_code" else "000001.SZ" if field == "ts_code" else "" for field in result_fields]]
        elif api_name == "index_member":
            rows = [[params.get("index_code", "") if field == "index_code" else "000001.SZ" if field == "con_code" else "20111010" if field == "in_date" else "N" if field == "is_new" else "" for field in result_fields]]
        elif api_name == "index_basic":
            rows = [["000300.SH" if field == "ts_code" else "sample" for field in result_fields]]
        elif api_name == "index_weight":
            rows = [[params.get("index_code", "") if field == "index_code" else "000001.SZ" if field == "con_code" else "20260630" if field == "trade_date" else "1.0" for field in result_fields]]
        return tushare_client.ApiResult(result_fields, rows)


class FundamentalClient:
    def __init__(self):
        self.calls = []

    def query(self, api_name, params=None, fields="", retries=5):
        params = params or {}
        self.calls.append((api_name, dict(params)))
        if api_name == "income_vip":
            result_fields = ["ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"]
            row = ["000001.SZ", "20200430", "20200430", params.get("period", "20200331"), "1", "1", "1"]
        elif api_name in {"dividend", "fina_audit", "fina_mainbz_vip"}:
            result_fields = ["ts_code", "ann_date", "end_date"]
            row = [params.get("ts_code", "000001.SZ"), "20200430", "20200331"]
        else:
            result_fields = ["ts_code"]
            row = ["000001.SZ"]
        return tushare_client.ApiResult(result_fields, [row])


class CalendarEventClient:
    def __init__(self):
        self.calls = []

    def query(self, api_name, params=None, fields="", retries=5):
        params = params or {}
        self.calls.append((api_name, dict(params)))
        result_fields = fields.split(",") if fields else []
        if api_name == "trade_cal":
            row = []
            for field in result_fields:
                if field == "exchange":
                    row.append(params.get("exchange", "SSE"))
                elif field == "cal_date":
                    row.append(params.get("end_date", "20260604"))
                elif field == "is_open":
                    row.append("1")
                elif field == "pretrade_date":
                    row.append("20260603")
                else:
                    row.append("")
            return tushare_client.ApiResult(result_fields, [row])
        if api_name == "margin_secs":
            rows = []
            for ts_code, exchange in (("600000.SH", "SSE"), ("000001.SZ", "SZSE"), ("830000.BJ", "BSE")):
                row = []
                for field in result_fields:
                    if field == "trade_date":
                        row.append(params.get("trade_date", "20260604"))
                    elif field == "ts_code":
                        row.append(ts_code)
                    elif field == "exchange":
                        row.append(exchange)
                    elif field == "name":
                        row.append("sample")
                    else:
                        row.append("")
                rows.append(row)
            return tushare_client.ApiResult(result_fields, rows)
        return tushare_client.ApiResult(result_fields, [])


class TieUnstablePagingClient:
    """stk_holdernumber's pager: rows sorted on a non-unique key whose tie
    order depends on the offset, so consecutive pages overlap and skip."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def query(self, api_name, params=None, fields="", retries=5):
        params = params or {}
        self.calls.append(dict(params))
        names = fields.split(",")
        served = [row for row in self.rows if params["start_date"] <= row["ann_date"] <= params["end_date"]]
        served.sort(key=lambda row: row["ts_code"], reverse=bool(params["offset"]))
        page = served[params["offset"]:params["offset"] + params["limit"]]
        return tushare_client.ApiResult(names, [[row.get(name, "") for name in names] for row in page])


class DownloadTest(LakeTestCase):
    def test_event_flow_trade_date_download_skips_non_trading_day(self):
        self._write_trade_cal_with_stamping_window("20260530", is_open="0")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20260530",
            end_date="20260530",
            datasets=["margin", "margin_detail", "margin_secs"],
            force=False,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=NoQueryClient()):
            self.assertEqual(download.download_event_flow(args), 0)

    def test_range_partition_skip_requires_sidecar_coverage(self):
        path = self.raw_dir / "anns_d" / "month=202605.parquet"
        existing = pd.DataFrame([{"ann_date": "20260528", "title": "old"}])
        tushare_io.write_parquet(
            path,
            existing,
            api_name="anns_d",
            params={"start_date": "20260501", "end_date": "20260528"},
            fields=list(existing.columns),
        )

        self.assertTrue(download.should_skip_existing_partition(
            path,
            force=False,
            requested_params={"start_date": "20260501", "end_date": "20260528"},
        ))
        self.assertFalse(download.should_skip_existing_partition(
            path,
            force=False,
            requested_params={"start_date": "20260501", "end_date": "20260529"},
        ))
        sidecar = path.with_suffix(path.suffix + ".meta.json")
        sidecar.unlink()
        self.assertFalse(download.should_skip_existing_partition(
            path,
            force=False,
            requested_params={"start_date": "20260501", "end_date": "20260528"},
        ))

    def test_sidecar_coverage_normalizes_date_and_datetime_bounds(self):
        path = self.raw_dir / "major_news" / "src=all" / "month=202605.parquet"
        existing = pd.DataFrame([{"pub_time": "2026-05-29 12:00:00", "title": "old"}])
        tushare_io.write_parquet(
            path,
            existing,
            api_name="major_news",
            params={"start_date": "2026-05-29 00:00:00", "end_date": "2026-05-29 23:59:59"},
            fields=list(existing.columns),
        )
        self.assertTrue(download.should_skip_existing_partition(
            path,
            force=False,
            requested_params={"start_date": "20260529000000", "end_date": "20260529235959"},
        ))

        tushare_io.write_parquet(
            path,
            existing,
            api_name="major_news",
            params={"start_date": "20260529000000", "end_date": "20260529000000"},
            fields=list(existing.columns),
        )
        self.assertFalse(download.should_skip_existing_partition(
            path,
            force=False,
            requested_params={"start_date": "20260529", "end_date": "20260529"},
        ))

        tushare_io.write_parquet(
            path,
            existing,
            api_name="major_news",
            params={"start_date": "20260529", "end_date": "20260529"},
            fields=list(existing.columns),
        )
        self.assertTrue(download.should_skip_existing_partition(
            path,
            force=False,
            requested_params={"start_date": "2026-05-29 00:00:00", "end_date": "2026-05-29 23:59:59"},
        ))

    def test_macro_month_loop_backfills_months_without_recorded_coverage_once(self):
        # A closed month whose sidecar records no coverage window must be
        # re-requested (its recorded pull may have stopped mid-month), then
        # skip forever once coverage is stamped. Skipping on bare file
        # existence would freeze such a partition permanently. The partitions
        # hold what the vendor answers again, so the content is unchanged and
        # only the sidecar records the coverage.
        spec = specs.MACRO_SPECS["cn_schedule"]
        for month in ("202604", "202605"):
            path = self.raw_dir / "cn_schedule" / f"month={month}.parquet"
            frame = stamps.augment_macro_frame(
                tushare_client.frame(FieldClient(MACRO_FIELDS).query("cn_schedule", {"m": month}, spec.fields)), spec
            )
            tushare_io.write_parquet(
                path,
                frame,
                api_name="cn_schedule",
                params={"m": month},
                fields=list(frame.columns),
            )

        client = FieldClient(MACRO_FIELDS)
        download.download_macro_month_loop(
            client,
            self.raw_dir,
            specs.MACRO_SPECS["cn_schedule"],
            "20260401",
            "20260529",
            False,
        )

        self.assertEqual([params["m"] for _, params in client.calls], ["202604", "202605"])
        for month, start, end in (("202604", "20260401", "20260430"), ("202605", "20260501", "20260529")):
            meta = json.loads((self.raw_dir / "cn_schedule" / f"month={month}.parquet.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["params"]["start_date"], start)
            self.assertEqual(meta["params"]["end_date"], end)

        # Second run: recorded coverage now satisfies both windows.
        client.calls.clear()
        download.download_macro_month_loop(
            client, self.raw_dir, specs.MACRO_SPECS["cn_schedule"], "20260401", "20260529", False,
        )
        self.assertEqual(client.calls, [])

    def test_uppercase_vendor_macro_schema_lands_as_the_declared_contract(self):
        # cn_pmi has answered its lowercase field request with the vendor's own
        # database schema since 2026-08-13 (uppercase columns plus ID/CREATE_*/
        # UPDATE_* row bookkeeping), flapping back and forth. Whatever casing
        # arrives, the partition must hold the declared contract: otherwise
        # `month` is absent, every row keeps the empty "missing_source_date"
        # stamp, and the whole table drops out of PIT replay.
        spec = specs.MACRO_SPECS["cn_pmi"]
        declared = spec.fields.split(",")
        series = declared[1:]

        class UppercaseVendorClient:
            def query(self, api_name, params=None, fields="", retries=5):
                columns = ["ID", "MONTH", "CREATE_TIME", "CREATE_BY", "UPDATE_TIME", "UPDATE_BY",
                           "PMI010100", *(name.upper() for name in series)]
                row = [553, "202607", "2026-09-03 12:39:39", None, "2026-09-03 12:39:39", None,
                       49.1, *(50.0 + index for index in range(len(series)))]
                return tushare_client.ApiResult(columns, [row])

        with redirect_stdout(io.StringIO()):
            download.download_macro_month_once(
                UppercaseVendorClient(), self.raw_dir, spec, "20260701", "20260731", False,
            )

        written = pd.read_parquet(self.raw_dir / "cn_pmi" / "range=202607_latest.parquet")
        self.assertEqual(list(written.columns), [*declared, "available_at", "available_at_rule"])
        self.assertEqual(written["month"].tolist(), ["202607"])
        self.assertEqual(
            written[series].iloc[0].tolist(), [50.0 + index for index in range(len(series))]
        )
        # Stamped like every other monthly macro release (cn_cpi 202607).
        self.assertEqual(written["available_at"].tolist(), ["2026-08-31 23:59:59+08:00"])
        self.assertEqual(written["available_at_rule"].tolist(), ["conservative_month_end_plus_31d"])

    def test_macro_response_without_the_date_column_is_refused(self):
        # Nothing downstream can repair an unstamped macro row, so a response
        # that carries no date column at all must fail the write, not land.
        class NoMonthClient:
            def query(self, api_name, params=None, fields="", retries=5):
                return tushare_client.ApiResult(["ID", "PMI010000"], [[553, 49.8]])

        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "returned no 'month' column"):
            download.download_macro_month_once(
                NoMonthClient(), self.raw_dir, specs.MACRO_SPECS["cn_pmi"],
                "20260701", "20260731", False,
            )
        self.assertFalse((self.raw_dir / "cn_pmi").exists())

    def test_window_merged_partition_preserves_rows_outside_refresh_window(self):
        path = self.raw_dir / "repurchase" / "month=202605.parquet"
        existing = pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20260501", "amount": 1},
            {"ts_code": "000002.SZ", "ann_date": "20260515", "amount": 2},
        ])
        tushare_io.write_parquet(
            path,
            existing,
            api_name="repurchase",
            params={"start_date": "20260501", "end_date": "20260531"},
            fields=list(existing.columns),
        )
        refreshed = pd.DataFrame([
            {"ts_code": "000002.SZ", "ann_date": "20260515", "amount": 20},
            {"ts_code": "000003.SZ", "ann_date": "20260516", "amount": 30},
        ])

        did_write, rows = download.write_window_merged_partition(
            path,
            refreshed,
            api_name="repurchase",
            params={"start_date": "20260510", "end_date": "20260520"},
            fields=list(refreshed.columns),
            key_columns=["ts_code", "ann_date"],
            date_columns=["ann_date"],
            start_date="20260510",
            end_date="20260520",
            revision_ledger=str(self.root / "revision_events.jsonl"),
        )

        merged = pd.read_parquet(path).sort_values("ts_code").reset_index(drop=True)
        self.assertTrue(did_write)
        self.assertEqual(rows, 3)
        self.assertEqual(merged["ts_code"].tolist(), ["000001.SZ", "000002.SZ", "000003.SZ"])
        self.assertEqual(merged["amount"].tolist(), [1, 20, 30])

    def test_macro_range_once_uses_retained_start_during_rolling_update(self):
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            tier="macro",
            start_date="20260504",
            macro_start_date="20200101",
            end_date="20260603",
            datasets=["cn_cpi"],
            force=True,
            page_limit=None,
            revision_ledger=str(self.root / "revision_events.jsonl"),
            min_interval_seconds=0,
            timeout_seconds=1,
        )
        client = FieldClient(MACRO_FIELDS)

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            self.assertEqual(download.download_macro(args), 0)

        cpi_calls = [params for api_name, params in client.calls if api_name == "cn_cpi"]
        self.assertEqual(cpi_calls[0]["start_m"], "202001")
        self.assertTrue((self.raw_dir / "cn_cpi" / "range=202001_latest.parquet").exists())

    def test_rolling_rewrite_keeps_a_range_file_backfilled_below_the_floor(self):
        # A backfill landed cn_cpi from 2014-07. The nightly rewrite asks from
        # the 2020 floor, but must re-request the file's own start and keep its
        # path: a rewrite from 202001 would prune the backfilled history away.
        backfilled = self.raw_dir / "cn_cpi" / "range=201407_latest.parquet"
        pd.DataFrame([{"month": "201407"}]).pipe(
            lambda df: tushare_io.write_parquet(backfilled, df, api_name="cn_cpi", params={}, fields=list(df.columns))
        )
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            tier="macro",
            start_date="20260504",
            macro_start_date="20200101",
            end_date="20260603",
            datasets=["cn_cpi"],
            force=True,
            page_limit=None,
            revision_ledger=str(self.root / "revision_events.jsonl"),
            min_interval_seconds=0,
            timeout_seconds=1,
        )
        class RangeClient(FieldClient):
            # Answers the whole requested range, as the vendor does.
            def query(self, api_name, params=None, fields="", retries=5):
                result = super().query(api_name, params, fields, retries)
                month = result.fields.index("month")
                first = list(result.items[0])
                first[month] = params["start_m"]
                return tushare_client.ApiResult(result.fields, [first, *result.items])

        client = RangeClient(MACRO_FIELDS)

        with redirect_stdout(io.StringIO()), patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            self.assertEqual(download.download_macro(args), 0)

        cpi_calls = [params for api_name, params in client.calls if api_name == "cn_cpi"]
        self.assertEqual(cpi_calls[0]["start_m"], "201407")
        remaining = sorted(path.name for path in (self.raw_dir / "cn_cpi").glob("range=*.parquet"))
        self.assertEqual(remaining, ["range=201407_latest.parquet"])
        # The audit expects the file the downloader keeps, not the floor's.
        expected = audit.expected_macro_paths(
            self.raw_dir, specs.MACRO_SPECS["cn_cpi"], "20200101", "20260603", args
        )
        self.assertEqual(expected, {backfilled})

    def test_lowering_the_start_of_a_stored_range_file_is_refused(self):
        # The range file's name carries its start: a backfill pull from an
        # earlier start would land a new partition outside the versioned path
        # (every row first seen today, the stored file and its first-seen
        # table orphaned), so it must stop before asking the vendor and leave
        # the stored series untouched.
        self.raw_dir.mkdir(parents=True)
        (self.raw_dir / ".raw_generation.json").write_text(json.dumps({
            "schema_version": 3, "state": "committed", "generation_id": "g", "versioned_since": "20260101",
        }), encoding="utf-8")

        def run(**overrides):
            args = argparse.Namespace(
                raw_dir=str(self.raw_dir), tier="macro", start_date="20260504", macro_start_date="20200101",
                end_date="20260603", datasets=["cn_cpi"], force=True, page_limit=None,
                revision_ledger=str(self.root / "revision_events.jsonl"), min_interval_seconds=0, timeout_seconds=1,
            )
            for key, value in overrides.items():
                setattr(args, key, value)
            client = FieldClient(MACRO_FIELDS)
            with redirect_stdout(io.StringIO()), patch.object(download, "load_token", return_value="token"), \
                    patch.object(download, "TuShareClient", return_value=client):
                download.download_macro(args)
            return client

        run()
        dataset = self.raw_dir / "cn_cpi"
        stored = {path.name: path.read_bytes() for path in dataset.iterdir()}
        self.assertEqual(sorted(stored), [
            "range=202001_latest.parquet", "range=202001_latest.parquet.first_seen", "range=202001_latest.parquet.meta.json",
        ])

        client = FieldClient(MACRO_FIELDS)
        with redirect_stdout(io.StringIO()), patch.object(download, "load_token", return_value="token"), \
                patch.object(download, "TuShareClient", return_value=client), \
                self.assertRaisesRegex(RuntimeError, r"cn_cpi: refusing to move the start of its range file from 202001 .* to 201407"):
            download.download_macro(argparse.Namespace(
                raw_dir=str(self.raw_dir), tier="macro", start_date="20140701", macro_start_date=None,
                history_floor="20140701", end_date="20260603", datasets=["cn_cpi"], force=True, page_limit=None,
                revision_ledger=str(self.root / "revision_events.jsonl"), min_interval_seconds=0, timeout_seconds=1,
            ))
        self.assertEqual(client.calls, [])
        self.assertEqual({path.name: path.read_bytes() for path in dataset.iterdir()}, stored)
        self.assertFalse((self.root / "raw_versions").exists())
        # A rolling pull from a later start still rewrites the stored file.
        self.assertEqual(run(start_date="20260601").calls[0][1]["start_m"], "202001")

    def test_fundamental_ann_month_windows_pull_full_natural_months(self):
        windows = download.fundamental_ann_month_windows("20200615", "20200705", {"202004"})
        self.assertEqual(windows, [
            ("20200401", "20200430", "202004"),
            ("20200601", "20200630", "202006"),
            ("20200701", "20200705", "202007"),
        ])

    def test_new_share_download_uses_the_local_trading_calendar(self):
        calendar_path = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2022.parquet"
        calendar_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"cal_date": "20220901", "is_open": "0"},
                {"cal_date": "20220930", "is_open": "1"},
                {"cal_date": "20221010", "is_open": "1"},
                {"cal_date": "20221011", "is_open": "1"},
                {"cal_date": "20221031", "is_open": "0"},
            ]
        ).to_parquet(calendar_path, index=False)

        class NewShareClient:
            def query(self, api_name, params=None, fields="", retries=5):
                if api_name != "new_share":
                    raise AssertionError(f"unexpected TuShare query: {api_name}")
                columns = fields.split(",")
                values = {
                    "ts_code": "301380.SZ",
                    "ipo_date": "20220930",
                    "issue_date": "20221019",
                    "ballot": "0.03",
                }
                return tushare_client.ApiResult(
                    columns,
                    [[values.get(column, "") for column in columns]])

        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20220901",
            end_date="20220930",
            datasets=["new_share"],
            force=True,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )
        with patch.object(download, "load_token", return_value="token"), patch.object(
            download, "TuShareClient", return_value=NewShareClient()
        ):
            self.assertEqual(download.download_event_flow(args), 0)

        frame = pd.read_parquet(self.raw_dir / "new_share" / "month=202209.parquet")
        self.assertEqual(frame.loc[0, "available_at"], "2022-10-11 23:59:59+08:00")

    def test_index_weight_download_pages_per_year_and_skips_covered_years(self):
        source_rows = {
            2020: 7005,  # forces two pages at the 7,000-row source clamp
            2021: 10,
        }

        class WeightClient:
            def __init__(self):
                self.calls = []

            def query(self, api_name, params=None, fields="", retries=5):
                assert api_name == "index_weight"
                self.calls.append(dict(params))
                year = int(params["start_date"][:4])
                rows = [
                    [params["index_code"], f"{index:06d}.SZ", f"{year}0131", 1.0]
                    for index in range(source_rows.get(year, 0))
                ]
                limit = min(int(params["limit"]), 7000)  # source-side clamp
                offset = int(params["offset"])
                page = rows[offset:offset + limit]
                return tushare_client.ApiResult(
                    fields=["index_code", "con_code", "trade_date", "weight"],
                    items=page)

        client = WeightClient()
        with patch.object(download, "DEFAULT_CN_INDEX_CODES", ["000300.SH"]):
            download.download_index_weight(
                client, self.raw_dir, "20190101", "20211231", specs.RESEARCH_HISTORY_FLOOR, False, None
            )
            base = self.raw_dir / "index_weight" / "index_code=000300.SH"
            # The 2019 window is floored away; both years land as partitions.
            self.assertFalse((base / "year=2019.parquet").exists())
            self.assertEqual(len(pd.read_parquet(base / "year=2020.parquet")), 7005)
            self.assertEqual(len(pd.read_parquet(base / "year=2021.parquet")), 10)
            offsets_2020 = [c["offset"] for c in client.calls if c["start_date"] == "20200101"]
            self.assertEqual(offsets_2020, [0, 7000])

            # Covered years skip without force; a later end date re-pulls only
            # the open year.
            calls_before = len(client.calls)
            download.download_index_weight(
                client, self.raw_dir, "20190101", "20211231", specs.RESEARCH_HISTORY_FLOOR, False, None
            )
            self.assertEqual(len(client.calls), calls_before)
            source_rows[2022] = 3
            download.download_index_weight(
                client, self.raw_dir, "20190101", "20221231", specs.RESEARCH_HISTORY_FLOOR, False, None
            )
            new_calls = client.calls[calls_before:]
            self.assertEqual({c["start_date"] for c in new_calls}, {"20220101"})
            self.assertEqual(len(pd.read_parquet(base / "year=2022.parquet")), 3)

            # A backfill run lowers the floor: the years below it land (from
            # the run's own start, mid-year included), the covered ones skip.
            source_rows[2018] = 4
            source_rows[2019] = 5
            calls_before = len(client.calls)
            download.download_index_weight(
                client, self.raw_dir, "20180701", "20191231", "20180701", False, None
            )
            new_calls = client.calls[calls_before:]
            self.assertEqual({c["start_date"] for c in new_calls}, {"20180701", "20190101"})
            self.assertEqual(len(pd.read_parquet(base / "year=2018.parquet")), 4)
            self.assertEqual(len(pd.read_parquet(base / "year=2019.parquet")), 5)

    def test_history_floor_is_lowered_only_by_an_explicit_backfill_flag(self):
        plain = argparse.Namespace()
        backfill = argparse.Namespace(history_floor="20140701")
        self.assertEqual(download.history_floor(specs.RESEARCH_HISTORY_FLOOR, plain), "20200101")
        self.assertEqual(download.history_floor(specs.RESEARCH_HISTORY_FLOOR, backfill), "20140701")
        # Never raised: a spec floor already below the flag stays where it is.
        self.assertEqual(download.history_floor("20100101", backfill), "20100101")
        parser = argparse.ArgumentParser()
        download.add_download_parser(parser.add_subparsers(dest="command"))
        parsed = parser.parse_args(["download", "--tier", "macro", "--history-floor", "20140701"])
        self.assertEqual(parsed.history_floor, "20140701")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["download", "--tier", "macro", "--history-floor", "2014-07-01"])

    def test_macro_backfill_lowers_the_spec_floor_of_the_named_datasets(self):
        # index_dailybasic keeps its own 2019 floor in every run but one that
        # names --history-floor; index_daily's 2010 spec floor is not raised.
        def run(**extra):
            args = argparse.Namespace(
                raw_dir=str(self.raw_dir),
                tier="macro",
                start_date="20170701",
                end_date="20171231",
                datasets=["index_dailybasic", "index_daily"],
                cn_index_code=["000905.SH"],
                force=False,
                page_limit=None,
                revision_ledger=str(self.root / "revision_events.jsonl"),
                min_interval_seconds=0,
                timeout_seconds=1,
                **extra,
                macro_start_date=None,
            )
            client = FieldClient(MACRO_FIELDS)
            with redirect_stdout(io.StringIO()), patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
                self.assertEqual(download.download_macro(args), 0)
            return sorted((api, params["start_date"]) for api, params in client.calls)

        self.assertEqual(run(), [("index_daily", "20170701")])
        self.assertEqual(
            run(history_floor="20140701"),
            [("index_dailybasic", "20170701")],  # index_daily's 2017 year is covered now
        )
        self.assertTrue((self.raw_dir / "index_dailybasic" / "ts_code=000905.SH" / "year=2017.parquet").exists())

    def test_default_daily_download_excludes_stk_auction(self):
        selected = common.selected_daily_datasets(argparse.Namespace(datasets=None))
        self.assertEqual(selected, specs.DAILY_DOWNLOAD_DATASETS)
        self.assertNotIn("stk_auction", selected)
        required = common.selected_daily_datasets(
            argparse.Namespace(datasets=None),
            default=specs.DAILY_REQUIRED_DATASETS,
        )
        self.assertIn("stk_auction", required)
        self.assertIn("stk_auction", specs.DAILY_REQUIRED_DATASETS)

    def test_generic_daily_cli_choices_exclude_stk_auction(self):
        parser = download.build_parser()
        rejected = (
            ["update", "--start-date", "20200102", "--daily-datasets", "stk_auction"],
            ["update", "--start-date", "20200102", "--refresh-daily-datasets", "stk_auction"],
            ["download", "--tier", "daily", "--refresh-daily-datasets", "stk_auction"],
        )
        for argv in rejected:
            with self.subTest(argv=argv), self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                parser.parse_args(argv)
        parsed = parser.parse_args([
            "update",
            "--start-date",
            "20200102",
            "--daily-datasets",
            "daily",
            "--refresh-daily-datasets",
            "adj_factor",
        ])
        self.assertEqual(parsed.daily_datasets, ["daily"])
        self.assertEqual(parsed.refresh_daily_datasets, ["adj_factor"])

    def test_generic_daily_path_refuses_stk_auction(self):
        self._write_trade_cal("20200102", is_open="1")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["stk_auction"],
            refresh_daily_datasets=[],
            force=False,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )
        with (
            patch.object(download, "load_token", return_value="token"),
            patch.object(download, "TuShareClient", return_value=NoQueryClient()),
            self.assertRaisesRegex(RuntimeError, "capture-open-auction"),
        ):
            download.download_daily(args)

    def _text_args(self, datasets: list[str]) -> argparse.Namespace:
        return argparse.Namespace(
            raw_dir=str(self.raw_dir), datasets=datasets, news_src=[], major_news_src=[],
            start_date="20261001", end_date="20261006", force=True, page_limit=None,
            min_interval_seconds=0, timeout_seconds=5,
            revision_ledger=str(self.root / "revision_events.jsonl"),
        )

    def test_each_text_dataset_goes_to_its_own_endpoint_with_its_own_token(self):
        # Through the real SDK, stopping only at the HTTP call: anns_d and
        # cctv_news go to the official service with TUSHARE_TOKEN_TMP,
        # report_rc stays on the relay with TUSHARE_TOKEN. Dummy tokens only.
        sent = []

        def post(url, **kwargs):
            payload = kwargs["json"]
            sent.append((url, payload["api_name"], payload["token"]))
            fields = specs.TEXT_SPECS[payload["api_name"]].fields.split(",")
            body = {"code": 0, "msg": "", "data": {"fields": fields, "items": []}}
            return SimpleNamespace(status_code=200, text=json.dumps(body), json=lambda: body)

        env = {"TUSHARE_TOKEN": "relay-dummy", "TUSHARE_TOKEN_TMP": "official-dummy",
               "TUSHARE_RELAY_URL": "https://relay.example"}
        with (
            chdir(self.root),
            patch.dict(os.environ, env, clear=True),
            patch.object(tushare_client.requests, "post", side_effect=post),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(download.download_text(self._text_args(["anns_d", "cctv_news", "report_rc"])), 0)
        routes = {(url.rsplit("/", 1)[0], api, token) for url, api, token in sent}
        self.assertEqual(routes, {
            (tushare_client.OFFICIAL_TUSHARE_URL, "anns_d", "official-dummy"),
            (tushare_client.OFFICIAL_TUSHARE_URL, "cctv_news", "official-dummy"),
            ("https://relay.example", "report_rc", "relay-dummy"),
        })
        self.assertTrue(tushare_client.OFFICIAL_TUSHARE_URL.startswith("https://"))
        self.assertTrue((self.raw_dir / "anns_d" / "month=202610.parquet").is_file())

    def test_refused_official_token_exits_77_before_any_write_and_never_prints_it(self):
        token = "official-dummy-5f3a"
        sent = []

        def post(url, **kwargs):
            sent.append(kwargs["json"]["api_name"])
            body = {"code": 40101, "msg": f"您的token不对，请确认。{token}", "data": None}
            return SimpleNamespace(status_code=200, text=json.dumps(body, ensure_ascii=False), json=lambda: body)

        out = io.StringIO()
        with (
            chdir(self.root),  # no .env here: the relay token is never needed
            patch.dict(os.environ, {"TUSHARE_TOKEN_TMP": token}, clear=True),
            patch.object(tushare_client.requests, "post", side_effect=post),
            patch.object(tushare_client.time, "sleep"),
            redirect_stdout(out),
        ):
            code = download.download_text(self._text_args(["anns_d"]))
        self.assertEqual(code, common.NO_MUTATION_FAILURE_EXIT_CODE)
        self.assertEqual(sent, ["anns_d"] * 5)  # the one-row proof, retried; no page request
        self.assertFalse((self.raw_dir / "anns_d").exists())
        self.assertIn("OfficialAccessError: anns_d", out.getvalue())
        self.assertIn(tushare_client.OFFICIAL_TOKEN_ENV, out.getvalue())
        self.assertNotIn(token, out.getvalue())

    def test_missing_official_token_exits_77_naming_the_variable(self):
        out = io.StringIO()
        with (
            chdir(self.root),
            patch.dict(os.environ, {}, clear=True),
            redirect_stdout(out),
        ):
            code = download.download_text(self._text_args(["anns_d"]))
        self.assertEqual(code, common.NO_MUTATION_FAILURE_EXIT_CODE)
        self.assertIn("TUSHARE_TOKEN_TMP is not set", out.getvalue())
        self.assertFalse((self.raw_dir / "anns_d").exists())

    def test_reference_refresh_datasets_force_selected_tables(self):
        self._write_trade_cal("20200102")
        for status in ("L", "D", "P"):
            path = self.raw_dir / "stock_basic" / f"list_status={status}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"ts_code": "999999.SZ", "list_status": status}]).to_parquet(path, index=False)
        namechange = self.raw_dir / "namechange" / "namechange.parquet"
        namechange.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"ts_code": "999999.SZ", "name": "old"}]).to_parquet(namechange, index=False)
        classify = self.raw_dir / "index_classify" / "src=SW2021.parquet"
        classify.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"index_code": "801999.SI", "level": "L1"}]).to_parquet(classify, index=False)
        member = self.raw_dir / "index_member_all" / "l1_code=801999.SI.parquet"
        member.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"l1_code": "801999.SI", "ts_code": "999999.SZ"}]).to_parquet(member, index=False)

        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            bak_start_date="20200102",
            skip_bak_basic=True,
            force=False,
            refresh_reference_datasets=["stock_basic", "namechange", "index_classify", "index_member_all"],
            namechange_recent_days=0,
            trade_cal_end_date=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )
        client = ReferenceClient()

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            self.assertEqual(download.download_reference(args), 0)

        called_apis = [api_name for api_name, _ in client.calls]
        self.assertEqual(called_apis.count("stock_basic"), 3)
        self.assertIn("namechange", called_apis)
        self.assertIn("index_classify", called_apis)
        self.assertIn("index_member_all", called_apis)
        self.assertIn("index_member", called_apis)
        # Both classification vintages refresh together.
        classify_srcs = {params.get("src") for api, params in client.calls if api == "index_classify"}
        self.assertEqual(classify_srcs, {"SW2014", "SW2021"})
        self.assertTrue((self.raw_dir / "index_classify" / "src=SW2014.parquet").exists())
        # Member refresh pulls current and departed members.
        member_is_new = {params.get("is_new") for api, params in client.calls if api == "index_member_all"}
        self.assertEqual(member_is_new, {"Y", "N"})
        self.assertTrue((self.raw_dir / "index_member" / "l1_code=801010.SI.parquet").exists())
        # New reference statics download on first run (files absent).
        for api in ("index_basic", "index_weight"):
            self.assertIn(api, called_apis)
        # The tables retired on 2026-11-21 are never requested.
        self.assertFalse({"stock_company", "ths_index", "ths_member"} & set(called_apis))

        # SW2014 membership is frozen history: a second run without force must
        # not re-request it while the classify refresh still runs.
        client.calls.clear()
        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            self.assertEqual(download.download_reference(args), 0)
        self.assertNotIn("index_member", [api for api, _ in client.calls])

    def test_a_namechange_window_replaces_the_recent_announcements_by_one_call(self):
        path = self.raw_dir / "namechange" / "namechange.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = ["ts_code", "name", "start_date", "end_date", "ann_date", "change_reason"]
        pd.DataFrame([
            ["000001.SZ", "平安银行", "20120801", None, "20120730", "改名"],
            ["000002.SZ", "万科A", "19910129", None, "19910129", None],
            ["000004.SZ", "ST国华", "20261110", None, "20261109", None],
        ], columns=columns).to_parquet(path, index=False)
        answer = [
            ["000004.SZ", "ST国华", "20261110", None, "20261109", "ST"],
            ["000002.SZ", "ST万科", "20261124", None, "20261123", "ST"],
        ]

        class WindowClient:
            def __init__(self):
                self.calls = []

            def query(self, api_name, params=None, fields="", retries=5):
                self.calls.append((api_name, dict(params or {})))
                return tushare_client.ApiResult(fields.split(","), [list(row) for row in answer])

        client = WindowClient()
        download.download_namechange(client, self.raw_dir, False, ledger=None, recent_since="20261024")
        self.assertEqual(client.calls, [("namechange", {"start_date": "20261024"})])
        table = pd.read_parquet(path)
        # The window's rows replace theirs (no second copy) and add the new
        # one; the rest of the table stays as the full loop wrote it.
        self.assertEqual(
            table[["ts_code", "name", "change_reason"]].values.tolist(),
            [["000001.SZ", "平安银行", "改名"], ["000002.SZ", "万科A", None], ["000002.SZ", "ST万科", "ST"],
             ["000004.SZ", "ST国华", "ST"]],
        )
        # Without a window an existing table is left alone, with no call.
        download.download_namechange(NoQueryClient(), self.raw_dir, False, ledger=None)

    def test_reference_refresh_does_not_overwrite_existing_index_classify_on_empty_response(self):
        self._write_trade_cal("20200102")
        for status in ("L", "D", "P"):
            stock_path = self.raw_dir / "stock_basic" / f"list_status={status}.parquet"
            stock_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"ts_code": "000001.SZ", "list_status": status}]).to_parquet(stock_path, index=False)
        path = self.raw_dir / "index_classify" / "src=SW2014.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        original = pd.DataFrame([{"index_code": "801010.SI", "level": "L1", "src": "SW2014"}])
        original.to_parquet(path, index=False)
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            bak_start_date="20200102",
            skip_bak_basic=True,
            force=False,
            refresh_reference_datasets=["index_classify"],
            namechange_recent_days=0,
            trade_cal_end_date=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=EmptyClient()):
            with self.assertRaisesRegex(RuntimeError, "index_classify params=.* required reference partition"):
                download.download_reference(args)
        self.assertTrue(pd.read_parquet(path).equals(original))

    def test_index_member_all_empty_refresh_keeps_existing_and_fails_on_missing(self):
        path = self.raw_dir / "index_member_all" / "l1_code=801010.SI.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        original = pd.DataFrame([{"l1_code": "801010.SI", "ts_code": "000001.SZ", "in_date": "20200101", "out_date": ""}])
        original.to_parquet(path, index=False)
        classify = pd.DataFrame([{"index_code": "801010.SI", "level": "L1"}])
        report = self.raw_dir.parent / "run_report.jsonl"
        with patch.dict(os.environ, {versions.RUN_REPORT_ENV: str(report)}):
            download.download_index_member_all(EmptyClient(), self.raw_dir, classify, True, ledger=None)
        self.assertTrue(pd.read_parquet(path).equals(original))
        # The kept partition is an open defect, not a silent skip.
        outcome = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual((outcome["dataset"], outcome["outcome"]), ("index_member_all", "defect"))
        # An absent partition with an empty vendor response must fail fast.
        classify_missing = pd.DataFrame([{"index_code": "801999.SI", "level": "L1"}])
        with self.assertRaisesRegex(RuntimeError, "zero rows for required reference partition"):
            download.download_index_member_all(EmptyClient(), self.raw_dir, classify_missing, True, ledger=None)

    def test_update_parser_force_refreshes_the_cheap_reference_tables_by_default(self):
        argv = [
            "download.py",
            "update",
            "--start-date",
            "20260601",
            "--end-date",
            "20260601",
        ]
        with patch.object(sys, "argv", argv):
            args = download.parse_args()

        self.assertEqual(args.refresh_reference_datasets, ["stock_basic", "namechange", "index_classify", "index_member_all"])
        self.assertEqual(args.refresh_daily_datasets, [])
        self.assertEqual(args.macro_start_date, "20200101")

    def test_update_all_dimensions_uses_retained_start_for_macro_context(self):
        args = argparse.Namespace(
            start_date="20260504",
            end_date="20260603",
            macro_start_date="20200101",
            bak_start_date=None,
            force=False,
            refresh_open_window=True,
            trade_cal_lookahead_days=7,
            raw_dir=str(self.raw_dir),
            page_limit=None,
            revision_ledger=str(self.root / "revision_events.jsonl"),
            reference_min_interval_seconds=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            refresh_reference_datasets=[],
            namechange_recent_days=0,
            daily_datasets=None,
            refresh_daily_datasets=[],
            macro_datasets=["cn_gdp"],
            global_datasets=["index_global"],
            event_datasets=[],
            include_board_trading=True,
            board_datasets=None,
            include_intraday=False,
            include_share_float_complete=False,
            include_text=True,
            include_global=True,
            text_datasets=[],
            fundamental_datasets=[],
            fundamental_refresh_period_count=0,
            fundamental_refresh_ann_month_count=0,
            fundamental_refresh_ts_code_datasets=[],
            fundamental_refresh_event_days=0,
            fundamental_dividend_probe_days=0,
        )
        seen = {}

        def capture(label, _fn, step_args, summary):
            seen[label] = step_args
            summary.append({"step": label, "exit_code": 0})

        with patch.object(download, "run_update_step", side_effect=capture):
            summary = []
            download.update_all_dimensions(args, summary)

        self.assertEqual(seen["daily"].start_date, "20260504")
        self.assertEqual(seen["event_flow"].start_date, "20260504")
        self.assertEqual(seen["macro"].start_date, "20260504")
        self.assertEqual(seen["macro"].macro_start_date, "20200101")
        self.assertEqual(seen["global"].start_date, "20260504")
        self.assertEqual(seen["global"].macro_start_date, "20200101")
        self.assertIn("text_evidence", seen)
        # Board datasets run after the fundamentals Paper needs, so a vendor
        # change to one of them cannot stop the fundamental refresh.
        self.assertEqual(list(seen)[-2:], ["fundamental", "board_trading"])

    def test_evening_update_excludes_the_natural_day_text_tier(self):
        # Text is landed by its own calendar-day job; running it inside the
        # trading-day evening update would both duplicate the calls and tie
        # weekend text to a job that never launches on weekends.
        args = argparse.Namespace(
            start_date="20260504",
            end_date="20260603",
            macro_start_date="20200101",
            bak_start_date=None,
            force=False,
            refresh_open_window=True,
            trade_cal_lookahead_days=7,
            raw_dir=str(self.raw_dir),
            page_limit=None,
            revision_ledger=str(self.root / "revision_events.jsonl"),
            reference_min_interval_seconds=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            refresh_reference_datasets=[],
            namechange_recent_days=0,
            daily_datasets=None,
            refresh_daily_datasets=[],
            macro_datasets=["cn_gdp"],
            global_datasets=["index_global"],
            event_datasets=[],
            include_board_trading=False,
            include_intraday=False,
            include_share_float_complete=False,
            include_text=False,
            include_global=False,
            text_datasets=[],
            fundamental_datasets=[],
            fundamental_refresh_period_count=0,
            fundamental_refresh_ann_month_count=0,
            fundamental_refresh_ts_code_datasets=[],
            fundamental_refresh_event_days=0,
            fundamental_dividend_probe_days=0,
        )
        seen = {}

        def capture(label, _fn, step_args, summary):
            seen[label] = step_args
            summary.append({"step": label, "exit_code": 0})

        with patch.object(download, "run_update_step", side_effect=capture):
            download.update_all_dimensions(args, [])

        self.assertNotIn("text_evidence", seen)
        self.assertNotIn("global", seen)
        self.assertIn("macro", seen)

    def test_daily_refresh_datasets_force_only_selected_trade_date_dataset(self):
        self._write_trade_cal("20200102")
        daily_rows = pd.DataFrame([{"trade_date": "20200102", "ts_code": "999999.SZ", "close": 1.0}])
        adj_rows = pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ", "adj_factor": 2.0}])
        tushare_io.write_parquet(
            self.raw_dir / "daily" / "trade_date=20200102.parquet",
            daily_rows,
            api_name="daily",
            params={"trade_date": "20200102"},
            fields=list(daily_rows.columns),
        )
        tushare_io.write_parquet(
            self.raw_dir / "adj_factor" / "trade_date=20200102.parquet",
            adj_rows,
            api_name="adj_factor",
            params={"trade_date": "20200102"},
            fields=list(adj_rows.columns),
        )

        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["daily", "adj_factor"],
            refresh_daily_datasets=["adj_factor"],
            revision_ledger=str(self.root / "revision_events.jsonl"),
            force=False,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
        )
        client = FieldClient(DAILY_FIELDS, default=0)

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(download.download_daily(args), 0)

        self.assertEqual([api_name for api_name, _ in client.calls], ["adj_factor"])
        self.assertIn("REVISION_ALERT", output.getvalue())
        # The alert is a compact one-liner; the full record lives in the ledger.
        self.assertIn('"dataset": "adj_factor"', output.getvalue())
        self.assertNotIn('"removed_keys_sample"', output.getvalue())
        ledger_lines = (self.root / "revision_events.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(ledger_lines), 1)
        ledger_event = json.loads(ledger_lines[0])
        self.assertEqual(ledger_event["schema_version"], 3)
        self.assertEqual(ledger_event["kind"], "restatement")
        self.assertTrue(ledger_event["write_id"])
        self.assertTrue(ledger_event["superseded_version"])
        self.assertEqual(set(pd.read_parquet(self.raw_dir / "daily" / "trade_date=20200102.parquet")["ts_code"]), {"999999.SZ"})

    def test_daily_force_refresh_keeps_the_partition_a_removal_replaces(self):
        self._write_trade_cal("20200102")
        ledger = self.root / "daily_key_removal.jsonl"
        for dataset in ("daily", "adj_factor"):
            original = pd.DataFrame([
                {"trade_date": "20200102", "ts_code": "000001.SZ", "close": 10.0, "adj_factor": 1.0},
                {"trade_date": "20200102", "ts_code": "000002.SZ", "close": 20.0, "adj_factor": 1.1},
            ])
            path = self.raw_dir / dataset / "trade_date=20200102.parquet"
            tushare_io.write_parquet(
                path,
                original,
                api_name=dataset,
                params={"trade_date": "20200102"},
                fields=list(original.columns),
            )
            # A forced re-pull that drops half of the keys is a restatement:
            # accepted, with the partition it replaces kept as a version.
            with redirect_stdout(io.StringIO()):
                download.download_trade_date_dataset(
                    FieldClient(DAILY_FIELDS, default=0),
                    self.raw_dir,
                    specs.DAILY_SPECS[dataset],
                    ["20200102"],
                    True,
                    5000,
                    ledger,
                )
            self.assertEqual(set(pd.read_parquet(path)["ts_code"]), {"000001.SZ"}, dataset)
            (superseded,) = versions.list_versions(path)
            kept = pd.read_parquet(versions.version_dir(path) / f"{superseded}.parquet")
            self.assertEqual(set(kept["ts_code"]), {"000001.SZ", "000002.SZ"}, dataset)

    def test_fundamental_update_refreshes_recent_periods_and_affected_ts_code_snapshots(self):
        stock_basic = self.raw_dir / "stock_basic" / "list_status=L.parquet"
        stock_basic.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"ts_code": "000001.SZ"}]).to_parquet(stock_basic, index=False)
        for dataset in ("dividend", "fina_audit", "fina_mainbz_vip"):
            path = self.raw_dir / dataset / "ts_code=000001.SZ.parquet"
            tushare_io.write_parquet(
                path,
                pd.DataFrame([{"ts_code": "000001.SZ", "ann_date": "20190101", "end_date": "20181231"}]),
                api_name=dataset,
                params={"ts_code": "000001.SZ"},
                fields=["ts_code", "ann_date", "end_date"],
            )
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200601",
            end_date="20200603",
            datasets=["dividend", "fina_audit", "fina_mainbz_vip", "income_vip"],
            force=False,
            page_limit=None,
            codes=None,
            max_codes=None,
            fundamental_refresh_period_count=2,
            fundamental_refresh_ann_month_count=0,
            fundamental_refresh_ts_code_datasets=["dividend", "fina_audit", "fina_mainbz_vip"],
            fundamental_refresh_event_days=90,
            fundamental_dividend_probe_days=0,
            min_interval_seconds=0,
            timeout_seconds=1,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )

        client = FundamentalClient()
        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            self.assertEqual(download.download_fundamental(args), 0)

        calls = [(api, params) for api, params in client.calls if params.get("offset") == 0]
        income_periods = [params["period"] for api, params in calls if api == "income_vip"]
        self.assertEqual(income_periods, ["20191231", "20200331"])
        refreshed_ts_code_calls = [(api, params.get("ts_code")) for api, params in calls if api in {"dividend", "fina_audit", "fina_mainbz_vip"}]
        self.assertEqual(refreshed_ts_code_calls, [("dividend", "000001.SZ"), ("fina_audit", "000001.SZ"), ("fina_mainbz_vip", "000001.SZ")])

    def test_fundamental_download_clamps_cashflow_to_measured_page_limit(self):
        # cashflow_vip returns at most 6,400 rows per call (probed 2026-08-10),
        # so the tier-default 7,000 request must clamp to the spec's measured
        # 6,000 while income_vip keeps paging at 7,000.
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20191231",
            end_date="20191231",
            datasets=["income_vip", "cashflow_vip"],
            force=True,
            page_limit=7000,
            max_codes=None,
            fundamental_refresh_period_count=0,
            fundamental_refresh_ann_month_count=0,
            fundamental_refresh_ts_code_datasets=[],
            fundamental_refresh_event_days=0,
            fundamental_dividend_probe_days=0,
            min_interval_seconds=0,
            timeout_seconds=1,
            revision_ledger=None,
        )
        client = FundamentalClient()
        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            self.assertEqual(download.download_fundamental(args), 0)
        limits = {(api, params["limit"]) for api, params in client.calls if api in {"income_vip", "cashflow_vip"}}
        self.assertEqual(limits, {("income_vip", 7000), ("cashflow_vip", 6000)})

    def test_fundamental_affected_codes_raise_on_corrupt_partition(self):
        """A corrupt partition must fail the targeted refresh loudly, not
        silently shrink the affected-codes set while the job publishes."""
        corrupt = self.raw_dir / "income_vip" / "period=20191231.parquet"
        corrupt.parent.mkdir(parents=True, exist_ok=True)
        corrupt.write_bytes(b"not a parquet file")

        with self.assertRaises(ValueError):
            download.recent_fundamental_event_codes(
                self.raw_dir,
                refresh_periods={"20191231"},
                refresh_months=set(),
                period_datasets=["income_vip"],
                ann_month_datasets=[],
                start_date="20200601",
                end_date="20200603",
            )

    def test_fundamental_download_uses_explicit_codes_for_ts_code_datasets(self):
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200601",
            end_date="20200603",
            datasets=["dividend", "fina_audit", "fina_mainbz_vip"],
            force=True,
            page_limit=None,
            max_codes=None,
            codes=["920126.BJ"],
            fundamental_refresh_period_count=0,
            fundamental_refresh_ann_month_count=0,
            fundamental_refresh_ts_code_datasets=[],
            fundamental_refresh_event_days=0,
            fundamental_dividend_probe_days=0,
            min_interval_seconds=0,
            timeout_seconds=1,
            revision_ledger=None,
        )
        client = FundamentalClient()

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            self.assertEqual(download.download_fundamental(args), 0)

        calls = [(api, params.get("ts_code")) for api, params in client.calls if params.get("offset") == 0]
        self.assertEqual(calls, [("dividend", "920126.BJ"), ("fina_audit", "920126.BJ"), ("fina_mainbz_vip", "920126.BJ")])

    def test_dividend_probe_uses_only_supported_date_params(self):
        client = FundamentalClient()

        codes = download.probe_recent_dividend_codes(client, "20200603", 1, page_limit=1000)

        self.assertEqual(codes, {"000001.SZ"})
        probe_params = [
            set(params) - {"limit", "offset"}
            for api_name, params in client.calls
            if api_name == "dividend"
        ]
        self.assertEqual(probe_params, [{"ann_date"}, {"imp_ann_date"}, {"ex_date"}, {"record_date"}])
        self.assertNotIn("pay_date", {key for params in probe_params for key in params})

    def test_event_flow_refreshes_trade_cal_before_same_day_margin_secs(self):
        self._write_trade_cal("20260603")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20260604",
            end_date="20260604",
            datasets=["margin_secs"],
            force=True,
            page_limit=None,
            revision_ledger=str(self.root / "revision_events.jsonl"),
            min_interval_seconds=0,
            timeout_seconds=1,
        )
        client = CalendarEventClient()

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            self.assertEqual(download.download_event_flow(args), 0)

        self.assertTrue((self.raw_dir / "margin_secs" / "trade_date=20260604.parquet").exists())
        self.assertIn(("trade_cal", {"exchange": "SSE", "start_date": "20260604", "end_date": "20260604"}), client.calls)
        margin_calls = [params for api_name, params in client.calls if api_name == "margin_secs"]
        self.assertEqual(margin_calls[0]["trade_date"], "20260604")

    def test_recent_fundamental_event_codes_filters_period_rows_by_visible_date(self):
        period_path = self.raw_dir / "income_vip" / "period=20200331.parquet"
        period_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200430", "end_date": "20200331"},
            {"ts_code": "000002.SZ", "ann_date": "20190430", "end_date": "20190331"},
        ]).to_parquet(period_path, index=False)

        codes = download.recent_fundamental_event_codes(
            self.raw_dir,
            {"20200331"},
            set(),
            ["income_vip"],
            [],
            "20200401",
            "20200603",
        )

        self.assertEqual(codes, {"000001.SZ"})

    def test_disclosure_schedule_rows_mark_codes_only_by_actual_disclosure(self):
        path = self.raw_dir / "disclosure_date" / "period=20200930.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200930", "end_date": "20200930", "pre_date": "20201020", "actual_date": None},
            {"ts_code": "000002.SZ", "ann_date": "20200930", "end_date": "20200930", "pre_date": "20201020", "actual_date": "20200602"},
            {"ts_code": "000003.SZ", "ann_date": "20200930", "end_date": "20200930", "pre_date": "20201020", "actual_date": "20200301"},
        ]).to_parquet(path, index=False)

        codes = download.recent_fundamental_event_codes(
            self.raw_dir, {"20200930"}, set(), ["disclosure_date"], [], "20200601", "20200930",
        )

        self.assertEqual(codes, {"000002.SZ"})

    def test_zero_ok_force_refresh_does_not_overwrite_existing_nonempty_partition(self):
        self._write_trade_cal("20200102")
        path = self.raw_dir / "limit_list_d" / "trade_date=20200102.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        original = pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ", "limit": "U"}])
        original.to_parquet(path, index=False)
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["limit_list_d"],
            revision_ledger=str(self.root / "zero_ok_revision_events.jsonl"),
            force=True,
            page_limit=10000,
            min_interval_seconds=0,
            timeout_seconds=1,
        )

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=EmptyClient()):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(download.download_board_trading(args), 0)

        # An empty answer for a non-empty partition is a defect: the current
        # version stays and the run goes on.
        self.assertIn("empty answer for a partition holding 1 rows", output.getvalue())
        self.assertTrue(pd.read_parquet(path).equals(original))
        ledger = (self.root / "zero_ok_revision_events.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(ledger), 1)
        self.assertEqual(json.loads(ledger[0])["kind"], "defect")

    def test_event_range_month_counts_only_partitions_it_wrote(self):
        # `written` decides the event/flow no-mutation exit contract, so a
        # partition the writer kept must never be counted as a mutation.
        spec = specs.EVENT_FLOW_SPECS["repurchase"]
        fresh = self.raw_dir / "repurchase" / "month=202002.parquet"
        with redirect_stdout(io.StringIO()):
            written = download.download_event_range_month(
                EmptyClient(), self.raw_dir, spec,
                [("20200201", "20200229", "202002")], True, 5000,
                str(self.root / "range_month_counts.jsonl"),
            )
        # An empty response on a NEW partition is a real write: the empty
        # partition is what the lake now carries.
        self.assertEqual(written, 1)
        self.assertTrue(fresh.exists())

        seed_partition(
            self.raw_dir / "repurchase" / "month=202003.parquet",
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20200315"],
                "end_date": [""],
                "proc": [""],
                "exp_date": [""],
            },
            "repurchase",
        )
        with redirect_stdout(io.StringIO()):
            written = download.download_event_range_month(
                EmptyClient(), self.raw_dir, spec,
                [("20200301", "20200331", "202003")], True, 5000,
                str(self.root / "range_month_counts.jsonl"),
            )
        # The same empty response over an existing non-empty partition is
        # refused by the empty guard and is NOT a mutation.
        self.assertEqual(written, 0)

    def test_event_range_month_replaces_by_the_supplier_filter_column(self):
        # top10_floatholders' start_date/end_date filter the report period
        # (end_date), not ann_date. A mid-month nightly window that dropped
        # existing rows by ann_date silently lost every row whose report period
        # precedes the window start (the request never re-serves it) and kept
        # the stale version of a re-served row whose ann_date the supplier
        # moved past the window.
        spec = specs.EVENT_FLOW_SPECS["top10_floatholders"]
        fields = spec.fields.split(",")

        def row(ts_code, ann_date, end_date, amount):
            values = {"ts_code": ts_code, "ann_date": ann_date, "end_date": end_date, "holder_name": "holder", "hold_amount": amount}
            return [values.get(field, "") for field in fields]

        class ReportPeriodClient:
            def __init__(self, rows):
                self.rows = rows

            def query(self, api_name, params=None, fields="", retries=5):
                params = params or {}
                served = [] if params.get("offset") else [
                    item for item in self.rows
                    if params["start_date"] <= item[fields.split(",").index("end_date")] <= params["end_date"]
                ]
                return tushare_client.ApiResult(fields.split(","), served)

        client = ReportPeriodClient([
            row("301575.SZ", "20260916", "20260911", 1.0),
            row("000001.SZ", "20260920", "20260915", 2.0),
            row("000002.SZ", "20261010", "20260930", 3.0),
        ])
        ledger = str(self.root / "report_period_revision_events.jsonl")
        with redirect_stdout(io.StringIO()):
            download.download_event_range_month(client, self.raw_dir, spec, [("20260901", "20260930", "202609")], True, 5000, ledger)
            # The supplier revises one row and corrects another's ann_date;
            # the 301575.SZ row is untouched and lies outside the next window.
            client.rows = [
                row("301575.SZ", "20260916", "20260911", 1.0),
                row("000001.SZ", "20260920", "20260915", 20.0),
                row("000002.SZ", "20261012", "20260930", 3.0),
            ]
            download.download_event_range_month(client, self.raw_dir, spec, [("20260913", "20260930", "202609")], True, 5000, ledger)

        merged = pd.read_parquet(self.raw_dir / "top10_floatholders" / "month=202609.parquet")
        self.assertEqual(
            sorted(zip(merged["ts_code"], merged["ann_date"], merged["end_date"], merged["hold_amount"], strict=True)),
            [
                ("000001.SZ", "20260920", "20260915", 20.0),
                ("000002.SZ", "20261012", "20260930", 3.0),
                ("301575.SZ", "20260916", "20260911", 1.0),
            ],
        )

    def test_event_range_month_bisects_windows_instead_of_trusting_offset_paging(self):
        spec = specs.EVENT_FLOW_SPECS["stk_holdernumber"]
        rows = [
            {"ts_code": f"{index:06d}.SZ", "ann_date": f"2026090{1 + index // 2}", "end_date": "20260820", "holder_num": index}
            for index in range(6)
        ]
        client = TieUnstablePagingClient(rows)
        with redirect_stdout(io.StringIO()):
            download.download_event_range_month(
                client, self.raw_dir, spec, [("20260901", "20260930", "202609")], True, 4,
                str(self.root / "paging_revision_events.jsonl"),
            )
        # Offset paging would land rows 0-3 plus a repeat of 0-1, and never 4-5.
        landed = pd.read_parquet(self.raw_dir / "stk_holdernumber" / "month=202609.parquet")
        self.assertEqual(sorted(landed["holder_num"].tolist()), list(range(6)))
        self.assertTrue(all(call["offset"] == 0 for call in client.calls))

    def test_event_range_month_records_a_day_its_pager_repeats_as_a_defect(self):
        # A row served on two pages means as many rows are missing: the answer
        # is defective, the month keeps its current version (here: none) and
        # the defect is recorded instead of ending the run.
        spec = specs.EVENT_FLOW_SPECS["stk_holdernumber"]
        client = TieUnstablePagingClient([
            {"ts_code": f"{index:06d}.SZ", "ann_date": "20260905", "end_date": "20260820", "holder_num": index}
            for index in range(5)
        ])
        ledger = self.root / "paging_revision_events.jsonl"
        with redirect_stdout(io.StringIO()):
            written = download.download_event_range_month(
                client, self.raw_dir, spec, [("20260901", "20260930", "202609")], True, 4, str(ledger),
            )
        self.assertEqual(written, 0)
        self.assertFalse((self.raw_dir / "stk_holdernumber" / "month=202609.parquet").exists())
        event = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual((event["kind"], event["partition"]), ("defect", "month=202609"))
        self.assertIn("on two pages", event["detail"])

    def test_event_flow_report_period_window_reaches_back_by_the_disclosure_lag(self):
        # A nightly window over announcements must request every report period
        # those announcements can belong to: the top-10 tables filter on the
        # period, and annual reports land up to 120 days after it ends.
        client = EmptyClient()
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20260914",
            end_date="20261014",
            datasets=["top10_floatholders", "repurchase"],
            force=True,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )
        with redirect_stdout(io.StringIO()), patch.object(download, "load_token", return_value="token"), patch.object(
            download, "TuShareClient", return_value=client
        ):
            download.download_event_flow(args)

        def first_window(api_name):
            return min((params["start_date"], params["end_date"]) for name, params in client.calls if name == api_name)

        self.assertEqual(first_window("top10_floatholders"), ("20260517", "20260531"))
        self.assertEqual(first_window("repurchase"), ("20260914", "20260930"))

    def test_required_event_flow_zero_rows_raise_instead_of_cron_ok(self):
        self._write_trade_cal_with_stamping_window("20200102")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["margin"],
            force=True,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            revision_ledger=common.REVISION_EVENTS_PATH,
            zero_rows_not_ready=False,
        )

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=EmptyClient()):
            with self.assertRaisesRegex(RuntimeError, "required event/flow partitions returned zero or incomplete rows"):
                download.download_event_flow(args)

    def test_event_flow_zero_rows_not_ready_exits_75_without_mutation(self):
        self._write_trade_cal_with_stamping_window("20200102")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["margin"],
            force=True,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            zero_rows_not_ready=True,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=EmptyClient()):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(download.download_event_flow(args), common.NO_MUTATION_RETRY_EXIT_CODE)

        self.assertIn("not_ready_no_mutation", output.getvalue())
        self.assertFalse((self.raw_dir / "margin" / "trade_date=20200102.parquet").exists())

    def test_margin_partial_exchange_day_is_not_ready_not_committed(self):
        # Exchanges publish independently; an SSE-only margin day poisons
        # market-wide aggregates and must ride the not-ready contract.
        class PartialMarginClient(EmptyClient):
            def query(self, api_name, params=None, fields="", retries=5):
                if api_name != "margin":
                    return super().query(api_name, params, fields, retries)
                self.calls.append((api_name, dict(params or {})))
                columns = fields.split(",")
                row = ["20260529" if col == "trade_date" else "SSE" if col == "exchange_id" else "1.0" for col in columns]
                return tushare_client.ApiResult(columns, [row])

        self._write_trade_cal_with_stamping_window("20260529")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20260529",
            end_date="20260529",
            datasets=["margin"],
            force=True,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            zero_rows_not_ready=True,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )
        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=PartialMarginClient()):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(download.download_event_flow(args), common.NO_MUTATION_RETRY_EXIT_CODE)
        self.assertIn("missing exchanges ['BSE', 'SZSE']", output.getvalue())
        self.assertFalse((self.raw_dir / "margin" / "trade_date=20260529.parquet").exists())

    def test_margin_committed_partial_partition_is_reattempted_and_audited(self):
        # Pre-BSE days need SSE+SZSE only; post-cutover days need all three.
        self.assertEqual(common.margin_missing_exchanges("20230210", ["SSE", "SZSE"]), [])
        self.assertEqual(common.margin_missing_exchanges("20230213", ["SSE", "SZSE"]), ["BSE"])

        margin_dir = self.raw_dir / "margin"
        margin_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"trade_date": ["20260529"], "exchange_id": ["SSE"], "rzye": [1.0]}).to_parquet(
            margin_dir / "trade_date=20260529.parquet", index=False
        )
        # A covering non-force run must not skip the committed partial day.
        client = EmptyClient()
        written, zero_skipped = download.download_event_trade_date_dataset(
            client, self.raw_dir, specs.EVENT_FLOW_SPECS["margin"], ["20260529"], False, None,
            trading_dates=["20260529", "20260601"],
        )
        self.assertEqual((written, zero_skipped), (0, 1))
        self.assertEqual([api for api, _ in client.calls], ["margin"])

        findings: list[tuple[str, str, str, dict]] = []
        audit.audit_margin_exchange_completeness(
            self.raw_dir, "margin", lambda sev, code, msg, details: findings.append((sev, code, msg, details))
        )
        self.assertEqual(len(findings), 1)
        severity, code, _, details = findings[0]
        self.assertEqual((severity, code), ("error", "margin_exchange_completeness"))
        self.assertEqual(details["incomplete_days"], [{"trade_date": "20260529", "missing_exchanges": ["BSE", "SZSE"]}])

    def test_margin_detail_partial_exchange_day_is_refused_and_audited(self):
        # margin_detail names its venue only through the ts_code suffix, so an
        # SSE-only day looks like a normal partition while silently dropping
        # ~55% of the market. It must ride the same contract as the summary.
        self.assertEqual(
            common.margin_family_missing_exchanges(
                "margin_detail", "20260529", pd.DataFrame({"ts_code": ["600000.SH"]})
            ),
            ["BSE", "SZSE"],
        )
        self.assertEqual(
            common.margin_family_missing_exchanges(
                "margin_detail",
                "20260529",
                pd.DataFrame({"ts_code": ["600000.SH", "000001.SZ", "830000.BJ"]}),
            ),
            [],
        )
        # Datasets outside the margin family keep no per-dataset branch at all.
        self.assertEqual(
            common.margin_family_missing_exchanges(
                "moneyflow", "20260529", pd.DataFrame({"ts_code": ["600000.SH"]})
            ),
            [],
        )

        class ShOnlyDetailClient(EmptyClient):
            def query(self, api_name, params=None, fields="", retries=5):
                if api_name != "margin_detail":
                    return super().query(api_name, params, fields, retries)
                self.calls.append((api_name, dict(params or {})))
                columns = fields.split(",")
                row = ["20260529" if col == "trade_date" else "600000.SH" if col == "ts_code" else "1.0" for col in columns]
                return tushare_client.ApiResult(columns, [row])

        written, zero_skipped = download.download_event_trade_date_dataset(
            ShOnlyDetailClient(), self.raw_dir, specs.EVENT_FLOW_SPECS["margin_detail"], ["20260529"], True, None,
            trading_dates=["20260529", "20260601"],
        )
        self.assertEqual((written, zero_skipped), (0, 1))
        self.assertFalse((self.raw_dir / "margin_detail" / "trade_date=20260529.parquet").exists())

        # A committed SH-only partition must be re-attempted, not skipped.
        detail_dir = self.raw_dir / "margin_detail"
        detail_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"trade_date": ["20260529"], "ts_code": ["600000.SH"], "rzye": [1.0]}).to_parquet(
            detail_dir / "trade_date=20260529.parquet", index=False
        )
        client = EmptyClient()
        written, zero_skipped = download.download_event_trade_date_dataset(
            client, self.raw_dir, specs.EVENT_FLOW_SPECS["margin_detail"], ["20260529"], False, None,
            trading_dates=["20260529", "20260601"],
        )
        self.assertEqual((written, zero_skipped), (0, 1))
        self.assertEqual([api for api, _ in client.calls], ["margin_detail"])

        findings: list[tuple[str, str, str, dict]] = []
        audit.audit_margin_exchange_completeness(
            self.raw_dir, "margin_detail", lambda sev, code, msg, details: findings.append((sev, code, msg, details))
        )
        self.assertEqual(len(findings), 1)
        severity, code, _, details = findings[0]
        self.assertEqual((severity, code), ("error", "margin_detail_exchange_completeness"))
        self.assertEqual(details["incomplete_days"], [{"trade_date": "20260529", "missing_exchanges": ["BSE", "SZSE"]}])

    def test_margin_secs_partial_exchange_day_is_refused_and_audited(self):
        # margin_secs carries its venue as a plain exchange column. The BSE
        # slice lags the vendor's SSE/SZSE feed intermittently, so a committed
        # two-exchange day must be re-attempted and audited, never frozen.
        self.assertEqual(
            common.margin_family_missing_exchanges(
                "margin_secs", "20260529", pd.DataFrame({"exchange": ["SSE", "SZSE"]})
            ),
            ["BSE"],
        )
        self.assertEqual(
            common.margin_family_missing_exchanges(
                "margin_secs", "20260529", pd.DataFrame({"exchange": ["SSE", "SZSE", "BSE"]})
            ),
            [],
        )

        class NoBseSecsClient(EmptyClient):
            def query(self, api_name, params=None, fields="", retries=5):
                if api_name != "margin_secs":
                    return super().query(api_name, params, fields, retries)
                self.calls.append((api_name, dict(params or {})))
                columns = fields.split(",")
                rows = [
                    ["20260529" if col == "trade_date" else exchange if col == "exchange" else "600000.SH" for col in columns]
                    for exchange in ("SSE", "SZSE")
                ]
                return tushare_client.ApiResult(columns, rows)

        written, zero_skipped = download.download_event_trade_date_dataset(
            NoBseSecsClient(), self.raw_dir, specs.EVENT_FLOW_SPECS["margin_secs"], ["20260529"], True, None
        )
        self.assertEqual((written, zero_skipped), (0, 1))
        self.assertFalse((self.raw_dir / "margin_secs" / "trade_date=20260529.parquet").exists())

        # A committed BSE-less partition must be re-attempted, not skipped.
        secs_dir = self.raw_dir / "margin_secs"
        secs_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"trade_date": ["20260529"] * 2, "ts_code": ["600000.SH", "000001.SZ"], "exchange": ["SSE", "SZSE"]}
        ).to_parquet(secs_dir / "trade_date=20260529.parquet", index=False)
        client = EmptyClient()
        written, zero_skipped = download.download_event_trade_date_dataset(
            client, self.raw_dir, specs.EVENT_FLOW_SPECS["margin_secs"], ["20260529"], False, None
        )
        self.assertEqual((written, zero_skipped), (0, 1))
        self.assertEqual([api for api, _ in client.calls], ["margin_secs"])

        findings: list[tuple[str, str, str, dict]] = []
        audit.audit_margin_exchange_completeness(
            self.raw_dir, "margin_secs", lambda sev, code, msg, details: findings.append((sev, code, msg, details))
        )
        self.assertEqual(len(findings), 1)
        severity, code, _, details = findings[0]
        self.assertEqual((severity, code), ("error", "margin_secs_exchange_completeness"))
        self.assertEqual(details["incomplete_days"], [{"trade_date": "20260529", "missing_exchanges": ["BSE"]}])

    def test_board_progress_counts_only_partitions_publish_wrote(self):
        spec = specs.BOARD_TRADING_SPECS["limit_list_d"]

        class LimitClient(EmptyClient):
            def query(self, api_name, params=None, fields="", retries=5):
                columns = fields.split(",")
                row = ["20200102" if col == "trade_date" else "000001.SZ" if col == "ts_code" else "U" for col in columns]
                return tushare_client.ApiResult(columns, [row])

        def run() -> str:
            out = io.StringIO()
            with redirect_stdout(out):
                download.download_board_trade_date_dataset(LimitClient(), self.raw_dir, spec, ["20200102"], True, 5000, None)
            return out.getvalue()

        self.assertIn("written=1 rows_written=1", run())
        # The same answer again: publish keeps the partition, so nothing is counted.
        self.assertIn("written=0 rows_written=0", run())

    def test_macro_static_full_writes_per_loop_registry_files(self):
        class RegistryClient(EmptyClient):
            def query(self, api_name, params=None, fields="", retries=5):
                self.calls.append((api_name, dict(params or {})))
                columns = fields.split(",")
                exchange = (params or {}).get("exchange", "")
                if exchange != "CFFEX":
                    return tushare_client.ApiResult(columns, [])
                row = ["IF2001.CFX" if col == "ts_code" else "20191018" if col == "list_date" else exchange if col == "exchange" else "x" for col in columns]
                return tushare_client.ApiResult(columns, [row])

        client = RegistryClient()
        spec = specs.MACRO_SPECS["fut_basic"]
        download.download_macro_static_full(client, self.raw_dir, spec, 10000)

        path = self.raw_dir / "fut_basic" / "exchange=CFFEX.parquet"
        self.assertTrue(path.exists())
        frame = pd.read_parquet(path)
        self.assertEqual(list(frame["ts_code"]), ["IF2001.CFX"])
        # available_at stamped from list_date so registry rows are PIT-gated.
        self.assertTrue(frame["available_at"].iloc[0].startswith("2019-10-18"))
        self.assertEqual({p.get("exchange") for _, p in client.calls}, set(spec.loop_values))

    def test_event_flow_not_ready_vetoed_by_trade_cal_refresh(self):
        # A trade_cal coverage refresh IS a lake write: exit 75 asserts "no
        # mutation" and must not fire even when every dataset was empty. The
        # run still failed to reach required coverage, so it reports 76 (commit
        # the generation, retry later) rather than success.
        self._write_trade_cal_with_stamping_window("20200102")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["margin"],
            force=True,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            zero_rows_not_ready=True,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )

        with patch.object(download, "load_token", return_value="token"), \
                patch.object(download, "TuShareClient", return_value=EmptyClient()), \
                patch.object(download, "ensure_trade_cal_coverage", return_value=True):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    download.download_event_flow(args),
                    common.MUTATED_NOT_READY_RETRY_EXIT_CODE,
                )

        self.assertIn("not_ready_after_mutation", output.getvalue())
        self.assertNotIn("not_ready_no_mutation", output.getvalue())

    def test_event_flow_extreme_shrink_is_staged_not_a_not_ready_day(self):
        # A non-empty answer removing most keys is staged for an operator; it
        # is never a "source not published yet" condition.
        self._write_trade_cal_with_stamping_window("20200102")
        path = self.raw_dir / "margin" / "trade_date=20200102.parquet"
        original = pd.DataFrame(
            [{"trade_date": "20200102", "exchange_id": f"EX{i:02d}", "rzye": 1.0} for i in range(30)]
        )
        tushare_io.write_parquet(path, original, api_name="margin", params={}, fields=list(original.columns))

        class ShrunkMarginClient(EmptyClient):
            def query(self, api_name, params=None, fields="", retries=5):
                if api_name == "margin":
                    columns = fields.split(",")
                    # Exchange-complete (SSE+SZSE pre-BSE) so the completeness
                    # guard passes and the shrink guard is what fires.
                    rows = [
                        ["20200102" if col == "trade_date" else exchange if col == "exchange_id" else 1.0 for col in columns]
                        for exchange in ("SSE", "SZSE")
                    ]
                    return tushare_client.ApiResult(columns, rows)
                return super().query(api_name, params, fields, retries)

        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["margin"],
            force=True,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            zero_rows_not_ready=True,
            revision_ledger=str(self.root / "shrink_revision_events.jsonl"),
        )

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=ShrunkMarginClient()):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(download.download_event_flow(args), 0)

        assert_staged(path, original, self.root / "shrink_revision_events.jsonl")

    def test_event_flow_zero_rows_not_ready_partial_write_reports_not_ready(self):
        self._write_trade_cal_with_stamping_window("20200102")

        class MarginOnlyClient(EmptyClient):
            def query(self, api_name, params=None, fields="", retries=5):
                if api_name == "margin":
                    columns = fields.split(",")
                    # Both pre-BSE required exchanges: the day is complete and commits.
                    rows = [
                        ["20200102" if col == "trade_date" else exchange if col == "exchange_id" else 1.0 for col in columns]
                        for exchange in ("SSE", "SZSE")
                    ]
                    return tushare_client.ApiResult(columns, rows)
                return super().query(api_name, params, fields, retries)

        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["margin", "margin_detail"],
            force=True,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            zero_rows_not_ready=True,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=MarginOnlyClient()):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    download.download_event_flow(args),
                    common.MUTATED_NOT_READY_RETRY_EXIT_CODE,
                )

        # margin committed, margin_detail still unpublished: the generation is
        # real, so this is never exit 75 -- but it is never success either, or
        # skip_if_already_ok would abandon margin_detail for this date.
        self.assertIn("not_ready_after_mutation", output.getvalue())
        self.assertTrue((self.raw_dir / "margin" / "trade_date=20200102.parquet").exists())
        self.assertFalse((self.raw_dir / "margin_detail" / "trade_date=20200102.parquet").exists())

    def test_board_trading_download_and_audit_use_dedicated_dimension(self):
        self._write_trade_cal("20231101")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20231101",
            end_date="20231101",
            datasets=["kpl_list", "limit_step", "limit_list_ths", "top_list", "hm_list", "ths_hot", "dc_hot"],
            force=False,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            kpl_tag=["涨停"],
            ths_limit_type=["涨停池"],
            ths_hot_market=["热股"],
            dc_hot_market=["A股市场"],
            dc_hot_type=["人气榜"],
            hot_is_new=["N"],
            revision_ledger=common.REVISION_EVENTS_PATH,
        )

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=FieldClient(BOARD_FIELDS, default=1.0)):
            self.assertEqual(download.download_board_trading(args), 0)

        self.assertTrue((self.raw_dir / "kpl_list" / f"tag={common.safe_partition_value('涨停')}" / "trade_date=20231101.parquet").exists())
        self.assertTrue((self.raw_dir / "limit_list_ths" / f"limit_type={common.safe_partition_value('涨停池')}" / "trade_date=20231101.parquet").exists())
        hot = pd.read_parquet(self.raw_dir / "ths_hot" / f"market={common.safe_partition_value('热股')}" / "is_new=N" / "trade_date=20231101.parquet")
        self.assertEqual(hot.loc[0, "available_at"], "2020-01-02 10:00:00+08:00")

        # limit_list_d is a regular board dataset: give it a valid partition
        # with the PIT stamp columns.
        limit_rows = pd.DataFrame([{
            "trade_date": "20231101", "ts_code": "000001.SZ", "limit": "U",
            "available_at": "2023-11-01 16:00:00+08:00",
            "available_at_rule": "official_16_from:trade_date",
        }])
        tushare_io.write_parquet(
            self.raw_dir / "limit_list_d" / "trade_date=20231101.parquet",
            limit_rows,
            api_name="limit_list_d",
            params={"trade_date": "20231101"},
            fields=list(limit_rows.columns),
        )
        status_path = self.root / "board_status.json"
        audit_args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20231101",
            end_date="20231101",
            datasets=["kpl_list", "limit_step", "limit_list_d", "limit_list_ths", "top_list", "hm_list", "ths_hot", "dc_hot"],
            kpl_tag=["涨停"],
            ths_limit_type=["涨停池"],
            ths_hot_market=["热股"],
            dc_hot_market=["A股市场"],
            dc_hot_type=["人气榜"],
            hot_is_new=["N"],
            output=str(status_path),
            revision_ledger=common.REVISION_EVENTS_PATH,
        )
        self.assertEqual(audit.audit_board_trading_only(audit_args), 0)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "ok")
        self.assertIn("kpl_list", status["datasets"])
        self.assertIn("limit_list_d", status["datasets"])
        self.assertEqual(status["datasets"]["limit_list_d"]["status"], "ok")

    def test_board_static_hm_list_refreshes_every_run(self):
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20231101",
            end_date="20231101",
            datasets=["hm_list"],
            force=False,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )
        client = FieldClient(BOARD_FIELDS, default=1.0)
        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            self.assertEqual(download.download_board_trading(args), 0)
            self.assertTrue((self.raw_dir / "hm_list" / "hm_list.parquet").exists())
            # The reference table must re-pull on every run, not skip once downloaded.
            self.assertEqual(download.download_board_trading(args), 0)
        self.assertEqual([api for api, _ in client.calls].count("hm_list"), 2)

    def test_cjk_partitioned_board_rerun_reuses_the_same_tree(self):
        """A second unforced run must skip, not fetch into a parallel tree."""
        self._write_trade_cal("20231101")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20231101",
            end_date="20231101",
            datasets=["ths_hot", "kpl_list"],
            force=False,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            kpl_tag=["涨停"],
            ths_hot_market=["热股"],
            hot_is_new=["N"],
            revision_ledger=common.REVISION_EVENTS_PATH,
        )
        client = FieldClient(BOARD_FIELDS, default=1.0)
        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            self.assertEqual(download.download_board_trading(args), 0)
            self.assertEqual(download.download_board_trading(args), 0)
        apis = [api for api, _ in client.calls]
        self.assertEqual(apis.count("ths_hot"), 1)
        self.assertEqual(apis.count("kpl_list"), 1)
        self.assertEqual([path.name for path in sorted((self.raw_dir / "ths_hot").iterdir())], ["market=%E7%83%AD%E8%82%A1"])
        self.assertEqual([path.name for path in sorted((self.raw_dir / "kpl_list").iterdir())], ["tag=%E6%B6%A8%E5%81%9C"])

    def test_board_trading_skips_non_trading_window(self):
        self._write_trade_cal("20260530", is_open="0")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20260530",
            end_date="20260530",
            datasets=["kpl_list", "limit_step", "limit_cpt_list"],
            force=True,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            kpl_tag=["涨停"],
            ths_limit_type=["涨停池"],
            ths_hot_market=["热股"],
            dc_hot_market=["A股市场"],
            dc_hot_type=["人气榜"],
            hot_is_new=["N"],
            revision_ledger=common.REVISION_EVENTS_PATH,
        )

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=NoQueryClient()):
            self.assertEqual(download.download_board_trading(args), 0)

    def test_same_day_empty_cctv_fetch_does_not_block_the_evening_refresh(self):
        # The 08:55 pre-open window now includes day D. cctv_news airs in the
        # evening, so the morning fetch of day D legitimately returns zero
        # rows; the zero-row partition it writes must not block the evening
        # --force refresh from landing the real transcript.
        spec = specs.TEXT_SPECS["cctv_news"]
        path = self.raw_dir / "cctv_news" / "date=20260809.parquet"

        class DayClient:
            def __init__(self, rows):
                self.rows = rows

            def query(self, api_name, params=None, fields="", retries=5):
                return tushare_client.ApiResult(fields.split(","), list(self.rows))

        with redirect_stdout(io.StringIO()):
            download.download_text_day(DayClient([]), self.raw_dir, spec, ["20260809"], True)
        self.assertEqual(tushare_io.parquet_rows(path), 0)

        with redirect_stdout(io.StringIO()):
            download.download_text_day(DayClient([["20260809", "标题", "内容"]]), self.raw_dir, spec, ["20260809"], True)
        frame = pd.read_parquet(path)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "title"], "标题")


class ManualProductionWriteRefusalTest(unittest.TestCase):
    """The cron runner is the ONLY production-lake write entrance (fence,
    per-job state, alerting, flock). Direct manual commands against the
    production data/raw are refused with the runner equivalent; cron children
    (TUSHARE_UPDATE_LOCK_HELD=1) and non-production raw dirs pass through."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        (self.root / "data" / "raw").mkdir(parents=True)
        self._old_cwd = os.getcwd()
        os.chdir(self.root)
        self._env = patch.dict(os.environ)
        self._env.start()
        os.environ.pop("TUSHARE_UPDATE_LOCK_HELD", None)

    def tearDown(self) -> None:
        self._env.stop()
        os.chdir(self._old_cwd)
        self._tmp.cleanup()

    def _run(self, argv: list[str], impl_name: str, impl_result) -> int:
        with (
            patch.object(sys, "argv", ["tushare_download.py", *argv]),
            patch.object(download, "_production_repo_root", lambda: self.root),
            patch.object(download, impl_name, return_value=impl_result),
            redirect_stdout(io.StringIO()),
        ):
            return download.main()

    def test_manual_production_write_is_refused_with_runner_equivalent(self) -> None:
        with self.assertRaisesRegex(SystemExit, "cron runner|tushare_cron_update"):
            self._run(["update", "--start-date", "20260101", "--end-date", "20260102"], "update_data", 0)
        self.assertFalse((self.root / "data" / "raw" / ".raw_generation.json").exists())

    def test_the_guard_covers_the_protected_tree_and_nothing_beside_it(self) -> None:
        protected = self.root / "data" / "pit" / "fundamental_events"
        for target, refused in (
            (protected, True),
            (protected / "dividend", True),
            (self.root / "data" / "pit" / "fundamental_events_dev", False),
            (self.root / "elsewhere", False),
            (None, False),
        ):
            self.assertEqual(tushare_lock.manual_production_write(self.root, protected, target), refused, target)

    def test_genuine_cron_child_with_inherited_lock_fd_passes(self) -> None:
        # Only an INHERITED, exclusively-held lock fd proves a runner child:
        # env flag + "somebody holds the lock" was forgeable by any manual run.
        import fcntl

        from autotrade.data_sources.tushare.lock import UPDATER_LOCK_FD_ENV

        lock_path = self.root / ".runtime" / "tushare" / "locks" / "tushare_update.lock"
        lock_path.parent.mkdir(parents=True)
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.environ[UPDATER_LOCK_FD_ENV] = str(fd)
            code = self._run(["update", "--start-date", "20260101", "--end-date", "20260102"], "update_data", 0)
            self.assertEqual(code, 0)
        finally:
            os.close(fd)

    def test_forged_marker_without_an_inherited_lock_fd_is_refused(self) -> None:
        import fcntl

        from autotrade.data_sources.tushare.lock import UPDATER_LOCK_FD_ENV

        lock_path = self.root / ".runtime" / "tushare" / "locks" / "tushare_update.lock"
        lock_path.parent.mkdir(parents=True)
        # (a) no fd at all
        os.environ[UPDATER_LOCK_FD_ENV] = "1"
        with self.assertRaisesRegex(SystemExit, "cron runner|tushare_cron_update"):
            self._run(["update", "--start-date", "20260101", "--end-date", "20260102"], "update_data", 0)
        # (b) an fd for the RIGHT file but nobody holds the lock exclusively
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)
        try:
            os.environ[UPDATER_LOCK_FD_ENV] = str(fd)
            with self.assertRaisesRegex(SystemExit, "cron runner|tushare_cron_update"):
                self._run(["update", "--start-date", "20260101", "--end-date", "20260102"], "update_data", 0)
        finally:
            os.close(fd)
        # (c) an fd pointing at an unrelated file, lock genuinely held elsewhere
        other = self.root / "unrelated.bin"
        other.write_bytes(b"x")
        holder = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)
        decoy = os.open(str(other), os.O_RDONLY)
        try:
            fcntl.flock(holder, fcntl.LOCK_EX)
            os.environ[UPDATER_LOCK_FD_ENV] = str(decoy)
            with self.assertRaisesRegex(SystemExit, "cron runner|tushare_cron_update"):
                self._run(["update", "--start-date", "20260101", "--end-date", "20260102"], "update_data", 0)
        finally:
            os.close(decoy)
            os.close(holder)

    def test_non_production_raw_dir_and_union_output_are_unaffected(self) -> None:
        other = self.root / "elsewhere" / "raw"
        other.mkdir(parents=True)
        code = self._run(
            ["update", "--start-date", "20260101", "--end-date", "20260102",
             "--raw-dir", str(other), "--union-output", str(other / "share_float_complete.parquet")],
            "update_data", 0,
        )
        self.assertEqual(code, 0)

    def test_production_union_output_is_refused_even_with_foreign_raw_dir(self) -> None:
        # --raw-dir elsewhere but --union-output under the production lake:
        # still a production write (the previously missed bypass).
        other = self.root / "elsewhere" / "raw"
        other.mkdir(parents=True)
        with self.assertRaisesRegex(SystemExit, "cron runner|tushare_cron_update"):
            self._run(
                ["update", "--start-date", "20260101", "--end-date", "20260102",
                 "--raw-dir", str(other),
                 "--union-output", str(self.root / "data" / "raw" / "share_float_complete" / "x.parquet")],
                "update_data", 0,
            )

    def test_production_raw_subdirectory_is_refused(self) -> None:
        with self.assertRaisesRegex(SystemExit, "cron runner|tushare_cron_update"):
            self._run(
                ["update", "--start-date", "20260101", "--end-date", "20260102",
                 "--raw-dir", str(self.root / "data" / "raw" / "daily")],
                "update_data", 0,
            )


def seed_partition(path: Path, columns: dict[str, list[str]], api_name: str) -> pd.DataFrame:
    """A committed partition holding ``columns``."""
    original = pd.DataFrame(columns)
    tushare_io.write_parquet(path, original, api_name=api_name, params={}, fields=list(original.columns))
    return original


def assert_staged(path: Path, original: pd.DataFrame, ledger: Path) -> None:
    """An answer removing most keys stays out of the visible partition: it is a
    staged version beside it, and the ledger names it."""
    assert pd.read_parquet(path).equals(original)
    staged = [version for version in versions.list_versions(path) if version.endswith("-staged")]
    assert len(staged) == 1
    events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert [(event["kind"], event["version"]) for event in events] == [("staged", staged[0])]


class AnswerClient(EmptyClient):
    """A vendor answering ``api_name`` with ``frame`` and anything else with
    no rows."""

    def __init__(self, api_name: str, frame: pd.DataFrame):
        super().__init__()
        self.api_name, self.frame = api_name, frame

    def query(self, api_name, params=None, fields="", retries=5):
        answer = super().query(api_name, params, fields, retries)
        if api_name != self.api_name:
            return answer
        return tushare_client.ApiResult(list(self.frame.columns), self.frame.values.tolist())


def run_tier(tier, client: EmptyClient, raw_dir: Path, ledger: Path, dataset: str, **extra):
    """One tier download of ``dataset`` on 20200102, forced, through the vendor ``client``."""
    args = argparse.Namespace(**{
        "raw_dir": str(raw_dir), "start_date": "20200102", "end_date": "20200102", "datasets": [dataset],
        "revision_ledger": str(ledger), "force": True, "page_limit": None, "min_interval_seconds": 0, "timeout_seconds": 1,
        **extra,
    })
    with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
        return tier(args)


def _codes(count: int) -> list[str]:
    return [f"{index:06d}.SZ" for index in range(count)]


# One row per landing loop: the dataset, the partition an interrupted write
# left without its commit identity, what it holds, and the pull (returning
# what that loop returns).
UNCOMMITTED_PARTITIONS = [
    pytest.param(
        "margin_secs", "trade_date=20260601",
        {"trade_date": ["20260601"] * 3, "ts_code": ["600000.SH", "000001.SZ", "830000.BJ"], "exchange": ["SSE", "SZSE", "BSE"]},
        lambda client, raw_dir: download.download_event_trade_date_dataset(
            client, raw_dir, specs.EVENT_FLOW_SPECS["margin_secs"], ["20260601"], False, None
        ),
        (1, 0), id="event_trade_date",
    ),
    pytest.param(
        "daily", "trade_date=20200102", {"trade_date": ["20200102"], "ts_code": ["000001.SZ"]},
        lambda client, raw_dir: download.download_trade_date_dataset(client, raw_dir, specs.DAILY_SPECS["daily"], ["20200102"], False, 5000, None),
        0, id="daily",
    ),
    pytest.param(
        "limit_list_d", "trade_date=20200102", {"trade_date": ["20200102"], "ts_code": ["000001.SZ"], "limit": ["U"]},
        lambda client, raw_dir: download.download_board_trade_date_dataset(
            client, raw_dir, specs.BOARD_TRADING_SPECS["limit_list_d"], ["20200102"], False, 5000, None
        ),
        None, id="board_trade_date",
    ),
    pytest.param(
        "cctv_news", "date=20200102", {"date": ["20200102"], "title": ["t"], "content": ["c"]},
        lambda client, raw_dir: download.download_text_day(client, raw_dir, specs.TEXT_SPECS["cctv_news"], ["20200102"], False),
        None, id="text_day",
    ),
    pytest.param(
        "sw_daily", "trade_date=20200102",
        {name: ["801010.SI" if name == "ts_code" else "20200102" if name == "trade_date" else "1"] for name in specs.MACRO_SPECS["sw_daily"].fields.split(",")},
        lambda client, raw_dir: download.download_macro_trade_date(client, raw_dir, specs.MACRO_SPECS["sw_daily"], "20200102", "20200102", False, 5000),
        None, id="macro_trade_date",
    ),
    pytest.param(
        "income_vip", "period=20200331",
        dict(zip(specs.FUNDAMENTAL_SPECS["income_vip"].key_columns, [["000001.SZ"], ["20200430"], ["20200430"], ["20200331"], ["1"], ["1"], ["1"]], strict=True)),
        lambda client, raw_dir: download.download_fundamental_period_dataset(
            client, raw_dir, specs.FUNDAMENTAL_SPECS["income_vip"], ["20200331"], False, 5000
        ),
        None, id="fundamental_period",
    ),
    # The reference tier's two loops that no scheduled run rewrites whole:
    # bak_basic before the update window and the frozen SW2014 membership.
    pytest.param(
        "bak_basic", "trade_date=20200102", {"trade_date": ["20200102"], "ts_code": ["000001.SZ"], "name": ["x"]},
        lambda client, raw_dir: download.download_bak_basic(client, raw_dir, ["20200102"], "20200102", False, None),
        None, id="bak_basic",
    ),
    pytest.param(
        "index_member", "l1_code=801010.SI",
        {"index_code": ["801010.SI"], "con_code": ["000001.SZ"], "in_date": ["20140101"], "out_date": ["20211213"], "is_new": ["N"]},
        lambda client, raw_dir: download.download_index_member_history(
            client, raw_dir, pd.DataFrame([{"index_code": "801010.SI", "level": "L1"}]), False, ledger=None
        ),
        None, id="index_member",
    ),
]


@pytest.mark.parametrize(("api_name", "partition", "held", "pull", "returned"), UNCOMMITTED_PARTITIONS)
def test_a_partition_without_its_commit_identity_is_pulled_again(tmp_path, api_name, partition, held, pull, returned):
    # An interrupted write leaves a parquet without (or with a stale) sidecar;
    # treating it as committed froze such partitions forever. Every landing
    # loop pulls it again (DATA_DOCUMENTATION §2.3), and the vendor's
    # identical answer commits the pair.
    raw_dir = tmp_path / "raw"
    write_trade_cal(raw_dir)
    path = raw_dir / api_name / f"{partition}.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(held).to_parquet(path, index=False)  # NO sidecar
    assert not tushare_io.committed_partition_intact(path)
    assert not download.should_skip_existing_partition(path, force=False)
    client = AnswerClient(api_name, pd.DataFrame(held))

    def run():
        client.calls.clear()
        with redirect_stdout(io.StringIO()):
            return pull(client, raw_dir)

    assert run() == returned
    assert [api for api, _ in client.calls] == [api_name]
    assert tushare_io.committed_partition_intact(path)
    # Once the pair is intact, the same covering run skips it.
    run()
    assert client.calls == []
    # A sidecar naming another write is an interrupted commit as well.
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    meta["write_id"] = "00000000-0000-0000-0000-000000000000"
    sidecar.write_text(json.dumps(meta), encoding="utf-8")
    assert not tushare_io.committed_partition_intact(path)
    run()
    assert [api for api, _ in client.calls] == [api_name]
    assert tushare_io.committed_partition_intact(path)
    # So is a parquet replaced under its sidecar: its footer no longer matches.
    pd.read_parquet(path).iloc[:2].to_parquet(path, index=False)
    assert not tushare_io.committed_partition_intact(path)


def _pull_zero_rows_ok_event(client, raw_dir, ledger):
    # Tolerating an EMPTY day says nothing about accepting an answer that
    # removes most of a partition.
    assert specs.EVENT_FLOW_SPECS["cyq_perf"].zero_rows_ok
    return run_tier(download.download_event_flow, client, raw_dir, ledger, "cyq_perf", zero_rows_not_ready=True)


# One row per landing loop: the dataset, its partition holding many keys, how
# many of them the forced pull's answer keeps, the pull, and what it returns
# (the board run goes on; the range loop counts nothing written).
SHRUNK_ANSWERS = [
    pytest.param(
        "limit_list_d", "trade_date=20200102", {"trade_date": ["20200102"] * 30, "ts_code": _codes(30), "limit": ["U"] * 30}, 2,
        lambda client, raw_dir, ledger: run_tier(download.download_board_trading, client, raw_dir, ledger, "limit_list_d", page_limit=10000),
        0, id="board_trade_date",
    ),
    pytest.param(
        "cyq_perf", "trade_date=20200102", {"trade_date": ["20200102"] * 30, "ts_code": _codes(30), "winner_rate": ["1"] * 30}, 2,
        _pull_zero_rows_ok_event, ANY, id="event_trade_date_zero_rows_ok",
    ),
    pytest.param(
        "repurchase", "month=202001",
        {"ts_code": _codes(30), "ann_date": ["20200115"] * 30, "end_date": [""] * 30, "proc": [""] * 30, "exp_date": [""] * 30}, 2,
        lambda client, raw_dir, ledger: download.download_event_range_month(
            client, raw_dir, specs.EVENT_FLOW_SPECS["repurchase"], [("20200101", "20200131", "202001")], True, 5000, str(ledger),
        ),
        0, id="event_range_month",
    ),
    pytest.param(
        "bak_basic", "trade_date=20200102", {"trade_date": ["20200102"] * 30, "ts_code": _codes(30), "name": ["x"] * 30}, 2,
        lambda client, raw_dir, ledger: download.download_bak_basic(client, raw_dir, ["20200102"], "20200102", True, ledger),
        None, id="bak_basic",
    ),
    pytest.param(
        "fina_mainbz_vip", "ts_code=000001.SZ",
        {**{col: [f"{col}{i}" for i in range(40)] for col in specs.FUNDAMENTAL_SPECS["fina_mainbz_vip"].key_columns}, "bz_sales": [1.0] * 40}, 3,
        lambda client, raw_dir, ledger: download.download_fundamental_ts_code_dataset(
            client, raw_dir, specs.FUNDAMENTAL_SPECS["fina_mainbz_vip"], ["000001.SZ"], True, 5000, revision_ledger=ledger,
        ),
        None, id="fundamental_ts_code",
    ),
]


@pytest.mark.parametrize(("api_name", "partition", "held", "kept", "pull", "returned"), SHRUNK_ANSWERS)
def test_an_answer_removing_most_keys_is_staged(tmp_path, api_name, partition, held, kept, pull, returned):
    # The transient-truncation shape the destructive-shrink guard exists for.
    raw_dir = tmp_path / "raw"
    write_trade_cal(raw_dir)
    path = raw_dir / api_name / f"{partition}.parquet"
    original = seed_partition(path, held, api_name)
    ledger = tmp_path / "revision_events.jsonl"
    with redirect_stdout(io.StringIO()):
        assert pull(AnswerClient(api_name, original.iloc[:kept]), raw_dir, ledger) == returned
    assert_staged(path, original, ledger)


if __name__ == "__main__":
    unittest.main()
