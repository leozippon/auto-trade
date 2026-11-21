"""The registry and the shared helpers: calendar, ledger path, partition names
(data_sources/tushare/specs.py, common.py).
"""

import json
import unittest
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

from autotrade.data_sources.tushare import common, specs

from .tushare_fixture import FieldClient, LakeTestCase, param

# One SSE calendar day: the end of the requested range, open.
TRADE_CAL_FIELDS = {
    "exchange": param("exchange", ""),
    "cal_date": param("end_date", "20260604"),
    "is_open": "1",
    "pretrade_date": "20260603",
}


class CommonTest(LakeTestCase):
    def test_default_revision_ledger_for_temp_raw_stays_local(self):
        ledger = common.resolve_revision_ledger(self.raw_dir, common.REVISION_EVENTS_PATH, repo_root=Path.cwd())

        self.assertEqual(ledger, self.root / "revision_events.jsonl")

    def test_load_stock_codes_keeps_only_valid_a_share_codes(self):
        path = self.raw_dir / "stock_basic" / "list_status=L.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ"},
                {"ts_code": "T00018.SH"},
                {"ts_code": "TS0018.SH"},
                {"ts_code": "920126.BJ"},
                {"ts_code": "bad"},
            ]
        ).to_parquet(path, index=False)

        self.assertEqual(common.load_stock_codes(self.raw_dir), ["000001.SZ", "920126.BJ"])

    def test_trade_cal_helpers_normalize_date_strings(self):
        path = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"cal_date": "2026-06-03", "is_open": "1"},
                {"cal_date": "20260604", "is_open": "0"},
                {"cal_date": "2026/06/05", "is_open": "1"},
            ]
        ).to_parquet(path, index=False)

        self.assertEqual(common.load_sse_open_dates(self.raw_dir, "20260603", "20260605"), ["20260603", "20260605"])
        self.assertEqual(common.latest_sse_calendar_date(self.raw_dir), "20260605")
        self.assertTrue(common.sse_trade_cal_covers(self.raw_dir, "20260603", "20260605"))

    def test_concat_rows_preserves_schema_when_all_inputs_are_empty(self):
        from autotrade.environment.data.pit import concat_rows

        left = pd.DataFrame({"ts_code": pd.Series(dtype=object), "close": pd.Series(dtype="float64")})
        right = pd.DataFrame({"ts_code": pd.Series(dtype=object), "volume": pd.Series(dtype="int64")})
        merged = concat_rows([left, right])
        self.assertEqual(list(merged.columns), ["ts_code", "close", "volume"])
        self.assertEqual(str(merged["close"].dtype), "float64")
        self.assertEqual(len(merged), 0)

        # Mixed inputs: an empty frame's extra columns still enter the schema.
        rows = pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.0]})
        widened = concat_rows([rows, right])
        self.assertIn("volume", widened.columns)
        self.assertEqual(len(widened), 1)
        self.assertTrue(widened["volume"].isna().all())

    def test_trade_cal_force_refresh_merges_without_shrinking_year_partition(self):
        path = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"exchange": "SSE", "cal_date": "20260603", "is_open": "1", "pretrade_date": "20260602"},
        ]).to_parquet(path, index=False)
        client = FieldClient(TRADE_CAL_FIELDS, apis={"trade_cal"})

        open_dates = common.download_trade_cal(client, self.raw_dir, "20260604", "20260604", force=True, ledger=None)

        refreshed = pd.read_parquet(path)
        self.assertEqual(sorted(refreshed["cal_date"].astype(str).tolist()), ["20260603", "20260604"])
        self.assertIn("20260604", open_dates)

    def test_trade_cal_past_day_revision_is_refused(self):
        path = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2020.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "exchange": ["SSE", "SSE"], "cal_date": ["20200102", "20200103"],
            "is_open": ["1", "1"], "pretrade_date": ["20200101", "20200102"],
        }).to_parquet(path, index=False)
        flipped = pd.DataFrame({
            "exchange": ["SSE", "SSE"], "cal_date": ["20200102", "20200103"],
            "is_open": ["0", "1"], "pretrade_date": ["20200101", "20200102"],
        })
        with self.assertRaises(RuntimeError) as ctx:
            common.merge_trade_cal_partition(path, flipped)
        self.assertIn("already-elapsed", str(ctx.exception))
        # Future-dated schedule publication merges freely.
        future = pd.DataFrame({
            "exchange": ["SSE"], "cal_date": ["29991231"], "is_open": ["0"], "pretrade_date": [""],
        })
        merged = common.merge_trade_cal_partition(path, future)
        self.assertEqual(len(merged), 3)

    def test_margin_exchange_check_fails_loudly_without_its_column(self):
        # Silently skipping an unverifiable partition is the failure mode this
        # guard exists to prevent, so an absent column must raise.
        with self.assertRaises(RuntimeError):
            common.margin_family_missing_exchanges(
                "margin_detail", "20260529", pd.DataFrame({"rzye": [1.0]})
            )

    def test_partition_names_are_reversible_and_never_truncate(self):
        """`safe_partition_value` is the single raw-lake naming authority.

        The pre-2026-09 sanitiser truncated `A股市场` to `A` and hex-digested
        all-CJK values, so post-2026-08-14 board downloads landed in a second
        tree that `should_skip_existing_partition` and the audit's expected-path
        builder could not see. Percent-encoding must stay reversible (the name
        decodes back to the vendor value) and injective (no two values share a
        directory), and must keep reproducing the names the lake already holds.
        """
        values = [
            *specs.BOARD_KPL_TAGS,
            *specs.BOARD_THS_LIMIT_TYPES,
            *specs.BOARD_THS_HOT_MARKETS,
            *specs.BOARD_DC_HOT_MARKETS,
            *specs.BOARD_DC_HOT_TYPES,
            *specs.NEWS_SOURCES,
            "000001.SH",
            "CFFEX",
            "all",
        ]
        names = [common.safe_partition_value(value) for value in values]
        self.assertEqual(len(set(names)), len(set(values)))
        for value, name in zip(values, names):
            self.assertEqual(unquote(name), value)
            self.assertNotIn("/", name)
        # Pinned against the history on disk (data docs §4).
        self.assertEqual(common.safe_partition_value("热股"), "%E7%83%AD%E8%82%A1")
        self.assertEqual(common.safe_partition_value("A股市场"), "A%E8%82%A1%E5%B8%82%E5%9C%BA")
        self.assertEqual(common.safe_partition_value("涨停"), "%E6%B6%A8%E5%81%9C")
        # ASCII values that need no escaping keep their plain directory name.
        self.assertEqual(common.safe_partition_value("000001.SH"), "000001.SH")
        self.assertEqual(common.safe_partition_value("  "), "empty")


class FetchStrategyTest(unittest.TestCase):
    def test_a_spec_holds_its_strategy_as_the_member_its_family_dispatches_on(self):
        spec = specs.DatasetSpec("income_vip", strategy="period")

        self.assertIs(spec.strategy, specs.FetchStrategy.PERIOD)
        # Recorded by value: the audit report and every message keep the string.
        self.assertEqual(json.dumps({"strategy": spec.strategy}), '{"strategy": "period"}')
        self.assertIs(specs.DatasetSpec("daily").strategy, specs.FetchStrategy.DAILY_FAMILY)

    def test_a_spec_naming_no_strategy_is_refused_when_built(self):
        with self.assertRaisesRegex(ValueError, "is not a valid FetchStrategy"):
            specs.DatasetSpec("income_vip", strategy="perod")


if __name__ == "__main__":
    unittest.main()
