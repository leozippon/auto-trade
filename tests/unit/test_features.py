# Consolidated unit tests: test_environment.py


# Source: test_auction_correction.py
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from autotrade.environment.data import PITDataStore
from autotrade.environment.data import (
    FundamentalEventsBuilder,
    FundamentalEventsConfig,
    audit_fundamental_events,
    month_aligned_replace_window,
)
from autotrade.environment.data.auction import (
    AuctionCorrectionConfig,
    apply_open_auction_correction,
    is_open_auction_time,
    market_bucket,
)
from autotrade.environment.data.pit import CorruptSidecarError


class AuctionCorrectionTest(unittest.TestCase):
    def test_market_bucket(self):
        self.assertEqual(market_bucket("000001.SZ"), "sz_main_00")
        self.assertEqual(market_bucket("300001.SZ"), "sz_gem_30")
        self.assertEqual(market_bucket("600000.SH"), "sh_main_60")
        self.assertEqual(market_bucket("688001.SH"), "sh_star_68")
        self.assertEqual(market_bucket("430001.BJ"), "bj")

    def test_apply_open_auction_correction_only_adjusts_sz_0930(self):
        frame = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_time": "2026-05-29 09:30:00", "vol": 1000.0, "amount": 2000.0},
                {"ts_code": "300001.SZ", "trade_time": "2026-05-29 09:30:00", "vol": 1000.0, "amount": 2000.0},
                {"ts_code": "600000.SH", "trade_time": "2026-05-29 09:30:00", "vol": 1000.0, "amount": 2000.0},
                {"ts_code": "000001.SZ", "trade_time": "2026-05-29 15:00:00", "vol": 1000.0, "amount": 2000.0},
            ]
        )

        out = apply_open_auction_correction(frame)

        self.assertAlmostEqual(out.loc[0, "vol_pit"], 760.0)
        self.assertAlmostEqual(out.loc[0, "amount_pit"], 1520.0)
        self.assertAlmostEqual(out.loc[1, "vol_pit"], 580.0)
        self.assertAlmostEqual(out.loc[1, "amount_pit"], 1160.0)
        self.assertAlmostEqual(out.loc[2, "vol_pit"], 1000.0)
        self.assertAlmostEqual(out.loc[3, "vol_pit"], 1000.0)
        self.assertEqual(out.loc[0, "auction_correction_rule"], "minute_0930_to_live_stk_auction_by_market_bucket")
        self.assertEqual(out.loc[2, "auction_correction_rule"], "none")

    def test_vectorized_auction_correction_matches_scalar_contract(self):
        frame = pd.DataFrame(
            {
                "ts_code": [" 000001.sz ", "300001.SZ", "600000.SH", "688001.SH", "430001.BJ", None, 0],
                "trade_time": [
                    " 2026-05-29 09:30:00 ",
                    "2026-05-29 09:31:00",
                    "09:30",
                    None,
                    "2026-05-29 09:30:59",
                    "2026-05-29 09:30:00",
                    930,
                ],
                "vol": ["100", "bad", 300, None, 500, 600, 700],
                "amount": [200, 400, "600", 800, None, 1200, 1400],
            },
            index=[9, 7, 5, 3, 1, -1, -3],
        )
        original = frame.copy(deep=True)
        config = AuctionCorrectionConfig(
            volume_factors={"sz_main_00": 0.5, "sh_main_60": 0.25, "other": 0.9},
            amount_factors={"sz_main_00": 0.75, "bj": 0.1, "other": 0.8},
        )

        out = apply_open_auction_correction(frame, config)

        expected_buckets = frame["ts_code"].map(market_bucket)
        expected_open = frame["trade_time"].map(is_open_auction_time)
        expected_vol_factors = pd.Series(
            [config.volume_factors.get(bucket, 1.0) if opened else 1.0 for bucket, opened in zip(expected_buckets, expected_open)],
            index=frame.index,
        )
        expected_amount_factors = pd.Series(
            [config.amount_factors.get(bucket, 1.0) if opened else 1.0 for bucket, opened in zip(expected_buckets, expected_open)],
            index=frame.index,
        )
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_series_equal(out["auction_market_bucket"], expected_buckets, check_names=False)
        pd.testing.assert_series_equal(out["auction_open_bar"], expected_open.astype(bool), check_names=False)
        pd.testing.assert_series_equal(out["auction_vol_correction_factor"], expected_vol_factors, check_names=False)
        pd.testing.assert_series_equal(out["auction_amount_correction_factor"], expected_amount_factors, check_names=False)
        pd.testing.assert_series_equal(
            out["vol_pit"], pd.to_numeric(frame["vol"], errors="coerce") * expected_vol_factors, check_names=False
        )
        pd.testing.assert_series_equal(
            out["amount_pit"], pd.to_numeric(frame["amount"], errors="coerce") * expected_amount_factors, check_names=False
        )



# Source: test_pit_store.py
class PITDataStoreTest(unittest.TestCase):
    def test_pit_store_handles_reversed_ranges_without_partition_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PITDataStore(Path(tmp))
            frame = store.read_trade_range("daily", "20200103", "20200101", columns=["trade_date", "ts_code"])
            self.assertTrue(frame.empty)
            self.assertEqual(list(frame.columns), ["trade_date", "ts_code"])


class FundamentalEventsBuilderTest(unittest.TestCase):
    def test_builds_available_month_events_and_audit_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            raw.mkdir()
            income_dir = raw / "income_vip"
            dividend_dir = raw / "dividend"
            mainbz_dir = raw / "fina_mainbz_vip"
            income_dir.mkdir()
            dividend_dir.mkdir()
            mainbz_dir.mkdir()
            pd.DataFrame([
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20200103",
                    "f_ann_date": "20200102",
                    "end_date": "20191231",
                    "report_type": "1",
                    "comp_type": "1",
                    "end_type": "4",
                }
            ]).to_parquet(income_dir / "period=20191231.parquet", index=False)
            pd.DataFrame([
                {
                    "ts_code": "000001.SZ",
                    "end_date": "20191231",
                    "ann_date": "",
                    "imp_ann_date": "20200104",
                    "ex_date": "20200110",
                    "record_date": "20200109",
                    "pay_date": "20200111",
                    "div_proc": "实施",
                    "cash_div_tax": 0.1,
                },
                {
                    "ts_code": "000002.SZ",
                    "end_date": "20191231",
                    "ann_date": "",
                    "imp_ann_date": "",
                    "ex_date": "20200110",
                    "record_date": "20200109",
                    "pay_date": "20200111",
                    "div_proc": "实施",
                    "cash_div_tax": 0.2,
                },
            ]).to_parquet(dividend_dir / "ts_code=000001.SZ.parquet", index=False)
            pd.DataFrame([
                {
                    "ts_code": "000001.SZ",
                    "end_date": "20191231",
                    "bz_item": "产品A",
                    "bz_code": "A",
                    "curr_type": "CNY",
                }
            ]).to_parquet(mainbz_dir / "ts_code=000001.SZ.parquet", index=False)

            builder = FundamentalEventsBuilder(raw)
            events = builder.build(FundamentalEventsConfig(
                start_date="20200101",
                end_date="20200131",
                datasets=("income_vip", "dividend", "fina_mainbz_vip"),
            ))

            self.assertEqual(set(events["dataset"]), {"income_vip", "dividend", "fina_mainbz_vip"})
            self.assertNotIn("000002.SZ", set(events["ts_code"]))
            self.assertTrue((events["available_month"] == "202001").all())
            mainbz = events[events["dataset"] == "fina_mainbz_vip"].iloc[0]
            self.assertEqual(mainbz["available_at_rule"], "fallback_joined_statement_available_at")

            output = Path(tmp) / "pit" / "fundamental_events"
            written = builder.write_partitioned(events, output)
            self.assertEqual(len(written), 3)
            report = audit_fundamental_events(
                output,
                FundamentalEventsConfig(start_date="20200101", end_date="20200131", datasets=("income_vip", "dividend", "fina_mainbz_vip")),
            )
            self.assertEqual(report["status"], "warning")

    def test_corrupt_sidecar_raises_instead_of_blanking_source_write_id(self):
        """A present-but-unparseable .meta.json is torn-write evidence; the
        build must raise (pit.parquet_meta contract), not stamp source_write_id=''
        and void the sidecar_write_id_mismatch audit."""
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            income_dir = raw / "income_vip"
            income_dir.mkdir(parents=True)
            partition = income_dir / "period=20191231.parquet"
            pd.DataFrame([
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20200103",
                    "f_ann_date": "20200102",
                    "end_date": "20191231",
                    "report_type": "1",
                    "comp_type": "1",
                    "end_type": "4",
                }
            ]).to_parquet(partition, index=False)
            partition.with_suffix(partition.suffix + ".meta.json").write_text("{not json", encoding="utf-8")

            builder = FundamentalEventsBuilder(raw)
            with self.assertRaises(CorruptSidecarError):
                builder.build(FundamentalEventsConfig(
                    start_date="20200101", end_date="20200131", datasets=("income_vip",),
                ))

    def test_forecast_revision_visible_only_from_its_own_ann_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            forecast_dir = raw / "forecast_vip"
            forecast_dir.mkdir(parents=True)
            pd.DataFrame([
                {
                    "ts_code": "600157.SH",
                    "ann_date": "20200110",
                    "first_ann_date": "20200110",
                    "end_date": "20191231",
                    "type": "预增",
                    "update_flag": "0",
                },
                {
                    "ts_code": "600157.SH",
                    "ann_date": "20200125",
                    "first_ann_date": "20200110",
                    "end_date": "20191231",
                    "type": "预增",
                    "update_flag": "1",
                },
            ]).to_parquet(forecast_dir / "ann_month=202001.parquet", index=False)

            builder = FundamentalEventsBuilder(raw)
            config = FundamentalEventsConfig(start_date="20200101", end_date="20200131", datasets=("forecast_vip",))
            events = builder.build(config).sort_values("available_at").reset_index(drop=True)

            self.assertEqual(list(events["available_at_rule"]), ["source:ann_date", "source:ann_date"])
            self.assertEqual(
                list(events["available_at"]),
                ["2020-01-10T18:00:00+08:00", "2020-01-25T18:00:00+08:00"],
            )

            output = Path(tmp) / "pit"
            builder.write_partitioned(events, output)
            # blank source_write_id (fixture has no .meta.json sidecars) keeps this at warning
            self.assertEqual(audit_fundamental_events(output, config)["status"], "warning")

            # Backdating a revision to first_ann_date is determinate lookahead
            # and must hard-fail the audit.
            path = output / "forecast_vip" / "available_month=202001.parquet"
            tampered = pd.read_parquet(path)
            tampered.loc[tampered["update_flag"] == "1", "available_at"] = "2020-01-10T18:00:00+08:00"
            tampered.to_parquet(path, index=False)
            report = audit_fundamental_events(output, config)
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["findings"][-1]["details"]["backdated_available_at_rows"], 1)

    def test_same_timestamp_statement_revision_wins_by_update_flag(self):
        """Same-announcement statement corrections share the whole business key;
        the surviving row must be the update_flag=1 revision regardless of the
        vendor response row order (previously raw row order decided)."""
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            cashflow_dir = raw / "cashflow_vip"
            cashflow_dir.mkdir(parents=True)
            base = {
                "ann_date": "20251030",
                "f_ann_date": "20251030",
                "end_date": "20250930",
                "report_type": "1",
                "comp_type": "1",
                "end_type": "3",
            }
            pd.DataFrame([
                # revision emitted BEFORE the original: row order must not decide
                {"ts_code": "000006.SZ", **base, "update_flag": "1", "free_cashflow": -2.148417e8},
                {"ts_code": "000006.SZ", **base, "update_flag": "0", "free_cashflow": None},
                # original emitted before the revision
                {"ts_code": "000008.SZ", **base, "update_flag": "0", "free_cashflow": None},
                {"ts_code": "000008.SZ", **base, "update_flag": "1", "free_cashflow": 5.0e7},
            ]).to_parquet(cashflow_dir / "period=20250930.parquet", index=False)

            builder = FundamentalEventsBuilder(raw)
            config = FundamentalEventsConfig(start_date="20251001", end_date="20251031", datasets=("cashflow_vip",))
            events = builder.build(config)

            output = Path(tmp) / "pit"
            builder.write_partitioned(events, output)
            partition = output / "cashflow_vip" / "available_month=202510.parquet"
            written = pd.read_parquet(partition)
            self.assertEqual(len(written), 2)
            self.assertEqual(list(written["update_flag"]), ["1", "1"])
            by_code = written.set_index("ts_code")["free_cashflow"]
            self.assertAlmostEqual(by_code["000006.SZ"], -2.148417e8, delta=1.0)
            self.assertAlmostEqual(by_code["000008.SZ"], 5.0e7, delta=1.0)

            # An incremental re-emission over a stale partition (old code could
            # have kept the update_flag=0 row) must also converge to the revision.
            stale = written.copy()
            stale["update_flag"] = "0"
            stale["free_cashflow"] = float("nan")
            stale.to_parquet(partition, index=False)
            builder.write_partitioned(events, output)
            healed = pd.read_parquet(partition)
            self.assertEqual(len(healed), 2)
            self.assertEqual(list(healed["update_flag"]), ["1", "1"])
            self.assertFalse(healed["free_cashflow"].isna().any())

    def test_fundamental_events_merge_partial_month_and_replace_complete_month(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pit"
            builder = FundamentalEventsBuilder(Path(tmp) / "raw")
            original = pd.DataFrame([
                {
                    "dataset": "dividend",
                    "ts_code": "000001.SZ",
                    "available_at": "2020-01-02T18:00:00+08:00",
                    "available_at_rule": "source:imp_ann_date_or_ann_date",
                    "available_month": "202001",
                    "business_key": "old",
                    "source_path": "/raw/dividend/ts_code=000001.SZ.parquet",
                    "source_write_id": "old",
                    "source_row_id": 0,
                }
            ])
            update = original.assign(ts_code="000002.SZ", business_key="new", source_write_id="new")

            builder.write_partitioned(original, output)
            builder.write_partitioned(update, output)
            merged = pd.read_parquet(output / "dividend" / "available_month=202001.parquet")
            self.assertEqual(set(merged["business_key"]), {"old", "new"})

            aligned_start, replace_months = month_aligned_replace_window("20200115", "20200131")
            self.assertEqual(aligned_start, "20200101")
            self.assertEqual(replace_months, {"202001"})
            builder.write_partitioned(update, output, replace_months=replace_months)
            replaced = pd.read_parquet(output / "dividend" / "available_month=202001.parquet")
            self.assertEqual(set(replaced["business_key"]), {"new"})

            # Replacing a month that still holds rows with an EMPTY build is
            # destruction, not an update: refused before any deletion.
            with self.assertRaisesRegex(RuntimeError, "refusing to replace history"):
                builder.write_partitioned(
                    pd.DataFrame(),
                    output,
                    replace_months=replace_months,
                    replace_datasets=("dividend",),
                )
            survived = pd.read_parquet(output / "dividend" / "available_month=202001.parquet")
            self.assertEqual(set(survived["business_key"]), {"new"})

    def test_build_script_refuses_to_replace_window_with_all_empty_build(self):
        """Catastrophically empty raw fundamentals must fail the build fast,
        not silently replace every partition of the derived layer with markers."""
        import argparse
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "build_pit_events_under_test", Path("scripts/data/build_pit_events.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                raw_dir=Path(tmp) / "raw",
                output_root=Path(tmp) / "events",
                start_date="20200101",
                end_date="20200131",
                dataset=None,
            )
            with self.assertRaisesRegex(RuntimeError, "refusing to replace the window"):
                module.run_build_fundamental_events(args)
            self.assertFalse((Path(tmp) / "events").exists())

    def test_zero_event_month_marker_keeps_audit_clean(self):
        """A month with no events inside the build window must audit as
        present-and-empty (no missing_months warning), while a month the
        builder never touched still warns."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "events"
            builder = FundamentalEventsBuilder(Path(tmp) / "raw")
            events = pd.DataFrame([
                {
                    "dataset": "express_vip",
                    "ts_code": "000001.SZ",
                    "available_at": "2020-01-15T18:00:00+08:00",
                    "available_at_rule": "source:ann_date",
                    "available_month": "202001",
                    "business_key": "k",
                    "source_path": "x",
                    "source_write_id": "",
                    "source_row_id": 0,
                }
            ])
            aligned_start, replace_months = month_aligned_replace_window("20200101", "20200229")
            builder.write_partitioned(
                events, output, replace_months=replace_months, replace_datasets=("express_vip",)
            )

            config = FundamentalEventsConfig(start_date=aligned_start, end_date="20200229", datasets=("express_vip",))
            report = audit_fundamental_events(output, config)
            checks = {finding["check"] for finding in report["findings"]}
            self.assertNotIn("express_vip_missing_months", checks)

            # Outside what was built (March), the month is genuinely missing.
            wider = FundamentalEventsConfig(start_date=aligned_start, end_date="20200331", datasets=("express_vip",))
            wider_report = audit_fundamental_events(output, wider)
            missing = [f for f in wider_report["findings"] if f["check"] == "express_vip_missing_months"]
            self.assertEqual(missing[0]["details"]["missing_months"], ["202003"])

    def test_fundamental_event_audit_rejects_dangerous_rules_and_wrong_source_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "events"
            path = root / "dividend" / "available_month=202001.parquet"
            path.parent.mkdir(parents=True)
            pd.DataFrame([
                {
                    "dataset": "dividend",
                    "ts_code": "000001.SZ",
                    "available_at": "2020-01-02T18:00:00+08:00",
                    "available_at_rule": "source:imp_ann_date_or_ann_date:ex_date",
                    "available_month": "202001",
                    "business_key": "bad",
                    "source_path": "/raw/fina_indicator_vip/period=20191231.parquet",
                    "source_write_id": "hash",
                    "source_row_id": 0,
                }
            ]).to_parquet(path, index=False)

            report = audit_fundamental_events(
                root,
                FundamentalEventsConfig(start_date="20200101", end_date="20200131", datasets=("dividend",)),
            )

            self.assertEqual(report["status"], "error")
            details = report["findings"][-1]["details"]
            self.assertEqual(details["disallowed_available_at_rule_rows"], 1)
            self.assertEqual(details["wrong_source_path_rows"], 1)

    def test_fundamental_event_audit_can_require_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "events"
            config = FundamentalEventsConfig(start_date="20200101", end_date="20200131", datasets=("dividend",))

            warning_report = audit_fundamental_events(root, config)
            required_report = audit_fundamental_events(root, config, require_partitions=True)

            self.assertEqual(warning_report["status"], "warning")
            self.assertEqual(required_report["status"], "error")
            self.assertIn(
                "fundamental_events_partitions",
                [finding["check"] for finding in required_report["findings"]],
            )

    def test_required_fundamental_event_audit_rejects_a_missing_month(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "events"
            FundamentalEventsBuilder(Path(tmp) / "raw").write_partitioned(
                pd.DataFrame(),
                root,
                replace_months={"202001"},
                replace_datasets=("dividend",),
            )

            report = audit_fundamental_events(
                root,
                FundamentalEventsConfig(start_date="20200101", end_date="20200229", datasets=("dividend",)),
                require_partitions=True,
            )

            self.assertEqual(report["status"], "error")
            self.assertIn("dividend_missing_months", [finding["check"] for finding in report["findings"]])

    def test_required_fundamental_event_audit_rejects_a_missing_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "events"
            FundamentalEventsBuilder(Path(tmp) / "raw").write_partitioned(
                pd.DataFrame(),
                root,
                replace_months={"202001"},
                replace_datasets=("dividend",),
            )

            report = audit_fundamental_events(
                root,
                FundamentalEventsConfig(
                    start_date="20200101",
                    end_date="20200131",
                    datasets=("dividend", "cashflow_vip"),
                ),
                require_partitions=True,
            )

            self.assertEqual(report["status"], "error")
            self.assertIn("cashflow_vip_partitions", [finding["check"] for finding in report["findings"]])


class UnitRegistryProjectionTest(unittest.TestCase):
    """The column-level unit registry: structure, coverage, projections."""

    @staticmethod
    def _export_units_module():
        import importlib.util

        repo_root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "export_units", repo_root / "scripts" / "dev" / "export_units.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _inventory_column_map() -> dict[tuple[str, str | None], list[str]]:
        repo_root = Path(__file__).resolve().parents[2]
        inventory = json.loads(
            (repo_root / "configs" / "data" / "snapshot_columns.json").read_text(encoding="utf-8")
        )
        column_map: dict[tuple[str, str | None], list[str]] = {
            (file, dataset): columns
            for file, datasets in inventory["files"].items()
            for dataset, columns in datasets.items()
        }
        # Builder-defined single-schema snapshot files (columns are a code
        # contract, pinned here; the snapshot build re-validates live).
        column_map[("daily.parquet", None)] = [
            "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change",
            "pct_chg", "vol", "amount", "turnover_rate", "turnover_rate_f",
            "volume_ratio", "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm",
            "total_share", "float_share", "free_share", "total_mv", "circ_mv",
            "up_limit", "down_limit", "adj_factor", "is_suspended",
        ]
        column_map[("auction.parquet", None)] = [
            "ts_code", "trade_date", "session", "price", "vol", "amount", "pre_close",
            "turnover_rate", "volume_ratio", "float_share", "available_at", "available_at_rule",
        ]
        column_map[("intraday_1min.parquet", None)] = [
            "ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount",
            "trade_date", "auction_market_bucket", "auction_open_bar", "vol_pit", "amount_pit",
            "auction_vol_correction_factor", "auction_amount_correction_factor",
            "auction_correction_rule",
        ]
        column_map[("corporate_actions.parquet", None)] = [
            "ts_code", "ex_date", "record_date", "pay_date", "div_listdate",
            "cash_per_share", "stock_per_share",
        ]
        column_map[("text_index.parquet", None)] = [
            "text_id", "dataset", "ts_codes", "title", "available_at", "library_file",
        ]
        column_map[("universe.parquet", None)] = [
            "ts_code", "exchange", "list_date", "market", "name", "l1_code", "l1_name",
        ]
        return column_map

    def test_every_inventory_column_resolves(self):
        from autotrade.environment.data.units import build_unit_reference

        # Full-column resolution over the committed vendor schema inventory:
        # an unregistered column or two overlapping rules both raise here.
        records = build_unit_reference(self._inventory_column_map())
        self.assertGreater(len(records), 1500)
        for record in records:
            if record["semantic_type"] == "numeric":
                self.assertTrue(record["source_unit"], record)
                self.assertEqual(
                    record["status"] == "unknown", record["source_unit"] == "unknown", record
                )
            else:
                self.assertIsNone(record["source_unit"], record)

    def test_registry_structure_and_selectable_dataset_coverage(self):
        from autotrade.environment.data.snapshot import SELECTABLE_DATASETS
        from autotrade.environment.data.units import (
            FIELD_RULES,
            NO_NUMERIC_DATASETS,
            rules_for,
        )

        keys = [(rule.file, rule.dataset, rule.columns) for rule in FIELD_RULES]
        self.assertEqual(len(keys), len(set(keys)), "duplicate registry rows")
        for rule in FIELD_RULES:
            self.assertIn(rule.status, {"verified", "official", "inferred", "unknown"}, rule.key())
            self.assertTrue(rule.columns, rule.key())
            if rule.semantic == "numeric":
                self.assertTrue(rule.source_unit, rule.key())
            if rule.factor is not None:
                self.assertTrue(rule.normalized_unit, rule.key())
            if rule.status == "verified":
                self.assertTrue(rule.evidence, rule.key())
        # Every SELECTABLE snapshot dataset — not just the default scope — is
        # either ruled under its OWN file (a fundamentals dataset registered
        # under events.parquet fails here) or a declared no-numeric dataset:
        # an experiment may opt into any of them and the registry is
        # fail-closed, so a missing rule would break that experiment's build.
        domain_files = (
            ("events.parquet", SELECTABLE_DATASETS["events"]),
            ("macro.parquet", SELECTABLE_DATASETS["macro"]),
            ("fundamentals.parquet", SELECTABLE_DATASETS["fundamentals"]),
        )
        for file, datasets in domain_files:
            for dataset in datasets:
                rules = rules_for(datasets=(dataset,))
                if dataset in NO_NUMERIC_DATASETS:
                    self.assertEqual(rules, (), dataset)
                    continue
                self.assertTrue(rules, f"no unit rules for selectable dataset {dataset}")
                self.assertEqual({rule.file for rule in rules}, {file}, dataset)

    def test_snapshot_excluded_columns_stay_classified_and_unmixed(self):
        from autotrade.environment.data.snapshot import SNAPSHOT_EXCLUDED_COLUMNS
        from autotrade.environment.data.units import FIELD_RULES, resolve_field

        # A column removed from the snapshot is still downloaded and audited,
        # so the registry must keep classifying it — dropping the rule would
        # break the raw-lake audit projections instead of the snapshot.
        for dataset, columns in SNAPSHOT_EXCLUDED_COLUMNS.items():
            files = {rule.file for rule in FIELD_RULES if rule.dataset == dataset}
            self.assertTrue(files, dataset)
            for file in files:
                for column in columns:
                    record = resolve_field(file, dataset, column)
                    self.assertEqual(record["column"], column)
        # A rule must be wholly excluded or wholly visible: a mixed rule would
        # make both the unit reference and the rendered doc ambiguous about
        # what the Agent can actually see.
        for rule in FIELD_RULES:
            excluded = set(SNAPSHOT_EXCLUDED_COLUMNS.get(rule.dataset or "", ()))
            covered = excluded.intersection(rule.columns)
            if covered:
                self.assertEqual(covered, set(rule.columns), rule.key())

    def test_conversion_rules_never_overlap(self):
        from autotrade.environment.data.units import (
            AUCTION_UNIT_CONVERSIONS,
            DAILY_UNIT_CONVERSIONS,
        )

        # A column multiplied by two factors would corrupt snapshot values.
        for table in (DAILY_UNIT_CONVERSIONS, AUCTION_UNIT_CONVERSIONS):
            columns = [column for column, _, _ in table]
            self.assertEqual(len(columns), len(set(columns)), table)

    def test_unit_corrections_are_pinned(self):
        from autotrade.environment.data.units import resolve_field

        expectations = [
            # (file, dataset, column, source_unit) — errors proven by
            # back-calculation/reconciliation must never regress.
            ("events.parquet", "share_float_complete", "float_share", "shares"),
            ("events.parquet", "repurchase", "high_limit", "CNY_per_share"),
            ("events.parquet", "cyq_perf", "cost_5pct", "CNY_per_share"),
            ("events.parquet", "cyq_perf", "winner_rate", "percent"),
            ("fundamentals.parquet", "fina_indicator_vip", "current_ratio", "multiple"),
            ("fundamentals.parquet", "fina_indicator_vip", "assets_turn", "times_per_period"),
            ("fundamentals.parquet", "fina_indicator_vip", "roe", "percent"),
            ("fundamentals.parquet", "fina_indicator_vip", "gross_margin", "CNY"),
            ("fundamentals.parquet", "balancesheet_vip", "total_share", "shares"),
            ("daily.parquet", None, "volume_ratio", "multiple"),
            ("daily.parquet", None, "pre_close", "CNY_per_share"),
            ("macro.parquet", "repo_daily", "close", "percent"),
        ]
        for file, dataset, column, unit in expectations:
            record = resolve_field(file, dataset, column)
            self.assertEqual(record["source_unit"], unit, (file, dataset, column))
        # fund_visitors stores comma-separated visitor NAMES — it must never
        # regress to a numeric count reading.
        surv = resolve_field("events.parquet", "stk_surv", "fund_visitors")
        self.assertEqual(surv["semantic_type"], "text")
        self.assertIsNone(surv["source_unit"])

    def test_status_tiers_reflect_their_evidence_class(self):
        from autotrade.environment.data.units import resolve_field

        # verified ⇒ a NAMED reconciliation against another source or known
        # external truth; magnitude-only readings stay inferred; vendor-doc
        # semantics validated by range stay official.
        verified = [
            ("events.parquet", "share_float_complete", "float_share", "daily_basic"),
            ("events.parquet", "repurchase", "high_limit", "close"),
            ("events.parquet", "moneyflow_dc", "close", "ratio 1.0000"),
            ("fundamentals.parquet", "forecast_vip", "net_profit_min", "ratio 1.0000"),
            ("fundamentals.parquet", "express_vip", "revenue", "income_vip"),
        ]
        from autotrade.environment.data.unit_rules import FIELD_RULES

        for file, dataset, column, evidence_fragment in verified:
            record = resolve_field(file, dataset, column)
            self.assertEqual(record["status"], "verified", (dataset, column))
            rules = [
                rule for rule in FIELD_RULES
                if rule.file == file and rule.dataset == dataset and column in rule.columns
            ]
            self.assertEqual(len(rules), 1, (dataset, column))
            self.assertIn(evidence_fragment, rules[0].evidence, (dataset, column))
        for dataset, column in (
            ("moneyflow_ind_dc", "net_amount"),
            ("moneyflow_ind_ths", "net_amount"),
            ("moneyflow_cnt_ths", "net_amount"),
        ):
            self.assertEqual(
                resolve_field("events.parquet", dataset, column)["status"], "inferred", dataset
            )
        self.assertEqual(
            resolve_field("fundamentals.parquet", "fina_indicator_vip", "current_ratio")["status"],
            "official",
        )

    def test_snapshot_column_map_reconciles_manifest_and_fails_on_corrupt_files(self):
        from autotrade.environment.data.units import snapshot_column_map

        with tempfile.TemporaryDirectory() as tmp:
            view = Path(tmp)
            pd.DataFrame(
                [{"dataset": "margin_secs", "ts_code": "000001.SZ", "orphan_col": 1.0}]
            ).to_parquet(view / "events.parquet", index=False)
            manifest = {
                "domains": {
                    "events": {"dataset_columns": {"margin_secs": ["dataset", "ts_code"]}}
                }
            }
            # A physical column not attributed to any dataset must fail loudly,
            # not silently under-cover the unit table.
            with self.assertRaises(ValueError) as ctx:
                snapshot_column_map(view, manifest)
            self.assertIn("orphan_col", str(ctx.exception))
            # events/macro require EXACT match: declaring a non-physical
            # column is also a broken manifest.
            manifest["domains"]["events"]["dataset_columns"]["margin_secs"] = [
                "dataset", "ts_code", "orphan_col", "future_col",
            ]
            with self.assertRaises(ValueError) as ctx:
                snapshot_column_map(view, manifest)
            self.assertIn("future_col", str(ctx.exception))
            # fundamentals attribution comes from the vendor schema and may
            # legitimately run ahead of window content (schema-forward).
            pd.DataFrame(
                [{"dataset": "fina_audit", "ts_code": "000001.SZ", "audit_fees": 1.0}]
            ).to_parquet(view / "fundamentals.parquet", index=False)
            manifest["domains"]["events"]["dataset_columns"]["margin_secs"] = [
                "dataset", "ts_code", "orphan_col",
            ]
            manifest["domains"]["fundamentals"] = {
                "dataset_columns": {
                    "fina_audit": ["dataset", "ts_code", "audit_fees", "future_col"]
                }
            }
            column_map = snapshot_column_map(view, manifest)
            self.assertIn("future_col", column_map[("fundamentals.parquet", "fina_audit")])

        with tempfile.TemporaryDirectory() as tmp:
            view = Path(tmp)
            (view / "broken.parquet").write_text("not parquet", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                snapshot_column_map(view, {"domains": {}})
            self.assertIn("broken.parquet", str(ctx.exception))
            self.assertNotIn(tmp, str(ctx.exception))

    def test_unresolved_column_fails_fast_with_full_listing(self):
        from autotrade.environment.data.units import UnresolvedUnitError, build_unit_reference

        with self.assertRaises(UnresolvedUnitError) as ctx:
            build_unit_reference({("events.parquet", "margin"): ["rzye", "made_up_a", "made_up_b"]})
        self.assertIn("made_up_a", str(ctx.exception))
        self.assertIn("made_up_b", str(ctx.exception))

    def test_audit_projections_derive_from_registry(self):
        from autotrade.data_sources.tushare.audit import (
            board_unit_rules,
            core_market_unit_rules,
            event_unit_rules,
            fundamental_unit_rules,
            macro_unit_rules,
        )
        from autotrade.data_sources.tushare.common import (
            BOARD_TRADING_DATASETS,
            EVENT_FLOW_SPECS,
            MACRO_SPECS,
        )
        from autotrade.environment.data.units import FIELD_RULES

        registry_records = [rule.to_record() for rule in FIELD_RULES]
        expected_event_ids = {
            "share_float_complete" if name == "share_float" else name for name in EVENT_FLOW_SPECS
        }
        domain_expectations = (
            (macro_unit_rules(), set(MACRO_SPECS)),
            (event_unit_rules(), expected_event_ids),
            (board_unit_rules(), set(BOARD_TRADING_DATASETS)),
        )
        for projection, expected_datasets in domain_expectations:
            self.assertEqual(set(projection), expected_datasets)
            for dataset, records in projection.items():
                for record in records:
                    self.assertIn(record, registry_records, dataset)
        for projection in (core_market_unit_rules(), fundamental_unit_rules()):
            for records in projection.values():
                for record in records:
                    self.assertIn(record, registry_records)

    def test_unit_reference_states_that_the_files_are_already_normalized(self):
        """`source_unit` is the vendor unit, not the unit in the file. A Fold
        that read `unit_reference.json` standalone measured daily.parquet in
        CNY/shares, saw `thousand_CNY`/`hands`, and filed the correct data
        contract as a defect; the standalone file must carry the clause that
        resolves it."""

        from autotrade.environment.data.summary import write_agent_data_summary

        with tempfile.TemporaryDirectory() as tmp:
            view = Path(tmp) / "decision"
            view.mkdir()
            pd.DataFrame(
                {
                    "ts_code": ["000957.SZ"],
                    "trade_date": ["20200102"],
                    "close": [6.73],
                    "vol": [10_693_000.0],
                    "amount": [71_948_000.0],
                }
            ).to_parquet(view / "daily.parquet", index=False)
            (view / "manifest.json").write_text('{"kind": "decision"}', encoding="utf-8")
            write_agent_data_summary(
                Path(tmp) / "data_summary.json",
                kind="decision",
                fold_id=None,
                views={"snapshot": (view, "/mnt/snapshot")},
            )
            payload = json.loads(
                (Path(tmp) / "unit_reference.json").read_text(encoding="utf-8")
            )

        from autotrade.environment.data.units import AGENT_UNIT_CONTRACT

        self.assertEqual(
            payload["normalized_files"], AGENT_UNIT_CONTRACT["normalized_files"]
        )
        amount = next(
            record
            for record in payload["records"]
            if record["file"] == "daily.parquet" and record["column"] == "amount"
        )
        # The record itself keeps stating the vendor unit and the applied factor.
        self.assertEqual(amount["source_unit"], "thousand_CNY")
        self.assertEqual(amount["normalized_unit"], "CNY")

    def test_unit_reference_scopes_itself_to_snapshot_columns(self):
        """A Fold looked up `suspend_d` — a vendor source named in the daily
        domain's `datasets` provenance, not a snapshot file — found no record,
        and filed the missing entry as a broken unit contract. The standalone
        file must say what it covers before it says that an absent column is
        broken."""

        from autotrade.environment.data.summary import write_agent_data_summary
        from autotrade.environment.data.units import AGENT_UNIT_CONTRACT

        payload = self._unit_reference_payload(write_agent_data_summary)
        self.assertEqual(payload["coverage"], AGENT_UNIT_CONTRACT["coverage"])
        self.assertIn("suspend_d", payload["coverage"])
        # The defect clause stays, but scoped: only a snapshot column can be missing.
        self.assertIn("snapshot column absent", payload["unknown_unit_policy"])
        # A source dataset genuinely has no record; that is the contract, not a gap.
        self.assertEqual(
            [record for record in payload["records"] if record["dataset"] == "suspend_d"], []
        )

    def test_is_suspended_record_states_what_the_flag_marks(self):
        """`daily.is_suspended` is set from the presence of a suspend_d row, so
        it marks resumption days and intraday halts — never a full-day
        suspension, which has no daily bar at all. A Fold pre-registered four
        candidates on "consecutive suspended bars >= 5", an event set that is
        empty by construction. The Agent-visible record must carry that."""

        from autotrade.environment.data.summary import write_agent_data_summary
        from autotrade.environment.data.units import resolve_field

        payload = self._unit_reference_payload(write_agent_data_summary)
        record = next(
            item
            for item in payload["records"]
            if item["file"] == "daily.parquet" and item["column"] == "is_suspended"
        )
        note = record["note"]
        self.assertEqual(note, resolve_field("daily.parquet", None, "is_suspended")["note"])
        for fragment in ("suspend_d", "NO daily row", "trading-day gaps"):
            self.assertIn(fragment, note)

    @staticmethod
    def _unit_reference_payload(write_agent_data_summary) -> dict:
        """`unit_reference.json` as an Agent reads it, for a minimal daily view."""
        with tempfile.TemporaryDirectory() as tmp:
            view = Path(tmp) / "decision"
            view.mkdir()
            pd.DataFrame(
                {
                    "ts_code": ["000957.SZ"],
                    "trade_date": ["20200102"],
                    "close": [6.73],
                    "is_suspended": [False],
                }
            ).to_parquet(view / "daily.parquet", index=False)
            (view / "manifest.json").write_text(
                '{"kind": "decision", "domains": {"daily": '
                '{"datasets": ["daily", "daily_basic", "adj_factor", "stk_limit", "suspend_d"]}}}',
                encoding="utf-8",
            )
            write_agent_data_summary(
                Path(tmp) / "data_summary.json",
                kind="decision",
                fold_id=None,
                views={"snapshot": (view, "/mnt/snapshot")},
            )
            return json.loads((Path(tmp) / "unit_reference.json").read_text(encoding="utf-8"))

    def test_agent_contract_is_a_pointer_not_a_copy(self):
        from autotrade.environment.data.units import AGENT_UNIT_CONTRACT

        self.assertEqual(
            AGENT_UNIT_CONTRACT["unit_reference"], "/mnt/artifacts/unit_reference.json"
        )
        # No embedded rule table: the contract stays a constant-size pointer.
        self.assertLess(len(json.dumps(AGENT_UNIT_CONTRACT)), 1200)

    def test_units_reference_doc_is_fresh(self):
        repo_root = Path(__file__).resolve().parents[2]
        module = self._export_units_module()
        committed = (repo_root / "docs" / "units-reference.md").read_text(encoding="utf-8")
        self.assertEqual(
            committed,
            module.render_units_markdown(),
            "docs/units-reference.md is stale; run scripts/dev/export_units.py",
        )

    def test_units_reference_marks_every_excluded_column_as_out_of_snapshot(self):
        from autotrade.environment.data.snapshot import SNAPSHOT_EXCLUDED_COLUMNS
        from autotrade.environment.data.units import FIELD_RULES

        module = self._export_units_module()
        rendered = module.render_units_markdown()
        # An excluded column never reaches the Agent, whether or not its
        # dataset is in the default scope. Reporting only "not loaded by
        # default" would advertise it as opt-in available.
        covered: set[str] = set()
        for rule in FIELD_RULES:
            excluded = set(SNAPSHOT_EXCLUDED_COLUMNS.get(rule.dataset or "", ()))
            if not excluded.intersection(rule.columns):
                continue
            covered.add(rule.dataset)
            self.assertEqual(
                module._visibility_note(rule),
                module._COLUMNS_NOT_IN_SNAPSHOT,
                rule.key(),
            )
            row = f"| `{rule.dataset}` | {'/'.join(rule.columns)} |"
            line = next(text for text in rendered.splitlines() if text.startswith(row))
            self.assertIn(module._COLUMNS_NOT_IN_SNAPSHOT, line)
            self.assertNotIn(module._DATASET_NOT_DEFAULT, line)
        # Non-vacuous: every dataset with an exclusion is reached above.
        self.assertEqual(covered, set(SNAPSHOT_EXCLUDED_COLUMNS))
