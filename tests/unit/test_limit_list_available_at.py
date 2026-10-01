import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from autotrade.data_sources.tushare import audit
from autotrade.data_sources.tushare import common as core


def limit_list_frame(rows):
    return pd.DataFrame(rows, columns=["trade_date", "ts_code", "close", "limit"])


class LimitListAvailableAtStampTest(unittest.TestCase):
    def test_limit_list_d_rows_stamp_the_shared_16_00_family_rule(self):
        frame = limit_list_frame([("20200102", "000001.SZ", 10.0, "U")])
        out = core.augment_board_frame(frame, core.BOARD_TRADING_SPECS["limit_list_d"])
        self.assertEqual(out.loc[0, "available_at"], "2020-01-02 16:00:00+08:00")
        self.assertEqual(out.loc[0, "available_at_rule"], "official_16_from:trade_date")
        # Single source: the official list must ride the exact limit_list_ths rule.
        self.assertIs(
            core.BOARD_TRADING_SPECS["limit_list_d"].availability,
            core.BOARD_TRADING_SPECS["limit_list_ths"].availability,
        )

    def test_malformed_trade_date_stays_unstamped_instead_of_guessing(self):
        frame = limit_list_frame([("", "000001.SZ", 10.0, "Z")])
        out = core.augment_board_frame(frame, core.BOARD_TRADING_SPECS["limit_list_d"])
        self.assertEqual(out.loc[0, "available_at"], "")
        self.assertEqual(out.loc[0, "available_at_rule"], "missing_source_time")

    def test_revision_sentinel_stamps_probe_so_stamped_partitions_are_not_false_revisions(self):
        # Board partitions carry available_at columns the raw API frame lacks;
        # the sentinel must compare like-for-like or every sampled limit_list_d
        # partition becomes a false REVISION_ALERT.
        class LimitListClient:
            def query(self, api_name, params=None, fields="", retries=5):
                names = fields.split(",") if fields else []
                values = {"trade_date": (params or {}).get("trade_date", ""), "ts_code": "000001.SZ",
                          "close": 10.0, "limit": "U"}
                return core.ApiResult(names, [[values.get(name) for name in names]])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            calendar = pd.DataFrame({"cal_date": ["20200102"], "is_open": ["1"]})
            (raw / "trade_cal" / "exchange=SSE").mkdir(parents=True)
            calendar.to_parquet(raw / "trade_cal" / "exchange=SSE" / "year=2020.parquet", index=False)
            spec = core.BOARD_TRADING_SPECS["limit_list_d"]
            stamped = core.augment_board_frame(
                pd.DataFrame([{name: {"trade_date": "20200102", "ts_code": "000001.SZ", "close": 10.0,
                                      "limit": "U"}.get(name) for name in spec.fields.split(",")}]),
                spec,
                {"trade_date": "20200102"},
            )
            (raw / "limit_list_d").mkdir(parents=True)
            stamped.to_parquet(raw / "limit_list_d" / "trade_date=20200102.parquet", index=False)
            args = argparse.Namespace(
                raw_dir=str(raw), start_date="20200102", end_date="20200102",
                datasets=["limit_list_d"], sample_size=0, seed=None, page_limit=10000,
                revision_ledger=str(root / "sentinel_events.jsonl"),
                output=str(root / "sentinel_summary.json"),
                fail_on_revision=False, min_interval_seconds=0, timeout_seconds=1,
            )
            with patch.object(audit, "load_token", return_value="token"), \
                    patch.object(audit, "TuShareClient", return_value=LimitListClient()):
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = audit.audit_revision_sentinel(args)
            self.assertEqual(exit_code, 0)
            self.assertNotIn("REVISION_ALERT", output.getvalue())
            summary = json.loads((root / "sentinel_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["metadata"]["totals"]["revision_events"], 0)

    def test_revision_sentinel_refuses_non_trade_date_board_datasets(self):
        with self.assertRaisesRegex(RuntimeError, "plain trade_date strategy"):
            audit.revision_sentinel_spec("kpl_list")
        with self.assertRaisesRegex(RuntimeError, "plain trade_date strategy"):
            audit.revision_sentinel_spec("not_a_dataset")
        self.assertIs(audit.revision_sentinel_spec("daily"), core.DAILY_SPECS["daily"])
        self.assertIs(
            audit.revision_sentinel_spec("limit_list_d"),
            core.BOARD_TRADING_SPECS["limit_list_d"],
        )


if __name__ == "__main__":
    unittest.main()
