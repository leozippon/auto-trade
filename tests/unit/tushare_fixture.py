"""Shared doubles for the TuShare ingest tests: fake vendors that record what
they are asked, a temp lake per test, and a committed generation record."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from autotrade.data_sources.tushare import generation
from autotrade.data_sources.tushare.client import ApiResult


def commit_generation(raw_dir: Path) -> dict:
    """A committed generation record, as a finished mutating job leaves it."""
    with redirect_stdout(io.StringIO()):
        return generation.finish_raw_generation_update(
            raw_dir, {}, {"job": "fixture"}, commit=True, datasets={}, issues=[]
        )


class EmptyClient:
    """A vendor that records every request in ``calls`` and answers no rows;
    a test's own vendor subclasses it to answer its scenario."""

    def __init__(self):
        self.calls = []

    def query(self, api_name, params=None, fields="", retries=5):
        self.calls.append((api_name, dict(params or {})))
        return ApiResult(fields.split(",") if fields else [], [])


def param(name, fallback):
    """A field answered with the request's parameter ``name`` (``fallback``
    when the request has none)."""
    return lambda api_name, params: params.get(name, fallback)


class FieldClient(EmptyClient):
    """A vendor that answers one row built field by field: ``fields`` maps a
    field to its value, or to a function of (api_name, params); every other
    field is ``default``. ``apis``, when given, names the only endpoints that
    answer; any other answers no rows."""

    def __init__(self, fields=None, *, default="", apis=None):
        super().__init__()
        self.fields = dict(fields or {})
        self.default = default
        self.apis = apis

    def query(self, api_name, params=None, fields="", retries=5):
        answer = super().query(api_name, params, fields, retries)
        if self.apis is not None and api_name not in self.apis:
            return answer
        params = params or {}
        row = []
        for name in answer.fields:
            value = self.fields.get(name, self.default)
            row.append(value(api_name, params) if callable(value) else value)
        return ApiResult(answer.fields, [row])


# One stock's row of any daily-market table.
DAILY_FIELDS = {"trade_date": param("trade_date", "20200102"), "ts_code": "000001.SZ", "adj_factor": 1.0}


class LakeTestCase(unittest.TestCase):
    """A test with its own temp lake: ``root`` holds ``raw_dir`` and the
    ledgers, and goes when the test ends."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw_dir = self.root / "raw"

    def tearDown(self):
        self.tmp.cleanup()

    def _write_trade_cal(self, trade_date="20200102", is_open="1"):
        path = self.raw_dir / "trade_cal" / "exchange=SSE" / f"year={trade_date[:4]}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"cal_date": trade_date, "is_open": is_open}]).to_parquet(path, index=False)

    def _write_trade_cal_with_stamping_window(self, trade_date="20200102", is_open="1"):
        """trade_date plus the 31 days a margin download stamps from (weekdays open)."""
        start = pd.Timestamp(trade_date)
        days = pd.date_range(start, start + pd.Timedelta(days=31))
        path = self.raw_dir / "trade_cal" / "exchange=SSE" / f"year={trade_date[:4]}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"cal_date": day.strftime("%Y%m%d"), "is_open": is_open if day == start else str(int(day.weekday() < 5))}
            for day in days
        ]).to_parquet(path, index=False)
