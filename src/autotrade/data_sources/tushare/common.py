#!/usr/bin/env python3
"""Shared TuShare constants, schemas, client, and utility helpers."""

from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import time
from bisect import bisect_right
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from autotrade.common.time import cn_today

# Raw-lake research contract values are owned by the environment's PIT layer;
# the ingest adapter imports them so producer and consumer cannot drift.
from autotrade.environment.data.contracts import (
    DOMAIN_STATUS_FILES,
    REFRESH_NODES,
    TEXT_DATASET_REFRESH_NODES,
    TEXT_NODE,
)
from autotrade.environment.data.pit import first_seen_at, partition_value

from .io import committed_partition_intact, parquet_meta, parquet_rows, read_many
from .specs import (
    BOARD_DC_HOT_MARKETS,
    BOARD_DC_HOT_TYPES,
    BOARD_HOT_IS_NEW,
    BOARD_KPL_TAGS,
    BOARD_THS_HOT_MARKETS,
    BOARD_THS_LIMIT_TYPES,
    BOARD_TRADING_DEFAULT_DATASETS,
    BOARD_TRADING_SPECS,
    DAILY_DOWNLOAD_DATASETS,
    DAILY_SPECS,
    DEFAULT_CN_INDEX_CODES,
    DEFAULT_FX_CODES,
    DEFAULT_GLOBAL_INDEX_CODES,
    EVENT_FLOW_DATASETS,
    EVENT_FLOW_SPECS,
    FUNDAMENTAL_DATASETS,
    FUNDAMENTAL_SPECS,
    GLOBAL_CONTEXT_DEFAULT_DATASETS,
    INTRADAY_DATASETS,
    MACRO_REGIME_DEFAULT_DATASETS,
    MACRO_SPECS,
    NEWS_SOURCES,
    STK_MINS_BY_DATE_DATASET,
    STK_MINS_DATASET,
    STK_MINS_REQUIRED_COLUMNS,
    TEXT_DATASETS,
    TEXT_SPECS,
    AvailabilityRule,
    DatasetSpec,
    FetchStrategy,
)
from .versions import DefectiveAnswer, publish

DEFAULT_TUSHARE_RELAY_URL = "https://fast.xiaodefa.cn"

# The official TuShare service, reached over TLS (the SDK's own default is the
# same host over plain HTTP, which would put the token on the wire in clear).
# Only datasets whose spec says ``official=True`` go here, with their own
# token; everything else stays on the relay with TUSHARE_TOKEN. The variable's
# name says the access may be temporary: without it those datasets alone stop.
OFFICIAL_TUSHARE_URL = "https://api.waditu.com/dataapi"
OFFICIAL_TOKEN_ENV = "TUSHARE_TOKEN_TMP"

# Status file names are owned by environment.data.contracts (one source for
# audit producers, snapshot gates, and research-release pinning).
CORE_MARKET_STATUS_PATH = "results/data_quality/" + DOMAIN_STATUS_FILES["daily"]
FUNDAMENTAL_RAW_STATUS_PATH = "results/data_quality/" + DOMAIN_STATUS_FILES["fundamentals_raw"]

TEXT_EVIDENCE_STATUS_PATH = "results/data_quality/" + DOMAIN_STATUS_FILES["text"]

INTRADAY_MINUTES_STATUS_PATH = "results/data_quality/" + DOMAIN_STATUS_FILES["intraday_1min"]

# Minimum seconds between vendor requests, for every entry point that talks to
# TuShare. 0.80s (75 req/min) is the rate the 2026-09-10 catch-up sustained for
# ~2.5 hours without incident. The former 0.22s (272 req/min) was inside the
# stated 400/min ceiling but now trips a SIX-MINUTE vendor cooldown, and
# ``TuShareClient.query`` retries five times over ~20 seconds — a budget no
# cooldown can fit inside, so every request in the window fails and the step
# dies. Anything faster than this must be a deliberate, supervised override.
MIN_REQUEST_INTERVAL_SECONDS = 0.80

# The largest offset the vendor serves: a request past it is refused (code
# 50101, "offset不能大于100000"), so offset paging reaches at most this offset
# plus one page. A per-day answer can grow past it unnoticed (a since-retired
# board-membership table did on 20261103: 105,452 rows at 8,000 a page);
# query_paged never requests past it.
VENDOR_MAX_OFFSET = 100_000

REVISION_EVENTS_PATH = "results/data_quality/revision_events.jsonl"

REVISION_SUMMARY_PATH = "results/data_quality/revision_summary.json"

# Partition outcomes (versions.publish) that changed the visible partition.
WRITTEN_OUTCOMES = frozenset({"appended", "replaced", "restated"})

MACRO_CONTEXT_STATUS_PATH = "results/data_quality/" + DOMAIN_STATUS_FILES["macro"]

EVENT_FLOW_STATUS_PATH = "results/data_quality/" + DOMAIN_STATUS_FILES["events"]

BOARD_TRADING_STATUS_PATH = "results/data_quality/" + DOMAIN_STATUS_FILES["board_trading"]

# Child-process contract: validation/polling ended before any raw write began.
# The cron runner may restore the previous committed generation and retry later.
NO_MUTATION_RETRY_EXIT_CODE = 75

# Child-process contract: the run failed before any raw write began, on a
# dataset whose own access is gone (the official-service datasets and
# OFFICIAL_TOKEN_ENV). Nothing changed, so the runner restores the previous
# committed generation instead of fencing the whole lake, and records the job
# as an error: losing that access stops those datasets alone, loudly.
NO_MUTATION_FAILURE_EXIT_CODE = 77

# Child-process contract: the lake DID mutate, but a required partition is still
# unpublished at the source. The generation must be committed (the writes are
# real), yet the run must not report success: an "ok" state lets
# skip_if_already_ok suppress every later attempt at the same resolved end date,
# which silently abandons the partition once the window moves on.
MUTATED_NOT_READY_RETRY_EXIT_CODE = 76

def resolve_revision_ledger(
    raw_dir: Path | str,
    revision_ledger: Path | str | None = REVISION_EVENTS_PATH,
    *,
    repo_root: Path | str | None = None,
) -> Path | None:
    """Resolve revision ledger path without letting temp raw dirs pollute production."""
    if revision_ledger in {None, ""}:
        return None
    root = Path(repo_root or Path.cwd()).resolve()
    raw_path = Path(raw_dir)
    if not raw_path.is_absolute():
        raw_path = root / raw_path
    raw_path = raw_path.resolve()
    ledger = Path(revision_ledger)
    if ledger.is_absolute():
        return ledger
    if ledger == Path(REVISION_EVENTS_PATH):
        production_raw = (root / "data/raw").resolve()
        if raw_path != production_raw:
            try:
                raw_path.relative_to(production_raw)
            except ValueError:
                return raw_path.parent / "revision_events.jsonl"
    return root / ledger


SHARE_FLOAT_ROW_LIMIT = 6000

SHARE_FLOAT_UNLOCK_TITLE_PATTERN = re.compile(
    r"限售|解禁|上市流通|解除限售|限售股份上市|限售股上市|首次公开发行限售|非公开发行限售|定向增发限售"
)


@dataclass
class ApiResult:
    fields: list[str]
    items: list[list[Any]]


class TuShareClient:
    """The TuShare data API over HTTP, pinned to one endpoint.

    The endpoint is the relay unless a caller names another (the official
    service, for the datasets that use it). Requests follow the vendor SDK's
    wire protocol (a JSON POST to ``<endpoint>/<api_name>``), sent here rather
    than through the SDK because the SDK answers an HTTP error status with an
    empty frame, indistinguishable from an empty result. The token lives only
    in this object and is scrubbed from any error this adapter raises.

    An answer that lacks a requested field (compared case-insensitively, as
    the macro tier normalises vendor casing) is a :class:`DefectiveAnswer`.
    """

    def __init__(
        self,
        token: str,
        min_interval: float = MIN_REQUEST_INTERVAL_SECONDS,
        timeout: int = 60,
        endpoint: str | None = None,
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise ValueError("TuShare timeout must be a positive integer number of seconds")
        self._token = token
        self.endpoint = (endpoint or load_relay_url()).rstrip("/")
        self.min_interval = max(0.0, float(min_interval))
        self.timeout = timeout
        self.last_call = 0.0

    def _post(self, api_name: str, params: dict[str, Any], fields: str) -> ApiResult:
        payload = {
            "api_name": api_name,
            "token": self._token,
            # The SDK sends its endpoint as ts_type_name; the services expect it.
            "params": {**params, "ts_type_name": self.endpoint},
            "fields": fields,
        }
        response = requests.post(f"{self.endpoint}/{api_name}", json=payload, timeout=self.timeout)
        if response.status_code >= 400:
            raise RuntimeError(f"{api_name} answered HTTP {response.status_code}: {response.text[:200]}")
        body = response.json()
        if body.get("code") != 0:
            raise RuntimeError(f"{api_name} answered code {body.get('code')}: {body.get('msg')}")
        # An empty answer still carries its fields and an empty item list; a
        # missing or null block is a malformed answer (the SDK raised on it),
        # never zero rows.
        data = body.get("data")
        if not (isinstance(data, dict) and isinstance(data.get("fields"), list) and isinstance(data.get("items"), list)):
            raise RuntimeError(f"{api_name} answered code 0 without a data block of fields and items")
        return ApiResult(list(data["fields"]), list(data["items"]))

    def query(self, api_name: str, params: dict[str, Any] | None = None, fields: str = "", retries: int = 5) -> ApiResult:
        for attempt in range(1, retries + 1):
            self._throttle()
            try:
                result = self._post(api_name, dict(params or {}), fields)
            except Exception as exc:  # noqa: BLE001 - every vendor error is retried, then re-raised scrubbed
                if attempt == retries:
                    # Raised without the chained original, whose text the
                    # traceback would print unscrubbed.
                    raise RuntimeError(
                        f"{api_name} failed after {retries} attempts: {self._scrub(str(exc))}"
                    ) from None
                time.sleep(2 * attempt)
                continue
            requested = {name.strip().lower() for name in fields.split(",") if name.strip()}
            missing = requested - {str(name).lower() for name in result.fields}
            if result.items and missing:
                raise DefectiveAnswer(
                    f"{api_name} answered without requested fields {sorted(missing)} (got {result.fields[:20]})"
                )
            return result
        raise RuntimeError(f"{api_name} failed unexpectedly")

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.monotonic()

    def _scrub(self, text: str) -> str:
        return text.replace(self._token, "<token>") if self._token else text


def load_token(repo_root: Path, name: str = "TUSHARE_TOKEN") -> str:
    """The token in environment variable ``name``, else its line in the ignored .env."""
    token = os.environ.get(name, "").strip()
    if token:
        return token
    env_file = repo_root / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{name}="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                if token:
                    return token
    raise RuntimeError(f"{name} is not set in the environment or ignored .env")


def official_client(repo_root: Path, min_interval: float, timeout: int) -> TuShareClient:
    """A client on the official service with its own token (``OFFICIAL_TOKEN_ENV``)."""
    return TuShareClient(
        load_token(repo_root, OFFICIAL_TOKEN_ENV), min_interval, timeout, endpoint=OFFICIAL_TUSHARE_URL
    )


def load_relay_url() -> str:
    return os.environ.get("TUSHARE_RELAY_URL", DEFAULT_TUSHARE_RELAY_URL).strip() or DEFAULT_TUSHARE_RELAY_URL

def frame(result: ApiResult) -> pd.DataFrame:
    return pd.DataFrame(result.items, columns=result.fields)

def query_paged(client: TuShareClient, api_name: str, params: dict[str, Any], fields: str = "", page_limit: int | None = 10000, allow_repeated_pages: bool = False) -> tuple[ApiResult, int]:
    """Every row the vendor serves for ``params``, by offset paging.

    An answer that runs past VENDOR_MAX_OFFSET raises."""
    page_limit = page_limit or 10000
    all_items: list[list[Any]] = []
    result_fields: list[str] = []
    page_payloads: set[str] = set()
    pages = 0
    offset = 0
    while True:
        page_params = dict(params)
        page_params.update({"limit": page_limit, "offset": offset})
        result = client.query(api_name, page_params, fields)
        if not result_fields:
            result_fields = result.fields
        elif result.items and result.fields != result_fields:
            raise DefectiveAnswer(f"{api_name} returned inconsistent fields while paging")
        page_payload = json.dumps(
            {"fields": result.fields, "items": result.items},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        if result.items and page_payload in page_payloads and not allow_repeated_pages:
            # A repeated full page normally means broken pagination. Sources
            # that serve heavily duplicated rows (eco_cal repeats each event
            # many times) legitimately produce identical pages; callers that
            # dedup by key opt out and rely on the offset cap below.
            raise DefectiveAnswer(
                f"{api_name} returned a repeated page while paging params={params}; "
                f"offset={offset} page_limit={page_limit}"
            )
        page_payloads.add(page_payload)
        all_items.extend(result.items)
        pages += 1
        if len(result.items) < page_limit:
            if not result.items and all_items:
                # The answer ended on a page boundary: an empty page is the
                # only sign it is complete, and an error answered as an empty
                # page reads the same. Read the boundary once more.
                again = client.query(api_name, page_params, fields)
                if again.items:
                    raise DefectiveAnswer(
                        f"{api_name} served rows at offset {offset} on a second read after an empty page; "
                        f"params={params} page_limit={page_limit}"
                    )
            break
        offset += page_limit
        if offset > VENDOR_MAX_OFFSET:
            # The vendor would refuse the next page and the rows past it are
            # unreachable with these params: fail instead of retrying a refusal.
            raise RuntimeError(
                f"{api_name} answer for params={params} runs past the vendor's offset cap "
                f"{VENDOR_MAX_OFFSET} (page_limit={page_limit}); narrow the request"
            )
    return ApiResult(result_fields, all_items), pages

def load_stock_codes(raw_dir: Path) -> list[str]:
    files = sorted((raw_dir / "stock_basic").glob("list_status=*.parquet"))
    if not files:
        raise RuntimeError("stock_basic partitions are missing")
    data = read_many(files, columns=["ts_code"])
    codes = data["ts_code"].dropna().astype(str).str.strip()
    valid = codes[codes.str.fullmatch(r"\d{6}\.(SH|SZ|BJ)")]
    return sorted(valid.unique().tolist())

def normalize_date_key(value: object) -> str:
    digits = re.sub(r"\D", "", str(value or ""))[:8]
    return digits if len(digits) == 8 else ""


def normalized_date_keys(values: pd.Series) -> pd.Series:
    """Vectorized twin of :func:`normalize_date_key` for whole-column passes.

    Same rule, same output for every input (a drift test pins the equivalence):
    non-digits dropped, first 8 digits kept, and anything that is not exactly 8
    digits -- missing, null, empty, malformed -- becomes ``""``. ``""`` sorts
    BELOW every real ``YYYYMMDD``, which is what makes an undated row lose the
    newest-dating comparison instead of winning it (``str(None) == "None"``
    sorted above every date and inverted the comparison)."""
    keys = values.astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 8)
    return keys.where(keys.str.len() == 8, "")


# One physical unlock (解禁) event, independent of its announcement date:
# ``ann_date`` is excluded because the provider re-dates announcements.
PHYSICAL_IDENTITY_COLUMNS = ["ts_code", "float_date", "holder_name", "share_type"]


def normalized_holder_keys(values: pd.Series) -> pd.Series:
    """Holder names with whitespace ignored (raw rows keep their source bytes).

    Restatements carry cosmetic spacing drift (observed live: "...企业 (有限合伙)"
    re-dated as "...企业(有限合伙)"), so whitespace must not split one holder
    into two identities."""
    return values.astype(str).str.replace(r"\s+", "", regex=True)


def physical_identity_index(frame: pd.DataFrame) -> pd.MultiIndex:
    """Physical-identity index for membership tests between two frames."""
    view = frame[PHYSICAL_IDENTITY_COLUMNS].copy()
    view["holder_name"] = normalized_holder_keys(view["holder_name"])
    return pd.MultiIndex.from_frame(view)


def physical_identity_codes(frame: pd.DataFrame) -> pd.Series:
    """Dense integer identity code per row -- same identity rule as
    :func:`physical_identity_index`, but for whole-baseline grouping where
    materializing 12M index tuples would be wasteful."""
    view = frame[PHYSICAL_IDENTITY_COLUMNS].copy()
    view["holder_name"] = normalized_holder_keys(view["holder_name"])
    return view.groupby(PHYSICAL_IDENTITY_COLUMNS, sort=False, dropna=False).ngroup()


def undated_duplicate_mask(frame: pd.DataFrame) -> np.ndarray:
    """Rows carrying no announcement date whose physical identity already holds
    a dated row of the identical ``float_share``.

    Such a row is the undated copy of an announcement the lake already has: it
    contributes no observation and double-counts the lot. Deleting it cannot
    move point-in-time visibility, because the undated row's availability was
    the conservative ``float_date`` fallback while the dated twin was already
    present and announced no later.

    Shared by the union and the one-off repair so the property is an invariant
    of the data rather than a fact about the current update schedule: a rescue
    fetch that returns an undated lot must not re-introduce the duplicate.
    """
    required = [*PHYSICAL_IDENTITY_COLUMNS, "ann_date", "float_share"]
    if frame.empty or any(column not in frame.columns for column in required):
        return np.zeros(len(frame), dtype=bool)
    ann = normalized_date_keys(frame["ann_date"]).to_numpy()
    identity = physical_identity_codes(frame).to_numpy()
    share = pd.to_numeric(frame["float_share"], errors="coerce").to_numpy()
    undated = (ann == "") & ~pd.isna(share)
    dated = ann != ""
    if not undated.any() or not dated.any():
        return np.zeros(len(frame), dtype=bool)
    twins = pd.MultiIndex.from_arrays([identity[dated], share[dated]])
    candidates = pd.MultiIndex.from_arrays([identity[undated], share[undated]])
    mask = np.zeros(len(frame), dtype=bool)
    mask[np.flatnonzero(undated)[candidates.isin(twins)]] = True
    return mask


def sse_calendar(raw_dir: Path) -> pd.DataFrame:
    """The lake's SSE trading calendar, the one reader of it in this package:
    ``cal_date`` (YYYYMMDD) and ``is_open`` (bool), rows without a parseable
    date dropped. Raises when the lake holds no usable calendar."""
    files = sorted((Path(raw_dir) / "trade_cal" / "exchange=SSE").glob("year=*.parquet"))
    if not files:
        raise RuntimeError(f"SSE trade_cal partitions are missing under {raw_dir}; run download --tier reference first")
    calendar = read_many(files, columns=["cal_date", "is_open"])
    if calendar.empty:
        raise RuntimeError(f"SSE trade_cal is empty under {raw_dir}; run download --tier reference first")
    calendar = pd.DataFrame({
        "cal_date": calendar["cal_date"].map(normalize_date_key),
        "is_open": calendar["is_open"].astype(str) == "1",
    })
    calendar = calendar[calendar["cal_date"] != ""]
    if calendar.empty:
        raise RuntimeError("SSE trade_cal has no parseable cal_date values; refresh reference trade_cal first")
    return calendar.reset_index(drop=True)

def load_sse_open_dates(raw_dir: Path, start_date: str, end_date: str, *, allow_empty: bool = False) -> list[str]:
    calendar = sse_calendar(raw_dir)
    available_min = str(calendar["cal_date"].min())
    available_max = str(calendar["cal_date"].max())
    if start_date < available_min or end_date > available_max:
        raise RuntimeError(
            f"SSE trade_cal covers {available_min}-{available_max}, not requested {start_date}-{end_date}; refresh reference trade_cal first"
        )
    mask = calendar["is_open"] & (calendar["cal_date"] >= start_date) & (calendar["cal_date"] <= end_date)
    dates = sorted(calendar.loc[mask, "cal_date"].tolist())
    if not dates and not allow_empty:
        raise RuntimeError(f"no SSE open dates found for {start_date}-{end_date}")
    return dates

def latest_sse_calendar_date(raw_dir: Path) -> str:
    return str(sse_calendar(raw_dir)["cal_date"].max())

def select_datasets(
    values: Iterable[str] | None,
    *,
    default: Iterable[str],
    allowed: Iterable[str],
    label: str,
) -> list[str]:
    """Resolve an optional CLI dataset list: fall back to ``default`` when
    empty and fail explicitly on names missing from ``allowed``."""
    datasets = list(values or default)
    invalid = sorted(set(datasets) - set(allowed))
    if invalid:
        raise RuntimeError(f"unknown {label} datasets: {invalid}; supported={sorted(allowed)}")
    return datasets

def selected_daily_datasets(args: argparse.Namespace, *, default: Iterable[str] | None = None) -> list[str]:
    return select_datasets(
        args.datasets,
        default=DAILY_DOWNLOAD_DATASETS if default is None else default,
        allowed=DAILY_SPECS,
        label="daily market",
    )

def selected_fundamental_datasets(values: Iterable[str] | None) -> list[str]:
    return select_datasets(values, default=FUNDAMENTAL_DATASETS, allowed=FUNDAMENTAL_SPECS, label="fundamental")

def parse_yyyymmdd(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()  # noqa: DTZ007 - calendar date, no time of day

def format_yyyymmdd(value: date) -> str:
    return value.strftime("%Y%m%d")

def month_windows(start_date: str, end_date: str) -> list[tuple[str, str, str]]:
    start = parse_yyyymmdd(start_date)
    end = parse_yyyymmdd(end_date)
    current = date(start.year, start.month, 1)
    windows: list[tuple[str, str, str]] = []
    while current <= end:
        last = date(current.year, current.month, calendar.monthrange(current.year, current.month)[1])
        window_start = max(current, start)
        window_end = min(last, end)
        windows.append((format_yyyymmdd(window_start), format_yyyymmdd(window_end), f"{current.year}{current.month:02d}"))
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return windows

def year_windows(start_date: str, end_date: str) -> list[tuple[str, str, str]]:
    """Calendar-year windows of ``start_date``..``end_date``: (start, end, year)."""
    return [
        (max(start_date, f"{year}0101"), min(end_date, f"{year}1231"), str(year))
        for year in range(int(start_date[:4]), int(end_date[:4]) + 1)
    ]

def quarter_periods(start_date: str, end_date: str) -> list[str]:
    start = parse_yyyymmdd(start_date)
    end = parse_yyyymmdd(end_date)
    periods: list[str] = []
    for year in range(start.year, end.year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            period = date(year, month, day)
            if start <= period <= end:
                periods.append(format_yyyymmdd(period))
    return periods

def yyyymmdd_to_month(value: str) -> str:
    return value[:6]

def month_end_from_yyyymm(value: str) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"\d{6}", text):
        raise ValueError(f"invalid yyyymm value: {value}")
    year = int(text[:4])
    month = int(text[4:6])
    day = calendar.monthrange(year, month)[1]
    return f"{year}{month:02d}{day:02d}"

def yyyymmdd_to_quarter(value: str) -> str:
    parsed = parse_yyyymmdd(value)
    quarter = (parsed.month - 1) // 3 + 1
    return f"{parsed.year}Q{quarter}"

def history_floor(floor: str, args: argparse.Namespace) -> str:
    """``floor`` lowered to a backfill run's ``--history-floor``, never raised."""

    lowered = getattr(args, "history_floor", None)
    return min(floor, lowered) if lowered else floor

def yyyymmdd_arg(value: str) -> str:
    """argparse type of a YYYYMMDD calendar date."""

    try:
        parse_yyyymmdd(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a YYYYMMDD date, got {value!r}") from None
    return value

def retained_range_start(dataset_dir: Path, start_date: str) -> str:
    """The first date a range dataset's canonical file must keep covering.

    A quarter_once/month_once pull rewrites ONE canonical
    ``range=<start>_latest`` file, so a rewrite from a later start than the
    file on disk holds would drop the history below it -- a backfill under the
    research-history floor included. The retained start is therefore never
    later than the start of the canonical file already there.
    """

    starts = [start_date]
    for path in dataset_dir.glob("range=*_latest.parquet"):
        token = path.name[len("range="):-len("_latest.parquet")]
        quarter = re.fullmatch(r"(\d{4})Q([1-4])", token)
        if quarter:
            starts.append(f"{quarter.group(1)}{(int(quarter.group(2)) - 1) * 3 + 1:02d}01")
        elif re.fullmatch(r"\d{6}", token):
            starts.append(f"{token}01")
    return min(starts)

def quarter_end_date(value: str) -> str:
    match = re.fullmatch(r"(\d{4})Q([1-4])", str(value).strip())
    if not match:
        return ""
    year = int(match.group(1))
    quarter = int(match.group(2))
    month, day = ((3, 31), (6, 30), (9, 30), (12, 31))[quarter - 1]
    return f"{year}{month:02d}{day:02d}"

def local_eod(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:8]} 23:59:59+08:00"

def period_available_at(value: str) -> tuple[str, str]:
    text = str(value or "").strip()
    if re.fullmatch(r"\d{8}", text):
        return local_eod(text), "conservative_date_eod"
    if re.fullmatch(r"\d{6}", text):
        month_end = parse_yyyymmdd(month_end_from_yyyymm(text))
        available = month_end + timedelta(days=31)
        return local_eod(format_yyyymmdd(available)), "conservative_month_end_plus_31d"
    if re.fullmatch(r"\d{4}Q[1-4]", text):
        quarter_end = quarter_end_date(text)
        available = parse_yyyymmdd(quarter_end) + timedelta(days=45)
        return local_eod(format_yyyymmdd(available)), "conservative_quarter_end_plus_45d"
    return "", "missing_source_date"

def combine_date_time_available(date_value: str, time_value: str) -> str:
    date_text = str(date_value or "").strip()
    time_text = str(time_value or "").strip()
    if not re.fullmatch(r"\d{8}", date_text):
        return ""
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", time_text)
    if not match:
        return ""
    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3) or 0)
    if hour > 23 or minute > 59 or second > 59:
        return ""
    return f"{date_text[:4]}-{date_text[4:6]}-{date_text[6:8]} {hour:02d}:{minute:02d}:{second:02d}+08:00"

def normalize_macro_columns(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Map one macro API response onto the dataset's declared field contract.

    ``spec.fields`` is the single source for a macro table's column names, but
    the vendor does not always answer in them: since 2026-08-13 cn_pmi has
    flapped between the documented lowercase fields and its own database schema
    (uppercase ``MONTH``/``PMI...`` plus ``ID``/``CREATE_*``/``UPDATE_*`` row
    bookkeeping), and the uppercase form costs every row its ``month`` key and
    therefore its availability stamp. Declared fields are matched
    case-insensitively and everything undeclared is dropped, so one dataset
    keeps one column contract whatever casing the API returns."""

    declared = [name.strip() for name in spec.fields.split(",") if name.strip()]
    if not declared:
        return df
    lowered: dict[str, str] = {}
    for column in df.columns:
        lowered.setdefault(str(column).lower(), str(column))
    keep: list[str] = []
    renamed: dict[str, str] = {}
    for name in declared:
        source = lowered.get(name.lower())
        if source is None:
            continue
        keep.append(source)
        if source != name:
            renamed[source] = name
    dropped = [str(column) for column in df.columns if str(column) not in keep]
    if not renamed and not dropped:
        return df
    print(json.dumps(
        {"note": "macro_vendor_schema_normalized", "dataset": spec.api_name,
         "renamed": renamed, "dropped": dropped},
        ensure_ascii=False, sort_keys=True,
    ))
    return df[keep].rename(columns=renamed)

def augment_macro_frame(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    out = normalize_macro_columns(df, spec).copy()
    if "available_at" not in out.columns:
        out["available_at"] = ""
    if "available_at_rule" not in out.columns:
        out["available_at_rule"] = "missing_source_date"
    if spec.date_column and len(out) and spec.date_column not in out.columns:
        # Without the date column every row silently keeps the empty
        # "missing_source_date" stamp and the whole table becomes invisible to
        # PIT replay: refuse the write instead of landing unstamped rows.
        raise RuntimeError(
            f"{spec.api_name} returned no {spec.date_column!r} column "
            f"(got {sorted(str(column) for column in df.columns)[:20]}); "
            "rows cannot be stamped with an availability time"
        )
    if spec.date_column and spec.date_column in out.columns:
        date_values = out[spec.date_column].astype(str).str.strip()
        if spec.time_column and spec.time_column in out.columns:
            time_values = out[spec.time_column].astype(str).str.strip()
            combined = [combine_date_time_available(day, clock) for day, clock in zip(date_values, time_values, strict=False)]
            combined_series = pd.Series(combined, index=out.index)
            mask = combined_series.astype(str).str.strip().ne("")
            out.loc[mask, "available_at"] = combined_series[mask]
            out.loc[mask, "available_at_rule"] = f"source:{spec.date_column}+{spec.time_column}"
        missing = out["available_at"].astype(str).str.strip().eq("")
        derived = date_values.map(period_available_at)
        available = derived.map(lambda item: item[0])
        rules = derived.map(lambda item: item[1])
        mask = missing & available.astype(str).str.strip().ne("")
        out.loc[mask, "available_at"] = available[mask]
        out.loc[mask, "available_at_rule"] = rules[mask]
    return out

def selected_macro_datasets(args: argparse.Namespace) -> list[str]:
    default = GLOBAL_CONTEXT_DEFAULT_DATASETS if args.tier == "global" else MACRO_REGIME_DEFAULT_DATASETS
    return select_datasets(args.datasets, default=default, allowed=MACRO_SPECS, label="macro/global")

def spec_page_limit(spec, requested: int | None) -> int:
    """Effective page size for one dataset spec: the caller's request clamped
    to the spec's documented limit; 10000 when the spec documents none."""
    if spec.page_limit is not None:
        return min(requested or spec.page_limit, spec.page_limit)
    return requested or 10000

# What a repeatable filter option (add_macro_filter_args, add_board_filter_args)
# means when it is not given; None is "no filter".
CLI_LIST_DEFAULTS: dict[str, list[str] | list[None]] = {
    "index_code": DEFAULT_GLOBAL_INDEX_CODES,
    "cn_index_code": DEFAULT_CN_INDEX_CODES,
    "fx_code": DEFAULT_FX_CODES,
    "eco_country": [None],
    "eco_currency": [None],
    "eco_event": [None],
    "kpl_tag": BOARD_KPL_TAGS,
    "ths_limit_type": BOARD_THS_LIMIT_TYPES,
    "ths_hot_market": BOARD_THS_HOT_MARKETS,
    "dc_hot_market": BOARD_DC_HOT_MARKETS,
    "dc_hot_type": BOARD_DC_HOT_TYPES,
    "hot_is_new": BOARD_HOT_IS_NEW,
}

def cli_values(args: argparse.Namespace, attr: str) -> list:
    """The non-blank values given for the repeatable option ``attr``, or its
    default (CLI_LIST_DEFAULTS) when none is given."""
    values = [str(value).strip() for value in getattr(args, attr, None) or [] if str(value).strip()]
    return values or list(CLI_LIST_DEFAULTS[attr])

def by_ts_code_universe(dataset: str, args: argparse.Namespace) -> list[str]:
    """The codes a by-ts_code macro dataset is pulled and audited over."""
    if dataset == "index_global":
        return cli_values(args, "index_code")
    if dataset in ("index_daily", "index_dailybasic"):
        return cli_values(args, "cn_index_code")
    if dataset == "fx_daily":
        return cli_values(args, "fx_code")
    raise RuntimeError(f"no ts_code universe defined for by-ts_code macro dataset {dataset!r}")

def write_frame(
    path: Path,
    df: pd.DataFrame,
    spec: DatasetSpec,
    params: dict[str, Any],
    revision_ledger: Path | str | None = None,
) -> tuple[bool, int]:
    """Publish one augmented answer at ``path`` under ``spec``'s key. Returns
    whether the visible partition changed and the rows that change wrote (0
    when ``publish`` wrote nothing)."""
    outcome = publish(
        path,
        df,
        api_name=spec.api_name,
        params=params,
        fields=list(df.columns),
        key_columns=list(spec.key_columns),
        ledger=revision_ledger,
        deduplicate=spec.unique_key,
    )
    written = outcome in WRITTEN_OUTCOMES
    return written, len(df) if written else 0

def selected_event_flow_datasets(args: argparse.Namespace) -> list[str]:
    return select_datasets(args.datasets, default=EVENT_FLOW_DATASETS, allowed=EVENT_FLOW_SPECS, label="event/flow")

def selected_board_trading_datasets(args: argparse.Namespace) -> list[str]:
    return select_datasets(getattr(args, "datasets", None), default=BOARD_TRADING_DEFAULT_DATASETS, allowed=BOARD_TRADING_SPECS, label="board-trading")

def selected_board_hot_is_new(args: argparse.Namespace) -> list[str]:
    values = [value.upper() for value in cli_values(args, "hot_is_new")]
    invalid = sorted(set(values) - {"Y", "N"})
    if invalid:
        raise RuntimeError(f"invalid hot is_new values: {invalid}; supported=['Y', 'N']")
    return values

def local_time(value: str, clock: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:8]} {clock}+08:00"

# BSE margin trading opened 2023-02-13; before that the exchange-level margin
# table legitimately carries SSE+SZSE only. A day missing a required exchange
# poisons market-wide aggregates (measured: 2026-06 Σrzye −49% craters from
# SSE-only days), so partial days must never be committed.
MARGIN_BSE_START = "20230213"

def margin_missing_exchanges(trade_date: str, present: Iterable[str]) -> list[str]:
    required = {"SSE", "SZSE"}
    if str(trade_date) >= MARGIN_BSE_START:
        required.add("BSE")
    return sorted(required - {str(item) for item in present})


# Full-market per-trade-date tables whose per-day stock universe must track
# the daily table. Measured 2020-2026 per-day floors: cyq_perf 0.9109 (2020
# risk-warning/select-layer gap), moneyflow 0.9401 (the vendor feed carries no
# BSE codes at all), stk_auction 0.9486; every other day sits higher. 0.90
# therefore trips only on a genuinely dropped batch of stocks, never on the
# known structural residuals.
FULL_MARKET_COVERAGE_MIN_RATIO = 0.90
FULL_MARKET_COVERAGE_DATASETS = (
    "bak_basic",
    "bak_daily",
    "cyq_perf",
    "moneyflow",
    "moneyflow_dc",
    "moneyflow_ths",
    "stk_auction",
)

# All margin tables publish per exchange and share the requirement above, but
# they name the exchange differently: the summary and the eligibility list
# carry it as a column, the per-stock detail only through the ts_code
# listing-venue suffix. Holding them here keeps one required-exchange rule and
# lets callers stay dataset-agnostic.
MARGIN_EXCHANGE_BY_TS_CODE_SUFFIX = {"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}
MARGIN_EXCHANGE_COLUMNS = {"margin": "exchange_id", "margin_detail": "ts_code", "margin_secs": "exchange"}

def margin_exchange_column(api_name: str) -> str | None:
    """Column identifying the exchange, or None outside the margin family."""
    return MARGIN_EXCHANGE_COLUMNS.get(str(api_name))

def margin_family_missing_exchanges(api_name: str, trade_date: str, frame: pd.DataFrame) -> list[str]:
    """Required exchanges absent from one margin-family trade-date partition.

    Empty for every other dataset, so callers need no per-dataset branch. An
    unreadable partition fails loudly instead of passing as complete: silently
    skipping the check is what let SSE-only margin_detail days be committed.
    """
    column = margin_exchange_column(api_name)
    if column is None:
        return []
    if column not in frame.columns:
        raise RuntimeError(
            f"{api_name} trade_date={trade_date} has no {column} column; "
            "exchange completeness cannot be verified"
        )
    values = [str(item) for item in frame[column]]
    if column == "ts_code":
        values = [MARGIN_EXCHANGE_BY_TS_CODE_SUFFIX.get(value.rsplit(".", 1)[-1], "") for value in values]
    return margin_missing_exchanges(trade_date, values)

def official_available_at(
    value: str,
    availability: AvailabilityRule | None,
    params: dict[str, Any] | None = None,
    trading_dates: Sequence[str] | None = None,
) -> tuple[str, str]:
    """Stamp the spec's official publication time, or conservative date EOD."""
    text = str(value or "").strip()
    if not re.fullmatch(r"\d{8}", text):
        return "", "missing_source_date"
    rule = availability
    if rule is not None and rule.only_when_is_new and (params or {}).get("is_new") != "Y":
        rule = None
    if rule is None:
        return local_eod(text), "conservative_date_eod"
    day = text
    if rule.trading_day_offset:
        if trading_dates is None:
            raise RuntimeError(f"{rule.rule} requires the A-share trading calendar")
        target_index = bisect_right(trading_dates, text) + rule.trading_day_offset - 1
        if target_index >= len(trading_dates):
            raise RuntimeError(
                f"A-share trading calendar has fewer than {rule.trading_day_offset} open dates after {text}"
            )
        day = str(trading_dates[target_index])
    elif rule.day_offset:
        day = format_yyyymmdd(parse_yyyymmdd(text) + timedelta(days=rule.day_offset))
    return local_time(day, rule.clock), rule.rule

def augment_event_frame(
    df: pd.DataFrame,
    spec: DatasetSpec,
    *,
    trading_dates: Sequence[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if "available_at" not in out.columns:
        out["available_at"] = ""
    if "available_at_rule" not in out.columns:
        out["available_at_rule"] = "missing_source_date"
    if spec.date_column and spec.date_column in out.columns:
        # The vendor has served a padded date ('20250512 ' in stk_holdernumber);
        # stored as served, it is a second spelling of the key for every reader.
        out[spec.date_column] = out[spec.date_column].map(lambda item: item.strip() if isinstance(item, str) else item)
        date_values = out[spec.date_column].astype(str).str.strip()
        derived = date_values.map(
            lambda item: official_available_at(item, spec.availability, trading_dates=trading_dates)
        )
        available = derived.map(lambda item: item[0])
        rules = derived.map(lambda item: item[1])
        mask = available.astype(str).str.strip().ne("")
        out.loc[mask, "available_at"] = available[mask]
        out.loc[mask, "available_at_rule"] = rules[mask]
    if spec.fallback_date_column and spec.fallback_date_column in out.columns:
        missing = out["available_at"].astype(str).str.strip().eq("")
        fallback_values = out[spec.fallback_date_column].astype(str).str.strip()
        derived = fallback_values.map(
            lambda item: official_available_at(item, spec.availability, trading_dates=trading_dates)
        )
        available = derived.map(lambda item: item[0])
        mask = missing & available.astype(str).str.strip().ne("")
        out.loc[mask, "available_at"] = available[mask]
        out.loc[mask, "available_at_rule"] = f"fallback_conservative_from:{spec.fallback_date_column}"
    return out

def normalize_source_datetime(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text in {"nan", "None"}:
        return ""
    if re.fullmatch(r"\d{8} \d{2}:\d{2}:\d{2}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]} {text[9:]}+08:00"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text):
        return f"{text}+08:00"
    return text

def augment_board_frame(df: pd.DataFrame, spec: DatasetSpec, params: dict[str, Any] | None = None) -> pd.DataFrame:
    out = df.copy()
    if "available_at" not in out.columns:
        out["available_at"] = ""
    if "available_at_rule" not in out.columns:
        out["available_at_rule"] = "missing_source_time"
    if spec.time_column and spec.time_column in out.columns:
        source_time = out[spec.time_column].map(normalize_source_datetime)
        mask = source_time.astype(str).str.strip().ne("")
        out.loc[mask, "available_at"] = source_time[mask]
        out.loc[mask, "available_at_rule"] = f"source:{spec.time_column}"
    if spec.date_column and spec.date_column in out.columns:
        missing = out["available_at"].astype(str).str.strip().eq("")
        date_values = out[spec.date_column].astype(str).str.strip()
        derived = date_values.map(lambda item: official_available_at(item, spec.availability, params))
        available = derived.map(lambda item: item[0])
        rules = derived.map(lambda item: item[1])
        mask = missing & available.astype(str).str.strip().ne("")
        out.loc[mask, "available_at"] = available[mask]
        out.loc[mask, "available_at_rule"] = rules[mask]
    if spec.strategy == FetchStrategy.STATIC_FULL and out["available_at"].astype(str).str.strip().eq("").all():
        out["available_at_rule"] = "static_reference_no_pit_time"
    return out

def selected_intraday_datasets(values: Iterable[str] | None) -> list[str]:
    aliases = {"stk_mins": STK_MINS_DATASET}
    normalized = [aliases.get(str(name), str(name)) for name in values or INTRADAY_DATASETS]
    return select_datasets(normalized, default=INTRADAY_DATASETS, allowed=INTRADAY_DATASETS, label="intraday minute")

def clean_yyyymmdd(value: Any) -> str:
    text = str(value or "").strip()
    return text if re.fullmatch(r"\d{8}", text) else ""

def date_overlaps(start_a: str, end_a: str, start_b: str, end_b: str) -> bool:
    return start_a <= end_b and start_b <= end_a


def load_minute_universe(raw_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    files = sorted((raw_dir / "stock_basic").glob("list_status=*.parquet"))
    if not files:
        raise RuntimeError("stock_basic partitions are missing; run download --tier reference first")
    cols = ["ts_code", "name", "market", "exchange", "list_status", "list_date", "delist_date"]
    df = read_many(files, columns=cols)
    if df.empty:
        raise RuntimeError("stock_basic is empty; cannot build full-A minute universe")
    df = df.dropna(subset=["ts_code"]).copy()
    df["ts_code"] = df["ts_code"].astype(str).str.strip()
    df = df[df["ts_code"].ne("")].drop_duplicates("ts_code", keep="first")
    if getattr(args, "codes", None):
        wanted = {str(code).strip() for code in args.codes if str(code).strip()}
        df = df[df["ts_code"].isin(wanted)]
    df = df.sort_values("ts_code").reset_index(drop=True)
    if getattr(args, "max_codes", None):
        df = df.head(int(args.max_codes))
    if df.empty:
        raise RuntimeError("minute universe is empty after code/max-code filtering")
    return df


def active_year_windows(row: pd.Series, start_date: str, end_date: str) -> list[tuple[int, str, str]]:
    list_date = clean_yyyymmdd(row.get("list_date")) or start_date
    delist_date = clean_yyyymmdd(row.get("delist_date")) or end_date
    active_start = max(start_date, list_date)
    active_end = min(end_date, delist_date)
    if active_start > active_end:
        return []
    windows: list[tuple[int, str, str]] = []
    for year in range(int(active_start[:4]), int(active_end[:4]) + 1):
        year_start = max(active_start, f"{year}0101")
        year_end = min(active_end, f"{year}1231")
        if date_overlaps(year_start, year_end, start_date, end_date):
            windows.append((year, year_start, year_end))
    return windows


def minute_datetime(value: str, *, end: bool = False) -> str:
    suffix = "15:00:00" if end else "09:00:00"
    return f"{value[:4]}-{value[4:6]}-{value[6:8]} {suffix}"

def augment_stk_mins_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=STK_MINS_REQUIRED_COLUMNS)
    out = df.copy()
    out["trade_time"] = out["trade_time"].astype(str).str.strip()
    parsed = pd.to_datetime(out["trade_time"], errors="coerce")
    out["trade_date"] = parsed.dt.strftime("%Y%m%d").fillna("")
    out["available_at"] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S+08:00").fillna("")
    out["available_at_rule"] = "source:trade_time_bar_close"
    sort_cols = [col for col in ["ts_code", "trade_time"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    for col in STK_MINS_REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = "" if col in {"ts_code", "trade_time", "trade_date", "available_at", "available_at_rule"} else pd.NA
    return out[STK_MINS_REQUIRED_COLUMNS]

def stk_mins_by_date_path(raw_dir: Path, dataset: str, trade_date: str) -> Path:
    return raw_dir / dataset / f"trade_date={trade_date}.parquet"

def normalize_stk_mins_by_date_frame(df: pd.DataFrame, trade_date: str) -> tuple[pd.DataFrame, dict[str, int]]:
    if df.empty:
        out = pd.DataFrame(columns=STK_MINS_REQUIRED_COLUMNS)
        return out, {"input_rows": 0, "filtered_rows": 0, "duplicate_rows_dropped": 0}
    out = df.copy()
    for col in STK_MINS_REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = "" if col in {"ts_code", "trade_time", "trade_date", "available_at", "available_at_rule"} else pd.NA
    out = out[STK_MINS_REQUIRED_COLUMNS]
    out["ts_code"] = out["ts_code"].astype(str).str.strip()
    out["trade_time"] = out["trade_time"].astype(str).str.strip()
    if "trade_date" not in out or out["trade_date"].astype(str).str.strip().eq("").all():
        parsed = pd.to_datetime(out["trade_time"], errors="coerce")
        out["trade_date"] = parsed.dt.strftime("%Y%m%d").fillna("")
    else:
        out["trade_date"] = out["trade_date"].astype(str).str.strip()
    before_filter = len(out)
    out = out[out["trade_date"] == trade_date].copy()
    if out["available_at"].astype(str).str.strip().eq("").any():
        parsed = pd.to_datetime(out["trade_time"], errors="coerce")
        mask = out["available_at"].astype(str).str.strip().eq("") & parsed.notna()
        out.loc[mask, "available_at"] = parsed[mask].dt.strftime("%Y-%m-%d %H:%M:%S+08:00")
    missing_rule = out["available_at_rule"].astype(str).str.strip().eq("")
    out.loc[missing_rule, "available_at_rule"] = "source:trade_time_bar_close"
    before_dedup = len(out)
    out = out.drop_duplicates(["ts_code", "trade_time"], keep="last")
    out = out.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)
    return out, {
        "input_rows": int(before_filter),
        "filtered_rows": int(before_filter - len(out)),
        "duplicate_rows_dropped": int(before_dedup - len(out)),
    }

def intraday_day_time_details(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "trade_time" not in df.columns:
        return {"unique_times": 0, "has_0930": False, "has_1500": False, "invalid_time_rows": len(df)}
    time_hhmm = df["trade_time"].astype(str).str.extract(r"(\d{2}:\d{2})", expand=False)
    valid = (
        (time_hhmm == "09:30")
        | ((time_hhmm >= "09:31") & (time_hhmm <= "11:30"))
        | ((time_hhmm >= "13:00") & (time_hhmm <= "15:00"))
    )
    times = set(time_hhmm.dropna().tolist())
    return {
        "unique_times": int(time_hhmm.nunique(dropna=True)),
        "has_0930": "09:30" in times,
        "has_1500": "15:00" in times,
        "invalid_time_rows": int((~valid.fillna(False)).sum()),
    }

def validate_stk_mins_by_date_frame(
    df: pd.DataFrame,
    trade_date: str,
    *,
    expected_codes: set[str] | None = None,
    min_rows: int = 0,
    allow_missing_codes: int = 0,
) -> tuple[bool, dict[str, Any]]:
    missing_columns = sorted(set(STK_MINS_REQUIRED_COLUMNS) - set(df.columns))
    details: dict[str, Any] = {
        "trade_date": trade_date,
        "rows": len(df),
        "unique_codes": int(df["ts_code"].nunique()) if "ts_code" in df.columns else 0,
        "missing_columns": missing_columns,
        "duplicate_key_rows": 0,
        "wrong_trade_date_rows": 0,
        "unparseable_trade_time_rows": 0,
        "unparseable_available_at_rows": 0,
        "min_rows": int(min_rows),
        "expected_codes": len(expected_codes or set()),
        "missing_expected_codes": 0,
        "extra_codes": 0,
        "missing_expected_sample": [],
        "extra_code_sample": [],
    }
    if missing_columns:
        return False, details
    details["duplicate_key_rows"] = int(df.duplicated(["ts_code", "trade_time"]).sum())
    details["wrong_trade_date_rows"] = int((df["trade_date"].astype(str) != trade_date).sum())
    parsed_trade = pd.to_datetime(df["trade_time"].astype(str), errors="coerce")
    details["unparseable_trade_time_rows"] = int(parsed_trade.isna().sum())
    available = df["available_at"].astype(str).str.strip()
    parsed_available = pd.to_datetime(available[available.ne("")], errors="coerce", utc=True)
    details["unparseable_available_at_rows"] = int(parsed_available.isna().sum())
    details.update(intraday_day_time_details(df))
    if expected_codes is not None:
        actual_codes = set(df["ts_code"].dropna().astype(str))
        missing = sorted(expected_codes - actual_codes)
        extra = sorted(actual_codes - expected_codes)
        details["missing_expected_codes"] = len(missing)
        details["extra_codes"] = len(extra)
        details["missing_expected_sample"] = missing[:20]
        details["extra_code_sample"] = extra[:20]
    ok = not any([
        missing_columns,
        details["duplicate_key_rows"],
        details["wrong_trade_date_rows"],
        details["unparseable_trade_time_rows"],
        details["unparseable_available_at_rows"],
        not details["has_0930"] if len(df) else False,
        not details["has_1500"] if len(df) else False,
        len(df) < min_rows,
        details["missing_expected_codes"] > allow_missing_codes,
    ])
    return bool(ok), details

def intraday_expected_codes_for_day(raw_dir: Path, args: argparse.Namespace, trade_date: str) -> set[str] | None:
    source = getattr(args, "expected_codes_source", "none")
    if source == "none":
        return None
    if getattr(args, "codes", None):
        codes = {str(code).strip() for code in args.codes if str(code).strip()}
    elif source in {"daily", "minute"}:
        if source == "minute":
            dataset = getattr(args, "output_dataset", STK_MINS_BY_DATE_DATASET)
            existing_path = stk_mins_by_date_path(raw_dir, dataset, trade_date)
            if existing_path.exists() and parquet_rows(existing_path) > 0:
                existing = pd.read_parquet(existing_path, columns=["ts_code"])
                codes = set(existing["ts_code"].dropna().astype(str))
                if getattr(args, "max_codes", None):
                    codes = set(sorted(codes)[: int(args.max_codes)])
                return codes
        daily_path = raw_dir / "daily" / f"trade_date={trade_date}.parquet"
        if not daily_path.exists():
            return set()
        if parquet_rows(daily_path) == 0:
            return set()
        daily = pd.read_parquet(daily_path, columns=["ts_code"])
        codes = set(daily["ts_code"].dropna().astype(str))
    else:
        universe = load_minute_universe(raw_dir, argparse.Namespace(codes=None, max_codes=None))
        active = []
        for _, row in universe.iterrows():
            list_date = clean_yyyymmdd(row.get("list_date")) or "00000000"
            delist_date = clean_yyyymmdd(row.get("delist_date")) or "99999999"
            if list_date <= trade_date <= delist_date:
                active.append(str(row["ts_code"]))
        codes = set(active)
    if getattr(args, "max_codes", None):
        codes = set(sorted(codes)[: int(args.max_codes)])
    return codes

def selected_text_datasets(
    values: Iterable[str] | None,
    *,
    news_src: list[str] | None = None,
    default: Iterable[str] | None = None,
) -> list[str]:
    datasets = select_datasets(
        values,
        default=TEXT_DATASETS if default is None else default,
        allowed=TEXT_SPECS,
        label="text",
    )
    if news_src is not None and "news" in datasets:
        selected_news_sources(news_src)
    return datasets

def selected_news_sources(values: list[str] | None) -> list[str]:
    requested = [str(value).strip() for value in values or [] if str(value).strip()]
    if not requested or any(value.lower() == "all" for value in requested):
        return list(NEWS_SOURCES)
    invalid = sorted(set(requested) - set(NEWS_SOURCES))
    if invalid:
        raise RuntimeError(f"unknown news source(s): {invalid}; official sources are {NEWS_SOURCES}")
    return requested

def date_range_days(start_date: str, end_date: str) -> list[str]:
    start = parse_yyyymmdd(start_date)
    end = parse_yyyymmdd(end_date)
    days = pd.date_range(start, end, freq="D")
    return days.strftime("%Y%m%d").tolist()

def as_datetime_window(value: str, *, end: bool = False) -> str:
    suffix = "23:59:59" if end else "00:00:00"
    return f"{value[:4]}-{value[4:6]}-{value[6:8]} {suffix}"

def safe_partition_value(value: str) -> str:
    """The single authority for raw-lake partition directory names.

    Percent-encoding is reversible and injective: distinct vendor values map to
    distinct directories and every directory decodes back to its value, so the
    downloader, the skip check and the audit's expected-path builder all resolve
    the same tree. It is also what the lake already holds for the Chinese board
    labels (`market=%E7%83%AD%E8%82%A1`, `tag=%E6%B6%A8%E5%81%9C`, ...).

    ASCII values that need no escaping (ts_code, exchange, news src, `all`) keep
    their plain names, so only the CJK partitions are affected by this rule.
    """
    text = str(value).strip()
    if not text:
        return "empty"
    return quote(text, safe="")

# A source timestamp is a credible publication time only when it sits near the
# announcement/report date. Backfilled history carries collection timestamps
# (e.g. 2025 rec_time on 2020 announcements) that must not gate PIT visibility.
# Nor is a time of exactly midnight one: it is a bare date (every 2015 anns_d
# rec_time, most official-service rows since 2026-09-14). Such a row is stamped
# from the lake's own first receipt when that lies in the same window, else
# like a row without a source time (see augment_text_frame). A source time
# that binds (DatasetSpec.source_time_binds) has no upper cap.
TEXT_TIME_PLAUSIBLE_BEFORE_DAYS = 1.0
TEXT_TIME_PLAUSIBLE_AFTER_DAYS = 3.0
# The stamp of a row whose publication time is unknowable: no decision reaches
# it, and it stays inside pandas' nanosecond range so every stamp parser and
# the audit read it.
NEVER_AVAILABLE_AT = "2200-01-01 00:00:00+08:00"


def _landing_run(instant: pd.Series, spec: DatasetSpec) -> pd.Series:
    """The launch of the dataset's latest landing run at or before each CN
    ``instant``: the cutoff the views release that run's rows at."""
    (node,) = TEXT_DATASET_REFRESH_NODES.get(spec.api_name, (TEXT_NODE,))
    start = REFRESH_NODES[node].start
    launch = instant.dt.normalize() + pd.Timedelta(hours=start.hour, minutes=start.minute, seconds=start.second)
    return launch.where(launch <= instant, launch - pd.Timedelta(days=1))


def augment_text_frame(df: pd.DataFrame, spec: DatasetSpec, first_seen: pd.Series | None = None) -> pd.DataFrame:
    """Stamp ``available_at`` and ``available_at_rule`` on a text frame.

    A source time within -1..+3 days of the date column is the publication
    time. A bare date (a source time of exactly 00:00:00) is not; such a row
    takes the launch of the landing run that first brought it into the lake
    when ``first_seen`` (per row, when the lake first held the key; NaT where
    it keeps no such time) puts that in the same window, so a title the 23:15
    run lands reaches the next morning, and the views still hold it until the
    instant itself. Every other row is stamped at the end of its date.

    Where the source time binds (``spec.source_time_binds``) any lag after the
    date is the publication time, a bare date is the end of that date, and a
    row without a source time is stamped ``NEVER_AVAILABLE_AT``."""
    df = df.copy()
    if "available_at" not in df.columns:
        df["available_at"] = ""
        df["available_at_rule"] = "missing_source_time"
    date_value = None
    if spec.date_column and spec.date_column in df.columns:
        date_value = df[spec.date_column].astype(str).str.strip()
    implausible = pd.Series(False, index=df.index)
    date_only = pd.Series(False, index=df.index)
    received = pd.Series(False, index=df.index)
    has_source_time = pd.Series(False, index=df.index)
    if spec.time_column and spec.time_column in df.columns:
        source_time = df[spec.time_column].map(normalize_source_datetime)
        mask = source_time.ne("") & source_time.ne("nan") & source_time.ne("None")
        has_source_time = mask
        if date_value is not None:
            base = pd.to_datetime(date_value, format="%Y%m%d", errors="coerce")
            parsed = pd.to_datetime(source_time.str.slice(0, 19), errors="coerce")
            has_source_time = mask & parsed.notna()
            lag_days = (parsed - base).dt.total_seconds() / 86400.0
            after_days = np.inf if spec.source_time_binds else TEXT_TIME_PLAUSIBLE_AFTER_DAYS
            plausible = lag_days.between(-TEXT_TIME_PLAUSIBLE_BEFORE_DAYS, after_days)
            implausible = mask & base.notna() & ~plausible.fillna(False)
            mask = mask & (base.isna() | plausible.fillna(False))
            date_only = mask & base.notna() & parsed.eq(parsed.dt.normalize())
            mask = mask & ~date_only
            if spec.source_time_binds:
                # A bare publication date is the end of that date, not a first
                # receipt: the lake can hold a key before its text is complete.
                received = date_only
                df.loc[received, "available_at"] = parsed[received].dt.strftime("%Y-%m-%d 23:59:59+08:00")
                df.loc[received, "available_at_rule"] = f"conservative_from:{spec.time_column}"
            elif first_seen is not None and date_only.any():
                landed = _landing_run(first_seen, spec)
                landed_lag = (landed.dt.tz_localize(None) - base).dt.total_seconds() / 86400.0
                received = date_only & landed_lag.between(
                    -TEXT_TIME_PLAUSIBLE_BEFORE_DAYS, TEXT_TIME_PLAUSIBLE_AFTER_DAYS
                ).fillna(False)
                df.loc[received, "available_at"] = landed[received].dt.strftime("%Y-%m-%d %H:%M:%S+08:00")
                df.loc[received, "available_at_rule"] = f"first_seen_for_date_only_{spec.time_column}"
        df.loc[mask, "available_at"] = source_time[mask]
        df.loc[mask, "available_at_rule"] = f"source:{spec.time_column}"
        df.loc[implausible | (date_only & ~received), "available_at"] = ""
    if date_value is not None:
        mask = (df["available_at"].astype(str).str.strip() == "") & date_value.str.fullmatch(r"\d{8}", na=False)
        suffix = " 22:00:00+08:00" if spec.api_name == "report_rc" else " 23:59:59+08:00"
        fallback = date_value.str.slice(0, 4) + "-" + date_value.str.slice(4, 6) + "-" + date_value.str.slice(6, 8) + suffix
        df.loc[mask, "available_at"] = fallback[mask]
        df.loc[mask, "available_at_rule"] = f"conservative_from:{spec.date_column}"
        df.loc[mask & implausible, "available_at_rule"] = (
            f"conservative_from:{spec.date_column}:implausible_{spec.time_column}"
        )
        df.loc[mask & date_only, "available_at_rule"] = (
            f"conservative_from:{spec.date_column}:date_only_{spec.time_column}"
        )
    if spec.source_time_binds:
        df.loc[~has_source_time, "available_at"] = NEVER_AVAILABLE_AT
        df.loc[~has_source_time, "available_at_rule"] = f"never:missing_{spec.time_column}"
    return df

def _rewrite_available_at(
    dataset_dir: Path,
    restamp: Callable[[pd.DataFrame, Path], pd.DataFrame],
    key_columns: Sequence[str],
) -> tuple[int, int]:
    """Re-derive the two stamp columns of every partition under ``dataset_dir``
    (``restamp`` gets the frame without them and the partition's path).

    A partition whose stamp or rule label differs is republished through the
    versioned write path: the restamped rows are a restatement, so the old
    partition is kept as a superseded version and the ledger records it. Only
    committed partitions are touched. Returns (files rewritten, rows whose stamp
    moved).
    """
    ledger = resolve_revision_ledger(dataset_dir.parent, REVISION_EVENTS_PATH, repo_root=dataset_dir.parent.parent.parent)
    files = 0
    changed_rows = 0
    for path in sorted(dataset_dir.rglob("*.parquet")):
        if not committed_partition_intact(path):
            raise RuntimeError(f"refusing to restamp a partition that is not a complete commit: {path}")
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        before = frame.get("available_at", pd.Series("", index=frame.index)).astype(str)
        repaired = restamp(frame.drop(columns=[c for c in ("available_at", "available_at_rule") if c in frame.columns]), path)
        meta = parquet_meta(path)
        outcome = publish(
            path,
            repaired,
            api_name=str(meta.get("api_name") or dataset_dir.name),
            params=dict(meta.get("params") or {}),
            fields=list(meta.get("fields") or repaired.columns),
            key_columns=key_columns,
            ledger=ledger,
            source="local_restamp",
            current=frame,
        )
        if outcome in WRITTEN_OUTCOMES:
            files += 1
            changed_rows += int((repaired["available_at"].astype(str) != before).sum())
    return files, changed_rows


def _restamp_text(frame: pd.DataFrame, path: Path, *, spec: DatasetSpec) -> pd.DataFrame:
    """A stored text partition under the current rule; its keys are all held,
    so their first-seen rows are the whole receipt record."""
    return augment_text_frame(frame, spec, first_seen_at(path, frame))


def _restamp_event(frame: pd.DataFrame, _path: Path, *, spec: DatasetSpec, trading_dates: list[str]) -> pd.DataFrame:
    return augment_event_frame(frame, spec, trading_dates=trading_dates)


def repair_text_available_at(raw_dir: str, datasets: list[str]) -> dict[str, Any]:
    """Re-derive available_at for existing text partitions under the current rule.

    Pure local rewrite of the two derived columns; source fields and sidecars
    stay untouched.
    """
    stats: dict[str, Any] = {"datasets": {}, "files_rewritten": 0, "rows_changed": 0}
    for dataset in datasets:
        spec = TEXT_SPECS.get(dataset)
        if spec is None:
            raise RuntimeError(f"unknown text dataset: {dataset}")
        dataset_dir = Path(raw_dir) / dataset
        if not dataset_dir.exists():
            raise RuntimeError(f"missing dataset directory: {dataset_dir}")
        files, changed_rows = _rewrite_available_at(dataset_dir, partial(_restamp_text, spec=spec), spec.key_columns)
        stats["datasets"][dataset] = {"files_rewritten": files, "rows_changed": changed_rows}
        stats["files_rewritten"] += files
        stats["rows_changed"] += changed_rows
    return stats


def repair_event_available_at(raw_dir: str, datasets: list[str]) -> dict[str, Any]:
    """Re-derive available_at for stored trade-date event partitions under the current rule.

    The same local rewrite as the text repair. Rules counted in open days read
    the SSE calendar, which must reach the open day after the newest partition.
    """
    raw = Path(raw_dir)
    stats: dict[str, Any] = {"datasets": {}, "files_rewritten": 0, "rows_changed": 0}
    for dataset in datasets:
        spec = EVENT_FLOW_SPECS.get(dataset)
        if spec is None or spec.strategy != FetchStrategy.TRADE_DATE:
            raise RuntimeError(f"event available_at repair covers trade-date event datasets; got {dataset}")
        partitions = sorted((raw / dataset).glob("trade_date=*.parquet"))
        if not partitions:
            raise RuntimeError(f"no partitions under {raw / dataset}")
        open_dates = load_sse_open_dates(raw, partition_value(partitions[0]), latest_sse_calendar_date(raw))
        files, changed_rows = _rewrite_available_at(
            raw / dataset, partial(_restamp_event, spec=spec, trading_dates=open_dates), spec.key_columns
        )
        stats["datasets"][dataset] = {"files_rewritten": files, "rows_changed": changed_rows}
        stats["files_rewritten"] += files
        stats["rows_changed"] += changed_rows
    return stats


def add_raw_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-dir", default="data/raw")

def add_macro_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-code", action="append", default=[], help="Global index ts_code for index_global; repeatable. Defaults to a curated major-index list.")
    parser.add_argument("--cn-index-code", action="append", default=[], help="A-share index ts_code for index_daily; repeatable. Defaults to the core benchmark set (上证指数/上证50/沪深300/中证500/中证1000/创业板指/科创50).")
    parser.add_argument("--fx-code", action="append", default=[], help="FX ts_code for fx_daily; repeatable. Defaults to USDCNH.FXCM.")
    parser.add_argument("--eco-country", action="append", default=[], help="Optional eco_cal country filter; repeatable. Omit for all countries.")
    parser.add_argument("--eco-currency", action="append", default=[], help="Optional eco_cal currency filter; repeatable. Omit for all currencies.")
    parser.add_argument("--eco-event", action="append", default=[], help="Optional eco_cal event fuzzy filter; repeatable. Omit for all events.")

def add_board_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--kpl-tag", action="append", default=[], help="开盘啦榜单 tag; repeatable. Defaults to 涨停/炸板/跌停/自然涨停/竞价.")
    parser.add_argument("--ths-limit-type", action="append", default=[], help="同花顺涨跌停榜单 limit_type; repeatable. Defaults to all official THS pools.")
    parser.add_argument("--ths-hot-market", action="append", default=[], help="同花顺热榜 market; repeatable. Defaults to 热股/行业板块/概念板块.")
    parser.add_argument("--dc-hot-market", action="append", default=[], help="东方财富热榜 market; repeatable. Defaults to A股市场.")
    parser.add_argument("--dc-hot-type", action="append", default=[], help="东方财富热榜 hot_type; repeatable. Defaults to 人气榜/飙升榜.")
    parser.add_argument("--hot-is-new", action="append", default=[], help="Hot-list is_new value Y or N; repeatable. Defaults to N for PIT rank_time snapshots.")

def add_runtime_args(parser: argparse.ArgumentParser, *, min_interval: float | None, timeout: int | None) -> None:
    parser.add_argument("--min-interval-seconds", type=float, default=min_interval)
    parser.add_argument("--timeout-seconds", type=int, default=timeout)

def add_intraday_by_date_common_args(
    parser: argparse.ArgumentParser,
    *,
    expected_codes_choices: list[str] | None = None,
    expected_codes_default: str = "none",
) -> None:
    expected_choices = expected_codes_choices or ["none", "daily", "active", "minute"]
    add_raw_arg(parser)
    parser.add_argument("--start-date", default="20200101")
    parser.add_argument("--end-date", default=cn_today())
    parser.add_argument("--output-dataset", default=STK_MINS_BY_DATE_DATASET)
    parser.add_argument("--codes", nargs="+", help="Optional explicit ts_code list for window tests or targeted refreshes.")
    parser.add_argument("--max-codes", type=int, help="Optional first-N stock_basic/daily codes for window tests.")
    parser.add_argument(
        "--expected-codes-source",
        choices=expected_choices,
        default=expected_codes_default,
        help=f"Coverage universe for validation: {', '.join(expected_choices)}.",
    )
    parser.add_argument("--min-rows-per-day", type=int, default=0)
    parser.add_argument("--allow-missing-codes", type=int, default=0)
    parser.add_argument("--existing-allow-missing-codes", type=int, default=50)
