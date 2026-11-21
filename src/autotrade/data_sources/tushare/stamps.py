"""When a vendor row becomes visible: the point-in-time stamping rules that give
every macro, event, board and text row its available_at, and the repairs that
restamp stored partitions through the versioned write path.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections.abc import Callable, Sequence
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autotrade.environment.data.contracts import (
    REFRESH_NODES,
    TEXT_DATASET_REFRESH_NODES,
    TEXT_NODE,
)
from autotrade.environment.data.pit import first_seen_at, partition_value

from .common import (
    REVISION_EVENTS_PATH,
    WRITTEN_OUTCOMES,
    format_yyyymmdd,
    latest_sse_calendar_date,
    load_sse_open_dates,
    month_end_from_yyyymm,
    parse_yyyymmdd,
    resolve_revision_ledger,
)
from .io import committed_partition_intact, parquet_meta
from .specs import (
    EVENT_FLOW_SPECS,
    TEXT_SPECS,
    AvailabilityRule,
    DatasetSpec,
    FetchStrategy,
)
from .versions import publish


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


def local_time(value: str, clock: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:8]} {clock}+08:00"


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
