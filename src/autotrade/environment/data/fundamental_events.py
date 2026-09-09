from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
import json

import pandas as pd
import pyarrow.parquet as pq

from autotrade.data_quality import build_quality_report, write_quality_report

from autotrade.environment.data.contracts import CN_TZ
from autotrade.environment.data.pit import concat_rows, parquet_meta, to_cn_timestamps, yyyymmdd


# Provenance/PIT columns the store stamps onto every fundamental event row on
# top of the raw vendor schema (single source; snapshot attribution and the
# schema-inventory exporter both reference it).
FUNDAMENTAL_SIDECAR_COLUMNS = (
    "dataset", "source_path", "source_write_id", "source_row_id", "business_key",
    "available_month", "available_at", "available_at_rule",
)

FUNDAMENTAL_EVENT_DATASETS = (
    "income_vip",
    "balancesheet_vip",
    "cashflow_vip",
    "fina_indicator_vip",
    "forecast_vip",
    "express_vip",
    "dividend",
    "fina_audit",
    "fina_mainbz_vip",
    "disclosure_date",
)

BUSINESS_KEYS = {
    "income_vip": ("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"),
    "balancesheet_vip": ("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"),
    "cashflow_vip": ("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"),
    "fina_indicator_vip": ("ts_code", "ann_date", "end_date"),
    "forecast_vip": ("ts_code", "ann_date", "end_date", "type", "first_ann_date", "update_flag"),
    "express_vip": ("ts_code", "ann_date", "end_date"),
    "dividend": ("ts_code", "end_date", "ann_date", "div_proc", "record_date", "ex_date", "pay_date"),
    "fina_audit": ("ts_code", "ann_date", "end_date"),
    "fina_mainbz_vip": ("ts_code", "end_date", "bz_item", "bz_code", "curr_type"),
    "disclosure_date": ("ts_code", "end_date", "ann_date", "pre_date", "actual_date"),
}

# Fundamental events are stamped at the CN evening clock: the vendor gives a
# date, not a time, and 18:00 is after the disclosure cut-off for that date.
_EVENT_CLOCK = time(18, 0)

RAW_PATTERNS = {
    "income_vip": "period=*.parquet",
    "balancesheet_vip": "period=*.parquet",
    "cashflow_vip": "period=*.parquet",
    "fina_indicator_vip": "period=*.parquet",
    "forecast_vip": "ann_month=*.parquet",
    "express_vip": "ann_month=*.parquet",
    "dividend": "ts_code=*.parquet",
    "fina_audit": "ts_code=*.parquet",
    "fina_mainbz_vip": "ts_code=*.parquet",
    "disclosure_date": "period=*.parquet",
}


@dataclass(frozen=True)
class FundamentalEventsConfig:
    start_date: str
    end_date: str
    datasets: tuple[str, ...] = field(default_factory=lambda: FUNDAMENTAL_EVENT_DATASETS)


class FundamentalEventsBuilder:
    """Build PIT-ready financial/event records from raw TuShare fundamental files."""

    def __init__(self, raw_dir: str | Path) -> None:
        self.raw_dir = Path(raw_dir)

    def build(self, config: FundamentalEventsConfig) -> pd.DataFrame:
        datasets = tuple(config.datasets or FUNDAMENTAL_EVENT_DATASETS)
        statement_availability = self._statement_availability()
        frames = [self._read_dataset(dataset, statement_availability) for dataset in datasets]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame(columns=self._event_columns())
        events = concat_rows(frames)
        events = events[events["available_at"].astype(str).str.strip().ne("")].copy()
        if events.empty:
            return pd.DataFrame(columns=self._event_columns())
        parsed = to_cn_timestamps(events["available_at"])
        start = pd.Timestamp(yyyymmdd(config.start_date)).tz_localize(CN_TZ)
        end = pd.Timestamp(yyyymmdd(config.end_date)).tz_localize(CN_TZ) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        window = (parsed >= start) & (parsed <= end)
        events = events[window].copy()
        if events.empty:
            return pd.DataFrame(columns=self._event_columns())
        events["available_month"] = parsed[window].dt.strftime("%Y%m")
        # Same-timestamp statement corrections share the entire business key
        # (update_flag is deliberately not part of statement keys), and the
        # write-side collapse keeps the LAST row per key: rank vendor versions
        # explicitly so the update_flag=1 revision survives instead of leaving
        # the outcome to raw response row order.
        if "update_flag" in events.columns:
            events["_version_rank"] = pd.to_numeric(events["update_flag"], errors="coerce").fillna(-1.0)
        else:
            events["_version_rank"] = -1.0
        events = events.sort_values(["available_at", "dataset", "ts_code", "business_key", "_version_rank"])
        events = events.drop(columns=["_version_rank"]).reset_index(drop=True)
        collapsed = int(events.duplicated(["dataset", "business_key", "available_at"]).sum())
        if collapsed:
            print(f"fundamental_events build: {collapsed} same-key duplicate rows collapse to the highest version on write")
        return events

    def write_partitioned(
        self,
        events: pd.DataFrame,
        output_root: str | Path,
        replace_months: set[str] | None = None,
        replace_datasets: tuple[str, ...] | None = None,
    ) -> list[Path]:
        output_root = Path(output_root)
        written: list[Path] = []
        replace_months = replace_months or set()
        if replace_datasets is None:
            replace_datasets = tuple(events["dataset"].dropna().astype(str).unique()) if not events.empty else ()
        # Refuse to destroy history: a replaced dataset that produced ZERO
        # events while its existing replace-window partitions still hold rows
        # means the raw side vanished or broke (a missing raw dir reads as
        # empty, not as an error). Replacing real rows with markers would
        # silently blank the dataset for every snapshot, so fail BEFORE any
        # deletion happens.
        incoming_rows = events["dataset"].astype(str).value_counts().to_dict() if not events.empty else {}
        for dataset in replace_datasets:
            if incoming_rows.get(str(dataset), 0):
                continue
            for month in sorted(replace_months):
                existing = output_root / dataset / f"available_month={month}.parquet"
                if existing.exists() and len(pd.read_parquet(existing, columns=["dataset"])) > 0:
                    raise RuntimeError(
                        f"dataset {dataset} produced no events for the replace window but "
                        f"{existing} still holds rows; refusing to replace history with empty markers"
                    )
        for dataset in replace_datasets:
            for month in replace_months:
                stale_path = output_root / dataset / f"available_month={month}.parquet"
                if stale_path.exists():
                    stale_path.unlink()
        covered: set[tuple[str, str]] = set()
        if not events.empty:
            for (dataset, month), group in events.groupby(["dataset", "available_month"], sort=True):
                path = output_root / dataset / f"available_month={month}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and month not in replace_months:
                    existing = pd.read_parquet(path)
                    group = concat_rows([existing, group])
                dedupe_cols = [col for col in ["dataset", "business_key", "available_at"] if col in group.columns]
                if dedupe_cols:
                    group = group.drop_duplicates(dedupe_cols, keep="last")
                tmp = path.with_suffix(path.suffix + ".tmp")
                group.to_parquet(tmp, index=False)
                tmp.replace(path)
                written.append(path)
                covered.add((str(dataset), str(month)))
        # A replaced month with no events gets an explicit zero-row marker so
        # "this month genuinely has no events" stays distinguishable from
        # "the builder never produced this month" (which the audit flags).
        empty_marker = pd.DataFrame({column: pd.Series(dtype="object") for column in self._event_columns()})
        for dataset in replace_datasets:
            for month in sorted(replace_months):
                if (dataset, month) in covered:
                    continue
                path = output_root / dataset / f"available_month={month}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(path.suffix + ".tmp")
                empty_marker.to_parquet(tmp, index=False)
                tmp.replace(path)
                written.append(path)
        return written

    def _read_dataset(self, dataset: str, statement_availability: dict[tuple[str, str], str]) -> pd.DataFrame:
        dataset_dir = self.raw_dir / dataset
        if not dataset_dir.exists():
            return pd.DataFrame(columns=self._event_columns())
        frames: list[pd.DataFrame] = []
        for path in sorted(dataset_dir.glob(RAW_PATTERNS[dataset])):
            df = pd.read_parquet(path)
            if df.empty:
                continue
            df = df.copy()
            df["dataset"] = dataset
            df["source_path"] = str(path)
            df["source_write_id"] = str(parquet_meta(path).get("write_id", ""))
            df["source_row_id"] = range(len(df))
            df["available_at"], df["available_at_rule"] = _available_at_for_frame(
                dataset, df, statement_availability
            )
            df["business_key"] = _business_key_frame(dataset, df)
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=self._event_columns())
        return concat_rows(frames)

    def _statement_availability(self) -> dict[tuple[str, str], str]:
        """Latest statement announcement per ``(ts_code, end_date)``.

        The fina_audit / fina_mainbz_vip rules join against this when the vendor
        left their own ann_date blank.
        """
        keyed: list[pd.DataFrame] = []
        for dataset in ("income_vip", "balancesheet_vip", "cashflow_vip", "fina_indicator_vip"):
            dataset_dir = self.raw_dir / dataset
            if not dataset_dir.exists():
                continue
            for path in sorted(dataset_dir.glob("period=*.parquet")):
                # The lookup needs the key and the announcement dates only; the
                # full ~100-column statement schema is read once, by
                # _read_dataset. Statement files carry no other rule input.
                wanted = ("ts_code", "end_date", "f_ann_date", "ann_date")
                present = set(pq.read_schema(path).names)
                df = pd.read_parquet(path, columns=[column for column in wanted if column in present])
                if df.empty:
                    continue
                available, _rules = _available_at_for_frame(dataset, df, {})
                frame = pd.DataFrame({
                    "ts_code": _column_text(df, "ts_code"),
                    "end_date": _clean_date_frame(df, "end_date"),
                    "available_at": available,
                })
                keyed.append(frame[frame["ts_code"].ne("") & frame["end_date"].ne("") & frame["available_at"].ne("")])
        combined = concat_rows(keyed) if keyed else pd.DataFrame()
        if combined.empty:
            return {}
        # Latest stamp wins per key, resolved once over the whole history: every
        # stamp is a same-clock ISO value, so a string sort is chronological.
        combined = combined.sort_values("available_at").drop_duplicates(["ts_code", "end_date"], keep="last")
        return dict(zip(zip(combined["ts_code"], combined["end_date"]), combined["available_at"]))

    @staticmethod
    def _event_columns() -> list[str]:
        return ["dataset", "ts_code", "available_at", "available_at_rule", "available_month", "business_key", "source_path", "source_write_id", "source_row_id"]


def audit_fundamental_events(events_root: str | Path, config: FundamentalEventsConfig, output: str | Path | None = None, require_partitions: bool = False) -> dict[str, object]:
    root = Path(events_root)
    datasets = tuple(config.datasets or FUNDAMENTAL_EVENT_DATASETS)
    start_ts = pd.Timestamp(yyyymmdd(config.start_date)).tz_localize(CN_TZ)
    end_ts = pd.Timestamp(yyyymmdd(config.end_date)).tz_localize(CN_TZ) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    expected_months = set(_month_keys_between(config.start_date, config.end_date))
    checks: list[dict[str, object]] = []
    total_rows = 0
    for dataset in datasets:
        files = sorted(
            path for path in (root / dataset).glob("available_month=*.parquet")
            if path.stem.split("=", 1)[1] in expected_months
        )
        file_months = {path.stem.split("=", 1)[1] for path in files}
        missing_months = sorted(expected_months - file_months)
        if not files:
            checks.append({"severity": "error" if require_partitions else "warning", "check": f"{dataset}_partitions", "message": "no PIT event partitions", "details": {"missing_months": missing_months}})
            continue
        if missing_months:
            checks.append({"severity": "error" if require_partitions else "warning", "check": f"{dataset}_missing_months", "message": "missing PIT event months in requested audit window", "details": {"missing_months": missing_months[:24], "missing_month_count": len(missing_months)}})
        rows = 0
        unparseable = 0
        wrong_partition = 0
        wrong_dataset = 0
        disallowed_rules = 0
        outside_window = 0
        blank_source_write_id = 0
        sidecar_write_id_mismatch = 0
        blank_source_path = 0
        wrong_source_path = 0
        missing_source_path = 0
        bad_source_row_id = 0
        duplicate_keys = 0
        backdated_available_at = 0
        for path in files:
            expected_dataset = path.parent.name
            df = pd.read_parquet(path)
            rows += len(df)
            missing = {"dataset", "ts_code", "available_at", "available_at_rule", "available_month", "business_key", "source_path", "source_write_id", "source_row_id"} - set(df.columns)
            if missing:
                checks.append({"severity": "error", "check": f"{dataset}_schema", "message": f"missing columns in {path}", "details": {"missing": sorted(missing)}})
                continue
            parsed = to_cn_timestamps(df["available_at"])
            unparseable += int(parsed.isna().sum())
            outside_window += int(((parsed < start_ts) | (parsed > end_ts)).sum())
            expected_month = path.stem.split("=", 1)[1]
            wrong_partition += int((df["available_month"].astype(str) != expected_month).sum())
            wrong_dataset += int((df["dataset"].astype(str) != expected_dataset).sum())
            disallowed_rules += int((~df["available_at_rule"].astype(str).map(_is_allowed_available_at_rule)).sum())
            blank_source_write_id += int(df["source_write_id"].astype(str).str.strip().eq("").sum())
            source_paths = df["source_path"].astype(str)
            blank_source_path += int(source_paths.str.strip().eq("").sum())
            wrong_source_path += int((~source_paths.map(lambda value: _source_path_matches_dataset(value, expected_dataset))).sum())
            path_exists = {value: Path(value).exists() for value in source_paths.unique()}
            missing_source_path += int((~source_paths.map(path_exists)).sum())
            bad_source_row_id += int(pd.to_numeric(df["source_row_id"], errors="coerce").isna().sum())
            # Event partitions carry millions of rows referencing thousands of
            # distinct raw files: read each sidecar once, then compare columns.
            # A missing raw file reads as "" (counted via missing_source_path);
            # a corrupt sidecar raises through parquet_meta.
            sidecar_write_ids = {}
            for value in source_paths.unique():
                raw_path = Path(str(value))
                sidecar_write_ids[value] = (
                    str(parquet_meta(raw_path).get("write_id", "")) if raw_path.exists() else ""
                )
            expected_write_id = source_paths.map(sidecar_write_ids)
            actual_write_id = df["source_write_id"].astype(str).str.strip()
            sidecar_write_id_mismatch += int(
                (expected_write_id.ne("") & actual_write_id.ne("") & expected_write_id.ne(actual_write_id)).sum()
            )
            duplicate_keys += int(df.duplicated(["dataset", "business_key", "available_at"], keep=False).sum())
            if expected_dataset in {"forecast_vip", "express_vip"} and "ann_date" in df.columns:
                ann = pd.to_datetime(df["ann_date"].astype(str).str.strip(), format="%Y%m%d", errors="coerce")
                ann = ann.dt.tz_localize(CN_TZ)
                # For forecast/express each version's ann_date IS its disclosure
                # date; available_at before it is determinate lookahead (the
                # first_ann_date bug class). Statement datasets are exempt:
                # f_ann_date legitimately precedes ann_date.
                backdated_available_at += int((parsed < ann).fillna(False).sum())
        total_rows += rows
        severity = (
            "error"
            if unparseable or wrong_partition or wrong_dataset or disallowed_rules or outside_window
            or blank_source_path or wrong_source_path or missing_source_path or bad_source_row_id
            or backdated_available_at or sidecar_write_id_mismatch
            else "warning" if duplicate_keys or blank_source_write_id else "info"
        )
        checks.append({
            "severity": severity,
            "check": f"{dataset}_pit_events",
            "message": f"{dataset} PIT event partition checks",
            "details": {
                "files": len(files),
                "rows": rows,
                "unparseable_available_at_rows": unparseable,
                "outside_audit_window_rows": outside_window,
                "wrong_available_month_rows": wrong_partition,
                "wrong_dataset_rows": wrong_dataset,
                "disallowed_available_at_rule_rows": disallowed_rules,
                "blank_source_write_id_rows": blank_source_write_id,
                "sidecar_write_id_mismatch_rows": sidecar_write_id_mismatch,
                "backdated_available_at_rows": backdated_available_at,
                "blank_source_path_rows": blank_source_path,
                "wrong_source_path_rows": wrong_source_path,
                "missing_source_path_rows": missing_source_path,
                "bad_source_row_id_rows": bad_source_row_id,
                "duplicate_dataset_business_key_available_at_rows": duplicate_keys,
            },
        })
    if require_partitions and total_rows == 0:
        checks.append({"severity": "error", "check": "fundamental_events_partitions", "message": "no PIT event rows found for required audit window", "details": {"start_date": config.start_date, "end_date": config.end_date}})
    report = build_quality_report(
        report_type="fundamental_events",
        scope={
            "data_root": str(root.resolve()),
            "start_date": config.start_date,
            "end_date": config.end_date,
            "datasets": list(datasets),
        },
        findings=checks,
        metadata={"rows": total_rows, "require_partitions": bool(require_partitions)},
    )
    if output:
        write_quality_report(output, report)
    return report


def read_fundamental_events(
    events_root: str | Path,
    max_available_at: str,
    datasets: tuple[str, ...] = FUNDAMENTAL_EVENT_DATASETS,
    *,
    min_available_at: str | None = None,
    require_partitions: bool = False,
    nat_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Read the PIT event store up to ``max_available_at``.

    ``nat_counts`` is the same out-parameter the snapshot's raw-window reader
    takes: an unparseable stamp fails both bounds and is dropped in the
    conservative direction (hidden, never leaked), and the per-dataset count is
    accumulated here so the caller can publish it as the domain manifest's
    ``unparseable_available_at_dropped``.
    """
    root = Path(events_root)
    datasets = tuple(datasets or ())
    if not datasets:
        return pd.DataFrame()
    if require_partitions and not root.exists():
        raise FileNotFoundError(f"missing PIT fundamental events root: {root}")
    max_ts = pd.Timestamp(max_available_at)
    min_ts = pd.Timestamp(min_available_at) if min_available_at else None
    min_month = min_ts.strftime("%Y%m") if min_ts is not None else None
    max_month = max_ts.strftime("%Y%m")
    frames: list[pd.DataFrame] = []
    available_partition_count = 0
    missing_datasets: list[str] = []
    for dataset in datasets:
        dataset_dir = root / dataset
        if not dataset_dir.exists():
            if require_partitions:
                missing_datasets.append(dataset)
            continue
        dataset_available_partition_count = 0
        for path in sorted(dataset_dir.glob("available_month=*.parquet")):
            month = path.stem.split("=", 1)[1]
            if month > max_month:
                continue
            dataset_available_partition_count += 1
            available_partition_count += 1
            if min_month is not None and month < min_month:
                continue
            df = pd.read_parquet(path)
            if not df.empty:
                frames.append(df)
        if require_partitions and dataset_available_partition_count == 0:
            missing_datasets.append(dataset)
    if require_partitions and missing_datasets:
        raise FileNotFoundError(
            f"missing PIT fundamental event partitions at or before {max_month} under {root} "
            f"for datasets: {', '.join(missing_datasets)}"
        )
    if require_partitions and available_partition_count == 0:
        raise FileNotFoundError(
            f"no PIT fundamental event partitions at or before {max_month} under {root} "
            f"for datasets: {', '.join(datasets)}"
        )
    if not frames:
        return pd.DataFrame()
    events = concat_rows(frames)
    parsed = to_cn_timestamps(events["available_at"])
    if nat_counts is not None:
        unparseable = parsed.isna()
        if unparseable.any():
            for dataset, count in events.loc[unparseable, "dataset"].astype(str).value_counts().items():
                nat_counts[str(dataset)] = nat_counts.get(str(dataset), 0) + int(count)
    visible = parsed <= max_ts
    if min_ts is not None:
        visible &= parsed >= min_ts
    return events[visible].copy()


# Statement history is millions of rows wide by ~100 vendor columns, so every
# rule below is a frame operation: a per-row dict of the whole vendor schema
# cost 18 minutes of the nightly PIT event build on its own.
def _available_at_for_frame(
    dataset: str, frame: pd.DataFrame, statement_availability: dict[tuple[str, str], str]
) -> tuple[pd.Series, pd.Series]:
    """``(available_at, available_at_rule)`` for one raw dataset's frame."""
    if dataset in {"income_vip", "balancesheet_vip", "cashflow_vip"}:
        return _first_available_frame(frame, ("f_ann_date", "ann_date"), "source:f_ann_date_or_ann_date")
    if dataset == "fina_indicator_vip":
        return _first_available_frame(frame, ("ann_date",), "source:ann_date")
    if dataset in {"forecast_vip", "express_vip"}:
        # Each VERSION becomes visible at its OWN announcement date. Using
        # first_ann_date backdated revised figures to the first announcement
        # (a Jan-2026 revision was visible at Aug-2024 in a real snapshot) —
        # determinate PIT lookahead. first_ann_date stays as a series attribute.
        return _first_available_frame(frame, ("ann_date",), "source:ann_date")
    if dataset == "dividend":
        available, rules = _first_available_frame(frame, ("imp_ann_date", "ann_date"), "source:imp_ann_date_or_ann_date")
        rules = rules.mask(available.eq(""), "missing_announcement_date_not_pit_visible")
        return available, rules
    if dataset in {"fina_audit", "fina_mainbz_vip"}:
        available, rules = _first_available_frame(frame, ("ann_date",), "source:ann_date")
        missing = available.eq("")
        if missing.any() and statement_availability:
            keys = zip(_column_text(frame, "ts_code")[missing], _clean_date_frame(frame, "end_date")[missing])
            joined = pd.Series(
                [statement_availability.get(key, "") for key in keys],
                index=frame.index[missing],
                dtype=object,
            )
            available = available.mask(missing, joined)
            rules = rules.mask(missing & available.ne(""), "fallback_joined_statement_available_at")
        return available, rules
    if dataset == "disclosure_date":
        return _first_available_frame(frame, ("ann_date", "actual_date", "pre_date"), "source:ann_or_conservative_disclosure_date")
    return (
        pd.Series("", index=frame.index, dtype=object),
        pd.Series("unsupported_dataset", index=frame.index, dtype=object),
    )


def _first_available_frame(frame: pd.DataFrame, columns: tuple[str, ...], rule: str) -> tuple[pd.Series, pd.Series]:
    """First non-blank date across ``columns``, stamped at the event clock.

    A column the vendor never wrote is all-blank, exactly as a missing dict key
    was; the rule string names the fallback column whenever it is not the first.
    """
    dates = pd.Series("", index=frame.index, dtype=object)
    rules = pd.Series("missing_source_date", index=frame.index, dtype=object)
    unresolved = pd.Series(True, index=frame.index)
    for column in columns:
        if not unresolved.any():
            break
        candidate = _clean_date_frame(frame, column)
        take = unresolved & candidate.ne("")
        dates = dates.mask(take, candidate)
        rules = rules.mask(take, rule if column == columns[0] else f"{rule}:{column}")
        unresolved &= ~take
    return _stamp_event_clock(dates), rules


def _stamp_event_clock(dates: pd.Series) -> pd.Series:
    """``_date_at`` over a cleaned YYYYMMDD series.

    Applied to the DISTINCT dates so the CN offset stays whatever the zone says
    for that day (China ran DST 1986-1991); a literal "+08:00" would not.
    """
    stamps = {value: _date_at(value, _EVENT_CLOCK) for value in dates.unique() if value}
    stamps[""] = ""
    return dates.map(stamps)


def _column_text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype=object)
    return frame[column].astype(str)


def _clean_date_frame(frame: pd.DataFrame, column: str) -> pd.Series:
    """``_clean_date`` over a column, absent columns reading as blank."""
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype=object)
    values = frame[column]
    text = values.astype(str).str.strip()
    blank = text.eq("") | text.str.lower().isin({"nan", "none", "nat"})
    dated = ~blank & text.str.fullmatch(r"\d{8}").fillna(False)
    out = pd.Series("", index=values.index, dtype=object)
    out = out.mask(dated, text)
    # Anything that is neither blank nor a bare YYYYMMDD (a vendor timestamp,
    # a date object) is rare: resolve those through the scalar rule on the
    # ORIGINAL value so the two paths cannot diverge on an odd form.
    odd = ~blank & ~dated
    if odd.any():
        out = out.mask(odd, values[odd].map(_clean_date))
    return out


def _is_allowed_available_at_rule(rule: str) -> bool:
    text = str(rule)
    allowed_rules = {
        "source:f_ann_date_or_ann_date",
        "source:f_ann_date_or_ann_date:ann_date",
        "source:ann_date",
        "source:imp_ann_date_or_ann_date",
        "source:imp_ann_date_or_ann_date:ann_date",
        "fallback_joined_statement_available_at",
        "source:ann_or_conservative_disclosure_date",
        "source:ann_or_conservative_disclosure_date:actual_date",
        "source:ann_or_conservative_disclosure_date:pre_date",
    }
    return text in allowed_rules


def _month_keys_between(start_date: str, end_date: str) -> list[str]:
    start = pd.Timestamp(yyyymmdd(start_date))
    end = pd.Timestamp(yyyymmdd(end_date))
    current = pd.Timestamp(year=start.year, month=start.month, day=1)
    months: list[str] = []
    while current <= end:
        months.append(current.strftime("%Y%m"))
        current = current + pd.DateOffset(months=1)
    return months


def month_aligned_replace_window(start_date: str, end_date: str) -> tuple[str, set[str]]:
    """Whole-month build window: start aligned to its month's first day, plus
    every month key in the window (current month included). Month partitions
    are then always replaced whole — merging a partial-window build would keep
    stale rows the raw layer no longer carries."""
    aligned = f"{yyyymmdd(start_date)[:6]}01"
    return aligned, set(_month_keys_between(aligned, end_date))


def _date_at(value: str, when: time) -> str:
    dt = datetime.strptime(value, "%Y%m%d").replace(hour=when.hour, minute=when.minute, second=when.second, tzinfo=CN_TZ)
    return dt.isoformat()


def _clean_date(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return ""
    try:
        return yyyymmdd(text)
    except ValueError:
        return ""


def _business_key_frame(dataset: str, frame: pd.DataFrame) -> pd.Series:
    """The canonical business-key JSON for every row of one dataset.

    Same bytes as ``json.dumps({"dataset": ..., "values": {...}},
    ensure_ascii=False, sort_keys=True, separators=(",", ":"))``: the object is
    assembled from sorted keys, and each column's DISTINCT values are escaped by
    ``json.dumps`` itself, so quoting stays the encoder's business.
    """
    keys = BUSINESS_KEYS.get(dataset, ("ts_code",))
    key_json = json.dumps(dataset, ensure_ascii=False)
    out = pd.Series(f'{{"dataset":{key_json},"values":{{', index=frame.index, dtype=object)
    for position, key in enumerate(sorted(set(keys))):
        values = _column_text(frame, key)
        escaped = values.map({value: json.dumps(value, ensure_ascii=False) for value in values.unique()})
        out = out + (("," if position else "") + json.dumps(key, ensure_ascii=False) + ":") + escaped
    return out + "}}"


def _source_path_matches_dataset(source_path: str, dataset: str) -> bool:
    parts = Path(str(source_path)).parts
    return dataset in parts
