"""share_float, the unlock schedule, pulled complete: the per-announcement-day
pulls, the per-code rescue of the answers the vendor caps, and the union
file that keeps the history the rolling pulls cannot reach, with its audit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from autotrade.common.records import read_json_object
from autotrade.common.time import utc_now
from autotrade.environment.data.pit import concat_rows

from .client import TuShareClient, frame, load_token
from .common import (
    WRITTEN_OUTCOMES,
    date_range_days,
    load_stock_codes,
    resolve_revision_ledger,
)
from .io import committed_partition_intact, parquet_rows
from .specs import EVENT_FLOW_SPECS, SHARE_FLOAT_FIELDS
from .stamps import augment_event_frame
from .versions import (
    answer_defects,
    list_versions,
    publish,
    report_complete,
    version_dir,
)

SHARE_FLOAT_ROW_LIMIT = 6000

SHARE_FLOAT_UNLOCK_TITLE_PATTERN = re.compile(
    r"限售|解禁|上市流通|解除限售|限售股份上市|限售股上市|首次公开发行限售|非公开发行限售|定向增发限售"
)


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


def query_share_float_to_path(
    client: TuShareClient,
    path: Path,
    params: dict[str, Any],
    source: str,
    force: bool,
    revision_ledger: Path | str | None = None,
) -> dict[str, Any]:
    if not force and committed_partition_intact(path):
        rows = parquet_rows(path)
        # Every return of this function carries the same keys: the caller reads
        # staged on each day, so omitting it here turned a plain resume
        # (partition present, no --force) into a KeyError.
        return {
            "path": str(path),
            "rows": rows,
            "skipped": True,
            "source_cap_risk": rows >= SHARE_FLOAT_ROW_LIMIT,
            "staged": False,
        }
    outcome = "defect"
    fetched = 0
    with answer_defects(path, ledger=revision_ledger):
        result = client.query("share_float", params, SHARE_FLOAT_FIELDS)
        meta_params = dict(params)
        meta_params["download_path"] = source
        meta_params["row_limit"] = SHARE_FLOAT_ROW_LIMIT
        df = augment_event_frame(frame(result), EVENT_FLOW_SPECS["share_float"])
        fetched = len(df)
        # An answer at the per-call cap is a truncated slice whose content
        # rotates as the vendor adds rows (2026-10: new IPO lock-ups of ~6,000
        # holder rows pushed older codes out of five ann_date answers). Its
        # missing keys are not a withdrawal, so it is published, never staged;
        # the ts_code rescue recovers what it could not return.
        outcome = publish(
            path,
            df,
            api_name="share_float",
            params=meta_params,
            fields=list(df.columns),
            key_columns=list(EVENT_FLOW_SPECS["share_float"].key_columns),
            ledger=revision_ledger,
            truncated=fetched >= SHARE_FLOAT_ROW_LIMIT,
        )
    written = outcome in WRITTEN_OUTCOMES
    rows = fetched if written else parquet_rows(path) if path.exists() else 0
    return {
        "path": str(path),
        "rows": rows,
        "skipped": not written,
        # Cap risk must consider the FETCHED response, not only the rows kept
        # on disk: a staged day whose new pull sits at the per-call cap is the
        # rotating-slice case the ts_code rescue exists for.
        "source_cap_risk": max(rows, fetched) >= SHARE_FLOAT_ROW_LIMIT,
        "staged": outcome == "staged",
    }

def download_share_float_ann_dates(client: TuShareClient, raw_dir: Path, args: argparse.Namespace, report: dict[str, Any]) -> list[str]:
    days = date_range_days(args.ann_start_date, args.ann_end_date)
    limit_hits: list[str] = []
    staged: list[str] = []
    written = 0
    skipped = 0
    total_rows = 0
    for index, day in enumerate(days, start=1):
        path = raw_dir / "share_float_ann_date" / f"ann_date={day}.parquet"
        result = query_share_float_to_path(
            client,
            path,
            {"ann_date": day},
            "ann_date",
            args.force,
            args.revision_ledger,
        )
        written += 0 if result["skipped"] else 1
        skipped += 1 if result["skipped"] else 0
        total_rows += int(result["rows"])
        if result["source_cap_risk"]:
            limit_hits.append(day)
        elif result["staged"]:
            # Only an uncapped answer is ever staged: the source dropped most of
            # the day's keys, which no ts_code rescue can repair. The staged
            # version is an open issue for an operator to promote or leave.
            staged.append(day)
        if index % 250 == 0:
            print(f"share_float ann_date {index}/{len(days)} skipped={skipped} written={written} rows_seen={total_rows} limit_hits={len(limit_hits)}")
    report["ann_date"] = {
        "start_date": args.ann_start_date,
        "end_date": args.ann_end_date,
        "days": len(days),
        "written": written,
        "skipped": skipped,
        "rows_seen": total_rows,
        "limit_hit_days": limit_hits,
        "staged_days": staged,
    }
    print(f"share_float ann_date done days={len(days)} skipped={skipped} written={written} rows_seen={total_rows} limit_hits={len(limit_hits)}")
    return limit_hits

def download_share_float_ts_code_rescue_by_date(
    client: TuShareClient,
    raw_dir: Path,
    date_codes: dict[str, list[str]],
    *,
    date_param: str,
    dataset_dir: str,
    source: str,
    force: bool,
    revision_ledger: Path | str | None = None,
) -> dict[str, Any]:
    written = 0
    skipped = 0
    total_rows = 0
    limit_hits: list[dict[str, Any]] = []
    no_candidate_dates = sorted(date for date, codes in date_codes.items() if not codes)
    total_tasks = sum(len(codes) for codes in date_codes.values())
    task_index = 0
    for day in sorted(date_codes):
        for code in date_codes[day]:
            task_index += 1
            path = raw_dir / dataset_dir / f"{date_param}={day}" / f"ts_code={code}.parquet"
            result = query_share_float_to_path(
                client,
                path,
                {date_param: day, "ts_code": code},
                source,
                force,
                revision_ledger,
            )
            written += 0 if result["skipped"] else 1
            skipped += 1 if result["skipped"] else 0
            total_rows += int(result["rows"])
            if result["source_cap_risk"]:
                limit_hits.append({"date": day, "ts_code": code, "rows": int(result["rows"])})
            if task_index % 500 == 0:
                print(
                    f"share_float {source} {task_index}/{total_tasks} skipped={skipped} "
                    f"written={written} rows_seen={total_rows} limit_hits={len(limit_hits)}"
                )
    return {
        "date_param": date_param,
        "dates": sorted(date_codes),
        "date_candidate_counts": {date: len(codes) for date, codes in sorted(date_codes.items())},
        "no_candidate_dates": no_candidate_dates,
        "tasks": total_tasks,
        "written": written,
        "skipped": skipped,
        "rows_seen": total_rows,
        "limit_hits": limit_hits,
    }

def unique_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result

def explicit_rescue_stock_codes(args: argparse.Namespace) -> list[str]:
    requested: list[str] = []
    requested.extend(args.rescue_code)
    codes_file = args.rescue_codes_file
    if codes_file:
        text = Path(codes_file).read_text(encoding="utf-8")
        requested.extend(re.split(r"[\s,]+", text))
    return unique_ordered(requested)

def selected_share_float_rescue_dates(args: argparse.Namespace, ann_limit_hits: list[str]) -> tuple[list[str], list[str]]:
    explicit = unique_ordered(args.rescue_ann_date)
    auto: list[str] = []
    skipped_auto: list[str] = []
    if args.rescue_ann_limit_hits:
        # Every limit-hit day: an unrescued capped day leaves the union without
        # the rows the vendor could not return. --max-rescue-calls bounds the
        # cost and fails the run explicitly when it would be exceeded.
        auto = ann_limit_hits
    elif ann_limit_hits:
        skipped_auto = ann_limit_hits
    return unique_ordered(explicit + auto), skipped_auto

def enforce_share_float_rescue_date_code_budget(args: argparse.Namespace, ann_date_codes: dict[str, list[str]], float_date_codes: dict[str, list[str]]) -> int:
    estimated_calls = sum(len(codes) for codes in ann_date_codes.values()) + sum(len(codes) for codes in float_date_codes.values())
    if args.max_rescue_calls is not None and estimated_calls > args.max_rescue_calls:
        raise RuntimeError(
            f"share_float rescue would make {estimated_calls} calls "
            f"({len(ann_date_codes)} ann_date dates and {len(float_date_codes)} float_date dates), "
            f"exceeding --max-rescue-calls={args.max_rescue_calls}"
        )
    return estimated_calls

def read_ts_codes_from_parquet(path: Path, *, extra_filter: tuple[str, str] | None = None) -> list[str]:
    if not path.exists() or parquet_rows(path) == 0:
        return []
    columns = ["ts_code"]
    if extra_filter:
        columns.append(extra_filter[0])
    df = pd.read_parquet(path, columns=list(dict.fromkeys(columns)))
    if extra_filter and extra_filter[0] in df.columns:
        df = df[df[extra_filter[0]].astype(str).str.strip() == extra_filter[1]]
    if "ts_code" not in df.columns:
        return []
    return unique_ordered(df["ts_code"].dropna().astype(str).str.strip().tolist())

def share_float_self_candidate_codes(raw_dir: Path, date_param: str, dates: list[str]) -> dict[str, list[str]]:
    """For an ann_date, every code a kept answer for the day carried: the
    visible partition, its kept versions (a truncated answer that replaced it
    leaves the codes it pushed out in the superseded one) and the day's ts_code
    rescue partitions, so a code once seen stays a candidate."""
    candidates: dict[str, list[str]] = {date: [] for date in dates}
    for day in dates:
        if date_param == "ann_date":
            path = raw_dir / "share_float_ann_date" / f"ann_date={day}.parquet"
            candidates[day].extend(read_ts_codes_from_parquet(path))
            for version in list_versions(path):
                candidates[day].extend(read_ts_codes_from_parquet(version_dir(path) / f"{version}.parquet"))
            rescue_dir = raw_dir / "share_float_ann_date_ts_code" / f"ann_date={day}"
            candidates[day].extend(
                item.name.removesuffix(".parquet").removeprefix("ts_code=")
                for item in sorted(rescue_dir.glob("ts_code=*.parquet"))
            )
        elif date_param == "float_date":
            # No self source: the legacy whole-day share_float/ dir was
            # archived out of the lake in 2026-05 and has no writer; float_date
            # rescue candidates come from the ann-path cross lookup instead.
            pass
        else:
            raise RuntimeError(f"unsupported share_float candidate date_param {date_param}")
        candidates[day] = unique_ordered(candidates[day])
    return candidates

def anns_unlock_candidate_codes(raw_dir: Path, ann_dates: list[str]) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {date: [] for date in ann_dates}
    dates_by_month: dict[str, set[str]] = {}
    for day in ann_dates:
        dates_by_month.setdefault(day[:6], set()).add(day)
    for month, days in sorted(dates_by_month.items()):
        path = raw_dir / "anns_d" / f"month={month}.parquet"
        if not path.exists() or parquet_rows(path) == 0:
            continue
        columns = ["ann_date", "ts_code", "title"]
        df = pd.read_parquet(path, columns=columns)
        df["ann_date"] = df["ann_date"].astype(str).str.strip()
        mask = df["ann_date"].isin(days) & df["title"].fillna("").astype(str).str.contains(SHARE_FLOAT_UNLOCK_TITLE_PATTERN)
        for day, group in df.loc[mask].groupby("ann_date"):
            candidates.setdefault(str(day), []).extend(group["ts_code"].dropna().astype(str).str.strip().tolist())
    return {date: unique_ordered(codes) for date, codes in candidates.items()}

def share_float_ann_path_float_candidate_codes(raw_dir: Path, float_dates: list[str], ann_start_date: str, ann_end_date: str) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {date: [] for date in float_dates}
    target_dates = set(float_dates)
    for day in date_range_days(ann_start_date, ann_end_date):
        path = raw_dir / "share_float_ann_date" / f"ann_date={day}.parquet"
        if not path.exists() or parquet_rows(path) == 0:
            continue
        df = pd.read_parquet(path, columns=["float_date", "ts_code"])
        df["float_date"] = df["float_date"].astype(str).str.strip()
        mask = df["float_date"].isin(target_dates)
        for float_date, group in df.loc[mask].groupby("float_date"):
            candidates.setdefault(str(float_date), []).extend(group["ts_code"].dropna().astype(str).str.strip().tolist())
    return {date: unique_ordered(codes) for date, codes in candidates.items()}

def merge_candidate_maps(*maps: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for mapping in maps:
        for date_key, codes in mapping.items():
            merged.setdefault(date_key, []).extend(codes)
    return {date_key: unique_ordered(codes) for date_key, codes in merged.items()}

def select_share_float_rescue_date_codes(
    raw_dir: Path,
    args: argparse.Namespace,
    *,
    date_param: str,
    dates: list[str],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    explicit = explicit_rescue_stock_codes(args)
    all_a_codes: list[str] = []
    detail: dict[str, Any] = {"date_param": date_param, "mode": args.rescue_universe, "dates": dates}
    if args.rescue_universe == "all_a":
        all_a_codes = load_stock_codes(raw_dir)
        if args.max_codes:
            all_a_codes = all_a_codes[: args.max_codes]
        return {date: all_a_codes for date in dates}, {**detail, "all_a_codes": len(all_a_codes)}
    if args.rescue_universe == "explicit":
        if not explicit:
            raise RuntimeError("--rescue-universe explicit requires --rescue-code or --rescue-codes-file")
        codes = explicit[: args.max_codes] if args.max_codes else explicit
        return {date: codes for date in dates}, {**detail, "explicit_codes": len(codes)}

    self_candidates = share_float_self_candidate_codes(raw_dir, date_param, dates)
    explicit_map = {date: explicit for date in dates}
    if date_param == "ann_date":
        anns_candidates = anns_unlock_candidate_codes(raw_dir, dates) if not args.no_anns_candidates else {date: [] for date in dates}
        # The float-dir -> ann cross lookup died with the archived whole-day
        # share_float/ dir; ann-path candidates come from self + anns sources.
        cross_candidates = {date: [] for date in dates}
    else:
        anns_candidates = {date: [] for date in dates}
        cross_candidates = (
            share_float_ann_path_float_candidate_codes(raw_dir, dates, args.ann_start_date, args.ann_end_date)
            if not args.no_cross_path_candidates
            else {date: [] for date in dates}
        )
    candidates = merge_candidate_maps(self_candidates, anns_candidates, cross_candidates, explicit_map)
    if args.max_codes:
        candidates = {date: codes[: args.max_codes] for date, codes in candidates.items()}
    detail.update({
        "self_candidate_counts": {date: len(self_candidates.get(date, [])) for date in dates},
        "anns_candidate_counts": {date: len(anns_candidates.get(date, [])) for date in dates},
        "cross_path_candidate_counts": {date: len(cross_candidates.get(date, [])) for date in dates},
        "explicit_codes": len(explicit),
        "final_candidate_counts": {date: len(candidates.get(date, [])) for date in dates},
        "max_codes": args.max_codes,
    })
    return candidates, detail

def share_float_union_files(raw_dir: Path, args: argparse.Namespace) -> list[tuple[Path, str]]:
    ann_start_date, ann_end_date = args.ann_start_date, args.ann_end_date
    float_start_date, float_end_date = args.float_start_date, args.float_end_date
    files: list[tuple[Path, str]] = []
    for day in date_range_days(ann_start_date, ann_end_date):
        path = raw_dir / "share_float_ann_date" / f"ann_date={day}.parquet"
        if path.exists():
            files.append((path, "ann_date"))
        rescue_dir = raw_dir / "share_float_ann_date_ts_code" / f"ann_date={day}"
        files.extend((path, "ann_date_ts_code") for path in sorted(rescue_dir.glob("ts_code=*.parquet")))
    if not args.skip_float_date_union:
        # Only the rescue partitions are readable on the float_date path: the
        # legacy whole-day dirs (share_float/, share_float_float_date/) were
        # archived out of the lake in 2026-05 and no writer can recreate them
        # (dev-phase removal 2026-08-02).
        for day in date_range_days(float_start_date, float_end_date):
            rescue_dir = raw_dir / "share_float_float_date_ts_code" / f"float_date={day}"
            files.extend((path, "float_date_ts_code") for path in sorted(rescue_dir.glob("ts_code=*.parquet")))
    return files

def _drop_stale_active_restatements(active: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep only the newest announcement of each physical event in ``active``.

    Only a NON-capped source may win the comparison: a capped (truncated)
    partition proves nothing about rows it could not return, so identities
    whose newest dating appears only in a capped file are left untouched and
    fall back to the conservative union. Distinct lots inside one winning
    ``(identity, ann_date)`` group are preserved.

    Datings are normalized first: an undated row (the float_date query path
    returns rows whose ``ann_date`` the provider left empty) ranks BELOW every
    real date, so it can never win and delete a dated announcement.
    """
    if active.empty:
        return active, 0
    keys = active[PHYSICAL_IDENTITY_COLUMNS].copy()
    keys["holder_name"] = normalized_holder_keys(keys["holder_name"])
    keys["ann"] = normalized_date_keys(active["ann_date"]).to_numpy()
    # "" marks a row whose source was capped, so it cannot win the comparison;
    # an identity with only capped sources gets newest == "" and, because no
    # date is ever "< ''", is never pruned (conservative union).
    keys["trusted_ann"] = keys["ann"].where(~active["source_cap_risk"].to_numpy(), "")
    newest = keys.groupby(PHYSICAL_IDENTITY_COLUMNS, dropna=False)["trusted_ann"].transform("max")
    stale = keys["ann"] < newest
    if not bool(stale.any()):
        return active, 0
    return active.loc[~stale.to_numpy()].reset_index(drop=True), int(stale.sum())


def write_share_float_union(raw_dir: Path, args: argparse.Namespace, report: dict[str, Any]) -> None:
    repo_root = Path.cwd().resolve()
    output = (repo_root / args.union_output).resolve()
    revision_ledger = resolve_revision_ledger(
        raw_dir,
        args.revision_ledger,
        repo_root=repo_root,
    )
    if not output.exists():
        raise RuntimeError(f"share_float_complete baseline is missing at {output}; restore it before updating")
    files = share_float_union_files(raw_dir, args)
    existing_rows = parquet_rows(output)
    active_frames: list[pd.DataFrame] = []
    for path, source in files:
        rows = parquet_rows(path)
        source_file = os.path.relpath(str(path), str(Path.cwd()))
        if rows == 0:
            continue
        df = pd.read_parquet(path)
        df["download_path"] = source
        # Repo-relative provenance: an absolute host path would flow into the
        # Agent-visible events snapshot (isolation contract, env docs §2.2).
        df["source_file"] = source_file
        df["source_cap_risk"] = rows >= SHARE_FLOAT_ROW_LIMIT
        active_frames.append(df)
    group_columns = ["ts_code", "ann_date", "float_date", "holder_name", "share_type"]
    identity_columns = SHARE_FLOAT_FIELDS.split(",")
    active = concat_rows(active_frames) if active_frames else pd.DataFrame(columns=identity_columns)
    missing_identity_columns = [column for column in identity_columns if column not in active.columns]
    if not active.empty and missing_identity_columns:
        raise RuntimeError(
            "share_float_complete active rows are missing identity columns: "
            + ", ".join(missing_identity_columns)
        )
    if not active.empty:
        active = active.sort_values("source_cap_risk", kind="stable")
        active = active.drop_duplicates(identity_columns).reset_index(drop=True)
    # A restatement's refresh window usually covers BOTH the stale and the
    # corrected ann_date partition, so the stale rows arrive in `active` too;
    # they must be dropped here or they would survive the union and make the
    # stale baseline group look "refreshed" (measured live: 301583.SZ stayed
    # at 72 rows across 3 ann_dates with baseline-only supersession).
    active, active_restatements_dropped = _drop_stale_active_restatements(active)
    safe_active_groups = (
        active.loc[~active["source_cap_risk"], group_columns].drop_duplicates()
        if not active.empty else pd.DataFrame(columns=group_columns)
    )
    # The canonical union is the durable historical baseline. A refreshed
    # coarse event group replaces the same baseline group; distinct source rows
    # inside that group remain separate. Unchanged history is inherited without
    # reopening archived process files.
    revision_old = None
    revision_new = None
    capped_groups_preserved = 0
    frames = [active] if not active.empty else []
    existing = pd.read_parquet(output)
    missing_baseline_columns = [column for column in identity_columns if column not in existing.columns]
    if missing_baseline_columns:
        raise RuntimeError(
            "share_float_complete baseline is missing identity columns: "
            + ", ".join(missing_baseline_columns)
        )
    existing_key_count = len(existing.drop_duplicates(group_columns))
    baseline = existing
    superseded_group_count = 0
    if not active.empty:
        active_groups = active[group_columns].drop_duplicates()
        refreshed_groups = pd.MultiIndex.from_frame(active_groups)
        safe_refreshed_groups = pd.MultiIndex.from_frame(safe_active_groups)
        baseline_groups = pd.MultiIndex.from_frame(baseline[group_columns])
        refreshed = baseline_groups.isin(refreshed_groups)
        safe_refreshed = baseline_groups.isin(safe_refreshed_groups)
        capped_groups_preserved = len(active_groups) - len(safe_active_groups)
        # Provider re-dating: the same physical unlock event -- identified by
        # (ts_code, float_date, holder_name, share_type), WITHOUT ann_date --
        # can reappear under a corrected announcement date (observed 2026-07:
        # 301583.SZ moved from ann_date 20260630 to 20260709). The stale-dated
        # baseline group is superseded so the unlock is never double-counted;
        # its rows are preserved in the revision event, where source revisions
        # belong.
        baseline_identity = physical_identity_index(baseline)
        # Only NON-capped identities may supersede baseline history, matching
        # the active-side rule: a truncated partition proves nothing about the
        # rows it could not return, so it must not be allowed to delete an
        # older announcement either. (Previously the active side refused capped
        # evidence while the baseline side accepted it -- inconsistent.)
        # An UNDATED active row carries no announcement date at all, so it may
        # only add: letting it supersede would delete the dated generation of
        # the same event and downgrade the baseline's PIT availability.
        dated_active_groups = safe_active_groups.loc[
            normalized_date_keys(safe_active_groups["ann_date"]) != ""
        ]
        active_identity = physical_identity_index(dated_active_groups)
        superseded = baseline_identity.isin(active_identity) & ~refreshed
        superseded_group_count = len(baseline.loc[superseded, group_columns].drop_duplicates())
        revision_old = baseline.loc[refreshed | superseded].reset_index(drop=True)
        preserved_capped = baseline.loc[refreshed & ~safe_refreshed].reset_index(drop=True)
        revision_new = concat_rows([active, preserved_capped]).drop_duplicates(identity_columns).reset_index(drop=True)
        baseline = baseline.loc[~(safe_refreshed | superseded)].reset_index(drop=True)
    else:
        revision_old = baseline.iloc[:0].copy()
        revision_new = active
    frames.append(baseline)
    union = concat_rows(frames) if frames else pd.DataFrame(columns=SHARE_FLOAT_FIELDS.split(","))
    before = len(union)
    undated_duplicates_dropped = 0
    if not union.empty:
        union = union.drop_duplicates(identity_columns).reset_index(drop=True)
        # The seven-field dedup above cannot collapse an undated row against
        # its dated twin (they differ in ann_date), so an undated rescue fetch
        # would re-add a lot the lake already carries and double the supply.
        # This keeps that class out by construction rather than relying on the
        # rescue path staying unused.
        undated_duplicates = undated_duplicate_mask(union)
        undated_duplicates_dropped = int(undated_duplicates.sum())
        if undated_duplicates_dropped:
            union = union.loc[~undated_duplicates].reset_index(drop=True)
    # Refreshed groups replace their complete baseline group, preserving
    # multiple distinct lots while removing superseded source revisions.
    new_key_count = len(union.drop_duplicates(group_columns)) if not union.empty else 0
    # Superseded re-dated groups are the only sanctioned shrink: each one is
    # replaced by the same physical event under its corrected ann_date.
    if new_key_count < existing_key_count - superseded_group_count:
        raise RuntimeError(
            "share_float_complete merge would shrink distinct event groups of "
            f"{output} from {existing_key_count} to {new_key_count} "
            f"(beyond {superseded_group_count} superseded re-dated groups) "
            f"using {len(files)} input files"
        )
    published = publish(
        output,
        union,
        api_name="share_float_complete",
        params={
            "strategy": "canonical_incremental_merge",
            "source_api": "share_float",
            "ann_start_date": args.ann_start_date,
            "ann_end_date": args.ann_end_date,
            "float_start_date": args.float_start_date,
            "float_end_date": args.float_end_date,
            "input_files": len(files),
        },
        fields=list(union.columns),
        key_columns=identity_columns,
        ledger=revision_ledger,
        source="share_float_union_merge",
        compare=(revision_old, revision_new),
        current=existing,
    )
    if published not in WRITTEN_OUTCOMES | {"unchanged"}:
        raise RuntimeError(f"share_float_complete merge was not published to {output}: {published}")
    report["union"] = {
        "output": str(output),
        "input_files": len(files),
        "rows_before_dedup": before,
        "rows_after_dedup": len(union),
        "previous_rows": existing_rows,
        "baseline_inherited": True,
        "capped_groups_preserved": capped_groups_preserved,
        "superseded_redated_groups": superseded_group_count,
        "active_stale_restatement_rows_dropped": active_restatements_dropped,
        "undated_duplicate_rows_dropped": undated_duplicates_dropped,
    }

def download_share_float_complete(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    args.revision_ledger = resolve_revision_ledger(raw_dir, args.revision_ledger, repo_root=repo_root)
    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    report: dict[str, Any] = {
        "created_at": utc_now("auto"),
        "raw_dir": str(raw_dir),
        "row_limit": SHARE_FLOAT_ROW_LIMIT,
        "scope": {
            "ann_start_date": args.ann_start_date,
            "ann_end_date": args.ann_end_date,
            "float_start_date": args.float_start_date,
            "float_end_date": args.float_end_date,
            "float_rescue_dates": args.float_rescue_date,
            "max_codes": args.max_codes,
        },
    }
    ann_limit_hits: list[str] = []
    if not args.skip_ann_date:
        ann_limit_hits = download_share_float_ann_dates(client, raw_dir, args, report)

    rescue_ann_dates, skipped_ann_rescue = selected_share_float_rescue_dates(args, ann_limit_hits)
    ann_date_codes: dict[str, list[str]] = {}
    float_date_codes: dict[str, list[str]] = {}
    if rescue_ann_dates or args.float_rescue_date:
        ann_detail: dict[str, Any] = {}
        float_detail: dict[str, Any] = {}
        if rescue_ann_dates:
            ann_date_codes, ann_detail = select_share_float_rescue_date_codes(raw_dir, args, date_param="ann_date", dates=rescue_ann_dates)
        if args.float_rescue_date:
            float_date_codes, float_detail = select_share_float_rescue_date_codes(raw_dir, args, date_param="float_date", dates=args.float_rescue_date)
        report["rescue_candidates"] = {"ann_date": ann_detail, "float_date": float_detail}
        report["rescue_estimated_calls"] = enforce_share_float_rescue_date_code_budget(args, ann_date_codes, float_date_codes)

    if rescue_ann_dates:
        report["ann_date_ts_code"] = download_share_float_ts_code_rescue_by_date(
            client,
            raw_dir,
            ann_date_codes,
            date_param="ann_date",
            dataset_dir="share_float_ann_date_ts_code",
            source="ann_date_ts_code",
            force=args.force,
            revision_ledger=args.revision_ledger,
        )
    report["ann_date_ts_code_skipped_limit_days"] = skipped_ann_rescue

    if args.float_rescue_date:
        report["float_date_ts_code"] = download_share_float_ts_code_rescue_by_date(
            client,
            raw_dir,
            float_date_codes,
            date_param="float_date",
            dataset_dir="share_float_float_date_ts_code",
            source="float_date_ts_code",
            force=args.force,
            revision_ledger=args.revision_ledger,
        )

    if args.write_union:
        write_share_float_union(raw_dir, args, report)
        report_complete(Path(args.union_output).parent.name)
    if not args.skip_ann_date:
        report_complete("share_float")

    if args.output:
        output = (repo_root / args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"share_float complete download process_report={output}")
    else:
        union = report.get("union") or {}
        print(
            "share_float complete download finished; "
            f"ann_limit_hits={len(report.get('ann_date', {}).get('limit_hit_days', []))} "
            f"ann_rescue_tasks={report.get('ann_date_ts_code', {}).get('tasks', 0)} "
            f"union_rows={union.get('rows_after_dedup', 'not_written')}; "
            "pass --output to write a process report"
        )
    return 0


def audit_share_float_complete_union(raw_dir: Path, add) -> None:
    union_path = raw_dir / "share_float_complete" / "share_float_complete.parquet"
    meta_path = union_path.with_suffix(union_path.suffix + ".meta.json")
    ann_rescue_files = sorted((raw_dir / "share_float_ann_date_ts_code").rglob("*.parquet"))
    float_rescue_files = sorted((raw_dir / "share_float_float_date_ts_code").rglob("*.parquet"))
    rescue_limit_files: list[str] = []
    rescue_zero_files = 0
    for path in ann_rescue_files + float_rescue_files:
        rows = parquet_rows(path)
        if rows == 0:
            rescue_zero_files += 1
        if rows >= SHARE_FLOAT_ROW_LIMIT:
            rescue_limit_files.append(str(path))

    base_details: dict[str, Any] = {
        "path": str(union_path),
        "ann_date_ts_code_files": len(ann_rescue_files),
        "float_date_ts_code_files": len(float_rescue_files),
        "rescue_zero_files": rescue_zero_files,
        "rescue_limit_files": len(rescue_limit_files),
        "rescue_limit_file_sample": rescue_limit_files[:10],
    }
    if not union_path.exists():
        add("warning", "share_float_complete_union", "share_float complete union file is missing; event_flow audit falls back to raw share_float partitions only", base_details)
        return

    rows = parquet_rows(union_path)
    schema = pq.ParquetFile(union_path).schema_arrow.names
    required_columns = ["ts_code", "ann_date", "float_date", "download_path", "source_file", "source_cap_risk"]
    missing_columns = [column for column in required_columns if column not in schema]
    meta_row_count = None
    meta_error = ""
    if meta_path.exists():
        try:
            meta_row_count = read_json_object(meta_path).get("row_count")
        except Exception as exc:  # noqa: BLE001 - unreadable sidecar is reported through meta_error
            meta_error = str(exc)

    details: dict[str, Any] = {
        **base_details,
        "rows": rows,
        "meta_exists": meta_path.exists(),
        "meta_row_count": meta_row_count,
        "meta_error": meta_error,
        "missing_columns": missing_columns,
    }
    # Business-identity duplication: the same PHYSICAL unlock event (ts_code +
    # float_date + holder + share_type, ann_date excluded, holder whitespace
    # normalized) must not survive under several announcement dates -- the
    # provider re-dates announcements, and a retained stale generation double
    # counts supply. Row/key checks cannot see this class, so it is checked
    # explicitly here; the incremental merge only fixes what its refresh window
    # touches, so historical residue needs a dedicated repair.
    identity_columns = ["ts_code", "ann_date", "float_date", "holder_name", "share_type"]
    # An absent identity column must surface, not silently skip the check --
    # silently passing unverifiable data is the failure mode audits exist for.
    identity_columns_missing = [column for column in identity_columns if column not in schema]
    details["identity_columns_missing"] = identity_columns_missing
    has_identity_columns = not identity_columns_missing
    stat_columns = [column for column in ("download_path", "source_cap_risk", "source_file") if column in schema]
    # One physical read serves both the identity check and the provenance
    # stats; each block keeps its own error attribution.
    read_columns = list(dict.fromkeys((identity_columns if has_identity_columns else []) + stat_columns))
    union_frame: pd.DataFrame | None = None
    union_read_error: Exception | None = None
    if read_columns:
        try:
            union_frame = pd.read_parquet(union_path, columns=read_columns)
        except Exception as exc:  # noqa: BLE001 - read failure is kept and re-raised into the audit finding
            union_read_error = exc
    if has_identity_columns:
        try:
            if union_frame is None:
                raise union_read_error
            frame = union_frame[identity_columns].copy()
            # Datings are normalized: an UNDATED row (float_date query path, the
            # provider left ann_date empty) must rank below every real date, or
            # str(None)=="None" sorts above them and the informative dated row
            # gets reported as the stale one.
            frame["ann_date"] = normalized_date_keys(frame["ann_date"])
            frame["holder_name"] = normalized_holder_keys(frame["holder_name"])
            physical = ["ts_code", "float_date", "holder_name", "share_type"]
            datings = frame.groupby(physical, sort=False)["ann_date"].nunique()
            multi = datings[datings > 1]
            details["cross_ann_date_identity_groups"] = len(multi)
            if len(multi):
                affected = frame.merge(multi.reset_index()[physical], on=physical, how="inner")
                newest = affected.groupby(physical, sort=False)["ann_date"].transform("max")
                stale = affected["ann_date"] < newest
                details["stale_redated_rows"] = int(stale.sum())
                # The two mechanisms need different repairs: an undated duplicate
                # is provable from the baseline alone, an older REAL dating is not.
                details["undated_duplicate_rows"] = int((stale & affected["ann_date"].eq("")).sum())
                details["cross_ann_date_sample"] = [
                    dict(zip(physical, values)) for values in multi.index[:5]
                ]
        except Exception as exc:  # noqa: BLE001 - failure is reported as identity_check_error
            details["identity_check_error"] = str(exc)
    try:
        if stat_columns and union_frame is None:
            raise union_read_error
        df = union_frame[stat_columns] if union_frame is not None else pd.DataFrame()
        if "download_path" in df.columns:
            details["download_path_counts"] = {str(key): int(value) for key, value in df["download_path"].value_counts(dropna=False).sort_index().items()}
        if "source_file" in df.columns:
            details["input_files_seen"] = int(df["source_file"].nunique(dropna=True))
        if "source_cap_risk" in df.columns:
            risk = df["source_cap_risk"].fillna(False)
            if risk.dtype != bool:
                risk = risk.astype(str).str.lower().isin({"true", "1", "yes"})
            details["source_cap_risk_rows"] = int(risk.sum())
    except Exception as exc:  # noqa: BLE001 - failure is reported as read_error
        details["read_error"] = str(exc)

    severity = "warning" if (
        rows == 0
        or missing_columns
        or identity_columns_missing
        or not meta_path.exists()
        or meta_error
        or details.get("source_cap_risk_rows", 0)
        or rescue_limit_files
        or details.get("cross_ann_date_identity_groups", 0)
        or details.get("identity_check_error")
    ) else "info"
    add(severity, "share_float_complete_union", "share_float complete union and rescue coverage", details)

def share_float_complete_union_exists(raw_dir: Path) -> bool:
    return (raw_dir / "share_float_complete" / "share_float_complete.parquet").exists()
