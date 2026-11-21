"""stk_auction, the opening call auction, landed whole: the pre-open capture that
publishes a day's frame once two reads agree and it is complete, and the
evening recheck that re-reads the latest open day and fills missed days.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autotrade.common.time import CN_TZ
from autotrade.environment.data.contracts import STK_AUCTION_PRICE_ABS_TOLERANCE
from autotrade.environment.data.pit import partition_value

from .client import ApiResult, TuShareClient, frame, load_token, query_paged
from .common import (
    NO_MUTATION_RETRY_EXIT_CODE,
    WRITTEN_OUTCOMES,
    load_sse_open_dates,
    normalize_date_key,
    resolve_revision_ledger,
)
from .io import committed_partition_intact
from .specs import DAILY_SPECS
from .versions import publish


def _auction_capture_min_rows(
    raw_dir: Path,
    trade_date: str,
    *,
    floor: int,
    ratio: float,
) -> int:
    """Completeness floor for one auction partition vs the previous one.

    Percentage-only by design: an absolute max-drop bound (previously
    prev-10 rows) dominated the ratio for full-market partitions and
    rejected legitimate provider-side coverage narrowing. The ratio is taken
    over stock rows only (``_auction_stock_rows``), so a vendor dropping its
    fund rows cannot read as truncation. The stability double-read still
    guards same-day flapping; the ratio guards truncation."""
    previous = sorted(
        path
        for path in (raw_dir / "stk_auction").glob("trade_date=*.parquet")
        if partition_value(path) < trade_date
    )
    previous_rows = (
        _auction_stock_rows(pd.read_parquet(previous[-1], columns=["ts_code"])["ts_code"])
        if previous
        else 0
    )
    return max(int(floor), math.floor(int(previous_rows) * float(ratio)))


# The auction frame also carries funds, ETFs, LOFs and convertibles, whose
# coverage the vendor narrows on its own schedule: about 1,100 rows went in
# 2026-07, and from 20260911 all 1,879 at once while every A-share code stayed.
# Completeness is therefore measured on A-share rows: Shanghai 6xxxxx,
# Shenzhen 0xxxxx/3xxxxx and Beijing, which lists no funds.
_AUCTION_STOCK_CODE = r"6\d{5}\.SH|[03]\d{5}\.SZ|\d{6}\.BJ"


def _auction_stock_rows(codes: pd.Series) -> int:
    return int(codes.astype(str).str.fullmatch(_AUCTION_STOCK_CODE).sum())


def _validate_auction_capture(frame_: pd.DataFrame, trade_date: str, *, min_rows: int) -> list[str]:
    required = set(DAILY_SPECS["stk_auction"].fields.split(","))
    missing = sorted(required.difference(frame_.columns))
    errors = [f"missing_columns={missing}"] if missing else []
    if frame_.empty:
        return [*errors, "empty_response"]
    dates = set(frame_["trade_date"].astype(str)) if "trade_date" in frame_.columns else set()
    if dates != {trade_date}:
        errors.append(f"unexpected_trade_dates={sorted(dates)}")
    if {"trade_date", "ts_code"}.issubset(frame_.columns):
        duplicates = int(frame_.duplicated(["trade_date", "ts_code"]).sum())
        if duplicates:
            errors.append(f"duplicate_keys={duplicates}")
        valid_codes = frame_["ts_code"].astype(str).str.fullmatch(r"\d{6}\.(SH|SZ|BJ)")
        invalid_codes = int((~valid_codes).sum())
        if invalid_codes:
            errors.append(f"invalid_codes={invalid_codes}")
    stock_rows = _auction_stock_rows(frame_["ts_code"]) if "ts_code" in frame_.columns else 0
    if stock_rows < int(min_rows):
        errors.append(f"stock_rows={stock_rows} below_min_rows={min_rows}")
    if {"price", "vol", "amount"}.issubset(frame_.columns):
        price = pd.to_numeric(frame_["price"], errors="coerce")
        volume = pd.to_numeric(frame_["vol"], errors="coerce")
        amount = pd.to_numeric(frame_["amount"], errors="coerce")
        finite_volume = pd.Series(np.isfinite(volume), index=frame_.index)
        finite_amount = pd.Series(np.isfinite(amount), index=frame_.index)
        invalid_volume = ~finite_volume | volume.lt(0)
        invalid_amount = ~finite_amount | amount.lt(0)
        if invalid_volume.any():
            errors.append(f"invalid_vol_rows={int(invalid_volume.sum())}")
        if invalid_amount.any():
            errors.append(f"invalid_amount_rows={int(invalid_amount.sum())}")

        valid_quantities = ~invalid_volume & ~invalid_amount
        traded = valid_quantities & (volume.gt(0) | amount.gt(0))
        matched_trade = valid_quantities & volume.gt(0) & amount.gt(0)
        inconsistent = traded & ~matched_trade
        if inconsistent.any():
            errors.append(f"inconsistent_trade_rows={int(inconsistent.sum())}")

        direct_price = pd.Series(np.isfinite(price), index=frame_.index) & price.gt(0)
        recovered_price = amount.div(volume)
        recoverable = (
            volume.gt(0)
            & amount.gt(0)
            & pd.Series(np.isfinite(recovered_price), index=frame_.index)
            & recovered_price.gt(0)
        )
        bad_trade_price = matched_trade & ~recoverable
        if bad_trade_price.any():
            errors.append(f"unrecoverable_trade_price_rows={int(bad_trade_price.sum())}")
        price_mismatch = (
            matched_trade
            & direct_price
            & recoverable
            & ~pd.Series(
                np.isclose(
                    price,
                    recovered_price,
                    rtol=1e-9,
                    atol=STK_AUCTION_PRICE_ABS_TOLERANCE,
                ),
                index=frame_.index,
            )
        )
        if price_mismatch.any():
            errors.append(f"inconsistent_trade_price_rows={int(price_mismatch.sum())}")
        no_trade = valid_quantities & volume.eq(0) & amount.eq(0)
        hidden_no_trade_price = no_trade & direct_price
        if hidden_no_trade_price.any():
            errors.append(f"hidden_no_trade_price_rows={int(hidden_no_trade_price.sum())}")
    return errors


def _auction_frame_snapshot(frame_: pd.DataFrame) -> str:
    ordered = frame_.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    return ordered.fillna("").astype(str).to_json(orient="split", force_ascii=False)


# Completeness floor shared by the pre-open capture CLI defaults and the
# evening recheck (which takes no knobs).
AUCTION_CAPTURE_MIN_ROWS_FLOOR = 1000
AUCTION_CAPTURE_MIN_PREVIOUS_DAY_RATIO = 0.85


def _publish_auction_frame(
    raw_dir: Path,
    repo_root: Path,
    trade_date: str,
    captured: pd.DataFrame,
    result: ApiResult,
    pages: int,
    page_limit,
    landing_job: str,
    revision_ledger_arg,
) -> tuple[str, str]:
    """Canonical-order publish with observed availability, shared by the
    pre-open capture and the evening recheck. Returns (outcome, landed_at)."""
    # Stable identity is key-based, so persist the same canonical order. A
    # source response that only reorders rows must remain byte-identical and
    # keep the first observed availability instead of looking like a revision.
    captured = captured.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    landed_at = datetime.now(UTC).astimezone(CN_TZ).isoformat()
    path = raw_dir / "stk_auction" / f"trade_date={trade_date}.parquet"
    availability = {
        "matched_at": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}T09:25:00+08:00",
        "available_at": landed_at,
        "rule": "observed:cn_open_auction_capture",
        "landing_job": landing_job,
        "row_count": len(captured),
    }
    revision_ledger = resolve_revision_ledger(raw_dir, revision_ledger_arg, repo_root=repo_root)
    spec = DAILY_SPECS["stk_auction"]
    outcome = publish(
        path,
        captured,
        api_name=spec.api_name,
        params={"trade_date": trade_date, "pagination": {"page_limit": page_limit, "pages": pages}},
        fields=result.fields,
        key_columns=list(spec.key_columns),
        ledger=revision_ledger,
        extra_metadata={"availability": availability},
    )
    return outcome, landed_at


def capture_open_auction(args: argparse.Namespace) -> int:
    """Poll the same-day opening-auction endpoint until a stable, complete frame lands."""
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    trade_date = normalize_date_key(args.trade_date)
    if not trade_date:
        raise ValueError(f"invalid --trade-date: {args.trade_date!r}")
    if not load_sse_open_dates(raw_dir, trade_date, trade_date, allow_empty=True):
        print(json.dumps({"status": "skipped_non_trading_day", "trade_date": trade_date}, ensure_ascii=False))
        return 0

    spec = DAILY_SPECS["stk_auction"]
    try:
        client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
        min_rows = _auction_capture_min_rows(
            raw_dir,
            trade_date,
            floor=args.min_rows,
            ratio=args.min_previous_day_ratio,
        )
        polling_started = time.monotonic()
        deadline = polling_started + max(0.0, float(args.max_wait_seconds))
        poll_interval = max(0.0, float(args.retry_delay_seconds))
        next_poll_at = polling_started
        required_stable_reads = max(1, int(args.stable_reads))
        stable_count = 0
        last_snapshot = ""
        accepted: tuple[pd.DataFrame, ApiResult, int] | None = None
        last_errors: list[str] = ["not_queried"]
        while True:
            try:
                result, pages = query_paged(
                    client,
                    spec.api_name,
                    {"trade_date": trade_date},
                    spec.fields,
                    args.page_limit,
                )
                candidate = frame(result).reset_index(drop=True)
                last_errors = _validate_auction_capture(candidate, trade_date, min_rows=min_rows)
            except Exception as exc:  # noqa: BLE001 - query failure is reported as a not-ready capture error
                last_errors = [f"query_error={type(exc).__name__}: {exc}"]
                stable_count = 0
                last_snapshot = ""
            else:
                if not last_errors:
                    snapshot = _auction_frame_snapshot(candidate)
                    stable_count = stable_count + 1 if snapshot == last_snapshot else 1
                    last_snapshot = snapshot
                    if stable_count >= required_stable_reads:
                        accepted = (candidate, result, pages)
                        break
                else:
                    stable_count = 0
                    last_snapshot = ""
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                break
            print(
                f"stk_auction trade_date={trade_date} not ready; errors={last_errors} "
                f"stable={stable_count}/{required_stable_reads}; retrying"
            )
            if poll_interval <= 0:
                continue
            next_poll_at += poll_interval
            if next_poll_at <= now:
                skipped_slots = math.floor((now - next_poll_at) / poll_interval) + 1
                next_poll_at += skipped_slots * poll_interval
            time.sleep(min(next_poll_at - now, remaining))
    except Exception as exc:  # noqa: BLE001 - setup failure is reported as a not-ready outcome without mutation
        print(json.dumps({
            "status": "not_ready_no_mutation",
            "trade_date": trade_date,
            "errors": [f"setup_error={type(exc).__name__}: {exc}"],
        }, ensure_ascii=False, sort_keys=True))
        return NO_MUTATION_RETRY_EXIT_CODE
    if accepted is None:
        print(json.dumps({
            "status": "not_ready_no_mutation",
            "trade_date": trade_date,
            "errors": last_errors,
        }, ensure_ascii=False, sort_keys=True))
        return NO_MUTATION_RETRY_EXIT_CODE

    captured, result, pages = accepted
    published, landed_at = _publish_auction_frame(
        raw_dir,
        repo_root,
        trade_date,
        captured,
        result,
        pages,
        args.page_limit,
        "cn_open_auction_capture_0927",
        args.revision_ledger,
    )
    if published not in WRITTEN_OUTCOMES | {"unchanged"}:
        raise RuntimeError(f"stk_auction trade_date={trade_date} complete frame was not published: {published}")
    print(json.dumps({
        "status": "ok",
        "trade_date": trade_date,
        "rows": len(captured),
        "pages": pages,
        "available_at": landed_at,
    }, ensure_ascii=False, sort_keys=True))
    return 0


def recheck_stk_auction(args: argparse.Namespace) -> int:
    """Validated evening stk_auction reconciliation inside the universal
    update, standardized with the margin family's evening self-heal shape:
    the latest open day is always re-read (late source corrections), and any
    EARLIER open day in the update window whose partition is missing or not an
    intact commit gets a validated capture attempt — a multi-day capture
    outage or torn write heals itself instead of waiting for a manual backfill. Healed days keep OBSERVED
    landing timestamps (auction visibility is deliberately not rule-stamped:
    a fixed morning stamp on an evening-healed day would show replay auction
    data at an open the live system provably lacked).

    Standard reads with the capture's completeness validation — no polling.
    An invalid or unchanged result never replaces an existing partition, and
    an invalid result never fails the whole update batch: gaps are escalated
    by the nightly audit (missing expected partition) instead of poisoning
    the batch's other datasets."""
    end_day = normalize_date_key(args.end_date)
    start_day = normalize_date_key(args.start_date)
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    open_days = load_sse_open_dates(raw_dir, start_day, end_day, allow_empty=True)
    if not open_days:
        print(json.dumps({"step": "stk_auction_recheck", "status": "skipped_non_trading_window", "start_date": start_day, "end_date": end_day}, ensure_ascii=False, sort_keys=True))
        return 0
    exit_code = 0
    for day in open_days:
        path = repo_root / args.raw_dir / "stk_auction" / f"trade_date={day}.parquet"
        if day != open_days[-1] and committed_partition_intact(path):
            continue  # latest day is always re-read; incomplete earlier days self-heal
        exit_code = _recheck_stk_auction_day(args, repo_root, raw_dir, day) or exit_code
    return exit_code

def _recheck_stk_auction_day(args: argparse.Namespace, repo_root: Path, raw_dir: Path, trade_date: str) -> int:
    outcome: dict[str, Any] = {"step": "stk_auction_recheck", "trade_date": trade_date}
    spec = DAILY_SPECS["stk_auction"]
    path = raw_dir / "stk_auction" / f"trade_date={trade_date}.parquet"
    existed = path.exists()
    candidate: pd.DataFrame | None = None
    result = None
    pages = 0
    try:
        client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
        min_rows = _auction_capture_min_rows(
            raw_dir,
            trade_date,
            floor=AUCTION_CAPTURE_MIN_ROWS_FLOOR,
            ratio=AUCTION_CAPTURE_MIN_PREVIOUS_DAY_RATIO,
        )
        result, pages = query_paged(client, spec.api_name, {"trade_date": trade_date}, spec.fields, args.page_limit)
        candidate = frame(result).reset_index(drop=True)
        errors = _validate_auction_capture(candidate, trade_date, min_rows=min_rows)
    except Exception as exc:  # noqa: BLE001 - query failure is reported as recheck_invalid
        errors = [f"query_error={type(exc).__name__}: {exc}"]
    if errors:
        print(json.dumps({
            **outcome, "status": "recheck_invalid", "errors": errors, "partition_exists": existed,
        }, ensure_ascii=False, sort_keys=True))
        return 0
    snapshot = _auction_frame_snapshot(candidate)
    if committed_partition_intact(path) and _auction_frame_snapshot(pd.read_parquet(path)) == snapshot:
        # Unchanged content on an intact commit keeps the first observed
        # availability and the original file untouched. A torn pair is rewritten
        # even when the payload matches, so the commit identity can self-heal.
        print(json.dumps({
            **outcome, "status": "recheck_unchanged", "rows": len(candidate),
        }, ensure_ascii=False, sort_keys=True))
        return 0
    published, landed_at = _publish_auction_frame(
        raw_dir,
        repo_root,
        trade_date,
        candidate,
        result,
        pages,
        args.page_limit,
        args.landing_job,
        args.revision_ledger,
    )
    if published not in WRITTEN_OUTCOMES:
        # Staged (a correction dropping most rows) or unchanged: the morning
        # partition stays visible; a staged answer is an open issue.
        print(json.dumps({**outcome, "status": f"recheck_{published}_kept_existing"}, ensure_ascii=False, sort_keys=True))
        return 0
    print(json.dumps({
        **outcome,
        "status": "recheck_published",
        "revised_existing": existed,
        "rows": len(candidate),
        "available_at": landed_at,
    }, ensure_ascii=False, sort_keys=True))
    return 0
