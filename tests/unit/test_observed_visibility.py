"""Observed-era visibility: a PIT view shows a row from the later of its
vendor-date rule and the instant the raw lake first held its business key
(docs/data-documentation.md §3.3). Keys visible before versioning started,
and keys first seen no later than their stamp, keep the vendor rule."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from autotrade.data_sources.tushare import versions
from autotrade.environment.data.fundamental_events import (
    FundamentalEventsBuilder,
    FundamentalEventsConfig,
    audit_fundamental_events,
    read_fundamental_events,
)
from autotrade.environment.data.pit import (
    FIRST_SEEN_COLUMN,
    canonical_keys,
    observation_path,
)
from autotrade.environment.data.snapshot import SnapshotBuilder, SnapshotConfig
from autotrade.environment.replay.timeview import Timeview
from autotrade.pipelines.pit_backend import _open_replay_rows

from .test_snapshot_builder import (
    build_raw,
    write,
    write_dividend_history,
    write_fundamental_status,
)

CN_TZ = ZoneInfo("Asia/Shanghai")
# Friday 2021-10-08 12:00 Beijing: after that morning's 09:45 disclosure run
# and 03:35 index build, before any further refresh node.
LATE = "2021-10-08T04:00:00.000000+00:00"
EARLY = "2021-09-01T00:00:00.000000+00:00"
HOLDER_KEYS = ["ts_code", "ann_date", "holder_name"]
TITLE_KEYS = ["ann_date", "ts_code", "title", "url"]
INCOME_KEYS = ["ts_code", "ann_date", "f_ann_date", "end_date"]
CONFIG = SnapshotConfig(
    events_datasets=("stk_holdertrade",), macro_datasets=(), text_datasets=("anns_d",),
    fundamental_datasets=("income_vip",), include_intraday=False, include_industry=False,
)
ANCHOR = datetime(2021, 10, 7, 23, 59, 59, tzinfo=CN_TZ)


def holder(name: str, ann_date: str) -> dict[str, object]:
    day = f"{ann_date[:4]}-{ann_date[4:6]}-{ann_date[6:]}"
    return {"ts_code": "000001.SZ", "ann_date": ann_date, "holder_name": name, "change_vol": 100.0,
            "available_at": f"{day} 19:00:00+08:00", "available_at_rule": "official_19_from:ann_date"}


def title(text: str, ann_date: str, received: str) -> dict[str, object]:
    return {"ann_date": ann_date, "ts_code": "000001.SZ", "title": text, "url": f"https://x/{text}",
            "rec_time": received, "available_at": f"{received}+08:00", "available_at_rule": "rec_time"}


def income(ann_date: str) -> dict[str, object]:
    return {"ts_code": "000001.SZ", "ann_date": ann_date, "f_ann_date": ann_date, "end_date": "20210630",
            "basic_eps": 0.5, "revenue": 1_000_000.0, "n_income": 100_000.0}


HOLDERS_SEPT = [holder("A", "20210910"), holder("B", "20210915")]
HOLDERS_OCT = [holder("C", "20211008")]
TITLES = [title("T1", "20210910", "2021-09-10 20:00:00"), title("T2", "20210915", "2021-09-15 20:00:00")]
INCOME = [income("20210910"), income("20210915")]
PARTITIONS = {
    "stk_holdertrade/month=202109.parquet": (HOLDERS_SEPT, HOLDER_KEYS),
    "stk_holdertrade/month=202110.parquet": (HOLDERS_OCT, HOLDER_KEYS),
    "anns_d/month=202109.parquet": (TITLES, TITLE_KEYS),
    "income_vip/period=20210630.parquet": (INCOME, INCOME_KEYS),
}


def publish(path: Path, records: list[dict[str, object]], keys: list[str], at: str) -> None:
    frame = pd.DataFrame(records)
    with patch.object(versions, "utc_now", return_value=at), redirect_stdout(io.StringIO()):
        versions.publish(path, frame, api_name=path.parent.name, params={}, fields=list(frame.columns),
                         key_columns=keys, ledger=None)


def build_index(raw: Path, events_root: Path) -> None:
    builder = FundamentalEventsBuilder(raw)
    events = builder.build(FundamentalEventsConfig("20210101", "20211231", datasets=("income_vip",)))
    builder.write_partitioned(
        events, events_root, replace_months={f"2021{month:02d}" for month in range(1, 13)},
        replace_datasets=("income_vip",),
    )
    write_dividend_history(events_root)


def lake(root: Path) -> tuple[Path, Path, Path]:
    """The synthetic lake, migrated before the periods of every fixture
    partition: writes record first-seen rows only for the versioned era."""
    raw = root / "raw"
    build_raw(raw)
    (raw / ".raw_generation.json").write_text(json.dumps({
        "schema_version": 3, "state": "committed", "generation_id": "g", "versioned_since": "20210601",
    }), encoding="utf-8")
    events_root = root / "pit" / "fundamental_events"
    status = root / "quality" / "fundamental_events_status.json"
    status.parent.mkdir(parents=True)
    write_fundamental_status(status)
    return raw, events_root, status


def views(root: Path, raw: Path, events_root: Path, status: Path) -> tuple[SnapshotBuilder, Path, Path]:
    builder = SnapshotBuilder(raw, events_root, status)
    decision, replay = root / "decision", root / "replay"
    builder.build_decision_snapshot(ANCHOR, decision, CONFIG)
    builder.build_replay_slot("20211008", "20211011", replay, label="paper", config=CONFIG, available_from=ANCHOR)
    return builder, decision, replay


def timeview(root: Path, decision: Path, replay: Path) -> Timeview:
    return Timeview(
        host_dir=root / "asof", snapshot_dir=decision, replay=_open_replay_rows(replay),
        replay_text_library_dir=replay / "text_library",
    )


def at(day: str, clock: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {clock}", tz=CN_TZ)


def visible(root: Path, domain: str, column: str) -> list[str]:
    return sorted(pd.read_parquet(root / "asof" / domain)[column].astype(str))


def test_a_key_the_vendor_added_late_is_visible_only_from_when_the_lake_first_held_it(tmp_path: Path) -> None:
    raw, events_root, status = lake(tmp_path)
    # A, T1 and the first statement were visible before versioning started;
    # the vendor added B, T2 and the second statement after the fact (stamps
    # in September, first held on Friday 12:00). C was first held at 12:00
    # too, before its own 19:00 stamp, so the vendor rule alone decides it.
    for name, (records, keys) in PARTITIONS.items():
        path = raw / name
        if len(records) > 1:
            write(path, pd.DataFrame(records[:1]))
            publish(path, records, keys, LATE)
        else:
            publish(path, records, keys, LATE)
    build_index(raw, events_root)
    builder, decision, replay = views(tmp_path, raw, events_root, status)

    # A decision before the lake held the late keys does not show them, a
    # later one does; neither view carries the replay's gate column.
    morning = tmp_path / "morning"
    builder.build_decision_snapshot(at("2021-10-08", "09:25:00").to_pydatetime(), morning, CONFIG)
    evening = tmp_path / "evening"
    builder.build_decision_snapshot(datetime(2021, 10, 8, 23, 59, 59, tzinfo=CN_TZ), evening, CONFIG)
    for view, names, titles, filings in ((morning, ["A"], ["T1"], ["20210910"]),
                                         (evening, ["A", "B", "C"], ["T1", "T2"], ["20210910", "20210915"])):
        events = pd.read_parquet(view / "events.parquet")
        assert sorted(events["holder_name"]) == names
        assert sorted(pd.read_parquet(view / "text_index.parquet")["title"]) == titles
        assert sorted(pd.read_parquet(view / "fundamentals.parquet")["ann_date"]) == filings
        for file in ("events.parquet", "text_index.parquet", "fundamentals.parquet"):
            assert FIRST_SEEN_COLUMN not in pq.read_schema(view / file).names

    # The replay slot carries the first-seen time of the late rows only, in
    # Beijing time; its vendor stamp is unchanged.
    slot = pd.read_parquet(replay / "events.parquet").set_index("holder_name")
    assert slot.loc["B", "available_at"] == "2021-09-15 19:00:00+08:00"
    assert slot.loc["B", FIRST_SEEN_COLUMN] == "2021-10-08 12:00:00+08:00"
    assert pd.isna(slot.loc["C", FIRST_SEEN_COLUMN])

    view = timeview(tmp_path, decision, replay)
    # 09:25 and 10:00: every refresh node has released B, T2 and the second
    # statement, but the lake did not hold them yet.
    for clock in ("09:25:00", "10:00:00"):
        view.refresh(at("2021-10-08", clock))
        assert visible(tmp_path, "events", "holder_name") == ["A"]
        assert visible(tmp_path, "text_index", "title") == ["T1"]
        assert visible(tmp_path, "fundamentals", "ann_date") == ["20210910"]
    # 12:30: no node has passed since 10:00, yet the first-seen time has.
    view.refresh(at("2021-10-08", "12:30:00"))
    assert visible(tmp_path, "events", "holder_name") == ["A", "B"]
    assert visible(tmp_path, "text_index", "title") == ["T1", "T2"]
    assert visible(tmp_path, "fundamentals", "ann_date") == ["20210910", "20210915"]
    # C follows its vendor rule: the disclosure run after its 19:00 stamp.
    view.refresh(at("2021-10-11", "09:25:00"))
    assert visible(tmp_path, "events", "holder_name") == ["A", "B", "C"]
    # The gate column never reaches a published part.
    for domain in ("events", "text_index", "fundamentals"):
        for part in (tmp_path / "asof" / domain).glob("*.parquet"):
            assert FIRST_SEEN_COLUMN not in pq.read_schema(part).names


def observe(path: Path, keys: list[str], first_seen: str | None) -> None:
    """An observation table beside ``path`` as the versioned write leaves it:
    every key first seen at ``first_seen``, or no row at all."""
    frame = canonical_keys(pd.read_parquet(path), keys) if first_seen else pd.DataFrame(columns=keys, dtype=str)
    frame[FIRST_SEEN_COLUMN] = first_seen
    pq.write_table(pa.Table.from_pandas(frame.astype(str), preserve_index=False), observation_path(path))
    if first_seen:
        path.with_suffix(".parquet.meta.json").write_text(f'{{"visible_since": "{first_seen}"}}', encoding="utf-8")


def built_files(root: Path, variant: str | None) -> dict[str, str]:
    """Hashes of every parquet one variant builds; each builds in the same
    directory, since the index records its source paths."""
    shutil.rmtree(root, ignore_errors=True)
    raw, events_root, status = lake(root)
    for name, (records, keys) in PARTITIONS.items():
        write(raw / name, pd.DataFrame(records))
        if variant is not None:
            observe(raw / name, keys, EARLY if variant == "earlier" else None)
    build_index(raw, events_root)
    _builder, decision, replay = views(root, raw, events_root, status)
    view = timeview(root, decision, replay)
    for moment in (at("2021-10-08", "09:25:00"), at("2021-10-08", "12:30:00"), at("2021-10-11", "09:25:00")):
        view.refresh(moment)
    files = [
        *events_root.rglob("*.parquet"), *decision.rglob("*.parquet"), *replay.rglob("*.parquet"),
        *(root / "asof").rglob("*.parquet"),
    ]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def test_views_without_a_later_first_seen_time_are_byte_identical(tmp_path: Path) -> None:
    """No observation table, an empty one, and one whose every key was first
    seen before its stamp build the same index, views and as-of parts."""
    plain = built_files(tmp_path / "lake", None)
    assert plain == built_files(tmp_path / "lake", "empty")
    assert plain == built_files(tmp_path / "lake", "earlier")
    assert any(name.startswith("asof/events/") for name in plain)


def test_the_index_files_a_late_financial_row_under_the_month_the_lake_first_held_it(tmp_path: Path) -> None:
    raw, events_root, _status = lake(tmp_path)
    path = raw / "income_vip" / "period=20210630.parquet"
    write(path, pd.DataFrame(INCOME[:1]))
    publish(path, INCOME, INCOME_KEYS, LATE)
    build_index(raw, events_root)
    late = pd.read_parquet(events_root / "income_vip" / "available_month=202110.parquet")
    assert late[["ann_date", "available_at", FIRST_SEEN_COLUMN]].to_dict("records") == [
        {"ann_date": "20210915", "available_at": "2021-09-15T18:00:00+08:00",
         FIRST_SEEN_COLUMN: "2021-10-08 12:00:00+08:00"},
    ]
    assert list(pd.read_parquet(events_root / "income_vip" / "available_month=202109.parquet")["ann_date"]) == ["20210910"]

    def read(until: str, **options) -> list[str]:
        rows = read_fundamental_events(events_root, until, datasets=("income_vip",), **options)
        return sorted(rows["ann_date"])

    assert read("2021-10-08T11:59:59+08:00") == ["20210910"]
    assert read("2021-10-08T12:00:00+08:00") == ["20210910", "20210915"]
    # The vendor rule alone (the broker's corporate-action truth) finds it by
    # its stamp, in the later month it is filed under.
    assert read("2021-09-30T23:59:59+08:00", observed=False) == ["20210910", "20210915"]
    report = audit_fundamental_events(
        events_root, FundamentalEventsConfig("20210101", "20211231", datasets=("income_vip",)), require_partitions=True,
    )
    assert report["status"] == "ok", report["findings"]


def test_the_broker_credits_a_dividend_whose_notice_the_lake_held_only_after_the_slot(tmp_path: Path) -> None:
    raw, events_root, status = lake(tmp_path)
    write_dividend_history(events_root)
    write(
        events_root / "dividend" / "available_month=202110.parquet",
        pd.DataFrame([{
            "dataset": "dividend", "ts_code": "000001.SZ", "end_date": "20210630", "div_proc": "实施",
            "ex_date": "20211011", "record_date": "20211008", "pay_date": "20211011", "div_listdate": None,
            "cash_div": 0.2, "cash_div_tax": 0.2, "stk_div": None, "stk_bo_rate": None, "stk_co_rate": None,
            "available_at": "2021-10-08T18:00:00+08:00", FIRST_SEEN_COLUMN: "2021-10-12 09:00:00+08:00",
            "available_at_rule": "source:imp_ann_date_or_ann_date", "available_month": "202110",
            "business_key": "d1", "source_path": "x", "source_write_id": "w", "source_row_id": 0,
        }]),
    )
    builder = SnapshotBuilder(raw, events_root, status)
    actions, _meta = builder._build_corporate_actions("20211008", "20211011", at("2021-10-11", "23:59:59"))
    assert actions[["ts_code", "ex_date", "cash_per_share"]].to_dict("records") == [
        {"ts_code": "000001.SZ", "ex_date": "20211011", "cash_per_share": 0.2},
    ]
