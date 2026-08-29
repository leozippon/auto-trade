"""Shared synthetic sandbox/snapshot fixtures for tool and pipeline tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from autotrade.environment.artifacts import artifact_fingerprint
from autotrade.environment.data.snapshot import finalize_snapshot_dir
from autotrade.environment.tools import ToolResult

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO_ROOT / "configs" / "agent_output_template"
TS_CODE = "000001.SZ"

# (trade_date, open, close) for the single fixture stock across all periods.
# The two leading days predate every validation period so the prior-day research
# snapshot always has an anchor: 20210930 anchors the 2021Q4 quarter validation
# (2021Q4 opens 20211008 after the National Day holiday) and 20201231 anchors the
# full-year 2021 validation. Neither enters a replay slot (no test period covers them).
PRICE_ROWS = [
    ("20201231", 9.5, 9.8),
    ("20210930", 9.8, 10.0),
    ("20211008", 10.0, 10.5),
    ("20211011", 10.6, 11.0),
    ("20211230", 11.1, 12.0),
    ("20220104", 12.0, 12.4),
    ("20220105", 12.5, 13.0),
    ("20220331", 13.1, 14.0),
    ("20220406", 13.2, 13.4),
    ("20220630", 13.5, 13.8),
    ("20260105", 14.0, 14.5),
    ("20260106", 14.6, 15.0),
    ("20260331", 15.1, 16.0),
]


def make_snapshot_dir(out_dir: Path, *, decision_date: str, kind: str) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{
            "trade_date": decision_date,
            "ts_code": TS_CODE,
            "open": 10.0,
            "close": 10.5,
            "vol": 100000.0,
            "amount": 1050000.0,
            "up_limit": 12.0,
            "down_limit": 8.0,
            "is_suspended": False,
        }]
    ).to_parquet(out_dir / "daily.parquet", index=False)
    pd.DataFrame([{"ts_code": TS_CODE, "name": "平安银行", "exchange": "SZSE"}]).to_parquet(
        out_dir / "universe.parquet", index=False
    )
    pd.DataFrame([{"dataset": "margin_secs", "trade_date": decision_date, "ts_code": TS_CODE}]).to_parquet(
        out_dir / "events.parquet", index=False
    )
    pd.DataFrame(columns=["dataset", "ts_code", "available_at", "end_date", "bz_item"]).to_parquet(
        out_dir / "fundamentals.parquet", index=False
    )
    pd.DataFrame(
        columns=["text_id", "dataset", "ts_codes", "title", "available_at", "library_file"]
    ).to_parquet(out_dir / "text_index.parquet", index=False)
    (out_dir / "text_library").mkdir(exist_ok=True)
    # External union snapshots must declare dataset ownership explicitly —
    # finalize never infers it from file content.
    return finalize_snapshot_dir(
        out_dir,
        kind=kind,
        decision_date=decision_date,
        domains={
            "events": {"dataset_columns": {"margin_secs": ["dataset", "trade_date", "ts_code"]}},
            "fundamentals": {
                "dataset_columns": {
                    "fina_mainbz_vip": ["dataset", "ts_code", "available_at", "end_date", "bz_item"]
                }
            },
        },
    )


def make_replay_dir(out_dir: Path, *, start: str, end: str, label: str) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"trade_date": d, "ts_code": TS_CODE, "open": o, "close": c, "up_limit": o * 1.2, "down_limit": o * 0.8, "is_suspended": False}
        for d, o, c in PRICE_ROWS
        if start <= d <= end
    ]
    pd.DataFrame(rows).to_parquet(out_dir / "daily.parquet", index=False)
    return finalize_snapshot_dir(out_dir, kind="replay_slot", label=label, period_start=start, period_end=end)


class FakeSnapshotProvider:
    """Synthetic SnapshotProvider for pipeline tests."""

    def decision_snapshot(self, decision_time: datetime, out_dir: Path) -> dict[str, object]:
        return make_snapshot_dir(Path(out_dir), decision_date=decision_time.strftime("%Y%m%d"), kind="decision_input")

    def replay_slot(self, start: str, end: str, out_dir: Path, *, label: str, available_from=None) -> dict[str, object]:
        return make_replay_dir(Path(out_dir), start=start, end=end, label=label)


class PassingModificationCheck:
    """A ``modification_check`` double that honours the formal contract.

    A formal call snapshots the artifact and refuses to replay a snapshot whose
    fingerprint is not the one the check published, so a double that publishes
    no fingerprint would let a test walk past a gate the real environment
    enforces. It reads the same trees the real tool would.
    """

    def __init__(
        self,
        output_dir: Path,
        models_dir: Path | None = None,
        **value: object,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir) if models_dir is not None else None
        self.value = value
        self.calls = 0

    def invoke(self, _arguments):
        self.calls += 1
        return ToolResult(
            True,
            value={
                **self.value,
                "fingerprint": artifact_fingerprint(self.output_dir, self.models_dir),
            },
        )
