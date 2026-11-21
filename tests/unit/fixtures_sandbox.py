"""Shared synthetic sandbox/snapshot fixtures for tool and pipeline tests."""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autotrade.common.time import utc_now
from autotrade.environment.artifacts import artifact_fingerprint
from autotrade.environment.container_stats import ContainerResourceMonitor
from autotrade.environment.data.snapshot import SnapshotConfig, _write_manifest
from autotrade.environment.data.units import validate_snapshot_units
from autotrade.environment.executor import DockerStrategyExecutor
from autotrade.environment.runtime import new_id
from autotrade.environment.sandbox import SandboxConfig
from autotrade.environment.tools import ToolResult
from autotrade.pipelines.config import SNAPSHOT_CACHE_FORMAT_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO_ROOT / "configs" / "agent_output_template"
TS_CODE = "000001.SZ"


@functools.cache
def _docker_available() -> bool:
    """Whether a Docker daemon answers here (asked once per run)."""

    executable = shutil.which("docker")
    if executable is None:
        return False
    try:
        completed = subprocess.run([executable, "info"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def real_docker(test):
    """Marks a test that talks to the host's real Docker daemon: skipped where
    none answers, and ``slow`` (outside the quick run)."""

    skip = pytest.mark.skipif(not _docker_available(), reason="Docker is unavailable")
    return pytest.mark.slow(skip(test))


def finalize_snapshot_dir(snapshot_dir: str | Path, **fields: object) -> dict[str, object]:
    """Stamp a test-assembled snapshot directory with a manifest.

    Directories containing union files must supply the builder's
    ``domains[...]["dataset_columns"]`` explicitly via ``fields`` — dataset
    ownership is never inferred from file content, and validation below fails
    without it. Same gate and same manifest writer as the builder: every column
    must classify in the unit registry.
    """
    snapshot_dir = Path(snapshot_dir)
    manifest: dict[str, object] = {"snapshot_id": new_id("snap"), "created_at": utc_now("auto"), **fields}
    validate_snapshot_units(snapshot_dir, manifest)
    _write_manifest(snapshot_dir, manifest)
    return manifest

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


def _executor_for_process(
    process: subprocess.Popen[bytes],
    *,
    config: SandboxConfig | None = None,
    drain_stderr: bool = False,
) -> DockerStrategyExecutor:
    executor = object.__new__(DockerStrategyExecutor)
    executor.config = config or SandboxConfig()
    executor.container_name = "test-container"
    executor._process = process
    executor._stdout_buffer = bytearray()
    executor._stderr_tail = deque()
    executor._stderr_size = 0
    executor._stderr_thread = None
    executor._closed = False
    executor.snapshot_dir = None
    executor.asof_dir = None
    executor.models_dir = None
    executor.state_dir = None
    executor.fit_schedule = None
    executor.context_state_dir = ""
    executor.context_models_dir = ""
    executor._fit_worker = None
    executor.gpu_indices = []
    executor._fit_seconds = []
    executor._decision_seconds = []
    executor._monitor = ContainerResourceMonitor(executor.container_name)
    executor._worker_reading = {}
    executor._call_in_flight = False
    executor._peak_state_bytes = None
    executor._reset_transport_state()
    if drain_stderr:
        thread = threading.Thread(target=executor._drain_stderr, daemon=True)
        executor._stderr_thread = thread
        thread.start()
    return executor


VALID_MAIN = """
def generate_orders(context):
    return []
"""


def write_artifact(root: Path, *, main: str = VALID_MAIN) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("readonly", encoding="utf-8")
    (root / "main.py").write_text(main, encoding="utf-8")
    (root / "notes.md").write_text("neutral when evidence is thin\n", encoding="utf-8")
    return root


def _pit_slot_paths(
    root: Path,
    *,
    decision: str,
    replay: str,
    generation_id: str,
    phase: str = "valid",
) -> tuple[Path, Path]:
    cache_root = root / "pit_views"
    snapshot = cache_root / "decision" / decision
    replay_slot = cache_root / "replay" / phase / replay
    snapshot.mkdir(parents=True)
    replay_slot.mkdir(parents=True)
    _write_provider_contract(cache_root, generation_id=generation_id)
    return snapshot, replay_slot


def _write_corporate_actions(replay: Path, rows: list[dict[str, object]] | None = None) -> None:
    """The slot's ex-date table; every built slot carries it, dividends or not."""

    columns = [
        "ts_code", "ex_date", "record_date", "pay_date", "div_listdate",
        "cash_per_share", "stock_per_share", "bonus_per_share",
    ]
    pd.DataFrame(rows or [], columns=columns).to_parquet(
        replay / "corporate_actions.parquet", index=False
    )


def _write_domains(snapshot: Path, replay: Path) -> None:
    pd.DataFrame(
        {
            "trade_date": ["20240101"],
            "ts_code": ["000001.SZ"],
            "open": [10.0],
            "close": [10.0],
            "pre_close": [10.0],
            "available_at": ["2024-01-01T17:30:00+08:00"],
        }
    ).to_parquet(snapshot / "daily.parquet", index=False)
    pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.0],
            "close": [10.0, 10.0],
            "pre_close": [10.0, 10.0],
            "available_at": ["2024-01-02T17:30:00+08:00", "2024-01-03T17:30:00+08:00"],
        }
    ).to_parquet(replay / "daily.parquet", index=False)
    _write_corporate_actions(replay)
    minute_columns = {
        "trade_date": ["20240101"],
        "ts_code": ["000001.SZ"],
        "trade_time": ["2024-01-01T15:00:00+08:00"],
        "close": [10.0],
        "available_at": ["2024-01-01T15:00:00+08:00"],
    }
    pd.DataFrame(minute_columns).to_parquet(snapshot / "intraday_1min.parquet", index=False)
    minute_replay = pd.DataFrame(
        {
            key: [
                value[0].replace("2024-01-01", "2024-01-02").replace("20240101", "20240102"),
                value[0].replace("2024-01-01", "2024-01-03").replace("20240101", "20240103"),
            ]
            if isinstance(value[0], str)
            else [value[0], value[0]]
            for key, value in minute_columns.items()
        }
    )
    pq.write_table(pa.Table.from_pandas(minute_replay, preserve_index=False), replay / "intraday_1min.parquet", row_group_size=1)

    _write_simple_domain(snapshot, replay, "auction", dataset=None, time="09:29:00")
    _write_simple_domain(snapshot, replay, "events", dataset="moneyflow", time="10:00:00")
    _write_simple_domain(snapshot, replay, "macro", dataset="cn_cpi", time="10:00:00")
    _write_simple_domain(snapshot, replay, "fundamentals", dataset="income_vip", time="10:00:00")

    snapshot_index = pd.DataFrame(
        {
            "dataset": ["news"],
            "text_id": ["old"],
            "title": ["visibletoken old"],
            "ts_codes": ["000001.SZ"],
            "library_file": ["news.parquet"],
            "available_at": ["2024-01-01T10:00:00+08:00"],
        }
    )
    replay_index = pd.DataFrame(
        {
            "dataset": ["news", "news"],
            "text_id": ["day1", "future"],
            "title": ["visibletoken day1", "visibletoken future"],
            "ts_codes": ["000001.SZ", "000001.SZ"],
            "library_file": ["news.parquet", "news.parquet"],
            "available_at": ["2024-01-02T10:00:00+08:00", "2024-01-03T10:00:00+08:00"],
        }
    )
    snapshot_index.to_parquet(snapshot / "text_index.parquet", index=False)
    replay_index.to_parquet(replay / "text_index.parquet", index=False)
    pd.DataFrame({"text_id": ["old"], "body": ["visibletoken old body"]}).to_parquet(
        snapshot / "text_library" / "news.parquet",
        index=False,
    )
    pd.DataFrame(
        {"text_id": ["day1", "future"], "body": ["visibletoken day1 body", "visibletoken future body"]}
    ).to_parquet(replay / "text_library" / "news.parquet", index=False)
    pd.DataFrame({"ts_code": ["000001.SZ"]}).to_parquet(snapshot / "universe.parquet", index=False)


def _write_provider_contract(
    cache_root: Path,
    *,
    generation_id: str,
    config: dict[str, object] | None = None,
) -> None:
    (cache_root / "provider.json").write_text(
        json.dumps(
            {
                "schema_version": SNAPSHOT_CACHE_FORMAT_VERSION,
                "generation_id": generation_id,
                "release_raw_dir": str(cache_root / "release" / generation_id / "raw"),
                "snapshot_config": config or SnapshotConfig().to_record(),
            }
        ),
        encoding="utf-8",
    )


def _write_simple_domain(
    snapshot: Path,
    replay: Path,
    name: str,
    *,
    dataset: str | None,
    time: str,
) -> None:
    base = {
        "trade_date": ["20240101"],
        "ts_code": ["000001.SZ"],
        "value": [1.0],
        "available_at": [f"2024-01-01T{time}+08:00"],
    }
    if dataset is not None:
        base["dataset"] = [dataset]
    pd.DataFrame(base).to_parquet(snapshot / f"{name}.parquet", index=False)
    current = {key: [value[0], value[0]] for key, value in base.items()}
    current["trade_date"] = ["20240102", "20240103"]
    current["available_at"] = [f"2024-01-02T{time}+08:00", f"2024-01-03T{time}+08:00"]
    pd.DataFrame(current).to_parquet(replay / f"{name}.parquet", index=False)
