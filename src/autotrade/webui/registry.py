"""Experiment discovery and read-model assembly for the HITL console.

Everything here is read-only over ``experiments/<id>/``: the append-only
ledger, the hitl/ control-plane files, and frozen artifacts. Unparseable
experiments still appear in listings so they can be deleted; they carry an
error note instead of metrics.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

from autotrade.environment.replay.style import STYLE_ARTIFACT_NAME, STYLE_SCHEMA_VERSION
from autotrade.pipelines.agent_inbox import INBOX_NAME, inbox_public_view
from autotrade.pipelines.fold_analysis import analysis_paths
from autotrade.pipelines.hitl_state import (
    ANALYSIS_DIR_NAME,
    CONTROL_NAME,
    HITL_DIR_NAME,
    PARAMS_NAME,
    SCHEDULE_NAME,
    STATUS_NAME,
    read_control,
    read_json,
    read_status,
    status_pid_alive,
)
from autotrade.pipelines.ledger import (
    ExperimentLedger,
    latest_fold_records,
    latest_heldout_records,
)
from autotrade.pipelines.prior import latest_prior_text, unified_meta_record

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
# Datasets whose partition coverage bounds the selectable backtest periods: the
# replay needs minute bars, so their intersection with the daily lake is the
# honest "data exists" window (issue: pre-coverage periods fail at runtime).
COVERAGE_DATASETS = ("daily", "stk_mins_1min_by_date")


def dataset_coverage(raw_dir: Path, dataset: str) -> tuple[str, str] | None:
    root = raw_dir / dataset
    if not root.is_dir():
        return None
    dates = [
        entry.name[len("trade_date="):-len(".parquet")]
        for entry in root.glob("trade_date=*.parquet")
    ]
    dates = [d for d in dates if len(d) == 8 and d.isdigit()]
    if not dates:
        return None
    return min(dates), max(dates)


def clamped_trading_days(repo_root: Path) -> list[str] | None:
    """SSE trading days clamped to the datasets' actual partition coverage, so
    the period pickers cannot offer periods without downloaded data. None when
    no calendar is available (dev/test roots): the pickers degrade to text."""
    try:
        from autotrade.pipelines.folds import load_sse_trading_days

        raw_dir = repo_root / "data" / "raw"
        days = load_sse_trading_days(raw_dir)
        coverages = [c for c in (dataset_coverage(raw_dir, name) for name in COVERAGE_DATASETS) if c]
        if coverages:
            low = max(c[0] for c in coverages)
            high = min(c[1] for c in coverages)
            days = [day for day in days if low <= day <= high]
        return days
    except Exception:  # noqa: BLE001 - schema must stay served without a calendar
        return None


ACTIVE_STATES = (
    "launching",
    "starting",
    "preparing",
    "running_session",
    "running_heldout",
    "waiting_user",
    "waiting_step_user",
    "waiting_user_reply",
    "paused",
)
# Manager-written stub state between spawn and the worker's first status write
# (interpreter start + imports take seconds). Stale = the worker never came up.
LAUNCH_GRACE_SECONDS = 180.0
# Fold-record fields whose content is test-period evidence (guarded view).
TEST_FIELDS = ("test_result",)
# Pointers that resolve to the same evidence on disk. They are stripped from
# every projection and never surface in the audit block either — the reveal
# gate would be pointless if the console handed out the path to the artifact.
_TEST_EVIDENCE_REFS = ("test_result_ref", "snapshot_ids")
# Operator-only keys plus retired sensitive configuration from historical
# params files. The read model never echoes any of them back out.
_PRIVATE_PARAMS = {
    "raw_dir",
    "fundamental_events_root",
    "fundamental_events_status",
    "pit_cache_root",
    "template_dir",
    "local_dev",
    "experiments_root",
    "work_root",
    "llm_api_key_env",
    "llm_env_file",
    "llm_base_url",
}


def read_ledger_records(experiment_dir: Path) -> list[dict[str, object]]:
    """The single validating reader shared with the pipeline: a ledger the
    pipeline would reject (unparseable line, wrong schema_version) must not
    drive console decisions either — inheritance, rollback, rerun and reveal
    all key off these records. Failures surface through the experiment's
    ``unreadable`` state rather than a silently skimmed partial view."""
    return ExperimentLedger(Path(experiment_dir) / "ledgers" / "experiment_ledger.jsonl").read()


def experiment_state(experiment_dir: Path) -> dict[str, object]:
    """Effective lifecycle state combining status.json and pid liveness.

    A missing/empty status.json reads as "created" — the pre-first-spawn
    state. This keeps the function total for the transient windows a listing
    thread can observe (hitl/ mid-mkdir during creation, mid-rmtree during
    delete); mutating operations re-check the control plane themselves.

    A status.json that cannot be read (corrupt JSON, foreign schema_version)
    reads as "unreadable" with a one-line error: never counted as running,
    still listed, still deletable (module invariant). Only this read-model
    combiner is total — control-plane mutation paths keep using the strict
    ``read_status`` and fail fast on the same file."""
    path = Path(experiment_dir) / HITL_DIR_NAME / STATUS_NAME
    try:
        status = read_status(path)
    except (OSError, ValueError) as exc:
        return {
            "kind": "hitl",
            "state": "unreadable",
            "worker_alive": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not status:
        return {"kind": "hitl", "state": "created", "worker_alive": False, "status": {}}
    alive = status_pid_alive(status)
    state = str(status.get("state") or "unknown")
    if state == "launching" and not alive:
        if _age_seconds(status.get("launched_at")) > LAUNCH_GRACE_SECONDS:
            state = "interrupted"
    elif state in ACTIVE_STATES and not alive:
        state = "interrupted"
    return {"kind": "hitl", "state": state, "worker_alive": alive, "status": status}


def _age_seconds(stamp: object) -> float:
    """Age of an ISO timestamp; unparseable stamps read as infinitely old."""
    try:
        then = datetime.fromisoformat(str(stamp))
    except (TypeError, ValueError):
        return float("inf")
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return (datetime.now(UTC) - then).total_seconds()


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _metric(record: Mapping[str, object], result_key: str, key: str) -> float | None:
    result = record.get(result_key)
    return _number(result.get(key)) if isinstance(result, Mapping) else None


def _metric_series(records: list[dict[str, object]], result_key: str, metric: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _metric(record, result_key, metric)
        if value is not None:
            values.append(value)
    return values


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _epoch_folds(folds: list[dict[str, object]], epoch_id: str | None) -> list[dict[str, object]]:
    return [record for record in folds if str(record.get("epoch_id")) == epoch_id]


def _compound(values: list[float]) -> float | None:
    if not values:
        return None
    total = 1.0
    for value in values:
        total *= 1.0 + value
    return total - 1.0


def _created_at(directory: Path, params: Mapping[str, object]) -> str | None:
    if params.get("_created_at"):
        return str(params["_created_at"])
    try:
        return datetime.fromtimestamp(directory.stat().st_mtime, UTC).isoformat(timespec="seconds")
    except OSError:
        return None


def test_results_revealed(
    experiment_dir: Path, records: list[dict[str, object]] | None = None
) -> bool:
    """P1-7: test/held-out results are hidden from the console until the
    researcher explicitly reveals them (which seals the experiment against
    further learning). Held-out is the terminal evaluation: once every
    scheduled held-out period is recorded, no learning session can ever
    follow, so results auto-reveal (and the same seal applies). Partial
    held-out does not auto-reveal — the worker may still need resume to
    finish the remaining periods. A missing control.json (transient
    mid-creation state) reads as an empty control plane: not revealed."""
    if read_control(Path(experiment_dir) / HITL_DIR_NAME / CONTROL_NAME).test_revealed:
        return True
    return heldout_complete(experiment_dir, records)


def heldout_complete(
    experiment_dir: Path, records: list[dict[str, object]] | None = None
) -> bool:
    """Every held-out period planned in the schedule has a durable ledger record."""
    schedule = read_json(Path(experiment_dir) / HITL_DIR_NAME / SCHEDULE_NAME)
    sessions = schedule.get("sessions") if isinstance(schedule.get("sessions"), list) else []
    planned = _planned_heldout_labels(sessions)
    if not planned:
        return False
    if records is None:
        records = read_ledger_records(experiment_dir)
    return planned <= _completed_heldout_labels(records)


def _planned_heldout_labels(sessions: list[object]) -> set[str]:
    return {
        str(period.get("label"))
        for session in sessions
        if isinstance(session, Mapping) and str(session.get("kind")) == "heldout"
        for period in (session.get("periods") if isinstance(session.get("periods"), list) else [])
        if isinstance(period, Mapping) and period.get("label")
    }


def _completed_heldout_labels(records: list[dict[str, object]]) -> set[str]:
    return {
        str(record.get("fold_id") or "").removeprefix("heldout_")
        for record in records
        if record.get("record_type") == "heldout"
    }


def _durable_session_progress(
    sessions: list[object] | None,
    records: list[dict[str, object]],
) -> tuple[int, int | None]:
    """Return progress from the durable plan and success ledger only.

    ``status.json`` is a replaceable worker heartbeat: startup and restart
    writers legitimately reset or briefly retain its counters, so it cannot
    prove cumulative completion. The persisted schedule supplies the bounded
    denominator and successful ledger records supply the numerator.
    """
    if sessions is None:
        return 0, None
    completed_keys = {
        str(record.get("session_key"))
        for record in records
        if record.get("record_type") in {"fold", "meta_learning"}
        and record.get("session_key")
    }
    completed_heldout = _completed_heldout_labels(records)
    completed = 0
    for session in sessions:
        if not isinstance(session, Mapping):
            continue
        if str(session.get("kind") or "") == "heldout":
            planned_heldout = _planned_heldout_labels([session])
            if planned_heldout and planned_heldout <= completed_heldout:
                completed += 1
            continue
        session_key = str(session.get("session_key") or session.get("key") or "")
        if session_key and session_key in completed_keys:
            completed += 1
    return completed, len(sessions)


def sealed_result_prefixes(experiment_dir: Path) -> tuple[str, ...]:
    """Central reveal gate for artifact routes (style, orders, CSV, result-name
    enumeration): result names / style prefixes starting with these carry
    test-period evidence and must stay invisible pre-reveal — respond 404 and
    filter listings, never confirm existence. Empty once revealed."""
    return () if test_results_revealed(experiment_dir) else ("test", "heldout")


def guarded_fold_view(record: Mapping[str, object]) -> dict[str, object]:
    """Fold record minus test-period evidence (shown separately, labelled)."""
    hidden = (*TEST_FIELDS, *_TEST_EVIDENCE_REFS)
    return {key: value for key, value in record.items() if key not in hidden}


def _public_meta_view(
    record: Mapping[str, object], *, current_prior: str | None = None
) -> dict[str, object]:
    return unified_meta_record(record, current_prior=current_prior)


def _public_heldout_view(record: Mapping[str, object], revealed: bool) -> dict[str, object]:
    # ``revealed`` is deliberately required: a reveal-defaulting default on a
    # reveal-gated projection would fail open at any future call site. A
    # held-out record IS test-period evidence, so only its identifying header
    # survives the guarded view.
    if revealed:
        return dict(record)
    return {
        "record_type": "heldout",
        "epoch_id": record.get("epoch_id"),
        "fold_id": record.get("fold_id"),
        "period": record.get("period"),
        "hidden": True,
    }


def _public_params(params: Mapping[str, object]) -> dict[str, object]:
    # params.json is also a worker-side ops channel where manager-owned roots
    # legitimately exist — never echo them back out through the read model.
    return {
        key: value
        for key, value in params.items()
        if not key.startswith("_") and key not in _PRIVATE_PARAMS
    }


def summarize_experiment(directory: Path) -> dict[str, object]:
    directory = Path(directory)
    summary: dict[str, object] = {"experiment_id": directory.name}
    try:
        state = experiment_state(directory)
        records = read_ledger_records(directory)
        folds = list(latest_fold_records(records).values())
        folds.sort(key=lambda row: (str(row.get("epoch_id")), str(row.get("test_period") or row.get("fold_id"))))
        heldout = latest_heldout_records(records)
        meta = [row for row in records if row.get("record_type") == "meta_learning"]
        params = read_json(directory / HITL_DIR_NAME / PARAMS_NAME)
        schedule = read_json(directory / HITL_DIR_NAME / SCHEDULE_NAME)
        sessions = schedule.get("sessions") if isinstance(schedule.get("sessions"), list) else None
        revealed = test_results_revealed(directory, records)
        epochs = sorted({str(row.get("epoch_id")) for row in folds if row.get("epoch_id")})
        latest_epoch = epochs[-1] if epochs else None
        completed_sessions, total_sessions = _durable_session_progress(sessions, records)
        status = state.get("status") or {}
        summary.update(state)
        summary.update(
            {
                "created_at": _created_at(directory, params),
                "current_session": status.get("session_key") if isinstance(status, Mapping) else None,
                "session_started_at": status.get("session_started_at") if isinstance(status, Mapping) else None,
                "environment_stage": status.get("environment_stage") if isinstance(status, Mapping) else None,
                "environment_stage_started_at": (
                    status.get("environment_stage_started_at") if isinstance(status, Mapping) else None
                ),
                "environment_progress": (
                    status.get("environment_progress") if isinstance(status, Mapping) else None
                ),
                "folds_recorded": len(folds),
                "meta_recorded": len(meta),
                "heldout_recorded": len(heldout),
                "completed_sessions": completed_sessions,
                "total_sessions": total_sessions,
                "test_revealed": revealed,
                # Epochs re-run the same fold calendar, so cumulative metrics
                # must never mix epochs (that compounds each quarter once per
                # epoch). Headline = the latest epoch; earlier epochs stay
                # comparable side by side in metrics_by_epoch.
                "metrics": {
                    "epoch_id": latest_epoch,
                    "cum_valid_return": _compound(
                        _metric_series(_epoch_folds(folds, latest_epoch), "validation_result", "total_return")
                    ),
                    "cum_test_return": _compound(
                        _metric_series(_epoch_folds(folds, latest_epoch), "test_result", "total_return")
                    ) if revealed else None,
                    "mean_test_sharpe": _mean(
                        _metric_series(_epoch_folds(folds, latest_epoch), "test_result", "sharpe")
                    ) if revealed else None,
                    "cum_heldout_return": _compound(
                        _metric_series(heldout, "result", "total_return")
                    ) if revealed else None,
                },
                "metrics_by_epoch": [
                    {
                        "epoch_id": epoch,
                        "folds": len(epoch_folds),
                        "cum_valid_return": _compound(
                            _metric_series(epoch_folds, "validation_result", "total_return")
                        ),
                        "cum_test_return": _compound(
                            _metric_series(epoch_folds, "test_result", "total_return")
                        ) if revealed else None,
                        "mean_test_sharpe": _mean(
                            _metric_series(epoch_folds, "test_result", "sharpe")
                        ) if revealed else None,
                    }
                    for epoch in epochs
                    for epoch_folds in [_epoch_folds(folds, epoch)]
                ],
                "fold_returns": [
                    {
                        "epoch_id": record.get("epoch_id"),
                        "fold_id": record.get("fold_id"),
                        "fold_status": record.get("fold_status"),
                        "valid_return": _metric(record, "validation_result", "total_return"),
                        "test_return": _metric(record, "test_result", "total_return") if revealed else None,
                        # Long-leg P&L return as recorded by the replay summary.
                        "valid_long": _metric(record, "validation_result", "long_return"),
                        "test_long": _metric(record, "test_result", "long_return") if revealed else None,
                    }
                    for record in folds
                ],
                "heldout_returns": [
                    {
                        "label": str(record.get("fold_id", "")).removeprefix("heldout_"),
                        "return": _metric(record, "result", "total_return"),
                    }
                    for record in sorted(heldout, key=lambda row: str(row.get("fold_id")))
                ] if revealed else [],
            }
        )
    except Exception as exc:  # noqa: BLE001 - broken experiments remain inspectable and deletable
        summary.update({"state": "unreadable", "worker_alive": False, "error": f"{type(exc).__name__}: {exc}"})
    return summary


def list_experiments(root: Path) -> list[dict[str, object]]:
    root = Path(root)
    if not root.is_dir():
        return []
    rows = [
        summarize_experiment(path)
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    return sorted(rows, key=lambda row: str(row.get("created_at") or ""), reverse=True)


def resolve_experiment_dir(root: Path, experiment_id: str) -> Path:
    if not _ID.fullmatch(experiment_id):
        raise ValueError(f"invalid experiment id: {experiment_id!r}")
    root = Path(root).resolve()
    path = (root / experiment_id).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"invalid experiment id: {experiment_id!r}")
    if not path.is_dir():
        raise KeyError(f"unknown experiment: {experiment_id}")
    return path


def experiment_detail(root: Path, experiment_id: str) -> dict[str, object]:
    directory = resolve_experiment_dir(root, experiment_id)
    detail = summarize_experiment(directory)
    if detail.get("state") == "unreadable":
        # Same isolation as the homepage list: a broken experiment renders as
        # a structured error detail (state + error), never a 500 — it stays
        # inspectable and deletable from the console.
        return {
            **detail,
            "params": {},
            "control": None,
            "schedule": {},
            "sessions": [],
            "test_revealed": False,
            "heldout_records": [],
            "ledger": [],
            "inbox": {"pending_count": 0, "queued_ids": []},
        }
    records = read_ledger_records(directory)
    folds = latest_fold_records(records)
    heldout = latest_heldout_records(records)
    meta_records = [
        row for row in records if row.get("record_type") == "meta_learning"
    ]
    latest_meta = meta_records[-1] if meta_records else None
    current_prior = latest_prior_text(records, experiment_dir=directory)
    # A fully recorded held-out means the terminal evaluation finished, so
    # results auto-reveal (mirrors test_results_revealed, reusing the records
    # already in hand).
    revealed = test_results_revealed(directory, records)
    hitl = directory / "hitl"
    params = read_json(hitl / "params.json")
    schedule = read_json(hitl / "schedule.json")
    plan = schedule.get("sessions") if isinstance(schedule.get("sessions"), list) else []
    sessions: list[dict[str, object]] = []
    for planned in plan:
        if not isinstance(planned, dict):
            continue
        entry = dict(planned)
        entry.setdefault("key", entry.get("session_key"))
        kind = str(entry.get("kind") or "")
        if kind == "meta":
            entry["kind"] = "meta_learning"
            kind = "meta_learning"
        if kind == "fold":
            record = folds.get((str(entry.get("epoch_id")), str(entry.get("fold_id"))))
            if record is not None:
                entry["record"] = guarded_fold_view(record)
                entry["analysis_available"] = analysis_available(
                    hitl, str(entry.get("epoch_id")), str(entry.get("fold_id"))
                )
        elif kind == "meta_learning":
            record = next(
                (
                    row
                    for row in reversed(records)
                    if row.get("record_type") == "meta_learning"
                    and (row.get("session_key") == entry.get("session_key") or row.get("fold_id") == entry.get("fold_id"))
                ),
                None,
            )
            if record is not None:
                entry["record"] = _public_meta_view(
                    record,
                    current_prior=current_prior if record is latest_meta else None,
                )
        elif kind == "heldout" and heldout:
            entry["records"] = [_public_heldout_view(row, revealed) for row in heldout]
        sessions.append(entry)
    control = read_control(hitl / "control.json")
    current_session = detail.get("current_session")
    inbox_session = (
        str(current_session) if isinstance(current_session, str) and current_session else None
    )
    return {
        **detail,
        "params": _public_params(params),
        "control": control.to_record(),
        "inbox": inbox_public_view(hitl / INBOX_NAME, session_key=inbox_session),
        # Effective reveal state (manual reveal OR held-out completed); the
        # UI must key on this, not on the raw control flag.
        "test_revealed": revealed,
        "schedule": schedule,
        "sessions": sessions,
        "heldout_records": [_public_heldout_view(row, revealed) for row in heldout],
        "ledger": [
            guarded_fold_view(row) if row.get("record_type") == "fold"
            else _public_heldout_view(row, revealed) if row.get("record_type") == "heldout"
            else _public_meta_view(
                row,
                current_prior=current_prior if row is latest_meta else None,
            ) if row.get("record_type") == "meta_learning"
            else dict(row)
            for row in records
        ],
    }


def fold_detail(root: Path, experiment_id: str, epoch_id: str, fold_id: str) -> dict[str, object]:
    directory = resolve_experiment_dir(root, experiment_id)
    records = read_ledger_records(directory)
    record = latest_fold_records(records).get((epoch_id, fold_id))
    if record is None:
        raise KeyError(f"no fold record for {epoch_id}/{fold_id}")
    hitl_dir = directory / HITL_DIR_NAME
    strategy_dir = record.get("frozen_strategy_artifact_path")
    md_path, meta_path = analysis_paths(hitl_dir / ANALYSIS_DIR_NAME, epoch_id, fold_id)
    return {
        "experiment_id": experiment_id,
        "epoch_id": epoch_id,
        "fold_id": fold_id,
        "record": guarded_fold_view(record),
        # Guarded test view: hidden entirely until the researcher reveals (and
        # thereby seals) the experiment; then shown collapsed with a warning.
        "test_audit": (
            {field: record.get(field) for field in TEST_FIELDS}
            if test_results_revealed(directory, records) else {"hidden": True}
        ),
        # Frozen artifacts are downloaded as ONE zip package (strategy.zip);
        # the console deliberately offers no per-file listing/download.
        "strategy_dir": str(strategy_dir) if strategy_dir else None,
        "analysis": {
            "available": md_path.exists(),
            "meta": read_json(meta_path) if meta_path.exists() else None,
        },
        "run_id": record.get("run_id"),
    }


def analysis_available(hitl_dir: Path, epoch_id: str, fold_id: str) -> bool:
    md_path, _meta = analysis_paths(hitl_dir / ANALYSIS_DIR_NAME, epoch_id, fold_id)
    return md_path.exists()


def fold_run_id(root: Path, experiment_id: str, epoch_id: str, fold_id: str) -> str:
    """Resolve a Fold run from the append-only ledger."""

    directory = resolve_experiment_dir(root, experiment_id)
    records = read_ledger_records(directory)
    record = latest_fold_records(records).get((epoch_id, fold_id))
    run_id = str(record.get("run_id") or "") if isinstance(record, Mapping) else ""
    if not _ID.fullmatch(run_id):
        raise KeyError(f"no fold record for {epoch_id}/{fold_id}")
    return run_id


def _selected_validation_ref(record: Mapping[str, object]) -> object:
    steps = [
        item
        for item in record.get("steps", [])
        if isinstance(item, Mapping) and item.get("validation_result_ref")
    ]
    selected = str(record.get("selected_step_id") or "")
    return next(
        (
            step.get("validation_result_ref")
            for step in steps
            if str(step.get("step_id") or "") == selected
        ),
        steps[-1].get("validation_result_ref") if steps else None,
    )


def style_payload(
    root: Path,
    experiment_id: str,
    *,
    run_id: str,
    prefix: str,
) -> dict[str, object]:
    """Read the canonical style sidecar selected by the trusted ledger."""

    if not _ID.fullmatch(run_id):
        raise ValueError("invalid run_id")
    if prefix not in {"valid", "test", "heldout"}:
        raise ValueError("prefix must be valid|test|heldout")
    experiment_dir = resolve_experiment_dir(root, experiment_id)
    # Reveal gate: pre-reveal, sealed prefixes answer exactly like a missing
    # rollup so test/held-out existence never leaks.
    if prefix.startswith(sealed_result_prefixes(experiment_dir)):
        raise KeyError("该运行没有已落盘的风格归因结果")
    records = read_ledger_records(experiment_dir)

    reference: object = None
    if prefix == "valid":
        fold = next(
            (
                row
                for row in latest_fold_records(records).values()
                if str(row.get("run_id") or "") == run_id
            ),
            None,
        )
        if fold is not None:
            reference = _selected_validation_ref(fold)
    elif prefix == "test":
        fold = next(
            (
                row
                for row in latest_fold_records(records).values()
                if str(row.get("run_id") or "") == run_id
            ),
            None,
        )
        if fold is not None:
            reference = fold.get("test_result_ref")
    elif prefix == "heldout":
        heldout = next(
            (
                row
                for row in latest_heldout_records(records)
                if str(row.get("run_id") or "") == run_id
            ),
            None,
        )
        if heldout is not None:
            reference = heldout.get("result_ref")

    result_file = _result_file(experiment_dir, reference)
    if result_file is None:
        raise KeyError("该运行没有已落盘的风格归因结果")
    sidecar = (result_file.parent / STYLE_ARTIFACT_NAME).resolve()
    if not sidecar.is_relative_to(experiment_dir.resolve()) or not sidecar.is_file():
        raise KeyError("该运行没有已落盘的风格归因结果")
    try:
        payload = read_json(sidecar)
    except (OSError, ValueError) as exc:
        raise KeyError("该运行没有已落盘的风格归因结果") from exc
    expected_mode = {"valid": "valid", "test": "frozen_test", "heldout": "heldout"}[prefix]
    if payload.get("schema_version") != STYLE_SCHEMA_VERSION or payload.get("mode") != expected_mode:
        raise KeyError("该运行没有已落盘的风格归因结果")
    return payload


def _result_file(experiment_dir: Path, reference: object) -> Path | None:
    if not isinstance(reference, str) or not reference:
        return None
    raw = Path(reference)
    path = raw.resolve() if raw.is_absolute() else (experiment_dir / raw).resolve()
    if not path.is_relative_to(experiment_dir.resolve()):
        return None
    if path.is_file():
        return path
    for name in ("result.json", "detailed_return.json"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    return None


def _fold_result_files(experiment_dir: Path, record: Mapping[str, object], *, revealed: bool) -> dict[str, Path]:
    files: dict[str, Path] = {}
    steps = [item for item in record.get("steps", []) if isinstance(item, dict)]
    selected = str(record.get("selected_step_id") or "")
    ordered = sorted(steps, key=lambda item: str(item.get("step_id") or ""))
    for step in ordered:
        path = _result_file(experiment_dir, step.get("validation_result_ref"))
        if path is None:
            continue
        label = path.parent.name
        if str(step.get("step_id") or "") == selected:
            files = {label: path, **files}
        else:
            files[label] = path
    if revealed:
        test = _result_file(experiment_dir, record.get("test_result_ref"))
        if test is not None:
            files[test.parent.name] = test
    run_id = str(record.get("run_id") or "")
    if run_id and Path(run_id).name == run_id:
        results = experiment_dir / "artifacts" / run_id / "results"
        sealed = sealed_result_prefixes(experiment_dir)
        if results.is_dir():
            for directory in sorted(path for path in results.iterdir() if path.is_dir()):
                # Pre-reveal: invisible, not just unselectable.
                if sealed and directory.name.startswith(sealed):
                    continue
                path = _result_file(experiment_dir, str(directory))
                if path is not None:
                    files.setdefault(directory.name, path)
    return files


def fold_orders(
    root: Path,
    experiment_id: str,
    epoch_id: str,
    fold_id: str,
    *,
    result: str | None = None,
    max_rows: int | None = 500,
) -> dict[str, object]:
    """Order stream + aggregate stats for one fold backtest result.

    Defaults to the selected Step's validation result; test results are only
    served when explicitly requested (the console keeps them inside the guarded
    audit block). ``max_rows`` caps the returned rows — ``row_count`` and the
    stats always describe the whole stream — and ``None`` returns every row
    (the CSV export).
    """
    experiment_dir = resolve_experiment_dir(root, experiment_id)
    records = read_ledger_records(experiment_dir)
    record = latest_fold_records(records).get((epoch_id, fold_id))
    if record is None:
        raise KeyError(f"no fold record for {epoch_id}/{fold_id}")
    files = _fold_result_files(
        experiment_dir, record, revealed=test_results_revealed(experiment_dir, records)
    )
    available = [name for name in files if not name.startswith("test")]
    test_results = [name for name in files if name.startswith("test")]
    selected = result or (available[0] if available else None)
    if selected is None:
        return {
            "result": None, "available": [], "test_results": test_results,
            "stats": _order_stats([]), "rows": [], "row_count": 0, "truncated": False,
        }
    if selected not in files:
        raise KeyError(f"unknown result: {selected}")
    try:
        payload = json.loads(files[selected].read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KeyError(f"unreadable result: {selected}") from exc
    rows = payload.get("executions") if isinstance(payload, dict) else None
    orders = [dict(item) for item in rows if isinstance(item, dict)] if isinstance(rows, list) else []
    return {
        "result": selected,
        "available": available,
        "test_results": test_results,
        "stats": _order_stats(orders),
        "rows": orders if max_rows is None else orders[:max_rows],
        "row_count": len(orders),
        "truncated": max_rows is not None and len(orders) > max_rows,
    }


def _order_trade_date(row: Mapping[str, object]) -> str:
    """``YYYYMMDD`` of a filled execution, from its matched (else planned) stamp."""
    stamp = str(row.get("matched_at") or row.get("execute_at") or "")
    return stamp[:10].replace("-", "") if len(stamp) >= 10 else ""


def _order_stats(orders: list[dict[str, object]]) -> dict[str, object]:
    """Aggregates the transaction pane renders (tiles, per-day bars, reject chips).

    Computed here, not in the browser: the console keeps the arithmetic on the
    server so every surface reads the same numbers off one projection.
    """
    filled = [row for row in orders if row.get("status") == "filled"]
    by_action: dict[str, int] = {}
    for row in orders:
        by_action[str(row.get("action") or "")] = by_action.get(str(row.get("action") or ""), 0) + 1
    reject_reasons: dict[str, int] = {}
    for row in orders:
        if row.get("status") != "rejected":
            continue
        reason = str(row.get("reason") or "unknown")
        reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
    daily: dict[str, list[float]] = {}
    for row in filled:
        day = _order_trade_date(row)
        if not day:
            continue
        amount = (_number(row.get("price")) or 0.0) * (_number(row.get("quantity")) or 0.0)
        entry = daily.setdefault(day, [0.0, 0.0])
        entry[0] += 1
        entry[1] += amount
    return {
        "orders": len(orders),
        "filled": len(filled),
        "rejected": sum(1 for row in orders if row.get("status") == "rejected"),
        "turnover": sum(
            (_number(row.get("price")) or 0.0) * (_number(row.get("quantity")) or 0.0) for row in filled
        ),
        "by_action": by_action,
        "reject_reasons": dict(sorted(reject_reasons.items(), key=lambda item: -item[1])[:6]),
        "daily": [
            {"trade_date": day, "filled_count": int(count), "amount": amount}
            for day, (count, amount) in sorted(daily.items())
        ],
    }


def fold_orders_csv(
    root: Path,
    experiment_id: str,
    epoch_id: str,
    fold_id: str,
    *,
    result: str,
) -> tuple[str, str]:
    # Uncapped: the export is the escape hatch from the table's row cap.
    payload = fold_orders(root, experiment_id, epoch_id, fold_id, result=result, max_rows=None)
    fields = (
        "symbol", "action", "quantity", "execute_at", "matched_at", "status",
        "price", "commission", "stamp_duty", "reason",
    )
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(payload["rows"])
    return f"{experiment_id}__{epoch_id}__{fold_id}__{result}.csv", stream.getvalue()
