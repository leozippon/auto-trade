"""The raw lake's generation record, ``<raw>/.raw_generation.json``: which
committed state the lake is in, each dataset's refresh time and vintage, and
the open issues (defective or staged partitions, pending history re-pulls).

A mutating job marks the record ``updating`` before its first write
(:func:`begin_raw_generation_update`), folds its children's partition
outcomes into it (:func:`fold_outcomes`, :func:`record_vintages`) and ends it
with :func:`finish_raw_generation_update`. The schema and states are owned by
``autotrade.environment.data.contracts``, which the PIT readers share.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from autotrade.common.records import read_json_object
from autotrade.common.time import utc_now
from autotrade.environment.data.contracts import (
    GENERATION_COMMITTED,
    GENERATION_SCHEMA_VERSION,
    GENERATION_UPDATING,
    RAW_GENERATION_FILENAME,
)

from .versions import dataset_vintage


def read_raw_generation_file(raw_dir: Path) -> dict:
    """Writer-side lenient read: a job must see an ``updating`` record too."""
    path = raw_dir / RAW_GENERATION_FILENAME
    if not path.exists():
        return {}
    try:
        return read_json_object(path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid raw generation record: {path}: {exc}") from exc


def _write_raw_generation_file(raw_dir: Path, payload: dict) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / RAW_GENERATION_FILENAME
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def begin_raw_generation_update(raw_dir: Path, transaction: dict) -> dict:
    """Mark the lake as being written before the first child process starts;
    the previous record is returned for the job's end.

    No state refuses a job. Every partition write is an atomic version swap,
    so a failed job leaves a consistent lake (older in what it did not reach,
    which the per-dataset refresh times show), and the updater lock already
    serialises writers. ``updating`` only tells consumers that the lake may
    differ from ``generation_id`` until this job commits."""
    previous = read_raw_generation_file(raw_dir)
    _write_raw_generation_file(raw_dir, {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "state": GENERATION_UPDATING,
        "generation_id": str(previous.get("generation_id", "")),
        "completed_at": str(previous.get("completed_at", "")),
        "updated_at": utc_now(),
        "transaction": {**transaction, "started_at": utc_now()},
        "datasets": dict(previous.get("datasets") or {}),
        "issues": list(previous.get("issues") or []),
        # The versioned era's start, which the job's writes read.
        **({"versioned_since": previous["versioned_since"]} if previous.get("versioned_since") else {}),
    })
    return previous


def finish_raw_generation_update(
    raw_dir: Path,
    previous: dict,
    transaction: dict,
    *,
    commit: bool,
    datasets: dict,
    issues: list,
) -> dict:
    """End a job: publish a new generation when the lake may have changed
    (``commit``), else restore the previous identity; either way record the
    per-dataset state and open issues the job leaves. The versioned era's
    start carries over.

    Restoring the identity is sound only over a record that was committed when
    the job began: one left ``updating`` by a crashed run (or none at all)
    describes a lake that may already differ from its id, so it always gets a
    new one."""
    commit = commit or previous.get("state") != GENERATION_COMMITTED
    payload = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "state": GENERATION_COMMITTED,
        "generation_id": uuid.uuid4().hex if commit else str(previous.get("generation_id", "")),
        "completed_at": utc_now() if commit else str(previous.get("completed_at", "")),
        "transaction": transaction if commit else previous.get("transaction") or {},
        "datasets": datasets,
        "issues": issues,
    }
    if previous.get("versioned_since"):
        payload["versioned_since"] = previous["versioned_since"]
    _write_raw_generation_file(raw_dir, payload)
    if commit:
        # One JSON line, like the runner's outcome lines.
        print(json.dumps({
            "note": "raw_generation_published",
            "generation_id": payload["generation_id"],
            "open_issues": len(issues),
            "path": str(raw_dir / RAW_GENERATION_FILENAME),
            "updated_at": utc_now(),
        }, ensure_ascii=False))
    return payload


def issue_key(issue: dict) -> tuple[str, str, str]:
    return str(issue.get("kind")), str(issue.get("dataset")), str(issue.get("partition") or "")


def fold_outcomes(datasets: dict, issues: list, outcomes: list[dict], *, at: str) -> tuple[set[str], set[str]]:
    """Apply one batch of outcomes to the record's datasets and issues;
    returns (datasets pulled, datasets restated).

    A defect or a staged version opens an issue for its partition; any other
    outcome for the partition closes them. A dataset is refreshed at ``at``
    only when its whole pull completed (a ``complete`` outcome, reported when
    the tier that pulls it returned without an error) and none of its
    partitions was defective or staged in the same batch: a job that failed
    after landing the daily core refreshes the core and leaves the steps it
    never finished at their earlier time."""
    open_issues = {issue_key(issue): dict(issue) for issue in issues}
    pulled: set[str] = set()
    complete: set[str] = set()
    withheld: set[str] = set()
    restated: set[str] = set()
    for outcome in outcomes:
        dataset, partition, kind = str(outcome["dataset"]), str(outcome["partition"]), str(outcome["outcome"])
        pulled.add(dataset)
        if kind == "complete":
            complete.add(dataset)
            continue
        if kind in {"defect", "staged"}:
            key = (kind, dataset, partition)
            issue = open_issues.get(key) or {"kind": kind, "dataset": dataset, "partition": partition, "since": outcome["at"]}
            issue.update({"detail": outcome.get("detail", ""), "version": outcome.get("version", "")})
            open_issues[key] = issue
            withheld.add(dataset)
            continue
        open_issues.pop(("defect", dataset, partition), None)
        open_issues.pop(("staged", dataset, partition), None)
        if kind == "restated":
            restated.add(dataset)
    for dataset in complete - withheld:
        datasets.setdefault(dataset, {})["refreshed_at"] = at
    issues[:] = sorted(open_issues.values(), key=issue_key)
    return pulled, restated


def record_vintages(raw_dir: Path, datasets: dict, names: set[str]) -> None:
    for name in sorted(names):
        if (raw_dir / name).is_dir():
            datasets.setdefault(name, {})["vintage"] = dataset_vintage(raw_dir / name)
