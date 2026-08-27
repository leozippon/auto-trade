"""The single experiment ledger (docs/pipeline-design.md §4.1).

One JSONL file per experiment. Records are distinguished by ``record_type``
(``fold`` / ``meta_learning`` / ``heldout``); Steps are lightweight
summaries inside the fold record's ``steps[]``, never separate files.
``attempt_failed`` records are appended when a run throws before its success
record — they carry the error evidence and are ignored by every reader that
selects the success types, so a failed attempt is re-runnable but auditable.

A ``fold`` or ``heldout`` row with ``state_changed_during_test=true`` is an
integrity failure, not a success: it is persisted before fail-fast so the
corruption is auditable, then every resume, retry, held-out, and parent
selection must refuse until a human rolls the dirty frozen trees back.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Mapping
from pathlib import Path

from autotrade.environment.runtime import sanitize_for_log, utc_now_iso

# Stamped on every appended record; bump when the record shape changes.
LEDGER_RECORD_SCHEMA_VERSION = 1
RECORD_TYPES = ("fold", "meta_learning", "heldout", "attempt_failed")
LINK_KEYS = ("experiment_id", "epoch_id", "fold_id", "run_id")
DURABLE_SUCCESS_TYPES = ("fold", "meta_learning", "heldout")
_INTEGRITY_RECORD_TYPES = frozenset({"fold", "heldout"})


class FrozenArtifactMutated(RuntimeError):
    """Frozen output/models changed during Test or Held-out.

    The integrity record is already in the ledger. Same-process retries,
    resume, rerun, held-out, and parent selection must refuse until a human
    rolls back the dirty trees.
    """


class FrozenArtifactRestoreFailed(FrozenArtifactMutated):
    """Mutation was detected, but pre-evaluation frozen bytes could not be restored.

    Worse than ``FrozenArtifactMutated``: the live trees must not be treated as
    clean. The integrity record is still written when the caller can append it.
    """


def is_frozen_artifact_mutation(record: Mapping[str, object]) -> bool:
    """True when a fold/held-out row flags frozen output/models as changed."""
    return (
        record.get("record_type") in _INTEGRITY_RECORD_TYPES
        and record.get("state_changed_during_test") is True
    )


def is_durable_success_record(
    record: Mapping[str, object],
    *,
    record_types: tuple[str, ...] | None = None,
) -> bool:
    """Fold/meta/held-out rows that may be treated as completed work."""
    types = record_types if record_types is not None else DURABLE_SUCCESS_TYPES
    if record.get("record_type") not in types:
        return False
    return not is_frozen_artifact_mutation(record)


def assert_no_frozen_artifact_mutation(records: list[dict[str, object]]) -> None:
    """Refuse further pipeline work while an integrity-failure row remains."""
    for record in records:
        if not is_frozen_artifact_mutation(record):
            continue
        phase = (
            "held-out" if record.get("record_type") == "heldout" else "frozen test"
        )
        raise FrozenArtifactMutated(
            "strategy or model artifacts changed during "
            f"{phase}; refuse retry, resume, rerun, held-out, and parent "
            "selection until the frozen trees are rolled back"
        )


def latest_fold_records(records: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    """Latest successful fold record per (epoch, fold): the ledger is append-only, so a
    re-run appends a superseding record. Formal consumers (reporting, console)
    must never double-count earlier attempts. Integrity-failure rows are not
    adopted as the official latest result."""
    latest: dict[tuple[str, str], dict[str, object]] = {}
    for record in records:
        if not is_durable_success_record(record, record_types=("fold",)):
            continue
        latest[(str(record.get("epoch_id")), str(record.get("fold_id")))] = record
    return latest


def latest_heldout_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Latest successful record per held-out period (a fold re-run replays held-out, so
    earlier period records are superseded, not removed). Integrity-failure rows
    are not adopted as the official latest result."""
    latest: dict[str, dict[str, object]] = {}
    for record in records:
        if not is_durable_success_record(record, record_types=("heldout",)):
            continue
        latest[str(record.get("fold_id"))] = record
    return [latest[key] for key in sorted(latest)]


class ExperimentLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: dict[str, object]) -> None:
        record_type = record.get("record_type")
        if record_type not in RECORD_TYPES:
            raise ValueError(f"unsupported record_type: {record_type!r}")
        missing = [key for key in LINK_KEYS if not record.get(key)]
        if missing:
            raise ValueError(f"ledger record missing link keys: {missing}")
        # Stamps come after the spread so a caller-supplied schema_version or
        # recorded_at can never override the ledger's own.
        payload = {
            **sanitize_for_log(record),
            "schema_version": LEDGER_RECORD_SCHEMA_VERSION,
            "recorded_at": utc_now_iso(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # flock + fsync: fold records regularly exceed the 8 KiB text buffer,
        # so an unlocked write reaches the file in multiple chunks and a
        # concurrent console read can see a torn final line; and a finished
        # run's record must survive power loss (the pipeline appends the
        # ledger record before treating the run as recorded).
        with self.path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def rewrite(self, records: list[dict[str, object]]) -> None:
        """Atomic full rewrite for migrations and Fold/Held-out rollback.

        The rolling-upgrade write-isolation guard is part of the primitive,
        not a procedural convention: the rewrite refuses while the owning
        experiment worker is alive, and refuses records that do not already
        carry the current schema stamp (a migration or rollback must hand over
        fully migrated records). Callers must go through this method instead of
        editing the file by hand.
        """
        # Local import: hitl_state pulls in the config/session stack, which
        # this module must not load for its plain append/read paths.
        from autotrade.pipelines.hitl_state import assert_no_live_writer

        assert_no_live_writer(self.path.parent.parent)
        for record in records:
            version = record.get("schema_version")
            if type(version) is not int or version != LEDGER_RECORD_SCHEMA_VERSION:
                raise ValueError(
                    f"rewrite requires fully migrated records; got schema_version {version!r}"
                )
        tmp = self.path.with_suffix(".jsonl.tmp")
        tmp.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def read(self, record_type: str | None = None) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        # Shared lock pairs with append's exclusive lock so a live console
        # read can never observe a half-written record. External corruption
        # (truncation, foreign writers) still fails fast in json.loads below.
        with self.path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                text = handle.read()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        for record in records:
            version = record.get("schema_version")
            # type() check: JSON true/1.0 must not pass as 1 (bool subclasses
            # int and floats compare equal), and "1" must not pass either.
            if type(version) is not int or version != LEDGER_RECORD_SCHEMA_VERSION:
                # Fail-fast, no legacy tolerance: a missing or unknown version
                # means a foreign/newer format that older code must not
                # silently misinterpret — migrate the ledger, don't guess.
                raise ValueError(
                    f"ledger record schema_version {version!r} != "
                    f"{LEDGER_RECORD_SCHEMA_VERSION} in {self.path}; migrate the ledger before reading"
                )
        if record_type is None:
            return records
        return [record for record in records if record.get("record_type") == record_type]
