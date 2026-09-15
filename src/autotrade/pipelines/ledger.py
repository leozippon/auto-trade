"""The single experiment ledger (docs/pipeline-design.md §4.1).

One JSONL file per experiment. The pipeline writes three record types:

- ``research_session``: the arm's one research session, its outcome and the
  Steps of every attempt; it carries the arm's ``frozen`` block when the
  nominee passed the freeze gate, otherwise ``arm_end``;
- ``forward``: the continuous forward and Held-out replay of the frozen
  artifact, its two slices and the graduation verdict;
- ``attempt_failed``: a session or replay that raised before its business
  record. It carries the error evidence and is ignored by every reader of the
  business types, so the attempt is re-runnable but auditable. A run killed
  outright cannot append it itself, so every run also leaves a host-only
  marker (:class:`RunMarkers`) that the next worker start turns into the
  missing ``attempt_failed``.

Every record carries the link keys ``experiment_id``, ``epoch_id``, ``fold_id``
and ``run_id``: ``epoch_id`` names the stage (``research`` or ``forward``) and
``fold_id`` the session key (``research`` or ``forward``). Both keep their
Fold-era names because renaming a persisted link key needs a schema bump.

A ``forward`` row with ``state_changed_during_test=true`` is an integrity
failure, not a verdict: it is persisted before fail-fast so the corruption is
auditable, then every resume and retry must refuse until a human rolls the
dirty frozen trees back.

``FOLD_ERA_RECORD_TYPES`` names the record types of archived Fold-era
ledgers only so the worker can refuse such a ledger; ``append`` accepts none of
them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from autotrade.environment.runtime import (
    append_versioned_jsonl,
    read_versioned_jsonl,
    utc_now_iso,
    write_json_atomic,
)

# Stamped on every appended record; bump when the record shape changes.
LEDGER_RECORD_SCHEMA_VERSION = 1
# The stage names the link key ``epoch_id`` carries.
RESEARCH_STAGE = "research"
FORWARD_STAGE = "forward"
# The two planned sessions' keys, also their ``fold_id``: the arm's one
# research session and the forward replay.
RESEARCH_SESSION_KEY = "research"
FORWARD_SESSION_KEY = "forward"
PIPELINE_RECORD_TYPES = ("research_session", "forward", "attempt_failed")
FOLD_ERA_RECORD_TYPES = (
    "fold",
    "meta_learning",
    "heldout",
    "deployment_adjustment",
    "terminated",
)
LINK_KEYS = ("experiment_id", "epoch_id", "fold_id", "run_id")
# The ``status`` of a forward record whose replay stopped at the strategy's own
# exception: the one replay failure that measures the strategy. Every other
# failure fails the attempt and never reaches a business record.
STRATEGY_ERROR = "strategy_error"
DURABLE_SUCCESS_TYPES = ("research_session", "forward")
_INTEGRITY_RECORD_TYPES = frozenset({"forward"})

# Host-only, never mounted into a sandbox and never Agent-visible.
RUN_MARKER_DIR = ".host/runs"
INTERRUPTED_RUN_ERROR = (
    "RunInterrupted: the run process exited before it wrote a ledger record"
)
# The marker exists to survive exactly the events that can also tear it (the
# atomic write renames without fsync, so a SIGKILL/OOM/host reset can leave a
# zero-length or truncated file). Such a marker still proves a run died, so it
# becomes an ``attempt_failed`` too, with its unknown link keys named as unknown.
UNREADABLE_RUN_MARKER_ERROR = (
    "RunMarkerUnreadable: the run process exited before it wrote a ledger "
    "record and its run marker could not be read back"
)
UNKNOWN_MARKER_LINK_KEY = "unknown"


class FrozenArtifactMutated(RuntimeError):
    """Frozen output/models changed during a replay of the frozen artifact.

    The integrity record is already in the ledger. Same-process retries and
    resume must refuse until a human rolls back the dirty trees.
    """


class FrozenArtifactRestoreFailed(FrozenArtifactMutated):
    """Mutation was detected, but pre-evaluation frozen bytes could not be restored.

    Worse than ``FrozenArtifactMutated``: the live trees must not be treated as
    clean. The integrity record is still written when the caller can append it.
    """


def is_frozen_artifact_mutation(record: Mapping[str, object]) -> bool:
    """True when a row flags frozen output/models as changed."""
    return (
        record.get("record_type") in _INTEGRITY_RECORD_TYPES
        and record.get("state_changed_during_test") is True
    )


def is_durable_success_record(
    record: Mapping[str, object],
    *,
    record_types: tuple[str, ...] | None = None,
) -> bool:
    """Business rows that may be treated as completed work."""
    types = record_types if record_types is not None else DURABLE_SUCCESS_TYPES
    if record.get("record_type") not in types:
        return False
    return not is_frozen_artifact_mutation(record)


def assert_no_frozen_artifact_mutation(records: list[dict[str, object]]) -> None:
    """Refuse further pipeline work while an integrity-failure row remains."""
    for record in records:
        if is_frozen_artifact_mutation(record):
            raise FrozenArtifactMutated(
                "strategy or model artifacts changed during the "
                f"{record.get('record_type')} replay; refuse retry and resume "
                "until the frozen trees are rolled back"
            )


def research_records(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """The research sessions' records, in the order they ran."""

    return [
        dict(record)
        for record in records
        if record.get("record_type") == "research_session"
    ]


def frozen_record(records: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    """The research session record that froze the arm's artifact, or None.

    An arm freezes at most once; a ledger holding two freezes is corrupt and
    raises rather than letting a reader pick one.
    """

    frozen = [record for record in research_records(records) if record.get("frozen")]
    if len(frozen) > 1:
        raise ValueError("the ledger holds more than one freeze for one arm")
    return frozen[0] if frozen else None


def forward_record(records: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    """The arm's forward verdict record, or None; an integrity row is not one."""

    rows = [
        dict(record)
        for record in records
        if is_durable_success_record(record, record_types=("forward",))
    ]
    if len(rows) > 1:
        raise ValueError("the ledger holds more than one forward verdict for one arm")
    return rows[0] if rows else None


def research_over(records: Sequence[Mapping[str, object]]) -> bool:
    """Whether research has ended: an artifact froze, or a session ended the arm."""

    return frozen_record(records) is not None or any(
        record.get("arm_end") for record in research_records(records)
    )


def experiment_verdict(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """The arm's verdict: ``graduated``/``discarded`` from its forward record,
    ``no_deliverable`` when research ended without a freeze, else None.

    The single source for the terminal status, the console, the graduated
    memory tier and Paper.
    """

    forward = forward_record(records)
    if forward is not None:
        verdict = forward.get("verdict")
        if not isinstance(verdict, Mapping):
            raise ValueError("forward record carries no verdict block")
        return {
            "status": str(verdict.get("status") or ""),
            "reasons": [str(reason) for reason in verdict.get("reasons") or ()],
        }
    if frozen_record(records) is not None:
        return None
    ended = next(
        (record for record in research_records(records) if record.get("arm_end")),
        None,
    )
    if ended is None:
        return None
    arm_end = ended["arm_end"]
    return {
        "status": "no_deliverable",
        "reasons": [str(arm_end.get("reason") or "") if isinstance(arm_end, Mapping) else ""],
    }


def paper_candidate(records: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    """The artifact Paper pins: the frozen artifact of a graduated arm, else None."""

    verdict = experiment_verdict(records)
    if verdict is None or verdict["status"] != "graduated":
        return None
    frozen = frozen_record(records)
    if frozen is None:
        raise ValueError("a graduated arm has no frozen record")
    block = frozen["frozen"]
    return {
        "artifact_id": str(block["artifact_id"]),
        "output_path": str(block["output_path"]),
        "models_path": str(block.get("models_path") or "") or None,
    }


class ExperimentLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: dict[str, object]) -> None:
        record_type = record.get("record_type")
        if record_type not in PIPELINE_RECORD_TYPES:
            raise ValueError(f"unsupported record_type: {record_type!r}")
        missing = [key for key in LINK_KEYS if not record.get(key)]
        if missing:
            raise ValueError(f"ledger record missing link keys: {missing}")
        if (
            record_type == "research_session"
            and record.get("frozen")
            and frozen_record(self.read()) is not None
        ):
            raise ValueError("the arm already froze an artifact; a second freeze is refused")
        if (
            record_type == "forward"
            and not is_frozen_artifact_mutation(record)
            and forward_record(self.read()) is not None
        ):
            raise ValueError("the arm already has its forward verdict")
        append_versioned_jsonl(
            self.path, record, schema_version=LEDGER_RECORD_SCHEMA_VERSION
        )

    def rewrite(self, records: list[dict[str, object]]) -> None:
        """Atomic full rewrite for migrations and console maintenance.

        The rolling-upgrade write-isolation guard is part of the primitive,
        not a procedural convention: the rewrite refuses while the owning
        experiment worker is alive, and refuses records that do not already
        carry the current schema stamp (a migration must hand over fully
        migrated records). Callers must go through this method instead of
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
        records = read_versioned_jsonl(
            self.path,
            schema_version=LEDGER_RECORD_SCHEMA_VERSION,
            label="ledger record",
        )
        if record_type is None:
            return records
        return [record for record in records if record.get("record_type") == record_type]


class RunMarkers:
    """In-flight run markers that make a killed run auditable.

    Research sessions and the forward replay append ``attempt_failed``
    themselves when they catch an exception, but a SIGKILL, an OOM kill or a host reset leaves no
    code to run: the trace file stops mid-event and the ledger under-reports the
    attempt. Each run therefore writes a marker holding its link keys before it
    starts and deletes it once its ledger record — a business record or an
    ``attempt_failed`` — is durable, so a leftover marker is exactly the
    evidence of a run that died silently.
    """

    def __init__(self, experiment_dir: str | Path) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.root = self.experiment_dir / RUN_MARKER_DIR

    def begin(self, attempt: Mapping[str, object]) -> None:
        missing = [key for key in LINK_KEYS if not attempt.get(key)]
        if missing:
            raise ValueError(f"run marker missing link keys: {missing}")
        self.root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.mkdir(exist_ok=True, mode=0o700)
        write_json_atomic(
            self.root / f"{attempt['run_id']}.json",
            {**dict(attempt), "started_at": utc_now_iso()},
        )

    def finish(self, run_id: str) -> None:
        (self.root / f"{run_id}.json").unlink(missing_ok=True)

    def recover(self, ledger: ExperimentLedger) -> list[dict[str, object]]:
        """Record every run that died before its ledger record, then forget it.

        Called once when an experiment worker starts, the only moment at which
        no run of this experiment is in flight. A marker whose run already
        reached the ledger (the process died between the append and the marker
        cleanup) is dropped without a second record, and every handled marker is
        removed, so repeated restarts never duplicate a record.

        An unreadable marker is handled the same way: it is still the evidence
        of a dead run, and it must never be able to fail every later worker
        start. Its file name is the run id, the experiment directory is the
        experiment id, and the link keys it cannot supply are recorded as
        unknown rather than guessed.
        """
        if not self.root.is_dir():
            return []
        recorded = {
            str(record.get("run_id"))
            for record in ledger.read()
            if record.get("run_id")
        }
        appended: list[dict[str, object]] = []
        for path in sorted(self.root.glob("*.json")):
            marker, unreadable = _read_run_marker(path)
            run_id = str(marker.get("run_id") or path.stem)
            if run_id not in recorded:
                record = {
                    **marker,
                    "record_type": "attempt_failed",
                    "experiment_id": marker.get("experiment_id")
                    or self.experiment_dir.name,
                    "epoch_id": marker.get("epoch_id") or UNKNOWN_MARKER_LINK_KEY,
                    "fold_id": marker.get("fold_id") or UNKNOWN_MARKER_LINK_KEY,
                    "run_id": run_id,
                    "error": (
                        f"{UNREADABLE_RUN_MARKER_ERROR}: {path.name}: {unreadable}"
                        if unreadable
                        else INTERRUPTED_RUN_ERROR
                    ),
                }
                ledger.append(record)
                appended.append(record)
            path.unlink(missing_ok=True)
        return appended


def _read_run_marker(path: Path) -> tuple[dict[str, object], str]:
    """A marker's fields, or ``({}, reason)`` when the file is not a JSON object."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON ({exc})"
    if not isinstance(payload, Mapping):
        return {}, f"payload is {type(payload).__name__}, not a JSON object"
    return dict(payload), ""
