"""The single experiment ledger (docs/PIPELINE_DESIGN.md §4.1).

One JSONL file per experiment. The pipeline writes three record types:

- ``research_session``: the experiment's one research session, its outcome and the
  Steps of every attempt; it carries the experiment's ``registered`` block when the
  nominee registered, otherwise ``experiment_end``;
- ``test_step``: one step of the experiment's validation record, appended by a
  session's ``test`` or by the host's completion (``validation_record``);
- ``forward``: the registered artifact's continuous forward and Held-out
  replay, the two slices and the experiment's readout (``validation_record.readout``);
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
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path

from autotrade.common.records import write_json_atomic
from autotrade.environment.runtime import (
    append_versioned_jsonl,
    read_versioned_jsonl,
    utc_now_iso,
)

# Stamped on every appended record; bump when the record shape changes. 2: the
# research session record's ``registration`` and ``registered`` blocks.
LEDGER_RECORD_SCHEMA_VERSION = 2


class Stage(StrEnum):
    """The stage a record belongs to: the link key ``epoch_id``."""

    RESEARCH = "research"
    FORWARD = "forward"


class SessionKind(StrEnum):
    """The experiment's two planned sessions, the research session and the forward
    replay: each one's kind and key, which is also its records' ``fold_id``."""

    RESEARCH = "research"
    FORWARD = "forward"


# One step of the experiment's validation record (``pipelines/validation_record.py``):
# appended by the session's ``test`` as the step completes, or by the host
# completing the record after the session; a step is recorded exactly once.
TEST_STEP_RECORD_TYPE = "test_step"
# How a research session ends the experiment without a registration (``experiment_end.status``).
# ``no_deliverable`` is a research outcome: the budget ran out before any test
# and no full-span candidate registers on its own. ``environment_blocked`` is not one: the
# host kept failing the session's replays on one span and ended the session
# (``session_tools.EnvironmentBlocked``), so nothing there was tested. Such an
# experiment is flagged on the console's home list.


class VerdictStatus(StrEnum):
    """An experiment's terminal status (:func:`experiment_verdict`): a forward record's
    readout, ``registered_with_record``, or the ``experiment_end`` status of a
    research session that ended the experiment."""

    REGISTERED_WITH_RECORD = "registered_with_record"
    NO_DELIVERABLE = "no_deliverable"
    ENVIRONMENT_BLOCKED = "environment_blocked"


# Also the research session's outcome that ends the experiment with this status.
ENVIRONMENT_BLOCKED = VerdictStatus.ENVIRONMENT_BLOCKED
EXPERIMENT_END_STATUSES = (VerdictStatus.NO_DELIVERABLE, VerdictStatus.ENVIRONMENT_BLOCKED)
PIPELINE_RECORD_TYPES = (
    "research_session",
    TEST_STEP_RECORD_TYPE,
    "forward",
    "attempt_failed",
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


class RegisteredArtifactMutated(RuntimeError):
    """Frozen output/models changed during a replay of the registered artifact.

    The integrity record is already in the ledger. Same-process retries and
    resume must refuse until a human rolls back the dirty trees.
    """


class RegisteredArtifactRestoreFailed(RegisteredArtifactMutated):
    """Mutation was detected, but pre-evaluation frozen bytes could not be restored.

    Worse than ``RegisteredArtifactMutated``: the live trees must not be treated as
    clean. The integrity record is still written when the caller can append it.
    """


def is_registered_artifact_mutation(record: Mapping[str, object]) -> bool:
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
    return not is_registered_artifact_mutation(record)


def assert_no_registered_artifact_mutation(records: list[dict[str, object]]) -> None:
    """Refuse further pipeline work while an integrity-failure row remains."""
    for record in records:
        if is_registered_artifact_mutation(record):
            raise RegisteredArtifactMutated(
                "strategy or model artifacts changed during the "
                f"{record.get('record_type')} replay; refuse retry and resume "
                "until the frozen trees are rolled back"
            )


def recorded_path(experiment_dir: str | Path, reference: str | Path) -> Path:
    """Where a path a record of ``experiment_dir`` names lives now.

    Records name what the pipeline wrote by the host path it was written at,
    and an experiment directory keeps its records when it moves (a renamed
    checkout, an experiment synced from another host). What is fixed is the path
    below the experiment's own directory, so a recorded absolute path is read
    as its tail after the last ``<experiments root>/<experiment id>`` pair,
    named as ``experiment_dir`` is now, under ``experiment_dir``; a relative
    reference is that tail already. Raises ``FileNotFoundError`` when the
    reference names no place inside the experiment, or nothing is there.
    """

    directory = Path(experiment_dir).resolve()
    outside = f"{directory.name}: recorded path outside the experiment: {reference}"
    tail = Path(reference)
    if tail.is_absolute():
        parts = tail.parts
        anchor = (directory.parent.name, directory.name)
        starts = [i for i in range(1, len(parts) - 1) if parts[i : i + 2] == anchor]
        if not starts:
            raise FileNotFoundError(outside)
        tail = Path(*parts[starts[-1] + 2 :])
    path = (directory / tail).resolve()
    if not path.is_relative_to(directory):
        raise FileNotFoundError(outside)
    if not path.exists():
        raise FileNotFoundError(
            f"{directory.name}: {tail.as_posix()} is recorded but not on disk"
        )
    return path


def research_records(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """The research sessions' records, in the order they ran."""

    return [
        dict(record)
        for record in records
        if record.get("record_type") == "research_session"
    ]


def registered_record(records: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    """The research session record that registered the experiment's artifact, or None.

    An experiment registers at most once; a ledger holding two registrations is corrupt and
    raises rather than letting a reader pick one.
    """

    rows = [record for record in research_records(records) if record.get("registered")]
    if len(rows) > 1:
        raise ValueError("the ledger holds more than one registration for one experiment")
    return rows[0] if rows else None


def forward_record(records: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    """The experiment's forward verdict record, or None; an integrity row is not one."""

    rows = [
        dict(record)
        for record in records
        if is_durable_success_record(record, record_types=("forward",))
    ]
    if len(rows) > 1:
        raise ValueError("the ledger holds more than one forward verdict for one experiment")
    return rows[0] if rows else None


def research_over(records: Sequence[Mapping[str, object]]) -> bool:
    """Whether research has ended: an artifact registered, or a session ended the experiment."""

    return registered_record(records) is not None or any(
        record.get("experiment_end") for record in research_records(records)
    )


def experiment_verdict(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """The experiment's terminal status: ``registered_with_record`` from its forward
    record's readout (nothing gates; the record and the reading are the
    status), the ``experiment_end`` status (:data:`EXPERIMENT_END_STATUSES`) when research
    ended without a registration, else None. The single source for the
    terminal status, the console and Paper.
    """

    forward = forward_record(records)
    if forward is not None:
        readout = forward.get("readout")
        if not isinstance(readout, Mapping):
            raise ValueError("forward record carries no readout")
        return {"status": str(readout["state"]), "reasons": []}
    if registered_record(records) is not None:
        return None
    ended = next(
        (record for record in research_records(records) if record.get("experiment_end")),
        None,
    )
    if ended is None:
        return None
    experiment_end = ended["experiment_end"]
    status = experiment_end.get("status") if isinstance(experiment_end, Mapping) else None
    if not isinstance(experiment_end, Mapping) or status not in EXPERIMENT_END_STATUSES:
        raise ValueError(f"research session record ends the experiment with an unknown status {status!r}")
    return {"status": status, "reasons": [str(experiment_end.get("reason") or "")]}


def paper_candidate(records: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    """The artifact a Paper account trades: the registered strategy of an experiment with its
    forward record (``registered_with_record``) whose forward replay completed (a
    book continues that replay's history); else None."""

    forward = forward_record(records)
    if forward is not None and forward.get("status") != "ok":
        return None
    verdict = experiment_verdict(records)
    if verdict is None or verdict["status"] != VerdictStatus.REGISTERED_WITH_RECORD:
        return None
    row = registered_record(records)
    if row is None:
        raise ValueError("an experiment with a forward record has no registered record")
    block = row["registered"]
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
            and record.get("registered")
            and registered_record(self.read()) is not None
        ):
            raise ValueError("the experiment already registered an artifact; a second registration is refused")
        if (
            record_type == "forward"
            and not is_registered_artifact_mutation(record)
            and forward_record(self.read()) is not None
        ):
            raise ValueError("the experiment already has its forward verdict")
        if record_type == TEST_STEP_RECORD_TYPE:
            step = record.get("step")
            if any(
                row.get("record_type") == TEST_STEP_RECORD_TYPE and row.get("step") == step
                for row in self.read()
            ):
                raise ValueError(f"the experiment already recorded test step {step!r}; a step is replayed once")
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
