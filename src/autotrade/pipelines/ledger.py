"""The single experiment ledger (docs/pipeline-design.md §4.1).

One JSONL file per experiment. The pipeline writes three record types:

- ``research_session``: one research session's outcome and Steps; the one
  that passed the freeze gate also carries the arm's ``frozen`` block, and the
  one that ended research without a freeze carries ``arm_end``;
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
``fold_id`` the session (``s1``, ``s2``, ... or ``forward``).

A ``forward`` row with ``state_changed_during_test=true`` is an integrity
failure, not a verdict: it is persisted before fail-fast so the corruption is
auditable, then every resume and retry must refuse until a human rolls the
dirty frozen trees back.

The console and the experiment reports still read the Fold-era record types
(``FOLD_ERA_RECORD_TYPES``) through the readers at the end of this module; the
pipeline never writes them and the worker refuses a ledger that holds one.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from autotrade.environment.replay.style import (
    STYLE_ARTIFACT_NAME,
    window_neutralized_excess,
)
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
# The forward replay's session key and ``fold_id``.
FORWARD_SESSION_KEY = "forward"
PIPELINE_RECORD_TYPES = ("research_session", "forward", "attempt_failed")
FOLD_ERA_RECORD_TYPES = (
    "fold",
    "meta_learning",
    "heldout",
    "deployment_adjustment",
    "terminated",
)
RECORD_TYPES = (*PIPELINE_RECORD_TYPES, *FOLD_ERA_RECORD_TYPES)
LINK_KEYS = ("experiment_id", "epoch_id", "fold_id", "run_id")
# The ``status`` of a forward record whose replay stopped at the strategy's own
# exception: the one replay failure that measures the strategy. Every other
# failure fails the attempt and never reaches a business record.
STRATEGY_ERROR = "strategy_error"
DURABLE_SUCCESS_TYPES = (
    "research_session",
    "forward",
    "fold",
    "meta_learning",
    "heldout",
    "deployment_adjustment",
)
_INTEGRITY_RECORD_TYPES = frozenset({"forward", "fold", "heldout"})

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
        if record_type not in RECORD_TYPES:
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

    Fold, Meta and Held-out runs append ``attempt_failed`` themselves when they
    catch an exception, but a SIGKILL, an OOM kill or a host reset leaves no
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


def finite_number(value: object) -> float | None:
    """``value`` as a finite float, or None when it is not a real number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


# --- Fold-era readers ---------------------------------------------------------
#
# The console and the experiment reports still read archived Fold-era ledgers
# through these; nothing in the pipeline calls them.


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


def _artifact_id(value: object) -> str | None:
    """A non-empty artifact id, or None when the record names none."""

    return value if isinstance(value, str) and value else None


def _epoch_folds(
    fold_records: list[dict[str, object]], *, epoch_id: str
) -> list[dict[str, object]]:
    """One Epoch's latest Fold records, in schedule order."""

    return sorted(
        (
            record
            for record in latest_fold_records(fold_records).values()
            if str(record.get("epoch_id")) == epoch_id
        ),
        key=lambda record: str(record.get("validation_period") or ""),
    )


@dataclass(frozen=True)
class Transition:
    """One walk-forward transition: whose it is, its result, and where it lives.

    ``result_ref`` points at the replay record the result was projected from,
    which is where the neutralized excess of a span is recomputed when the
    record itself does not carry one (:func:`transition_neutralized_excess`).
    ``failure`` is empty unless the replay failed: then it is the row's
    ``failure`` tag (``STRATEGY_ERROR``), or ``"unclassified"`` for a failed
    row written before failures were classified.
    """

    artifact_id: str | None
    result: object
    result_ref: str
    failure: str = ""


def baseline_anchor_artifacts(fold_records: list[dict[str, object]]) -> frozenset[str]:
    """Every artifact frozen as a baseline anchor (docs/pipeline-design.md §2.2).

    An anchor is the lineage's control -- the weak baseline a parentless Fold
    must freeze so the next Fold has something to beat, or a repair or re-tune
    of it that the nominating Fold declared a control -- not a candidate
    anyone judged worth shipping. The Fold record's ``baseline_anchor`` label
    is the single handle on that: the transitions it replayed score a control
    and are excluded from graduation, and the experiment refuses to deliver
    one to Held-out.
    """

    return frozenset(
        str(record["frozen_strategy_artifact_id"])
        for record in latest_fold_records(fold_records).values()
        if record.get("baseline_anchor") is True
        and _artifact_id(record.get("frozen_strategy_artifact_id"))
    )


def _transition_rows(
    folds: list[dict[str, object]], *, test_stage: bool
) -> list[Transition]:
    """The transitions of one Epoch, in schedule order.

    Without a Test stage a transition is the host's ``parent_control`` of every
    Fold after the Epoch's first, and the artifact it scores is the parent that
    control replayed. With a Test stage it is each Fold's frozen Test, which
    scores that Fold's own frozen artifact. The single source for both the
    chain-wide count (:func:`walk_forward_transitions`) and the shipped
    artifact's own count (:func:`final_artifact_transitions`), so the two can
    never disagree about which artifact a transition belongs to. The id is
    ``None`` when the record names none, which matches no artifact.

    Every transition the schedule produced, anchors included: the callers drop
    the ones that replayed an anchor (:func:`_counted`) but still need the full
    row count to tell "this schedule confirmed nothing" from "this schedule
    could confirm nothing".
    """

    if test_stage:
        return [
            Transition(
                _artifact_id(record.get("frozen_strategy_artifact_id")),
                record.get("test_result"),
                str(record.get("test_result_ref") or ""),
                _failure(record.get("test_result")),
            )
            for record in folds
        ]
    return [
        _parent_control_transition(record.get("parent_control")) for record in folds[1:]
    ]


def _counted(rows: list[Transition], anchors: frozenset[str]) -> list[Transition]:
    """The transitions that score something: an anchor's do not.

    A baseline anchor is a control the experiment never delivers, so its
    forward record leaves the numerator and the denominator alike -- counting
    it either way would let a placebo's record stand in for a candidate's.
    """

    return [row for row in rows if row.artifact_id not in anchors]


def _parent_control_transition(control: object) -> Transition:
    """One host parent control as the transition it is."""

    control = control if isinstance(control, Mapping) else {}
    return Transition(
        _artifact_id(control.get("parent_strategy_artifact_id")),
        transition_result(control),
        str(control.get("validation_result_ref") or ""),
        _failure(control),
    )


def _failure(block: object) -> str:
    """How a parent control or frozen Test failed, when it did."""

    if not isinstance(block, Mapping) or block.get("status") != "failed":
        return ""
    return str(block.get("failure") or "unclassified")


def parent_control_excess(control: object) -> float | None:
    """The neutralized excess one parent control is graded on, for readers that
    hold a single Fold record rather than an Epoch's rows (the console row, the
    report). Same function, same number as the count above it."""

    return transition_neutralized_excess(_parent_control_transition(control))


def epoch_transitions(
    fold_records: list[dict[str, object]], *, epoch_id: str, test_stage: bool
) -> list[Transition]:
    """One Epoch's transitions as the counts were taken on them.

    The public seam over :func:`_transition_rows` for readers that need the
    transitions themselves rather than the totals -- the console averages the
    same figures the count is taken on, so a tile and the number above it
    cannot describe different sets.
    """

    return _counted(
        _transition_rows(
            _epoch_folds(fold_records, epoch_id=epoch_id), test_stage=test_stage
        ),
        baseline_anchor_artifacts(fold_records),
    )


def walk_forward_transitions(
    fold_records: list[dict[str, object]], *, epoch_id: str, test_stage: bool
) -> dict[str, object]:
    """Out-of-sample transitions of one Epoch for graduation term (b).

    Without a Test stage a transition is the host's ``parent_control`` of every
    Fold after the Epoch's first: the previous Fold's frozen strategy replayed
    on this Fold's Validation window, scored on its new period alone when the
    window trails over several (``transition_result``). With a Test stage it is
    each Fold's frozen Test. A transition counts as positive only when its
    size/beta-neutralized excess is > 0 (``transition_neutralized_excess``);
    how the rest are counted is :func:`_counts`.

    This is the development *chain's* record: the transitions it counts mostly
    replay earlier artifacts of the lineage, not the one Held-out ships. What
    that shipped artifact proved forward on its own is
    :func:`final_artifact_transitions`. Transitions that replayed a baseline
    anchor are not part of either: an anchor is a control the experiment never
    delivers, so its forward record says nothing about a graduating strategy.
    """
    folds = _epoch_folds(fold_records, epoch_id=epoch_id)
    scheduled = _transition_rows(folds, test_stage=test_stage)
    rows = _counted(scheduled, baseline_anchor_artifacts(fold_records))
    if test_stage:
        source = "frozen_test"
        percentiles: list[float] = []
    else:
        source = "parent_control"
        # The null percentile of each counted transition on the span it is
        # scored on; a control that ran no null control contributes nothing.
        percentiles = [
            value
            for value in (
                finite_number(
                    (transition_null_control(record.get("parent_control")) or {}).get(
                        "excess_percentile"
                    )
                )
                for record in folds[1:]
            )
            if value is not None
        ]
    return {
        "source": source,
        "epoch_id": epoch_id,
        # What the schedule produced, before the anchors came out: a term that
        # counts nothing because every transition replayed a control is not the
        # same as a schedule with no transitions to count.
        "scheduled": len(scheduled),
        **_counts(rows),
        # Diagnostic beside the count: where the transitions sat inside
        # random-name replays of their own trade skeletons, on average. Never
        # part of the term; None when no transition carried a null control.
        "mean_excess_percentile": (
            sum(percentiles) / len(percentiles) if percentiles else None
        ),
    }


def final_artifact_transitions(
    fold_records: list[dict[str, object]],
    *,
    epoch_id: str,
    test_stage: bool,
    artifact_id: str,
) -> dict[str, object]:
    """The transitions of one Epoch that scored ``artifact_id`` itself.

    Graduation's walk-forward term counts the whole chain, and a mechanism
    first frozen in the Epoch's last Fold inherits none of that record: every
    transition replayed the parent it replaced. This is the subset that
    actually replayed the artifact Held-out ships, counted the same way
    (:func:`_transition_rows`), so an artifact with no forward quarter of its
    own is visible as ``transitions == 0`` rather than hidden behind the
    chain's average.
    """

    rows = _counted(
        _transition_rows(
            _epoch_folds(fold_records, epoch_id=epoch_id), test_stage=test_stage
        ),
        baseline_anchor_artifacts(fold_records),
    )
    own = [row for row in rows if row.artifact_id == artifact_id]
    return {
        "artifact_id": artifact_id,
        "epoch_id": epoch_id,
        **_counts(own),
    }


def frozen_selection(
    fold_records: list[dict[str, object]], *, artifact_id: str
) -> dict[str, object] | None:
    """Selection evidence of the Fold that froze ``artifact_id``.

    The strategy a Held-out replays was nominated out of one Fold's search,
    and that Fold's record is the only place the search's deflated-Sharpe
    probability (§2.4) and the frozen node's own null percentile live. Both
    are diagnostics; the verdict carries them beside its gating metrics so a
    graduation is read together with how much of it selection alone explains.
    ``None`` when no Fold record froze the artifact (a Meta-regularized
    artifact that never went through a Fold), and each field is ``None`` when
    the record predates its block.
    """

    matches = [
        record
        for record in latest_fold_records(fold_records).values()
        if record.get("frozen_strategy_artifact_id") == artifact_id
    ]
    if not matches:
        return None
    record = matches[-1]
    selection = record.get("selection_statistics")
    selection = selection if isinstance(selection, Mapping) else {}
    null = record.get("null_control")
    null = null if isinstance(null, Mapping) else {}
    return {
        "fold_id": record.get("fold_id"),
        "candidates_evaluated": _count(selection.get("candidates_evaluated")),
        "deflated_sharpe_probability": finite_number(selection.get("deflated_sharpe_probability")),
        # N as the formula actually used it: the finite trial Sharpes, which is
        # not always ``candidates_evaluated`` (a kept parent joins the trials,
        # a non-finite Sharpe drops out). A probability from two trials barely
        # deflates anything, so the count has to be read beside it.
        "deflated_sharpe_trials": _count(selection.get("trials")),
        "validation_excess_percentile": finite_number(null.get("excess_percentile")),
    }


def transition_result(control: object) -> Mapping[str, object] | None:
    """The result one parent control is graded on: its new period, else the window.

    A Fold whose Validation window trails over several periods has already seen
    all but its last one, so its ``parent_control`` records ``step_result`` --
    that last period alone -- and the transition is scored on it. A single-period
    window (and every ledger written before the rolling schedule) has none, and
    the whole ``validation_result`` is the transition. The whole window is never
    a fallback for a missing step: a trailing window's control without its own
    quarter is refused before it is recorded (``experiment._step_result``).
    The single source for both the graduation term and the report, so the two
    can never disagree.
    """

    if not isinstance(control, Mapping):
        return None
    step = control.get("step_result")
    if isinstance(step, Mapping):
        return step
    result = control.get("validation_result")
    return result if isinstance(result, Mapping) else None


def transition_null_control(control: object) -> Mapping[str, object] | None:
    """The null-control block of the span ``transition_result`` scores.

    A control graded on its ``step_result`` is ranked against the ``step``
    sub-block of its null control, never the whole window's; otherwise the
    window's block is the one. None when no null control ran.
    """

    if not isinstance(control, Mapping):
        return None
    null = control.get("null_control")
    if not isinstance(null, Mapping):
        return None
    if isinstance(control.get("step_result"), Mapping):
        step = null.get("step")
        return step if isinstance(step, Mapping) else None
    return null


def transition_neutralized_excess(transition: Transition) -> float | None:
    """The size/beta-neutralized excess one transition is graded on.

    The single source for every reader of a transition's sign: the graduation
    terms, the report and the console all call it, so none of them can grade on
    a different number. CSI 300 fell 21.6% in 2022 and rose 16-18% in the 2024
    and 2025 quarters, so a raw excess over it says as much about which quarter
    a transition landed in as about the strategy; the neutralized figure is the
    one that survives that.

    Read in this order, and never past it: the figure the record carries; else
    the same figure recomputed on the graded span from the replay's own style
    sidecar (records written before results carried a per-quarter figure);
    else ``None``. ``None`` means the transition's sign is unknown, which the
    counts report as ``unmeasured`` and the verdict fails on -- falling back to
    the raw excess would silently grade graduation on the number this rule
    exists to stop using.
    """

    result = transition.result
    if not isinstance(result, Mapping) or result.get("status") == "failed":
        return None
    benchmark = result.get("benchmark")
    recorded = finite_number(
        benchmark.get("neutralized_excess_return") if isinstance(benchmark, Mapping) else None
    )
    if recorded is not None:
        return recorded
    return _derived_excess(
        transition.result_ref,
        str(result.get("start") or ""),
        str(result.get("end") or ""),
    )


@lru_cache(maxsize=512)
def _derived_excess(result_ref: str, start: str, end: str) -> float | None:
    """The neutralized excess of one span, recomputed from the style sidecar.

    Cached because the console re-reads the same transitions on every list and
    detail request and a replay result never changes once written (a re-run
    writes a new result directory): without it one experiment's twelve
    transitions cost ~40 ms of sidecar parsing and regression per pass.
    """

    path = Path(result_ref or "")
    if not result_ref:
        return None
    if path.name == "result.json":
        path = path.parent
    try:
        analysis = json.loads((path / STYLE_ARTIFACT_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(analysis, Mapping):
        return None
    return window_neutralized_excess(analysis, start=start, end=end)


def _counts(rows: Sequence[Transition]) -> dict[str, int]:
    """How many transitions were counted, positive, failed, and unmeasured.

    Every row is a transition. Positive means a neutralized excess > 0.
    ``failed`` rows are replays the strategy's own code crashed: a measurement
    (the strategy cannot run forward there), so they stay completed,
    non-positive transitions. ``unmeasured`` rows have no sign at all -- a
    result whose neutralized excess could not be established, or a failed
    replay the ledger never classified (rows written before failures were
    typed, which may have been a timeout) -- and fail the verdict explicitly
    instead of being graded as negatives. A Fold with no parent to replay
    (after a ``baseline_missing`` one) proved nothing and is not positive.
    """

    positive = failed = unmeasured = 0
    for row in rows:
        if row.failure == STRATEGY_ERROR:
            failed += 1
        elif row.failure:
            unmeasured += 1
        else:
            value = transition_neutralized_excess(row)
            if value is None:
                unmeasured += isinstance(row.result, Mapping)
            elif value > 0:
                positive += 1
    return {
        "transitions": len(rows),
        "positive_excess": positive,
        "failed": failed,
        "unmeasured": unmeasured,
    }


def _count(value: object) -> int | None:
    """``value`` as a count, or None when the record does not carry one."""

    return value if isinstance(value, int) and not isinstance(value, bool) else None


def terminated_record(records: list[dict[str, object]]) -> dict[str, object] | None:
    """The ``terminated`` row of an arm that ended itself, or None.

    Single source for "this experiment is over before Held-out": the runner
    stops scheduling sessions, the worker skips Held-out and the verdict reads
    ``terminated`` off the same row.
    """
    return next(
        (record for record in records if record.get("record_type") == "terminated"),
        None,
    )


def latest_deployment_record(
    records: list[dict[str, object]],
) -> dict[str, object] | None:
    """The latest ``deployment_adjustment`` row (append-only, latest wins)."""
    latest = None
    for record in records:
        if is_durable_success_record(record, record_types=("deployment_adjustment",)):
            latest = record
    return latest
