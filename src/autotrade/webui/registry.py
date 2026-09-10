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
from typing import NamedTuple

from autotrade.environment.replay.style import STYLE_ARTIFACT_NAME, STYLE_SCHEMA_VERSION
from autotrade.pipelines.agent_inbox import INBOX_NAME, inbox_public_view
from autotrade.pipelines.config import AcceptanceRules
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
    experiment_verdict,
    is_durable_success_record,
    is_frozen_artifact_mutation,
    latest_fold_records,
    latest_deployment_record,
    latest_heldout_records,
    paper_candidate,
    transition_null_control,
    transition_result,
    walk_forward_transitions,
)
from autotrade.pipelines.worker import _ALLOWED_PARAMS
from autotrade.pipelines.meta_schedule import meta_record_session_key
from autotrade.pipelines.skills import latest_skills_snapshot

from .public_identity import PublicIdentity

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
    "initializing",
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
# ``parent_control`` is deliberately not one of them: the host's parent control
# is a Validation over this Fold's own development window, so it stays visible
# before the reveal exactly like the Fold's own Validation result.
TEST_FIELDS = ("test_result",)
# Pointers that resolve to the same evidence on disk. They are stripped from
# every projection and never surface in the audit block either — the reveal
# gate would be pointless if the console handed out the path to the artifact.
_TEST_EVIDENCE_REFS = ("test_result_ref", "snapshot_ids")
# The sealed calendar itself. A fold record names the Test window it will be
# scored on; publishing it before the reveal would hand out the held-out dates
# the params view and the session list already seal.
_SEALED_FOLD_PERIODS = ("test_period", "test_decision_time")
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
# Held-out is the one calendar the console must not publish before the reveal.
# The development window is public research scope: the session list already
# labels every Fold with its own period (one regular Fold per period of the
# window, ``fold_2022``..``fold_2025`` by default), so sealing the same dates
# here sealed nothing.
_SEALED_PERIOD_PARAMS = {
    "heldout_first_period",
    "heldout_last_period",
}


class UnsupportedParamsError(ValueError):
    """params.json names parameters the worker no longer accepts (a legacy
    experiment). The listing flags it unreadable instead of crashing."""


def _require_supported_params(params: Mapping[str, object]) -> None:
    unknown = sorted(
        key for key in params if not str(key).startswith("_") and key not in _ALLOWED_PARAMS
    )
    if unknown:
        raise UnsupportedParamsError(f"unsupported experiment parameters: {unknown}")


def read_ledger_records(experiment_dir: Path) -> list[dict[str, object]]:
    """The single validating reader shared with the pipeline: a ledger the
    pipeline would reject (unparseable line, wrong schema_version) must not
    drive console decisions either — inheritance, rollback, rerun and reveal
    all key off these records. Failures surface through the experiment's
    ``unreadable`` state rather than a silently skimmed partial view."""
    return ExperimentLedger(Path(experiment_dir) / "ledgers" / "experiment_ledger.jsonl").read()


# Worker stdout/stderr, repo-relative and inside the ignored logs/ tree so a
# crashed session stays diagnosable without ever entering the repository.
WORKER_LOG_DIR = "logs/workers"


def worker_log_ref(experiment_id: str) -> str:
    """Repo-relative worker log location published to status.json and the API."""
    return f"{WORKER_LOG_DIR}/{experiment_id}.log"


def worker_log_for(experiment_dir: Path, repo_root: Path | None = None) -> str:
    """The worker log for this experiment, when it exists on disk.

    The manager writes ``worker_log`` into the transient ``launching`` status
    and the worker's own first status write replaces it, so a running or
    crashed experiment carried no log reference at all. Deriving it from the
    experiment id keeps every state diagnosable; the file has to exist, so a
    never-launched experiment still advertises nothing.
    """

    directory = Path(experiment_dir)
    root = Path(repo_root) if repo_root is not None else directory.parents[1]
    ref = worker_log_ref(directory.name)
    return ref if (root / ref).is_file() else ""


def experiment_state(
    experiment_dir: Path, *, repo_root: Path | None = None
) -> dict[str, object]:
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
    log_ref = worker_log_for(experiment_dir, repo_root)
    if not status:
        result: dict[str, object] = {
            "kind": "hitl",
            "state": "created",
            "worker_alive": False,
            "status": {},
        }
        if log_ref:
            result["worker_log"] = log_ref
        return result
    alive = status_pid_alive(status)
    state = str(status.get("state") or "unknown")
    if state == "launching" and not alive:
        if _age_seconds(status.get("launched_at")) > LAUNCH_GRACE_SECONDS:
            state = "interrupted"
    elif state in ACTIVE_STATES and not alive:
        state = "interrupted"
    result = {"kind": "hitl", "state": state, "worker_alive": alive, "status": status}
    if log_ref:
        result["worker_log"] = log_ref
    return result


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


def _excess_return(result: object) -> float | None:
    """One replay result's return over its benchmark, ``None`` without both.

    The same subtraction ``ledger._excess_positive`` grades a transition on, so
    an excess the console prints and the count the graduation term reads can
    never come from different arithmetic.
    """
    if not isinstance(result, Mapping):
        return None
    benchmark = result.get("benchmark")
    total = _number(result.get("total_return"))
    bench = (
        _number(benchmark.get("benchmark_return")) if isinstance(benchmark, Mapping) else None
    )
    return total - bench if total is not None and bench is not None else None


def _parent_control_view(record: Mapping[str, object]) -> dict[str, object] | None:
    """Console metrics of one Fold's host parent control.

    The parent control replays the inherited parent unchanged on this Fold's
    Validation window, so it is the Fold's baseline and the previous Fold's
    walk-forward evidence. The numbers are the ones that transition is graded
    on (``ledger.transition_result``): the Fold's new period alone when the
    window trails over several, the whole window otherwise — so this row and
    the transition counts above it can never tell different stories. That span
    travels with them (``source`` and its bounds, the same three fields
    ``reporting.walk_forward_report`` publishes), because a trailing window
    scores the control on less ground than the Fold's own Validation row covers
    and the console must not present the two as one span. ``None`` for a Fold
    that inherited no parent; a failed control keeps its status and carries no
    numbers.
    """
    control = record.get("parent_control")
    if not isinstance(control, Mapping):
        return None
    result = transition_result(control) or {}
    # The branch transition_result took: the step row carries its own bounds,
    # the whole window is the fold record's validation period.
    stepped = isinstance(control.get("step_result"), Mapping)
    window_start, _, window_end = str(record.get("validation_period") or "").partition("..")
    scored_start = result.get("start") if stepped else window_start
    scored_end = result.get("end") if stepped else window_end
    return {
        "status": control.get("status"),
        "source": "step_result" if stepped else "validation_result",
        "period_start": str(scored_start) if scored_start else None,
        "period_end": str(scored_end) if scored_end else None,
        "return": _number(result.get("total_return")),
        "excess_return": _excess_return(result),
        "sharpe": _number(result.get("sharpe")),
        "max_drawdown": _number(result.get("max_drawdown")),
        # Where that excess sits inside random-name replays of the control's own
        # trade skeleton, on the same span. None when no null control ran.
        "excess_percentile": _number(
            (transition_null_control(control) or {}).get("excess_percentile")
        ),
    }


def _selection_view(record: Mapping[str, object]) -> dict[str, object] | None:
    """Console projection of one Fold's selection statistics.

    How many candidates the Fold replayed on its Validation window, and the
    deflated-Sharpe probability of the candidate it froze
    (``pipelines/ledger.deflated_sharpe``). Validation-only development
    evidence, never sealed. ``None`` for a Fold record written before the
    block existed.
    """
    block = record.get("selection_statistics")
    if not isinstance(block, Mapping):
        return None
    return {
        "candidates_evaluated": _count(block.get("candidates_evaluated")),
        "trials": _count(block.get("trials")),
        "deflated_sharpe_probability": _number(
            block.get("deflated_sharpe_probability")
        ),
        "sharpe_star": _number(block.get("sharpe_star")),
        "unavailable_reason": block.get("unavailable_reason"),
    }


def _count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _walk_forward_view(
    records: list[dict[str, object]],
    epoch_id: str,
    epoch_folds: list[dict[str, object]],
    *,
    test_stage: bool,
    revealed: bool,
) -> dict[str, object] | None:
    """Graduation term (b) for one Epoch (``ledger.walk_forward_transitions``).

    Without a Test stage the transitions are the host's parent controls, which
    are development evidence and readable while the experiment runs. With a
    Test stage they are the frozen Test results, so the counts stay sealed
    until the reveal like every other Test number. ``required`` is the same
    two-thirds bar the acceptance rules apply, served so the console states the
    threshold without restating the rule (``None`` without transitions).

    Two diagnostics ride beside the count, because a bare 5/7 says nothing
    about how much ground those five won: ``mean_excess`` is the average excess
    the counted transitions carried, and ``mean_excess_percentile`` is where
    those excesses sat inside random-name replays of their own trade skeletons
    (the ledger's own figure, the one the verdict diagnostics publish). Neither
    gates anything. The excesses are read off exactly the transitions the
    ledger counted — each Fold's frozen Test with a Test stage, otherwise every
    Fold after the Epoch's first (``ledger._transition_rows``) — so the mean
    can never average a wider set than the count it sits next to.
    """
    if test_stage and not revealed:
        return None
    counts = walk_forward_transitions(records, epoch_id=epoch_id, test_stage=test_stage)
    scored = (
        [record.get("test_result") for record in epoch_folds]
        if test_stage
        else [transition_result(record.get("parent_control")) for record in epoch_folds[1:]]
    )
    return {
        "source": counts["source"],
        "transitions": counts["transitions"],
        "positive_excess": counts["positive_excess"],
        "required": AcceptanceRules.walk_forward_consistency(counts).get("required"),
        "mean_excess": _mean(
            [value for value in map(_excess_return, scored) if value is not None]
        ),
        "mean_excess_percentile": _number(counts["mean_excess_percentile"]),
    }


def _public_diagnostics(
    block: object, identity: PublicIdentity
) -> dict[str, object] | None:
    """Verdict diagnostics with the raw Fold token swapped for its public ref.

    ``ledger.frozen_selection`` names the Fold that froze the shipped artifact
    by its host id (``fold_2025Q4``), which is exactly the token the console
    never returns: Fold identity leaves here as ``fold_ref_`` and is shown to
    people as the period label the session list already carries.
    """
    if not isinstance(block, Mapping):
        return None
    fold_id = block.get("frozen_fold_id")
    return {
        **{key: value for key, value in block.items() if key != "frozen_fold_id"},
        "frozen_fold_ref": identity.fold_ref(fold_id) if fold_id else None,
    }


def _public_verdict(
    records: list[dict[str, object]], identity: PublicIdentity
) -> dict[str, object] | None:
    """Graduation verdict with term (b) and its diagnostics surfaced beside it.

    The pipeline computes the walk-forward term and the verdict diagnostics
    once over the final Epoch and stamps the same blocks into every Held-out
    period's verdict, so the console publishes them once next to the verdict
    instead of only inside the periods. The diagnostics carry the shipped
    artifact's own forward record (``ledger.final_artifact_transitions``),
    which is the only thing that says whether term (b)'s chain-wide count is
    about this strategy or about the ones it replaced.
    """
    verdict = experiment_verdict(records, strict=False)
    if verdict is None:
        return None
    periods = [
        {**period, "diagnostics": _public_diagnostics(period["diagnostics"], identity)}
        if isinstance(period, Mapping) and isinstance(period.get("diagnostics"), Mapping)
        else period
        for period in verdict.get("periods") or ()
    ]
    stamped = {
        key: next(
            (
                dict(period[key])
                for period in periods
                if isinstance(period, Mapping) and isinstance(period.get(key), Mapping)
            ),
            None,
        )
        for key in ("walk_forward", "diagnostics")
    }
    return {**verdict, "periods": periods, **stamped}


def _paper_candidate_view(
    directory: Path, records: list[dict[str, object]]
) -> dict[str, object] | None:
    """The Paper candidate with the launch command that pins it.

    Paths are written relative to the repository root the console serves
    (``experiments/<id>/...``), which is where ``run_paper.py`` is run from;
    an experiment tree outside it keeps absolute paths.
    """
    candidate = paper_candidate(records)
    if candidate is None:
        return None
    repo_root = directory.parent.parent

    def relative(value: object) -> str | None:
        if not value:
            return None
        path = Path(str(value))
        return str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path)

    strategy = relative(candidate["output_path"])
    models = relative(candidate["models_path"])
    models_dir = models if models and Path(str(candidate["models_path"])).is_dir() else None
    command = [
        "python scripts/paper/run_paper.py",
        f"--strategy {strategy}/main.py" if strategy else "--strategy <missing>",
        f"--strategy-revision {candidate['artifact_id']}",
        *([f"--models-dir {models_dir}"] if models_dir else []),
    ]
    return {
        "artifact_id": candidate["artifact_id"],
        "source": candidate["source"],
        "graduated_artifact_id": candidate["graduated_artifact_id"],
        "strategy_path": f"{strategy}/main.py" if strategy else None,
        "models_dir": models_dir,
        "command": " ".join(command),
    }


def walk_forward_folds(folds: list[dict[str, object]]) -> list[dict[str, object]]:
    """Folds in walk-forward order: by Validation window, then by Fold id.

    The one ordering everything that walks a Fold calendar uses — the read
    model's per-Fold rows, the chained equity series and
    ``ledger.walk_forward_transitions`` — so a curve, the tile above it and the
    transition table can never pair the same Folds differently.
    """
    return sorted(
        folds,
        key=lambda row: (
            str(row.get("validation_period") or ""),
            str(row.get("fold_id") or ""),
        ),
    )


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
    """Every planned held-out period has a durable success record.

    Integrity-flagged held-out rows are never completed work and never auto-reveal
    or seal the experiment. A later or earlier success for the same period cannot
    wash a remaining ``state_changed_during_test`` row.
    """
    schedule = read_json(Path(experiment_dir) / HITL_DIR_NAME / SCHEDULE_NAME)
    sessions = schedule.get("sessions") if isinstance(schedule.get("sessions"), list) else []
    planned = _planned_heldout_labels(sessions)
    if not planned:
        return False
    if records is None:
        records = read_ledger_records(experiment_dir)
    if any(
        is_frozen_artifact_mutation(record)
        for record in records
        if record.get("record_type") == "heldout"
    ):
        return False
    return planned <= _completed_heldout_labels(records)


def _planned_heldout_labels(sessions: list[object]) -> set[str]:
    return {
        str(period.get("label"))
        for session in sessions
        if isinstance(session, Mapping) and str(session.get("kind")) == "heldout"
        for period in (session.get("periods") if isinstance(session.get("periods"), list) else [])
        if isinstance(period, Mapping) and period.get("label")
    }


def _heldout_period_label(record: Mapping[str, object]) -> str:
    fold_id = str(record.get("fold_id") or "")
    if fold_id.startswith("heldout_"):
        return fold_id.removeprefix("heldout_")
    return str(record.get("period") or "")


def _completed_heldout_labels(records: list[dict[str, object]]) -> set[str]:
    tainted = {
        _heldout_period_label(record)
        for record in records
        if is_frozen_artifact_mutation(record)
    }
    tainted.discard("")
    labels: set[str] = set()
    for record in latest_heldout_records(records):
        label = _heldout_period_label(record)
        if label and label not in tainted:
            labels.add(label)
    return labels


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
        if is_durable_success_record(
            record, record_types=("fold", "meta_learning", "deployment_adjustment")
        )
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


# The evaluation backends name every result directory ``f"{mode}_{uuid4().hex}"``
# (pipelines/pit_backend.py, pipelines/local_backend.py). The console speaks a
# shorter public vocabulary — valid | test | heldout — so this mapping is the
# single place that knows how a public prefix is spelled on disk.
RESULT_MODES: dict[str, str] = {
    "valid": "valid",
    "test": "frozen_test",
    "heldout": "heldout",
}
# Public prefixes whose results carry test-period evidence.
SEALED_PREFIXES: tuple[str, ...] = ("test", "heldout")


def result_dir_prefixes(prefixes: tuple[str, ...]) -> tuple[str, ...]:
    """Result-directory name prefixes for the given public prefixes."""
    return tuple(f"{RESULT_MODES[prefix]}_" for prefix in prefixes)


def sealed_prefixes(experiment_dir: Path) -> tuple[str, ...]:
    """Central reveal gate for artifact routes (style, orders, CSV, result-name
    enumeration): results under these public prefixes carry test-period
    evidence and must stay invisible pre-reveal — respond 404 and filter
    listings, never confirm existence. Empty once revealed."""
    return () if test_results_revealed(experiment_dir) else SEALED_PREFIXES


def guarded_fold_view(
    record: Mapping[str, object], *, test_revealed: bool
) -> dict[str, object]:
    """Fold record minus test-period evidence (shown separately, labelled).

    Result payloads and their on-disk pointers are always stripped — the
    console shows them through the labelled ``test_audit`` block instead. The
    Test window and its decision time are stripped only until the reveal, the
    same boundary ``_public_params`` and ``public_session`` apply to the very
    same dates.
    """
    hidden = (*TEST_FIELDS, *_TEST_EVIDENCE_REFS)
    if not test_revealed:
        hidden = (*hidden, *_SEALED_FOLD_PERIODS)
    return {key: value for key, value in record.items() if key not in hidden}


def _public_params(
    params: Mapping[str, object], *, test_revealed: bool
) -> dict[str, object]:
    # params.json is also a worker-side ops channel where manager-owned roots
    # and the sealed Test/Held-out calendar legitimately exist. Neither may be
    # echoed before its public release boundary.
    return {
        key: value
        for key, value in params.items()
        if not key.startswith("_")
        and key not in _PRIVATE_PARAMS
        and (test_revealed or key not in _SEALED_PERIOD_PARAMS)
    }


def summarize_experiment(directory: Path) -> dict[str, object]:
    directory = Path(directory)
    summary: dict[str, object] = {"experiment_id": directory.name}
    try:
        identity = PublicIdentity(directory)
        state = experiment_state(directory)
        records = read_ledger_records(directory)
        folds = walk_forward_folds(list(latest_fold_records(records).values()))
        heldout = latest_heldout_records(records)
        skills_snapshot = latest_skills_snapshot(records, experiment_dir=directory)
        params = read_json(directory / HITL_DIR_NAME / PARAMS_NAME)
        _require_supported_params(params)
        schedule = read_json(directory / HITL_DIR_NAME / SCHEDULE_NAME)
        sessions = schedule.get("sessions") if isinstance(schedule.get("sessions"), list) else None
        revealed = test_results_revealed(directory, records)
        test_stage = bool(params.get("test_stage"))
        epochs = sorted({str(row.get("epoch_id")) for row in folds if row.get("epoch_id")})
        latest_epoch = epochs[-1] if epochs else None
        completed_sessions, total_sessions = _durable_session_progress(sessions, records)
        # Cumulative Validation/Test return is the final value of the very
        # series the console draws, not a product of per-Fold window returns:
        # a rolling Validation window trails over several quarters, so
        # compounding whole windows counts the shared quarters again and again.
        # One function serves both, so the tile and the chart cannot diverge.
        from . import equity  # deferred: equity reads this module's read model

        cumulative = {
            epoch: {
                key: equity.walk_forward_final(
                    directory, records, epoch_id=epoch, key=key
                )
                for key in ("valid", *(("test",) if revealed else ()))
            }
            for epoch in epochs
        }
        raw_status = state.get("status")
        status = identity.public_status(raw_status) if isinstance(raw_status, Mapping) else {}
        public_state = identity.public_record(
            {key: value for key, value in state.items() if key != "status"},
            heldout_revealed=False,
        )
        if raw_status is not None:
            public_state["status"] = status
        summary.update(public_state)
        summary.update(
            {
                "created_at": _created_at(directory, params),
                "current_session": status.get("session_key"),
                "current_session_label": status.get("session_label"),
                "session_started_at": status.get("session_started_at"),
                "environment_stage": status.get("environment_stage"),
                "environment_stage_started_at": status.get("environment_stage_started_at"),
                "environment_progress": status.get("environment_progress"),
                "folds_recorded": len(folds),
                "heldout_recorded": len(heldout),
                "skills": {
                    "count": skills_snapshot.stats.count,
                    "files": skills_snapshot.stats.files,
                    "bytes": skills_snapshot.stats.bytes,
                },
                "completed_sessions": completed_sessions,
                "total_sessions": total_sessions,
                "test_revealed": revealed,
                # Graduation verdict from the Held-out records; sealed like
                # every other Held-out number until the reveal.
                "verdict": _public_verdict(records, identity) if revealed else None,
                # The artifact Paper pins and the command that pins it
                # (ledger.paper_candidate); None unless graduated.
                "paper_candidate": _paper_candidate_view(directory, records) if revealed else None,
                "metrics": {
                    "epoch_id": latest_epoch,
                    "cum_valid_return": cumulative.get(latest_epoch, {}).get("valid"),
                    "cum_test_return": cumulative.get(latest_epoch, {}).get("test"),
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
                        "cum_valid_return": cumulative[epoch].get("valid"),
                        "cum_test_return": cumulative[epoch].get("test"),
                        "mean_test_sharpe": _mean(
                            _metric_series(epoch_folds, "test_result", "sharpe")
                        ) if revealed else None,
                        "walk_forward": _walk_forward_view(
                            records,
                            epoch,
                            epoch_folds,
                            test_stage=test_stage,
                            revealed=revealed,
                        ),
                    }
                    for epoch in epochs
                    for epoch_folds in [_epoch_folds(folds, epoch)]
                ],
                "fold_returns": [
                    {
                        "epoch_id": record.get("epoch_id"),
                        "fold_ref": identity.fold_ref(record.get("fold_id")),
                        "fold_status": record.get("fold_status"),
                        # Which strategy this Fold left in force. The Fold's
                        # headline numbers and its curve are both that
                        # strategy's, so both read this one label instead of
                        # re-deriving the branch from fold_status.
                        "strategy_in_force": strategy_in_force(record).source,
                        # Development evidence: the Fold's baseline, never sealed.
                        "parent_control": _parent_control_view(record),
                        # How wide the search behind this Fold's frozen
                        # candidate was, and how much of its Sharpe that width
                        # alone explains.
                        "selection": _selection_view(record),
                    }
                    # ``folds`` is already in walk-forward order, the same
                    # order ledger.walk_forward_transitions counts in, so
                    # labels cannot mis-pair with the transition counts above.
                    for record in folds
                ],
            }
        )
    except Exception as exc:  # noqa: BLE001 - broken experiments remain inspectable and deletable
        summary.update(
            {
                "state": "unreadable",
                "worker_alive": False,
                "error": f"{type(exc).__name__}: experiment state is unreadable",
            }
        )
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
        return {
            **detail,
            "params": {},
            "control": None,
            "sessions": [],
            "test_revealed": False,
            "inbox": {"pending_count": 0, "queued_ids": []},
        }
    identity = PublicIdentity(directory)
    records = read_ledger_records(directory)
    folds = latest_fold_records(records)
    heldout = latest_heldout_records(records)
    revealed = test_results_revealed(directory, records)
    hitl = directory / HITL_DIR_NAME
    params = read_json(hitl / PARAMS_NAME)
    sessions: list[dict[str, object]] = []
    for planned in identity.sessions:
        entry = identity.public_session(planned, heldout_revealed=revealed)
        kind = str(planned.get("kind") or "")
        raw_key = str(planned.get("_raw_key") or "")
        if kind == "fold":
            raw_fold = str(planned.get("fold_id") or raw_key.partition("/")[2])
            record = folds.get((str(planned.get("epoch_id")), raw_fold))
            if record is not None:
                entry["record"] = identity.public_record(
                    guarded_fold_view(record, test_revealed=revealed),
                    heldout_revealed=revealed,
                )
                entry["analysis_available"] = analysis_available(
                    hitl,
                    str(planned.get("epoch_id")),
                    identity.fold_ref(raw_fold),
                )
        elif kind == "meta":
            record = next(
                (
                    row
                    for row in reversed(records)
                    if row.get("record_type") == "meta_learning"
                    and meta_record_session_key(row) == raw_key
                ),
                None,
            )
            if record is not None:
                entry["record"] = identity.public_record(
                    record, heldout_revealed=revealed
                )
        elif kind == "heldout" and heldout:
            entry["records"] = [
                identity.public_record(row, heldout_revealed=revealed)
                for row in heldout
            ]
        elif kind == "deployment_adjustment":
            record = latest_deployment_record(records)
            if record is not None:
                entry["record"] = identity.public_record(
                    record, heldout_revealed=revealed
                )
        sessions.append(entry)
    control = read_control(hitl / CONTROL_NAME)
    raw_state = experiment_state(directory)
    raw_status = raw_state.get("status")
    raw_current = raw_status.get("session_key") if isinstance(raw_status, Mapping) else None
    return {
        **detail,
        "params": _public_params(params, test_revealed=revealed),
        "control": identity.public_control(control.to_record()),
        "inbox": inbox_public_view(
            hitl / INBOX_NAME,
            session_key=str(raw_current) if isinstance(raw_current, str) and raw_current else None,
        ),
        "test_revealed": revealed,
        "sessions": sessions,
    }


def resolve_fold_record(
    root: Path, experiment_id: str, epoch_id: str, fold_ref: str
) -> tuple[Path, PublicIdentity, list[dict[str, object]], dict[str, object]]:
    """Resolve one public fold reference to its trusted host ledger record."""

    directory = resolve_experiment_dir(root, experiment_id)
    identity = PublicIdentity(directory)
    raw_fold_id = identity.raw_fold_id(epoch_id, fold_ref)
    records = read_ledger_records(directory)
    record = latest_fold_records(records).get((epoch_id, raw_fold_id))
    if record is None:
        raise KeyError(f"no fold record for {epoch_id}/{fold_ref}")
    return directory, identity, records, record


def fold_strategy_dir(
    root: Path, experiment_id: str, epoch_id: str, fold_ref: str
) -> Path:
    directory, _identity, _records, record = resolve_fold_record(
        root, experiment_id, epoch_id, fold_ref
    )
    value = record.get("frozen_strategy_artifact_path")
    if not isinstance(value, str) or not value:
        raise KeyError("fold has no frozen strategy artifact on disk")
    raw = Path(value)
    strategy_dir = raw.resolve() if raw.is_absolute() else (directory / raw).resolve()
    if not strategy_dir.is_relative_to(directory.resolve()) or not strategy_dir.is_dir():
        raise KeyError("fold has no frozen strategy artifact on disk")
    return strategy_dir


def fold_detail(root: Path, experiment_id: str, epoch_id: str, fold_ref: str) -> dict[str, object]:
    directory, identity, records, record = resolve_fold_record(
        root, experiment_id, epoch_id, fold_ref
    )
    hitl_dir = directory / HITL_DIR_NAME
    md_path, meta_path = analysis_paths(hitl_dir / ANALYSIS_DIR_NAME, epoch_id, fold_ref)
    revealed = test_results_revealed(directory, records)
    analysis_meta = read_json(meta_path) if meta_path.exists() else None
    return {
        "experiment_id": experiment_id,
        "epoch_id": epoch_id,
        "fold_ref": fold_ref,
        "record": identity.public_record(
            guarded_fold_view(record, test_revealed=revealed),
            heldout_revealed=revealed,
        ),
        "test_audit": (
            {
                **identity.public_record(
                    {"record_type": "fold", **{field: record.get(field) for field in TEST_FIELDS}},
                    heldout_revealed=True,
                ),
                # Result-directory name of the revealed test evaluation, so the
                # console links its order export by the id the read-model
                # actually serves instead of guessing a name.
                "result": _result_name(directory, record.get("test_result_ref")),
            }
            if revealed else {"hidden": True}
        ),
        "strategy_available": bool(record.get("frozen_strategy_artifact_path")),
        "analysis": {
            "available": md_path.exists(),
            "meta": identity.public_analysis_meta(analysis_meta)
            if isinstance(analysis_meta, Mapping)
            else None,
        },
        "run_ref": identity.run_ref(record.get("run_id")) if record.get("run_id") else None,
        "trace_ref": identity.trace_ref(record.get("run_id")) if record.get("run_id") else None,
    }


def analysis_available(hitl_dir: Path, epoch_id: str, fold_ref: str) -> bool:
    md_path, _meta = analysis_paths(hitl_dir / ANALYSIS_DIR_NAME, epoch_id, fold_ref)
    return md_path.exists()


def fold_run_id(root: Path, experiment_id: str, epoch_id: str, fold_ref: str) -> str:
    """Resolve a public Fold reference to the raw host run identity."""

    _directory, _identity, _records, record = resolve_fold_record(
        root, experiment_id, epoch_id, fold_ref
    )
    run_id = str(record.get("run_id") or "")
    if not _ID.fullmatch(run_id):
        raise KeyError(f"no fold record for {epoch_id}/{fold_ref}")
    return run_id


# A Fold that froze nothing keeps its inherited parent as the lineage head, so
# the host's parent control -- that parent replayed on this Fold's Validation
# window -- is the strategy the Fold left in force.
_PARENT_KEPT_STATUSES = frozenset({"no_update", "no_valid_backtest"})


class StrategyInForce(NamedTuple):
    """The strategy one Fold left in force, and the replay that records it.

    ``source`` is the single label everything downstream reads: the Fold's own
    ``frozen_candidate``, the inherited ``parent_control``, or ``none``. The
    console labels its Fold numbers and its Fold curve with it, so the two can
    never claim different strategies. ``reference`` is that replay's result
    artifact, ``None`` when the record names none.
    """

    source: str
    reference: object


def strategy_in_force(record: Mapping[str, object]) -> StrategyInForce:
    """Which strategy one Fold left in force, and its Validation replay.

    A ``frozen`` Fold nominated a Step -- possibly the host's parent control,
    when it re-froze its parent -- and that Step's replay is the answer.
    ``no_update`` and ``no_valid_backtest`` froze nothing, so the parent stays
    the lineage head and the host's parent control is the answer; a nominated
    but hard-rejected candidate never is, even though the record still names
    it in ``selected_step_id``. ``baseline_missing`` left no strategy at all
    and answers ``none``, and so does an unknown status.

    A named artifact that cannot be read is still the strategy in force: the
    source stands and the reference is the one that failed, so the console can
    say the replay is unreadable instead of that the Fold traded nothing.
    """

    status = str(record.get("fold_status") or "")
    if status == "frozen":
        selected = str(record.get("selected_step_id") or "")
        return StrategyInForce(
            "frozen_candidate",
            next(
                (
                    step.get("validation_result_ref")
                    for step in record.get("steps", [])
                    if isinstance(step, Mapping)
                    and str(step.get("step_id") or "") == selected
                ),
                None,
            ),
        )
    if status in _PARENT_KEPT_STATUSES:
        control = record.get("parent_control")
        return StrategyInForce(
            "parent_control",
            control.get("validation_result_ref")
            if isinstance(control, Mapping)
            else None,
        )
    return StrategyInForce("none", None)


def style_payload(
    root: Path,
    experiment_id: str,
    *,
    run_ref: str,
    prefix: str,
) -> dict[str, object]:
    """Read the canonical style sidecar selected by one public run reference."""

    if prefix not in RESULT_MODES:
        raise ValueError("prefix must be valid|test|heldout")
    experiment_dir = resolve_experiment_dir(root, experiment_id)
    identity = PublicIdentity(experiment_dir)
    try:
        run_id = identity.raw_run_id(run_ref)
    except (KeyError, ValueError) as exc:
        raise ValueError("invalid run reference") from exc
    # Reveal gate: pre-reveal, sealed prefixes answer exactly like a missing
    # rollup so test/held-out existence never leaks.
    if prefix in sealed_prefixes(experiment_dir):
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
            reference = strategy_in_force(fold).reference
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
    expected_mode = RESULT_MODES[prefix]
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
    candidate = path / "result.json"
    return candidate if candidate.is_file() else None


def _result_name(experiment_dir: Path, reference: object) -> str | None:
    """Result-directory name (``frozen_test_<hex>``) behind one result ref."""
    path = _result_file(experiment_dir, reference)
    return path.parent.name if path is not None else None


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
        sealed = result_dir_prefixes(sealed_prefixes(experiment_dir))
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
    experiment_dir, _identity, records, record = resolve_fold_record(
        root, experiment_id, epoch_id, fold_id
    )
    files = _fold_result_files(
        experiment_dir, record, revealed=test_results_revealed(experiment_dir, records)
    )
    guarded = result_dir_prefixes(SEALED_PREFIXES)
    available = [name for name in files if not name.startswith(guarded)]
    selected = result or (available[0] if available else None)
    if selected is None:
        return {
            "result": None, "available": [],
            "stats": _order_stats([]), "rows": [], "row_count": 0,
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
        "stats": _order_stats(orders),
        "rows": orders if max_rows is None else orders[:max_rows],
        "row_count": len(orders),
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
        "price", "commission", "stamp_duty", "realized_pnl", "reason",
    )
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(payload["rows"])
    return f"{experiment_id}__{epoch_id}__{fold_id}__{result}.csv", stream.getvalue()
