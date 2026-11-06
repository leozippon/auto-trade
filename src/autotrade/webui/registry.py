"""Experiment discovery and read models for the research console.

Everything here is read-only over ``experiments/<id>/``: the append-only
ledger, the hitl/ control-plane files and the result artifacts the ledger
names. Unparseable experiments still appear in listings so they can be
deleted; they carry an error note instead of metrics.

An experiment researches in sessions, registers at most once and is then replayed once
over forward and Held-out (docs/PIPELINE_DESIGN.md). The replay is out of reach
while it runs: the console serves a result only when a ledger record names it,
and the forward record that names the replay is written together with its
readout.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

from autotrade.agent.runner import (
    API_SPEND_EXHAUSTED,
    DEADLINE_GRACE_EXHAUSTED,
    LLM_CALL_BUDGET_EXHAUSTED,
)
from autotrade.common.ids import is_experiment_id
from autotrade.common.records import (
    dataclass_record,
    read_json_object,
)
from autotrade.environment.replay.style import (
    STYLE_ARTIFACT_NAME,
    STYLE_SCHEMA_VERSION,
)
from autotrade.environment.step_tree import StepTree
from autotrade.environment.tools.report_issue import (
    ISSUE_REPORTS_NAME,
    issue_reports_path,
    read_issue_reports,
)
from autotrade.pipelines import validation_record
from autotrade.pipelines.agent_inbox import INBOX_NAME, inbox_public_view
from autotrade.pipelines.calendar import (
    GEOMETRY_PARAMETERS,
    ResearchGeometry,
)
from autotrade.pipelines.config import acceptance_for, rolling_default
from autotrade.pipelines.experiment import validation_statistics
from autotrade.pipelines.experiment_params import PARAMETERS, read_param
from autotrade.pipelines.hitl_state import (
    ARCHIVED_NAME,
    CONTROL_NAME,
    HITL_DIR_NAME,
    PARAMS_NAME,
    STATUS_NAME,
    WEB_CREATE_DEFAULTS,
    ExperimentState,
    experiment_is_archived,
    read_control,
    read_gpu_claim,
    read_status,
    status_pid_alive,
)
from autotrade.pipelines.ledger import (
    ENVIRONMENT_BLOCKED,
    ExperimentLedger,
    SessionKind,
    Stage,
    VerdictStatus,
    experiment_verdict,
    forward_record,
    paper_candidate,
    recorded_path,
    registered_record,
    research_records,
)
from autotrade.pipelines.pit_views_seed import FORWARD_PHASE, RESEARCH_PHASE
from autotrade.pipelines.research_session import session_role_quotas
from autotrade.pipelines.session_resume import STEP_SIDECAR_DIR
from autotrade.pipelines.session_tools import BATCH_FAILURE_ENVIRONMENT
from autotrade.pipelines.skills import latest_skills_snapshot
from autotrade.pipelines.verdict import (
    pooled_reading,
    residual_threshold,
    summary_excess_at_cost_stress,
)

from .public_identity import PublicIdentity
from .traces import resolve_trace_path, trace_stats

ACTIVE_STATES = (
    ExperimentState.LAUNCHING,
    ExperimentState.INITIALIZING,
    ExperimentState.RUNNING_SESSION,
    ExperimentState.PAUSED,
)
# Manager-written stub state between spawn and the worker's first status write
# (interpreter start + imports take seconds). Stale = the worker never came up.
LAUNCH_GRACE_SECONDS = 180.0
# Operator-only keys. The read model never echoes any of them back out.
_PRIVATE_PARAMS = {
    "raw_dir",
    "fundamental_events_root",
    "fundamental_events_status",
    "pit_cache_root",
    "template_dir",
    "local_dev",
    "experiments_root",
    "work_root",
    "llm_env_file",
}
# Replay result modes a style sidecar may carry: research and forward replays.
_RESULT_MODES = frozenset({RESEARCH_PHASE, FORWARD_PHASE})
# Experiment stages, in order: researching, registered with the forward replay pending or
# running, readout recorded.
STAGES = ("research", "forward", "readout")
# How an experiment ended, in the order the homepage lists them. The console says an
# ending in this one vocabulary wherever it says it at all, so the ledger's own
# words (``no_deliverable``, the session outcomes) never have to be read twice.
# An experiment whose session the host ended on the environment (``environment_blocked``)
# is no research outcome and keeps that word of its own.
ENDING_STATES = (
    "registered_with_record",
    "budget_exhausted",
    ENVIRONMENT_BLOCKED,
    "broken",
)
_ENDING_ORDER = {state: index for index, state in enumerate(ENDING_STATES)}


def _recorded_reasons(reasons: object) -> list[str]:
    """Recorded reason tokens as recorded, once each."""

    return list(dict.fromkeys(str(token) for token in reasons or ()))  # type: ignore[union-attr]


# The budgets a research session can run out of, named as the reason line.
_BUDGET_EXHAUSTED = {
    DEADLINE_GRACE_EXHAUSTED: "研究时长用尽",
    LLM_CALL_BUDGET_EXHAUSTED: "模型调用次数用尽",
    API_SPEND_EXHAUSTED: "API 花费上限用尽",
}
# Sentence enders of an Agent-authored reason, whose first sentence is the line
# the console shows.
_SENTENCE_END = re.compile(r"[。！？!?\n]")


class UnsupportedParamsError(ValueError):
    """params.json names parameters no parameter declares. The listing flags
    it unreadable instead of crashing."""


def _require_supported_params(params: Mapping[str, object]) -> None:
    unknown = sorted(
        key for key in params if not str(key).startswith("_") and key not in PARAMETERS
    )
    if unknown:
        raise UnsupportedParamsError(f"unsupported experiment parameters: {unknown}")


def read_ledger_records(experiment_dir: Path) -> list[dict[str, object]]:
    """The single validating reader shared with the pipeline: a ledger the
    pipeline would reject (unparseable line, wrong schema_version) must not
    drive console decisions either. Failures surface through the experiment's
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
    and the worker's own first status write replaces it, so deriving it from
    the experiment id keeps every state diagnosable; the file has to exist, so a
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
    state. A status.json that cannot be read (corrupt JSON, foreign
    schema_version) reads as "unreadable" with a one-line error: never counted
    as running, still listed, still deletable. Only this read-model combiner is
    total — control-plane mutation paths keep using the strict ``read_status``
    and fail fast on the same file."""
    path = Path(experiment_dir) / HITL_DIR_NAME / STATUS_NAME
    try:
        status = read_status(path)
    except (OSError, ValueError) as exc:
        return {
            "kind": "hitl",
            "state": ExperimentState.UNREADABLE,
            "worker_alive": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    log_ref = worker_log_for(experiment_dir, repo_root)
    if not status:
        result: dict[str, object] = {
            "kind": "hitl",
            "state": ExperimentState.CREATED,
            "worker_alive": False,
            "status": {},
        }
    else:
        alive = status_pid_alive(status)
        state = str(status.get("state") or "unknown")
        if state == ExperimentState.LAUNCHING and not alive:
            if _age_seconds(status.get("launched_at")) > LAUNCH_GRACE_SECONDS:
                state = ExperimentState.INTERRUPTED
        elif state in ACTIVE_STATES and not alive:
            state = ExperimentState.INTERRUPTED
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


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _created_at(directory: Path, params: Mapping[str, object]) -> str | None:
    if params.get("_created_at"):
        return str(params["_created_at"])
    try:
        return datetime.fromtimestamp(directory.stat().st_mtime, UTC).isoformat(timespec="seconds")
    except OSError:
        return None


def _public_params(params: Mapping[str, object]) -> dict[str, object]:
    # params.json is also a worker-side ops channel where manager-owned roots
    # legitimately exist; they never leave the host.
    return {
        key: value
        for key, value in params.items()
        if not key.startswith("_") and key not in _PRIVATE_PARAMS
    }


def experiment_stage(records: Sequence[Mapping[str, object]]) -> str:
    """Which of :data:`STAGES` the experiment is in, read off the ledger alone."""

    if experiment_verdict(records) is not None:
        return "readout"
    if registered_record(records) is not None:
        return "forward"
    return "research"


def _reason_line(text: str, limit: int = 80) -> str:
    """One line of an Agent-authored reason: its first sentence, cut to
    ``limit`` characters with an ellipsis."""

    head = _SENTENCE_END.split(text.strip(), maxsplit=1)[0].strip()
    return head if len(head) <= limit else head[: limit - 1] + "…"


def _percent(value: object) -> str:
    """One signed percentage of the ending's reason line."""

    number = _number(value)
    return "—" if number is None else f"{number * 100:+.2f}%"


def _attempt_failures(
    records: Sequence[Mapping[str, object]], phase: str | None = None
) -> list[Mapping[str, object]]:
    """The recorded failed attempts, of one phase or of the whole experiment."""

    return [
        row
        for row in records
        if row.get("record_type") == "attempt_failed"
        and (phase is None or row.get("phase") == phase)
    ]


def experiment_ending(
    identity: PublicIdentity,
    records: Sequence[Mapping[str, object]],
    state: Mapping[str, object],
    forward: Mapping[str, object] | None,
) -> dict[str, str] | None:
    """How the experiment ended — one of :data:`ENDING_STATES` and a one-line reason —
    or ``None`` while it can still run.

    Every ending the console shows comes from here, so no page classifies one
    for itself. A worker the host or the environment broke ended the process
    rather than the research, so its state is read before the ledger. Otherwise
    the ledger decides: a registered strategy with its record by its readout
    line (``forward`` is :func:`_forward_view`), one whose session the host ended
    on the environment by the host's own reason, and one whose budget ran out
    before anything registered by the budget that did.
    """

    if str(state.get("state") or "") == ExperimentState.FAILED:
        error = str(_mapping(state.get("status")).get("error") or "")
        if not error:
            failures = _attempt_failures(records)
            error = str(failures[-1].get("error") or "") if failures else ""
        line = identity.public_text(error).splitlines()
        return {"state": "broken", "reason": line[0] if line else ""}
    verdict = experiment_verdict(records)
    if verdict is None:
        return None
    if verdict["status"] == VerdictStatus.REGISTERED_WITH_RECORD:
        return {"state": "registered_with_record", "reason": _readout_line(_mapping(_mapping(forward).get("readout")))}
    if verdict["status"] == VerdictStatus.ENVIRONMENT_BLOCKED:
        # The host's reason names the span and the error, like a broken
        # worker's: the error is what the operator acts on, so it is not cut.
        reasons = verdict.get("reasons")
        text = str(reasons[0]) if isinstance(reasons, list) and reasons else ""
        line = identity.public_text(text).splitlines()
        return {"state": ENVIRONMENT_BLOCKED, "reason": line[0] if line else ""}
    session = _mapping(
        next((row for row in research_records(records) if row.get("experiment_end")), None)
    )
    finish = str(session.get("finish_reason") or "")
    return {"state": "budget_exhausted", "reason": _BUDGET_EXHAUSTED.get(finish, finish)}


def _readout_line(readout: Mapping[str, object]) -> str:
    """One line of a registered strategy's readout: the validation record's residual
    IR, its p-value and candidate changes, then the forward lower bound."""

    record = _mapping(readout.get("record"))
    forward = _mapping(readout.get("forward"))
    ratio, p_value = _number(record.get("residual_information_ratio")), _number(record.get("p_value"))
    bound = _number(forward.get("lower_bound"))
    error = _mapping(forward.get("strategy_error"))
    return (
        f"验证记录 残差 IR {'—' if ratio is None else f'{ratio:.2f}'}"
        f" · p {'—' if p_value is None else f'{p_value:.2f}'}"
        f" · 换候选 {record.get('candidate_changes', '—')} 次"
        + (f" · 前推下界 {_percent(bound)}" if not error else f" · 前推 {error.get('slice')} 段策略异常")
    )


def experiment_attention(
    identity: PublicIdentity, directory: Path, ending: Mapping[str, object] | None
) -> dict[str, str] | None:
    """What the home list flags for the operator on this experiment, as one line, or
    ``None``.

    Two things raise it and nothing else does: an experiment the host ended on the
    environment, which needs a fix and a re-run rather than a reading, and an
    experiment that can still run while an issue report its session filed is
    unresolved -- until ``scripts/experiments/resolve_feedback.py issue``
    answers it or the experiment ends.
    """

    if ending is not None:
        return {"reason": str(ending["reason"])} if ending["state"] == ENVIRONMENT_BLOCKED else None
    try:
        reports = read_issue_reports(issue_reports_path(directory))
    except (OSError, ValueError):
        return {"reason": "问题报告无法读取"}
    unresolved = [report for report in reports if "resolved_at" not in report]
    if not unresolved:
        return None
    summary = _reason_line(identity.public_text(str(unresolved[-1].get("summary") or "")))
    return {"reason": f"{len(unresolved)} 条问题报告未处置：{summary}"}


def _result_name(reference: object) -> str | None:
    """Public name of one replay result: its result directory's name."""

    if not isinstance(reference, str) or not reference:
        return None
    path = Path(reference)
    return path.parent.name if path.name == "result.json" else path.name


def _forward_view(
    identity: PublicIdentity, record: Mapping[str, object] | None
) -> dict[str, object] | None:
    """The forward record: replay span, slice statistics and readout, as the
    record stores them. ``None`` until the record exists, which is also when
    the readout does."""

    if record is None:
        return None
    null = _mapping(record.get("null_control"))
    return {
        "recorded_at": record.get("recorded_at"),
        "error": identity.public_text(str(record.get("error") or "")) or None,
        "replay": dict(_mapping(record.get("replay"))),
        "result": _result_name(record.get("result_ref")),
        "slices": {
            name: dict(block)
            for name, block in _mapping(record.get("slices")).items()
            if isinstance(block, Mapping)
        },
        "refits_executed": record.get("refits_executed"),
        # Diagnostic: the forward slice's excess among random-name replays.
        "null_percentile": _number(_mapping(null.get("step")).get("excess_percentile")),
        # The experiment's readout (``validation_record.readout``).
        "readout": dict(_mapping(record.get("readout"))),
    }


def _paper_candidate_view(records: list[dict[str, object]]) -> dict[str, object] | None:
    """The registered artifact the experiment's Paper account is opened from."""
    candidate = paper_candidate(records)
    if candidate is None:
        return None
    return {"artifact_id": candidate["artifact_id"]}


def summarize_experiment(directory: Path) -> dict[str, object]:
    directory = Path(directory)
    summary: dict[str, object] = {"experiment_id": directory.name}
    try:
        identity = PublicIdentity(directory)
        state = experiment_state(directory)
        records = read_ledger_records(directory)
        skills = latest_skills_snapshot(records, experiment_dir=directory).stats
        params = read_json_object(directory / HITL_DIR_NAME / PARAMS_NAME, missing_ok=True)
        _require_supported_params(params)
        raw_status = state.get("status")
        status = identity.public_status(raw_status) if isinstance(raw_status, Mapping) else {}
        summary.update(
            identity.public_record({key: value for key, value in state.items() if key != "status"})
        )
        if raw_status is not None:
            summary["status"] = status
        forward = _forward_view(identity, forward_record(records))
        ending = experiment_ending(identity, records, state, forward)
        summary.update(
            {
                "created_at": _created_at(directory, params),
                "skills": {"count": skills.count, "files": skills.files, "bytes": skills.bytes},
                "stage": experiment_stage(records),
                "registered_session": _registered_session(records),
                "record": _record_view(records, params),
                "research_result": _research_result(records),
                "research_outcome": next(
                    (row.get("outcome") for row in research_records(records)), None
                ),
                "budget": _budget_totals(params),
                "budget_used": _budget_used(directory, records, raw_status),
                "verdict": experiment_verdict(records),
                "ending": ending,
                "attention": experiment_attention(identity, directory, ending),
                "forward": forward,
                "paper_candidate": _paper_candidate_view(records),
            }
        )
    except Exception as exc:  # noqa: BLE001 - broken experiments remain inspectable and deletable
        summary.update(
            {
                "state": ExperimentState.UNREADABLE,
                "worker_alive": False,
                "error": f"{type(exc).__name__}: experiment state is unreadable",
            }
        )
    return summary


def _budget_totals(params: Mapping[str, object]) -> dict[str, object]:
    """The experiment's research budget, keyed like the trace's ``budget_used`` block.

    The create form persists only what differs from the defaults, so the
    effective ceilings are the worker defaults overlaid with ``params.json``.
    The sub-agents' ceiling is their quota of the shared call budget
    (:func:`session_role_quotas`), not a parameter of its own.
    """

    effective = {**WEB_CREATE_DEFAULTS, **params}
    minutes = _number(effective.get("max_research_minutes"))
    calls = _number(effective.get("max_llm_calls"))
    return {
        "inference_seconds": minutes * 60.0 if minutes is not None else None,
        "llm_calls": calls,
        "null_controls": _number(effective.get("max_null_controls")),
        "api_spend_usd": _number(
            effective.get("max_api_spend_usd", rolling_default("max_api_spend_usd"))
        ),
        "subagent_calls": (
            float(session_role_quotas(int(calls))[0]) if calls is not None and calls > 0 else None
        ),
    }


def _budget_used(
    directory: Path,
    records: Sequence[Mapping[str, object]],
    raw_status: object,
) -> dict[str, object] | None:
    """What the research session has spent: the ledger's block once it is
    recorded, else the last block its trace carries (all a session that
    ended unrecorded, terminated or with its worker lost, leaves), else
    nothing."""

    for record in research_records(records):
        block = record.get("budget_used")
        if isinstance(block, Mapping):
            return dict(block)
    status = _mapping(raw_status)
    if status.get("session_key") != SessionKind.RESEARCH:
        return None
    run_id = status.get("run_id")
    path = resolve_trace_path(directory, str(run_id)) if isinstance(run_id, str) else None
    if path is None:
        return None
    block = trace_stats(path).get("budget_used")
    return dict(block) if isinstance(block, Mapping) else None


def research_budget_used(directory: Path) -> dict[str, object] | None:
    """:func:`_budget_used` for the status poll, which reads no summary. An
    unreadable ledger answers nothing here; the listing already flags it."""

    directory = Path(directory)
    try:
        records = read_ledger_records(directory)
    except (OSError, ValueError):
        return None
    return _budget_used(directory, records, experiment_state(directory).get("status"))


def _ending_rank(row: Mapping[str, object]) -> int:
    """Listing order: the experiments that can still run, then the ended ones in
    :data:`ENDING_STATES` order."""

    ending = _mapping(row.get("ending"))
    if not ending:
        return 0
    return 1 + _ENDING_ORDER.get(str(ending.get("state")), len(ENDING_STATES))


# While no worker is alive a listing row is a function of a few files: the
# status, the ledger, the issue reports (resolving one clears the row's
# attention line), the params, the public-identity map and whether the worker
# log exists. The home page polls the listing every 5 s over every experiment
# ever created, so such a row is kept per process and re-derived only when one
# of those files changes; a row with a live worker, or one still launching (its
# state ages), is derived on every poll.
_SUMMARY_SOURCES = (
    f"{HITL_DIR_NAME}/{STATUS_NAME}",
    "ledgers/experiment_ledger.jsonl",
    f"ledgers/{ISSUE_REPORTS_NAME}",
    f"{HITL_DIR_NAME}/{PARAMS_NAME}",
    ".host/agent-refs.json",
)
_SUMMARY_CACHE: dict[Path, tuple[tuple[object, ...], dict[str, object]]] = {}
_LISTING_LOCK = threading.Lock()


def _summary_signature(directory: Path) -> tuple[object, ...]:
    signature: list[object] = [bool(worker_log_for(directory))]
    for relative in _SUMMARY_SOURCES:
        try:
            info = (directory / relative).stat()
        except FileNotFoundError:
            signature.append(None)
        else:
            signature.append((info.st_ino, info.st_mtime_ns, info.st_ctime_ns, info.st_size))
    return tuple(signature)


def _listing_row(directory: Path) -> tuple[dict[str, object], tuple[object, ...] | None]:
    """The row, and the signature it is kept under; ``None`` for a row derived
    afresh on every poll."""

    # Signed before reading, so a change racing the read only re-derives.
    signature = _summary_signature(directory)
    cached = _SUMMARY_CACHE.get(directory)
    if cached is not None and cached[0] == signature:
        return cached[1], signature
    row = summarize_experiment(directory)
    if row.get("worker_alive") is False and row.get("state") not in (ExperimentState.LAUNCHING, ExperimentState.UNREADABLE):
        _SUMMARY_CACHE[directory] = (signature, row)
        return row, signature
    _SUMMARY_CACHE.pop(directory, None)
    return row, None


# The console's side of archiving (``hitl_state.ARCHIVED_NAME``): the home page
# lists an archived experiment only when asked for the archived ones, the default
# listing never summarizes it, its experiment page still opens, and the manager
# never starts its worker again. The consistency check
# (scripts/dev/check_verdicts.py), the round fill queue and Paper see the experiment
# as before.


def archive_marker(directory: Path) -> dict[str, object] | None:
    """The experiment's archive marker (when and why), or None when it is not archived."""

    path = Path(directory) / HITL_DIR_NAME / ARCHIVED_NAME
    return read_json_object(path, missing_ok=True) if path.is_file() else None


# Rows kept by another process were derived by the code that process ran, so a
# listing digest never matches across a console restart.
_PROCESS_NONCE = uuid.uuid4().hex


def _listing(
    root: Path, *, archived: bool = False
) -> tuple[list[dict[str, object]], set[str], str, int]:
    """Every row in listing order (the archived experiments' when ``archived``, else
    every other experiment's), the ids of the rows derived afresh, a digest of all
    the kept ones (ids and signatures) and of which experiments are archived, and how
    many experiments are archived.

    One listing at a time per process. The server answers requests from a
    thread pool, and two overlapping listings would otherwise both derive the
    rows of a cold process — repeated by every tab that asks meanwhile — and
    one would prune the summary cache while the other fills it.
    """

    if not root.is_dir():
        return [], set(), "", 0
    with _LISTING_LOCK:
        directories = [
            path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
        ]
        shelved = {path.name for path in directories if experiment_is_archived(path)}
        rows: list[dict[str, object]] = []
        fresh: set[str] = set()
        kept: list[tuple[str, tuple[object, ...]]] = []
        for path in directories:
            if (path.name in shelved) != archived:
                continue
            row, signature = _listing_row(path)
            rows.append(row)
            if signature is None:
                fresh.add(path.name)
            else:
                kept.append((path.name, signature))
        for gone in {path for path in _SUMMARY_CACHE if path.parent == root} - set(directories):
            _SUMMARY_CACHE.pop(gone, None)
    rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    rows.sort(key=_ending_rank)
    digest = hashlib.sha256(
        repr((_PROCESS_NONCE, archived, sorted(kept), sorted(shelved))).encode()
    ).hexdigest()[:16]
    return rows, fresh, digest, len(shelved)


def experiment_listing(
    root: Path, kept: str | None = None, *, archived: bool = False
) -> dict[str, object]:
    """The home page's listing: every row in order, the ranking of the registered
    strategies with a readout,
    ``archived``, how many experiments are archived, and ``kept``, the digest of the
    rows that change only when their files do (every experiment without a live
    worker) and of which experiments are archived. With ``archived`` the rows are the
    archived experiments' instead of every other experiment's.

    The home page polls it every few seconds, and those rows are nearly all of
    it. A poll that sends back the digest it last received, and finds it
    unchanged, is answered with the rows derived afresh (live workers,
    launching or unreadable experiments) and the order of every id; it keeps the
    other rows it already holds. Any change to a kept row — an experiment ending,
    created, deleted, archived or restored, any of its files rewritten, or a
    console restart — changes the digest, and the poll gets every row again.
    """

    rows, fresh, digest, shelved = _listing(Path(root), archived=archived)
    payload: dict[str, object] = {
        "experiments": rows,
        "ranking": ranking(rows),
        "kept": digest,
        "archived": shelved,
    }
    if kept is not None and kept == digest:
        payload["experiments"] = [row for row in rows if row["experiment_id"] in fresh]
        payload["order"] = [row["experiment_id"] for row in rows]
    return payload


def ranking(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """The registered strategies with a readout, ranked by their validation record's
    residual IR (a record the significance test could not read sorts last), each with
    its forward reading and whether a Paper account may be opened from it.

    Research-period numbers never rank, and an experiment without a forward record is
    not listed: the record and the forward reading are the whole evidence.
    Ties keep the listing order.
    """

    ranked: list[tuple[int, float, int, dict[str, object]]] = []
    for position, row in enumerate(rows):
        readout = _mapping(_mapping(row.get("forward")).get("readout"))
        if not readout:
            continue
        record = _mapping(readout.get("record"))
        ratio = _number(record.get("residual_information_ratio"))
        ranked.append(
            (
                ratio is None,
                -(ratio or 0.0),
                position,
                {
                    "experiment_id": row["experiment_id"],
                    "record": {
                        key: record.get(key)
                        for key in ("residual_information_ratio", "p_value", "years", "steps", "candidate_changes", "start", "end")
                    },
                    "forward": dict(_mapping(readout.get("forward"))),
                    "paper_candidate": row.get("paper_candidate") is not None,
                },
            )
        )
    return [entry for *_order, entry in sorted(ranked, key=lambda item: item[:3])]


def resolve_experiment_dir(root: Path, experiment_id: str) -> Path:
    if not is_experiment_id(experiment_id):
        raise ValueError(f"invalid experiment id: {experiment_id!r}")
    root = Path(root).resolve()
    path = (root / experiment_id).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"invalid experiment id: {experiment_id!r}")
    if not path.is_dir():
        raise KeyError(f"unknown experiment: {experiment_id}")
    return path


def _cost_stress_multiplier(directory: Path) -> float:
    """The experiment's cost-stress multiplier as its worker resolves it."""

    params = read_json_object(directory / HITL_DIR_NAME / PARAMS_NAME, missing_ok=True)
    return acceptance_for(params).cost_stress_multiplier


@dataclass(frozen=True)
class ValidationView:
    """One recorded Validation as the console shows it: its span, headline
    replay metrics, the raw excess over the experiment's benchmark, and the graded
    figures, the active excess at the experiment's cost stress among them."""

    step_id: object
    span: object
    total_return: float | None
    excess_return: float | None
    excess_at_cost_stress: float | None
    sharpe: float | None
    max_drawdown: float | None
    plain_excess: float | None
    information_ratio: float | None
    # ``active`` when the figures are of the node's return minus its
    # random baseline, ``absolute`` when its replay carries none.
    series: object

    @classmethod
    def of(cls, row: Mapping[str, object], multiplier: float) -> ValidationView:
        """The view of one step row, the cost stress at ``multiplier``."""

        summary = _mapping(row.get("summary"))
        graded = _mapping(row.get("graded"))
        return cls(
            step_id=row.get("step_id"),
            span=row.get("span"),
            total_return=_number(summary.get("total_return")),
            excess_return=_number(_mapping(summary.get("benchmark")).get("excess_return")),
            excess_at_cost_stress=(
                summary_excess_at_cost_stress(graded, summary, cost_stress_multiplier=multiplier)
                if _number(graded.get("plain_excess")) is not None
                else None
            ),
            sharpe=_number(summary.get("sharpe")),
            max_drawdown=_number(summary.get("max_drawdown")),
            plain_excess=_number(graded.get("plain_excess")),
            information_ratio=_number(graded.get("information_ratio")),
            series=graded.get("series"),
        )


# A recorded Validation is immutable: its sidecar and its result's style file
# are read once per node and kept for the process lifetime, so a listing poll
# of a running experiment re-reads nothing until the session records another node.
_LIVE_STEP_CACHE: dict[tuple[str, str], dict[str, object]] = {}


def _live_steps(directory: Path) -> list[dict[str, object]]:
    """The research session's Validations recorded so far, from the host
    sidecars ``session_resume.record_step_sidecar`` writes, shaped like the
    ledger's step rows with the graded figures the Pipeline records
    (``experiment.validation_statistics``): what names a running session's
    results to the console before the session is recorded."""

    root = directory / STEP_SIDECAR_DIR
    if not root.is_dir():
        return []
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*.json")):
        key = (str(directory), path.stem)
        row = _LIVE_STEP_CACHE.get(key)
        if row is None:
            record = read_json_object(path, missing_ok=True)
            try:
                reference = str(
                    recorded_path(directory, str(record.get("result_ref") or ""))
                )
                figures = validation_statistics(reference)
            except OSError:
                # The replay's style file is written before its sidecar; a
                # node whose files are not readable yet is read next time.
                continue
            row = {
                "step_id": record.get("step_id"),
                "revision_id": record.get("revision_id"),
                "span": record.get("span"),
                "summary": dict(_mapping(record.get("summary"))),
                "validation_result_ref": reference,
                "graded": figures,
                **{name: record.get(name) for name in ("control", "batch_id")},
            }
            _LIVE_STEP_CACHE[key] = row
        rows.append(row)
    return rows


def _registered_session(records: Sequence[Mapping[str, object]]) -> str | None:
    """The research session that registered the experiment's artifact, or ``None``.

    The listing's one fact about the registration: the experiment card draws the
    register step from it instead of inferring a registration from the stage and the
    verdict.
    """

    row = registered_record(records)
    return str(row["session_key"]) if row is not None else None


def _research_result(records: Sequence[Mapping[str, object]]) -> str | None:
    """The research-period result the console draws beside the forward replay:
    the registered strategy's own validation, once the experiment registered."""

    row = registered_record(records)
    if row is None:
        return None
    return _result_name(_mapping(row.get("frozen")).get("research_result_ref"))


def _record_alpha(params: Mapping[str, object]) -> float:
    """The level the experiment's record is read at, from its own stamped rules; an
    experiment with a record was created under the rolling validation, whose rules
    state it."""

    return acceptance_for(params).alpha


def _record_view(records: Sequence[Mapping[str, object]], params: Mapping[str, object]) -> dict[str, object] | None:
    """The experiment's validation record so far, read as one series: the steps the
    ledger holds, the test horizon they reached, the significance test over them
    (``validation_record.record_significance``) and how often the candidate
    changed. ``None`` before the first step is recorded: nothing is read off a
    replay the ledger does not name."""

    tests = validation_record.test_step_records(records)
    if not tests:
        return None
    significance = validation_record.record_significance([row["entry"] for row in tests], alpha=_record_alpha(params))  # type: ignore[misc]
    return {
        **significance,
        "start": str(tests[0]["window"]["start"]),  # type: ignore[index]
        "end": str(tests[-1]["window"]["end"]),  # type: ignore[index]
        "test_horizon": str(tests[-1]["test_horizon"]),
        "candidate_changes": validation_record.candidate_changes(records),
    }


def _record_detail(
    identity: PublicIdentity, records: Sequence[Mapping[str, object]], params: Mapping[str, object]
) -> dict[str, object] | None:
    """The record for the experiment page: :func:`_record_view` with one row per
    step (its window, test horizon, candidate by public ref, source and entry
    without the daily rows) and the daily curve the stitched entries give --
    the book and its random baseline compounded, with each step's bounds and
    whether its candidate changed, for the chart's markers."""

    view = _record_view(records, params)
    if view is None:
        return None
    tests = validation_record.test_step_records(records)
    rows: list[dict[str, object]] = []
    dates: list[str] = []
    book: list[float] = []
    baseline: list[float] = []
    steps: list[dict[str, object]] = []
    book_level = baseline_level = 1.0
    previous: str | None = None
    for record in tests:
        candidate = _mapping(record.get("candidate"))
        entry = _mapping(record.get("entry"))
        fingerprint = str(candidate.get("fingerprint") or "")
        changed = previous is not None and fingerprint != previous
        previous = fingerprint
        rows.append(
            {
                "step": record.get("step"),
                "window": dict(_mapping(record.get("window"))),
                "test_horizon": record.get("test_horizon"),
                "candidate": identity.public_text(str(candidate.get("step_id") or "")),
                "changed": changed,
                "source": record.get("source"),
                **{key: value for key, value in entry.items() if key != "daily"},
            }
        )
        for day, strategy, baseline_value, *_legs in entry.get("daily") or ():  # type: ignore[misc]
            book_level *= 1.0 + float(strategy)
            baseline_level *= 1.0 + float(baseline_value)
            dates.append(str(day))
            book.append(book_level - 1.0)
            baseline.append(baseline_level - 1.0)
        window = _mapping(record.get("window"))
        steps.append({"step": record.get("step"), "start": window.get("start"), "end": window.get("end"), "changed": changed})
    return {**view, "rows": rows, "curve": {"dates": dates, "book": book, "baseline": baseline, "steps": steps}}


def _verdict_thresholds(params: Mapping[str, object]) -> dict[str, object]:
    """How the experiment's record will be read, from its own stamped rules, so the
    console states it before any record exists: the residual statistic's
    level, the research years, the step length and the threshold for the whole
    validation period. A geometry date the params leave out is the worker's
    default (``experiment_params.read_param``)."""

    rules = acceptance_for(params)
    geometry = ResearchGeometry(**{name: str(read_param(params, name)) for name in GEOMETRY_PARAMETERS})
    validation_years = len(geometry.research_years) - rules.research_years
    return {
        "alpha": rules.alpha,
        "research_years": rules.research_years,
        "validation_years": validation_years,
        "test_step_months": rules.test_step_months,
        "threshold": residual_threshold(float(validation_years), rules.alpha),
    }


# The forward record of each experiment's ledger, by the ledger's size and mtime: the
# readouts below read every experiment, and an experiment page polls them.
_FORWARD_RECORDS: dict[Path, tuple[tuple[int, int], Mapping[str, object] | None]] = {}


def _experiment_forward_records(root: Path) -> tuple[list[tuple[str, Mapping[str, object]]], int]:
    """The forward record of every experiment under ``root`` that has one, by
    experiment id, archived experiments included, and how many experiments' ledgers could
    not be read (a readout counts them rather than skip them silently)."""

    found: list[tuple[str, Mapping[str, object]]] = []
    unreadable = 0
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        ledger = directory / "ledgers" / "experiment_ledger.jsonl"
        if not ledger.is_file():
            continue
        stat = ledger.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = _FORWARD_RECORDS.get(ledger)
        if cached is None or cached[0] != signature:
            try:
                cached = (signature, forward_record(read_ledger_records(directory)))
            except (OSError, ValueError):
                unreadable += 1
                continue
            _FORWARD_RECORDS[ledger] = cached
        if cached[1] is not None:
            found.append((directory.name, cached[1]))
    return found, unreadable


def _record_readouts(root: Path, record: Mapping[str, object] | None) -> dict[str, object] | None:
    """What every experiment under ``root`` read over this experiment's validation window and
    forward months, beside its own forward ``record``; ``None`` before that
    record.

    ``validation_window``: every registered strategy's validation record over the
    same validation period -- its residual IR, years, one-sided null p-value
    (``verdict.null_p_value``), steps and candidate changes -- and how many
    there are. No reading is adjusted for the others and no threshold moves with
    their count; the count is what a p-value is weighed against.
    ``forward_pool``: every registered strategy's residual reading over the same
    forward months, pooled (``verdict.pooled_reading``): one forward year
    cannot certify a strategy. A strategy that raised in its forward replay
    has no reading, and is counted under ``raised`` rather than left out
    silently.
    """

    mine = _mapping(_mapping(record).get("readout"))
    if not mine:
        return None
    window = _mapping(mine.get("record"))

    def forward_months(row: Mapping[str, object]) -> tuple[object, object]:
        replay = _mapping(row.get("replay"))
        return replay.get("start"), replay.get("forward_end")

    months = forward_months(_mapping(record))
    records, unreadable = _experiment_forward_records(root)
    readings: list[dict[str, object]] = []
    pooled: list[dict[str, object]] = []
    raised = 0
    for experiment_id, other in records:
        readout = _mapping(other.get("readout"))
        if not readout:
            continue
        block = _mapping(readout.get("record"))
        if (block.get("start"), block.get("end")) == (window.get("start"), window.get("end")):
            ratio, years = _number(block.get("residual_information_ratio")), _number(block.get("years"))
            readings.append(
                {
                    "experiment_id": experiment_id,
                    "residual_information_ratio": ratio,
                    "years": years,
                    "p_value": _number(block.get("p_value")),
                    "steps": block.get("steps"),
                    "candidate_changes": block.get("candidate_changes"),
                }
            )
        if forward_months(other) == months:
            slice_reading = _mapping(_mapping(readout.get("forward")).get("residual_reading"))
            if slice_reading:
                pooled.append(dict(slice_reading))
            else:
                raised += 1
    readings.sort(key=lambda row: (row["p_value"] is None, row["p_value"] or 0.0))
    return {
        "validation_window": {
            "start": window.get("start"),
            "end": window.get("end"),
            "count": len(readings),
            "readings": readings,
        },
        "forward_pool": {
            "start": months[0],
            "end": months[1],
            **pooled_reading(pooled),
            "raised": raised,
        },
        "unreadable_experiments": unreadable,
    }


def _seed_gate_view(identity: PublicIdentity, block: object) -> dict[str, object] | None:
    """The registration check's seed-replicate reading (``experiment._seed_replicates``)
    -- the replicates it read, each one's seed line and IR or the problem that
    refused it -- or ``None`` for a check that never read the nominee."""

    seeds = _mapping(block)
    if not seeds:
        return None
    return {
        "trains_a_model": bool(seeds.get("trains_a_model")),
        "replicates": [
            {
                "step_id": entry.get("step_id"),
                "seed_line": entry.get("seed_line"),
                "information_ratio": _number(entry.get("information_ratio")),
                "problem": identity.public_text(str(entry.get("problem") or "")) or None,
            }
            for entry in seeds.get("replicates") or ()
            if isinstance(entry, Mapping)
        ],
    }


def _research_session_view(
    directory: Path, identity: PublicIdentity, record: Mapping[str, object]
) -> dict[str, object]:
    steps = [row for row in record.get("steps") or () if isinstance(row, Mapping)]
    gate = _mapping(record.get("freeze_gate"))
    run_id = str(record.get("run_id") or "")
    multiplier = _cost_stress_multiplier(directory)
    return {
        "outcome": record.get("outcome"),
        "recorded_at": record.get("recorded_at"),
        # Agent-authored: through the same projection as every traced string.
        "reason": identity.public_text(str(record.get("reason") or "")) or None,
        "finish_reason": record.get("finish_reason"),
        "run_ref": identity.run_ref(run_id),
        # Whether the last attempt's trace is on disk: the page asks for its
        # counters and replay only then.
        "trace": resolve_trace_path(directory, run_id) is not None,
        "run_wall_seconds": _number(record.get("run_wall_seconds")),
        "nominated_step_id": record.get("nominated_step_id"),
        "registration": (
            {
                "passed": bool(gate.get("passed")),
                "reasons": _recorded_reasons(gate.get("reasons")),
                "mandate": {
                    key: _number(value)
                    for key, value in _mapping(gate.get("mandate")).items()
                },
                # The limits this nomination was actually judged against, as
                # its own record states them, so a later change to any of them
                # does not restate the experiment's gate and a limit the record does
                # not carry is not drawn.
                "thresholds": {
                    key: _number(value)
                    for key, value in _mapping(gate.get("thresholds")).items()
                },
                "seed_replicates": _seed_gate_view(identity, gate.get("seed_replicates")),
            }
            if gate
            else None
        ),
        "registered": bool(record.get("frozen")),
        "experiment_end": identity.public_record(record["experiment_end"])  # type: ignore[arg-type]
        if isinstance(record.get("experiment_end"), Mapping)
        else None,
        "validations": [dataclass_record(ValidationView.of(row, multiplier)) for row in steps],
        "attempts": record.get("attempts"),
        "budget_used": _mapping(record.get("budget_used")) or None,
    }


def _registered_view(
    identity: PublicIdentity, records: Sequence[Mapping[str, object]]
) -> dict[str, object] | None:
    """The registered artifact and the research statistics it was registered on."""

    row = registered_record(records)
    if row is None:
        return None
    block = _mapping(row.get("frozen"))
    fit = _mapping(block.get("fit_plan"))
    return {
        "session_key": row.get("session_key"),
        "strategy_ref": identity.strategy_ref(block["artifact_id"]),
        "source_step_id": block.get("source_step_id"),
        "result": _result_name(block.get("research_result_ref")),
        "days": block.get("days"),
        "plain_excess": _number(block.get("plain_excess")),
        "tracking_error": _number(block.get("tracking_error")),
        "information_ratio": _number(block.get("information_ratio")),
        "forward_mde": _number(block.get("forward_mde")),
        "null_percentile": _number(_mapping(block.get("null_control")).get("excess_percentile")),
        "fit": bool(fit.get("fit")),
        "refit_period": fit.get("refit_period"),
        "blocks": [dict(item) for item in block.get("blocks") or () if isinstance(item, Mapping)],
        # Registered beside replicates: the gate's reading of them, for the seed
        # mean beside the nominee's IR and their threshold. Absent on one seed.
        "seed_replicates": (
            _seed_gate_view(identity, _mapping(row.get("freeze_gate")).get("seed_replicates"))
            if block.get("seed_replicates")
            else None
        ),
    }


def _failed_attempts(
    identity: PublicIdentity, records: Sequence[Mapping[str, object]], phase: str
) -> dict[str, object]:
    """How many attempts of ``phase`` failed before their record, and the last
    failure's reason: the retries the verdict view accounts for."""

    failed = _attempt_failures(records, phase)
    last = failed[-1] if failed else None
    return {
        "failed": len(failed),
        "last_error": identity.public_text(str(last.get("error") or "")) or None if last else None,
    }


def _failed_validations(identity: PublicIdentity, directory: Path) -> dict[str, object]:
    """How many of the research session's Validations failed, how many of those
    the environment failed (their replay-years were refunded), and the last
    one's reason. A failed Validation is no attempt failure and leaves no
    ledger row: its one record is its dead-end node in the step tree (written
    while ``record_failed_attempts`` is on, the default), the same nodes the
    step-tree view marks failed."""

    failed = [
        node
        for node in StepTree(directory / "steps").nodes()
        if node.get("status") == "failed"
    ]
    last = failed[-1] if failed else None
    return {
        "failed": len(failed),
        "environment": sum(
            1
            for node in failed
            if _mapping(node.get("metadata")).get("cause") == BATCH_FAILURE_ENVIRONMENT
        ),
        "last_error": identity.public_text(str(last.get("error") or "")) or None if last else None,
    }


def experiment_detail(root: Path, experiment_id: str) -> dict[str, object]:
    directory = resolve_experiment_dir(root, experiment_id)
    detail = {**summarize_experiment(directory), "archived": archive_marker(directory)}
    if detail.get("state") == ExperimentState.UNREADABLE:
        return {
            **detail,
            "params": {},
            "control": None,
            "sessions": [],
            "registered": None,
            "inbox": {"pending_count": 0, "queued_ids": []},
        }
    identity = PublicIdentity(directory)
    records = read_ledger_records(directory)
    research = research_records(records)
    hitl = directory / HITL_DIR_NAME
    params = read_json_object(hitl / PARAMS_NAME, missing_ok=True)
    sessions: list[dict[str, object]] = []
    for planned in identity.sessions:
        key = str(planned["session_key"])
        entry: dict[str, object] = {"key": key, "kind": planned["kind"]}
        record = next((row for row in research if row.get("session_key") == key), None)
        if record is not None:
            entry["record"] = _research_session_view(directory, identity, record)
        if planned["kind"] == "forward":
            replay = dict(_mapping(planned.get("replay")))
            entry["replay"] = replay
            entry["thresholds"] = _verdict_thresholds(params)
        sessions.append(entry)
    raw_status = _mapping(experiment_state(directory).get("status"))
    current = raw_status.get("session_key")
    return {
        **detail,
        "params": _public_params(params),
        "control": identity.public_control(read_control(hitl / CONTROL_NAME).to_record()),
        # The cards claimed for the experiment's live worker: a session attaches some
        # of these and no other. A dead worker's claim holds nothing (a resume
        # writes a fresh one), so it is not shown.
        "gpu_claim": list(read_gpu_claim(hitl) or ()) if detail.get("worker_alive") else [],
        "inbox": inbox_public_view(
            hitl / INBOX_NAME,
            session_key=str(current) if isinstance(current, str) and current else None,
        ),
        "sessions": sessions,
        "registered": _registered_view(identity, records),
        "record": _record_detail(identity, records, params),
        "readouts": _record_readouts(root, forward_record(records)),
        # Two different failures, kept apart: a forward replay attempt that
        # raised before its record -- selected by the pipeline's stage, not by
        # the provider phase the replay reads (``FORWARD_PHASE``) -- and a
        # research Validation that failed.
        "replay_attempts": _failed_attempts(identity, records, Stage.FORWARD),
        "research_validation_failures": _failed_validations(identity, directory),
    }


def registered_strategy_dir(root: Path, experiment_id: str) -> Path:
    """The registered artifact's ``output`` tree, validated inside the experiment."""

    directory = resolve_experiment_dir(root, experiment_id)
    row = registered_record(read_ledger_records(directory))
    value = _mapping(_mapping(row).get("frozen")).get("output_path")
    if not isinstance(value, str) or not value:
        raise KeyError("the experiment has no registered artifact")
    return recorded_path(directory, value)


def _ledger_result_refs(
    directory: Path, records: Sequence[Mapping[str, object]]
) -> Iterator[object]:
    """Every replay result the experiment has recorded: the research session's
    Validations — from the ledger once the session is recorded, from the host
    sidecars while it runs — and, once it is recorded with its readout, the
    forward replay. The forward replay itself has no sidecar, so nothing of it
    is served while it runs."""

    for record in research_records(records):
        for row in record.get("steps") or ():
            if isinstance(row, Mapping):
                yield row.get("validation_result_ref")
    for row in _live_steps(directory):
        yield row.get("validation_result_ref")
    forward = forward_record(records)
    if forward is not None:
        yield forward.get("result_ref")


def ledger_result(root: Path, experiment_id: str, name: str) -> Path:
    """The ``result.json`` of the ledger-named result called ``name``.

    A result directory no record names — a replay still running, or one a
    failed attempt left behind — raises ``KeyError`` exactly like a missing
    one; a named one whose file is not in the experiment directory raises
    ``FileNotFoundError`` (:func:`recorded_path`).
    """

    directory = resolve_experiment_dir(root, experiment_id)
    for reference in _ledger_result_refs(directory, read_ledger_records(directory)):
        if _result_name(reference) == name:
            return recorded_path(directory, str(reference))
    raise KeyError(f"unknown result: {name}")


def result_style(root: Path, experiment_id: str, name: str) -> dict[str, object]:
    """The canonical style sidecar of one ledger-named result.

    A name no ledger record carries raises ``KeyError`` and the route answers
    404. A named result that carries no usable sidecar is an expected state,
    not a failure, so it answers ``available: false`` the way the
    sidecar's own sections report an unavailable figure. The console can then
    tell "nothing to show" from "no such result" without reading an error
    message. A sidecar that is there but cannot be read or parsed answers
    ``available: false`` with reason ``unreadable``, so a load failure is not
    shown as a result that never had one.
    """

    sidecar = ledger_result(root, experiment_id, name).parent / STYLE_ARTIFACT_NAME
    try:
        payload = read_json_object(sidecar, missing_ok=True)
    except (OSError, ValueError):
        return {"available": False, "reason": "unreadable"}
    if (
        payload.get("schema_version") != STYLE_SCHEMA_VERSION
        or payload.get("mode") not in _RESULT_MODES
    ):
        return {"available": False, "reason": "no_style_artifact"}
    payload["available"] = True
    return payload


def result_orders(
    root: Path, experiment_id: str, name: str, *, max_rows: int | None = 500
) -> dict[str, object]:
    """Order stream + aggregate stats for one ledger-named result.

    ``max_rows`` caps the returned rows — ``row_count`` and the stats always
    describe the whole stream — and ``None`` returns every row (the CSV export).
    """
    path = ledger_result(root, experiment_id, name)
    try:
        payload = read_json_object(path)
    except (OSError, ValueError) as exc:
        raise KeyError(f"unreadable result: {name}") from exc
    rows = payload.get("executions") if isinstance(payload, dict) else None
    orders = [dict(item) for item in rows if isinstance(item, dict)] if isinstance(rows, list) else []
    return {
        "result": name,
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


def result_orders_csv(root: Path, experiment_id: str, name: str) -> tuple[str, str]:
    # Uncapped: the export is the escape hatch from the table's row cap.
    payload = result_orders(root, experiment_id, name, max_rows=None)
    fields = (
        "symbol", "action", "quantity", "execute_at", "matched_at", "status",
        "price", "commission", "stamp_duty", "dividend_tax", "realized_pnl", "reason",
    )
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(payload["rows"])
    return f"{experiment_id}__{name}.csv", stream.getvalue()
