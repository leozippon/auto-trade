"""Experiment discovery and read models for the research console.

Everything here is read-only over ``experiments/<id>/``: the append-only
ledger, the hitl/ control-plane files and the result artifacts the ledger
names. Unparseable experiments still appear in listings so they can be
deleted; they carry an error note instead of metrics.

An arm researches in sessions, freezes at most once and is then replayed once
over forward and Held-out (docs/pipeline-design.md). The replay stays sealed by
construction: the console serves a result only when a ledger record names it,
and the forward record that names the replay is written together with its
verdict, so nothing of the replay is readable while it runs.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

from autotrade.agent.runner import DEADLINE_GRACE_EXHAUSTED, LLM_CALL_BUDGET_EXHAUSTED
from autotrade.environment.replay.style import STYLE_ARTIFACT_NAME, STYLE_SCHEMA_VERSION
from autotrade.pipelines.agent_inbox import INBOX_NAME, inbox_public_view
from autotrade.pipelines.calendar import FULL_SPAN
from autotrade.pipelines.config import acceptance_for
from autotrade.pipelines.experiment import freeze_gate_for, neutralized
from autotrade.pipelines.hitl_state import (
    CONTROL_NAME,
    HITL_DIR_NAME,
    PARAMS_NAME,
    STATUS_NAME,
    WEB_CREATE_DEFAULTS,
    read_control,
    read_json,
    read_status,
    status_pid_alive,
)
from autotrade.pipelines.ledger import (
    RESEARCH_SESSION_KEY,
    ExperimentLedger,
    experiment_verdict,
    forward_record,
    frozen_record,
    paper_candidate,
    research_records,
)
from autotrade.pipelines.pit_views_seed import FORWARD_PHASE, RESEARCH_PHASE
from autotrade.pipelines.session_resume import STEP_SIDECAR_DIR
from autotrade.pipelines.skills import latest_skills_snapshot
from autotrade.pipelines.worker import _ALLOWED_PARAMS

from .public_identity import PublicIdentity
from .traces import resolve_trace_path, trace_stats

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
ACTIVE_STATES = ("launching", "initializing", "running_session", "paused")
# Manager-written stub state between spawn and the worker's first status write
# (interpreter start + imports take seconds). Stale = the worker never came up.
LAUNCH_GRACE_SECONDS = 180.0
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
# Replay result modes a style sidecar may carry: research and forward replays.
_RESULT_MODES = frozenset({RESEARCH_PHASE, FORWARD_PHASE})
# Arm stages, in order: researching, frozen with the forward replay pending or
# running (sealed), verdict recorded.
STAGES = ("research", "forward", "verdict")
# How an arm ended, in the order the homepage lists them. The console says an
# ending in this one vocabulary wherever it says it at all, so the ledger's own
# words (``graduated``/``discarded``/``no_deliverable``, the session outcomes)
# never have to be read twice into the same three blurred endings.
ENDING_STATES = ("graduated", "rejected", "no_edge", "budget_exhausted", "broken")
_ENDING_ORDER = {state: index for index, state in enumerate(ENDING_STATES)}
# Which graduation criterion each failed verdict token is, as
# docs/pipeline-design.md §3.2 numbers them: a rejected arm's reason names the
# criteria that refused it. F6 and H4 each cover two recorded conditions.
_CRITERION_CODES = {
    "forward_strategy_error": "F1",
    "forward_lower_bound_not_positive": "F2",
    "forward_recency_negative": "F3",
    "forward_max_drawdown_exceeded": "F4",
    "forward_active_drawdown_exceeded": "F4",
    "forward_not_positive_at_cost_stress": "F5",
    "forward_too_few_round_trips": "F6",
    "forward_exposure_below_floor": "F6",
    "forward_tracking_error_above_cap": "F7",
    "forward_beta_outside_band": "F7",
    "heldout_strategy_error": "H1",
    "heldout_excess_below_tolerance": "H2",
    "heldout_max_drawdown_exceeded": "H3",
    "heldout_active_drawdown_exceeded": "H3",
    "heldout_exposure_below_floor": "H4",
}
# The two budgets a research session can run out of, named as the reason line.
_BUDGET_EXHAUSTED = {
    DEADLINE_GRACE_EXHAUSTED: "研究时长用尽",
    LLM_CALL_BUDGET_EXHAUSTED: "模型调用次数用尽",
}
# Sentence enders of an Agent-authored reason, whose first sentence is the line
# the console shows.
_SENTENCE_END = re.compile(r"[。！？!?\n]")


class UnsupportedParamsError(ValueError):
    """params.json names parameters the worker no longer accepts (a Fold-era
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
    else:
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


def arm_stage(records: Sequence[Mapping[str, object]]) -> str:
    """Which of :data:`STAGES` the arm is in, read off the ledger alone."""

    if experiment_verdict(records) is not None:
        return "verdict"
    if frozen_record(records) is not None:
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
    """The recorded failed attempts, of one phase or of the whole arm."""

    return [
        row
        for row in records
        if row.get("record_type") == "attempt_failed"
        and (phase is None or row.get("phase") == phase)
    ]


def arm_ending(
    identity: PublicIdentity,
    records: Sequence[Mapping[str, object]],
    state: Mapping[str, object],
) -> dict[str, str] | None:
    """How the arm ended — one of :data:`ENDING_STATES` and a one-line reason —
    or ``None`` while it can still run.

    Every ending the console shows comes from here, so no page classifies one
    for itself. A worker the host or the environment broke ended the process
    rather than the research, so its state is read before the ledger. Otherwise
    the verdict decides: a graduate is named by the two numbers that carried
    it, a replay that refused one by the criteria it failed, and an arm that
    never reached a replay by how its research session ended — the Agent's own
    ``no_edge``, an exhausted budget, or a nomination the freeze gate refused.
    """

    if str(state.get("state") or "") == "failed":
        error = str(_mapping(state.get("status")).get("error") or "")
        if not error:
            failures = _attempt_failures(records)
            error = str(failures[-1].get("error") or "") if failures else ""
        line = identity.public_text(error).splitlines()
        return {"state": "broken", "reason": line[0] if line else ""}
    verdict = experiment_verdict(records)
    if verdict is None:
        return None
    if verdict["status"] == "graduated":
        slices = _mapping(_mapping(forward_record(records)).get("slices"))
        forward = _mapping(slices.get("forward"))
        heldout = _mapping(slices.get("heldout"))
        return {
            "state": "graduated",
            "reason": (
                f"前推 80% 下界 {_percent(forward.get('lower_bound'))}"
                f" · Held-out 超额 {_percent(heldout.get('neutralized_excess'))}"
            ),
        }
    if verdict["status"] == "discarded":
        codes = dict.fromkeys(
            _CRITERION_CODES.get(str(token), str(token))
            for token in verdict.get("reasons") or ()
        )
        return {"state": "rejected", "reason": " · ".join(codes)}
    session = _mapping(
        next((row for row in research_records(records) if row.get("arm_end")), None)
    )
    if session.get("outcome") == "no_edge":
        reason = identity.public_text(str(session.get("reason") or ""))
        return {"state": "no_edge", "reason": _reason_line(reason)}
    if session.get("outcome") == "deadline":
        finish = str(session.get("finish_reason") or "")
        return {"state": "budget_exhausted", "reason": _BUDGET_EXHAUSTED.get(finish, finish)}
    return {"state": "rejected", "reason": "提名未通过冻结门"}


def _result_name(reference: object) -> str | None:
    """Public name of one replay result: its result directory's name."""

    if not isinstance(reference, str) or not reference:
        return None
    path = Path(reference)
    return path.parent.name if path.name == "result.json" else path.name


def _forward_view(
    identity: PublicIdentity, record: Mapping[str, object] | None
) -> dict[str, object] | None:
    """The forward record: replay span, slice statistics and verdict.

    ``None`` until the record exists, which is also when the verdict does.
    """

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
        "verdict": dict(_mapping(record.get("verdict"))),
    }


def _paper_candidate_view(
    directory: Path, records: list[dict[str, object]]
) -> dict[str, object] | None:
    """The Paper candidate with the command that creates a book
    (``scripts/paper/run_paper.py init``, run from the repository root)."""
    candidate = paper_candidate(records)
    if candidate is None:
        return None
    return {
        "artifact_id": candidate["artifact_id"],
        "command": (
            "python scripts/paper/run_paper.py init "
            f"--experiment {directory.name} --artifact {candidate['artifact_id']}"
        ),
    }


def summarize_experiment(directory: Path) -> dict[str, object]:
    directory = Path(directory)
    summary: dict[str, object] = {"experiment_id": directory.name}
    try:
        identity = PublicIdentity(directory)
        state = experiment_state(directory)
        records = read_ledger_records(directory)
        skills = latest_skills_snapshot(records, experiment_dir=directory).stats
        params = read_json(directory / HITL_DIR_NAME / PARAMS_NAME)
        _require_supported_params(params)
        raw_status = state.get("status")
        status = identity.public_status(raw_status) if isinstance(raw_status, Mapping) else {}
        summary.update(
            identity.public_record({key: value for key, value in state.items() if key != "status"})
        )
        if raw_status is not None:
            summary["status"] = status
        best = _research_best(directory, records)
        summary.update(
            {
                "created_at": _created_at(directory, params),
                "skills": {"count": skills.count, "files": skills.files, "bytes": skills.bytes},
                "stage": arm_stage(records),
                "frozen_session": _frozen_session(records),
                "research_best": best,
                "research_result": _research_result(records, best),
                "research_outcome": next(
                    (row.get("outcome") for row in research_records(records)), None
                ),
                "budget": _budget_totals(params),
                "budget_used": _budget_used(directory, records, raw_status),
                "verdict": experiment_verdict(records),
                "ending": arm_ending(identity, records, state),
                "forward": _forward_view(identity, forward_record(records)),
                "paper_candidate": _paper_candidate_view(directory, records),
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


def _budget_totals(params: Mapping[str, object]) -> dict[str, object]:
    """The arm's research budget, keyed like the trace's ``budget_used`` block.

    The create form persists only what differs from the defaults, so the
    effective ceilings are the worker defaults overlaid with ``params.json``.
    """

    effective = {**WEB_CREATE_DEFAULTS, **params}
    minutes = _number(effective.get("max_research_minutes"))
    return {
        "inference_seconds": minutes * 60.0 if minutes is not None else None,
        "llm_calls": _number(effective.get("max_llm_calls")),
        "replay_years": _number(effective.get("max_replay_years")),
        "null_controls": _number(effective.get("max_null_controls")),
    }


def _budget_used(
    directory: Path,
    records: Sequence[Mapping[str, object]],
    raw_status: object,
) -> dict[str, object] | None:
    """What the research session has spent: the ledger's block once it is
    recorded, else the last block its live trace carries, else nothing."""

    for record in research_records(records):
        block = record.get("budget_used")
        if isinstance(block, Mapping):
            return dict(block)
    status = _mapping(raw_status)
    if status.get("session_key") != RESEARCH_SESSION_KEY or not status_pid_alive(status):
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
    """Listing order: the arms that can still run, then the ended ones in
    :data:`ENDING_STATES` order."""

    ending = _mapping(row.get("ending"))
    if not ending:
        return 0
    return 1 + _ENDING_ORDER.get(str(ending.get("state")), len(ENDING_STATES))


def list_experiments(root: Path) -> list[dict[str, object]]:
    """Every experiment, newest first inside each :func:`_ending_rank` group."""

    root = Path(root)
    if not root.is_dir():
        return []
    rows = [
        summarize_experiment(path)
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    rows.sort(key=_ending_rank)
    return rows


def best_experiment(rows: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    """The homepage's best experiment: the graduate with the highest forward
    bootstrap lower bound.

    Only a graduate is offered. Every other ending is an arm the pipeline
    refused, and an arm still researching has no out-of-sample evidence at all,
    so with no graduate the homepage names no best experiment rather than
    crowning the least bad ending. Research-period numbers never rank. Ties
    keep the listing order.
    """

    ranked: list[tuple[float, int, Mapping[str, object]]] = []
    for position, row in enumerate(rows):
        if _mapping(row.get("ending")).get("state") != "graduated":
            continue
        forward = _mapping(_mapping(_mapping(row.get("forward")).get("slices")).get("forward"))
        bound = _number(forward.get("lower_bound"))
        if bound is not None:
            ranked.append((-bound, position, row))
    if not ranked:
        return None
    _bound, _position, row = min(ranked, key=lambda item: (item[0], item[1]))
    return {"experiment_id": row["experiment_id"]}


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


def _step_view(row: Mapping[str, object]) -> dict[str, object]:
    """One recorded Validation: its span, headline replay metrics and the
    neutralised figures the freeze gate counts."""

    summary = _mapping(row.get("summary"))
    neutral = _mapping(row.get("neutralized"))
    return {
        "step_id": row.get("step_id"),
        "span": row.get("span"),
        "total_return": _number(summary.get("total_return")),
        "sharpe": _number(summary.get("sharpe")),
        "max_drawdown": _number(summary.get("max_drawdown")),
        "neutralized_excess": _number(neutral.get("neutralized_excess")),
        "information_ratio": _number(neutral.get("information_ratio")),
        # ``active`` when the figures are of the node's return minus its
        # zero-skill panel; absent on a node recorded before panels existed.
        "series": neutral.get("series"),
    }


def _best_candidate(
    earlier: Sequence[Mapping[str, object]], steps: Sequence[Mapping[str, object]]
) -> dict[str, object] | None:
    """The session's full-span Validation with the highest neutralised IR, and
    the deflated Sharpe probability the freeze gate would give it, with the
    trials it was deflated against.

    Both come from the pipeline's own gate over the arm as it stood when the
    session ended (``experiment.freeze_gate_for``), so they are the numbers a
    nomination of this node would have been judged on. ``None`` when the
    session ran no measurable full-span Validation.
    """

    full = [
        row
        for row in steps
        if row.get("span") == FULL_SPAN
        # A control can never be nominated, so it is never the arm's best.
        and row.get("control") is not True
        and _number(_mapping(row.get("neutralized")).get("information_ratio")) is not None
    ]
    if not full:
        return None
    best = max(full, key=lambda row: float(row["neutralized"]["information_ratio"]))  # type: ignore[index]
    try:
        gate = freeze_gate_for(earlier, steps, best)
    except (OSError, ValueError):
        gate = {}
    dsr = _mapping(gate.get("deflated_sharpe"))
    return {
        **_step_view(best),
        "result": _result_name(best.get("validation_result_ref")),
        "deflated_sharpe_probability": _number(dsr.get("deflated_sharpe_probability")),
        "trials": dsr.get("trials"),
    }


def _frozen_session(records: Sequence[Mapping[str, object]]) -> str | None:
    """The research session that froze the arm's artifact, or ``None``.

    The listing's one fact about the freeze: the experiment card draws the
    freeze step from it instead of inferring a freeze from the stage and the
    verdict.
    """

    row = frozen_record(records)
    return str(row["session_key"]) if row is not None else None


def _research_best(
    directory: Path, records: Sequence[Mapping[str, object]]
) -> dict[str, object] | None:
    """The arm's best full-span candidate so far, for the listing.

    The recorded research session's, through the same :func:`_best_candidate`
    the experiment page reads, so both surfaces name one number; while the
    session still runs, the same rule over the Validations it has recorded so
    far (:func:`_live_steps`), so the card, the tiles and the curve follow the
    research as it happens and agree with the record once it lands. ``None``
    while no measurable full-span Validation exists — the card then shows no
    research evidence rather than a dash.
    """

    research = research_records(records)
    for position in reversed(range(len(research))):
        steps = [row for row in research[position].get("steps") or () if isinstance(row, Mapping)]
        best = _best_candidate(research[:position], steps)
        if best is not None:
            return {"session_key": research[position].get("session_key"), **best}
    if research:
        return None
    best = _live_best(directory)
    return {"session_key": RESEARCH_SESSION_KEY, **best} if best is not None else None


# A recorded Validation is immutable: its sidecar and its result's style file
# are read once per node and kept for the process lifetime, and the best
# candidate (with the freeze gate the Pipeline computes for it) is kept per
# node set, so a listing poll of a running arm re-reads nothing until the
# session records another node.
_LIVE_STEP_CACHE: dict[tuple[str, str], dict[str, object]] = {}
_LIVE_BEST_CACHE: dict[tuple[str, tuple[str, ...]], dict[str, object] | None] = {}


def _live_steps(directory: Path) -> list[dict[str, object]]:
    """The research session's Validations recorded so far, from the host
    sidecars ``session_resume.record_step_sidecar`` writes, shaped like the
    ledger's step rows with the neutralised figures the Pipeline records
    (``experiment.neutralized``), so the same selection applies to both."""

    root = directory / STEP_SIDECAR_DIR
    if not root.is_dir():
        return []
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*.json")):
        key = (str(directory), path.stem)
        row = _LIVE_STEP_CACHE.get(key)
        if row is None:
            record = read_json(path)
            reference = str(record.get("result_ref") or "")
            try:
                figures = neutralized(reference)
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
                "neutralized": figures,
                # What the freeze gate's trial family reads; absent (no
                # control, undeclared) on a sidecar written before them.
                **{
                    name: record.get(name)
                    for name in ("control", "batch_id", "offline_trials")
                },
            }
            _LIVE_STEP_CACHE[key] = row
        rows.append(row)
    return rows


def _live_best(directory: Path) -> dict[str, object] | None:
    steps = _live_steps(directory)
    key = (str(directory), tuple(sorted(str(row.get("step_id")) for row in steps)))
    if key not in _LIVE_BEST_CACHE:
        _LIVE_BEST_CACHE[key] = _best_candidate([], steps)
    return _LIVE_BEST_CACHE[key]


def _research_result(
    records: Sequence[Mapping[str, object]], best: Mapping[str, object] | None
) -> str | None:
    """The research-period result the console draws: the frozen artifact's own
    validation once the arm froze, else the best full-span candidate's."""

    row = frozen_record(records)
    if row is not None:
        return _result_name(_mapping(row.get("frozen")).get("research_result_ref"))
    return str(best["result"]) if best and best.get("result") else None


def _months_between(start: object, end: object) -> int | None:
    """Calendar months from ``start``'s month through ``end``'s, YYYYMMDD."""

    try:
        first, last = str(start), str(end)
        return (int(last[:4]) * 12 + int(last[4:6])) - (int(first[:4]) * 12 + int(first[4:6])) + 1
    except ValueError:
        return None


def _verdict_thresholds(
    params: Mapping[str, object], replay: Mapping[str, object]
) -> dict[str, object]:
    """The graduation thresholds the replay will be held to, from the arm's
    own stamped rules, so the console lists the criteria before the replay
    has run. The forward record's own block replaces them once it exists."""

    effective = {**WEB_CREATE_DEFAULTS, **params}
    months = _months_between(replay.get("start"), replay.get("forward_end"))
    rules = acceptance_for(effective)
    # An arm created since the five limits reached the create form states all
    # of them; an earlier arm's params.json carries its equity drawdown limit
    # alone, and that is all it is held to. Statistical bars always resolve
    # (missing keys take today's defaults) so the listing matches the worker.
    # Freeze-gate knobs stay off this preview: they are not forward criteria.
    limits: dict[str, object] = {"max_drawdown": _number(effective.get("max_drawdown"))}
    if "active_max_drawdown" in params:
        record = rules.to_record()
        limits = {
            key: record[key]
            for key in (
                "max_drawdown",
                "active_max_drawdown",
                "tracking_error_cap",
                "beta_min",
                "beta_max",
            )
        }
    return {
        "forward_confidence": rules.forward_confidence,
        "recency_months": rules.recency_months,
        **limits,
        "cost_stress_multiplier": rules.cost_stress_multiplier,
        "min_round_trips": (
            rules.min_round_trips_per_month * months if months else None
        ),
        "min_mean_gross": rules.min_mean_gross,
        "heldout_tolerance_z": rules.heldout_tolerance_z,
    }


def _research_session_view(
    directory: Path,
    identity: PublicIdentity,
    earlier: Sequence[Mapping[str, object]],
    record: Mapping[str, object],
) -> dict[str, object]:
    steps = [row for row in record.get("steps") or () if isinstance(row, Mapping)]
    gate = _mapping(record.get("freeze_gate"))
    run_id = str(record.get("run_id") or "")
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
        "trials_to_date": record.get("trials_to_date"),
        "nominated_step_id": record.get("nominated_step_id"),
        "freeze_gate": (
            {
                "passed": bool(gate.get("passed")),
                "reasons": [str(reason) for reason in gate.get("reasons") or ()],
                "deflated_sharpe_probability": _number(
                    _mapping(gate.get("deflated_sharpe")).get("deflated_sharpe_probability")
                ),
                "full_span_validations": _number(gate.get("full_span_validations")),
                "series": gate.get("series"),
                "information_ratio": _number(gate.get("information_ratio")),
                "positive_years": _number(gate.get("positive_years")),
                "active_max_drawdown": _number(gate.get("active_max_drawdown")),
                "mandate": {
                    key: _number(value)
                    for key, value in _mapping(gate.get("mandate")).items()
                },
                # The limits this nomination was actually judged against, as
                # its own record states them, so a later change to any of them
                # does not restate the arm's gate and a limit the record does
                # not carry is not drawn.
                "thresholds": {
                    key: _number(value)
                    for key, value in _mapping(gate.get("thresholds")).items()
                },
            }
            if gate
            else None
        ),
        "froze": bool(record.get("frozen")),
        "arm_end": identity.public_record(record["arm_end"])  # type: ignore[arg-type]
        if isinstance(record.get("arm_end"), Mapping)
        else None,
        "validations": [_step_view(row) for row in steps],
        "best": _best_candidate(earlier, steps),
        "attempts": record.get("attempts"),
        "budget_used": _mapping(record.get("budget_used")) or None,
    }


def _frozen_view(
    identity: PublicIdentity, records: Sequence[Mapping[str, object]]
) -> dict[str, object] | None:
    """The frozen artifact and the research statistics it was frozen on."""

    row = frozen_record(records)
    if row is None:
        return None
    block = _mapping(row.get("frozen"))
    dsr = _mapping(block.get("deflated_sharpe"))
    fit = _mapping(block.get("fit_plan"))
    return {
        "session_key": row.get("session_key"),
        "strategy_ref": identity.strategy_ref(block["artifact_id"]),
        "source_step_id": block.get("source_step_id"),
        "result": _result_name(block.get("research_result_ref")),
        "days": block.get("days"),
        "neutralized_excess": _number(block.get("neutralized_excess")),
        "tracking_error": _number(block.get("tracking_error")),
        "information_ratio": _number(block.get("information_ratio")),
        "deflated_sharpe_probability": _number(dsr.get("deflated_sharpe_probability")),
        "trials": dsr.get("trials"),
        "sharpe_star": _number(dsr.get("sharpe_star")),
        "full_span_validations": block.get("full_span_validations"),
        "forward_mde": _number(block.get("forward_mde")),
        "null_percentile": _number(_mapping(block.get("null_control")).get("excess_percentile")),
        "fit": bool(fit.get("fit")),
        "refit_period": fit.get("refit_period"),
        "blocks": [dict(item) for item in block.get("blocks") or () if isinstance(item, Mapping)],
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


def experiment_detail(root: Path, experiment_id: str) -> dict[str, object]:
    directory = resolve_experiment_dir(root, experiment_id)
    detail = summarize_experiment(directory)
    if detail.get("state") == "unreadable":
        return {
            **detail,
            "params": {},
            "control": None,
            "sessions": [],
            "frozen": None,
            "inbox": {"pending_count": 0, "queued_ids": []},
        }
    identity = PublicIdentity(directory)
    records = read_ledger_records(directory)
    research = research_records(records)
    hitl = directory / HITL_DIR_NAME
    params = read_json(hitl / PARAMS_NAME)
    sessions: list[dict[str, object]] = []
    for planned in identity.sessions:
        key = str(planned["session_key"])
        entry: dict[str, object] = {"key": key, "kind": planned["kind"]}
        position = next(
            (index for index, row in enumerate(research) if row.get("session_key") == key),
            None,
        )
        if position is not None:
            entry["record"] = _research_session_view(
                directory, identity, research[:position], research[position]
            )
        if planned["kind"] == "forward":
            replay = dict(_mapping(planned.get("replay")))
            entry["replay"] = replay
            entry["thresholds"] = _verdict_thresholds(params, replay)
        sessions.append(entry)
    raw_status = _mapping(experiment_state(directory).get("status"))
    current = raw_status.get("session_key")
    return {
        **detail,
        "params": _public_params(params),
        "control": identity.public_control(read_control(hitl / CONTROL_NAME).to_record()),
        "inbox": inbox_public_view(
            hitl / INBOX_NAME,
            session_key=str(current) if isinstance(current, str) and current else None,
        ),
        "sessions": sessions,
        "frozen": _frozen_view(identity, records),
        "replay_attempts": _failed_attempts(identity, records, FORWARD_PHASE),
    }


def frozen_strategy_dir(root: Path, experiment_id: str) -> Path:
    """The frozen artifact's ``output`` tree, validated inside the experiment."""

    directory = resolve_experiment_dir(root, experiment_id)
    row = frozen_record(read_ledger_records(directory))
    value = _mapping(_mapping(row).get("frozen")).get("output_path")
    if not isinstance(value, str) or not value:
        raise KeyError("the experiment has no frozen artifact")
    strategy_dir = (directory / value).resolve()
    if not strategy_dir.is_relative_to(directory) or not strategy_dir.is_dir():
        raise KeyError("the experiment has no frozen artifact on disk")
    return strategy_dir


def _ledger_result_refs(
    directory: Path, records: Sequence[Mapping[str, object]]
) -> Iterator[object]:
    """Every replay result the arm has recorded: the research session's
    Validations — from the ledger once the session is recorded, from the host
    sidecars while it runs — and, once it is recorded with its verdict, the
    forward replay. The replay itself has no sidecar, so it stays sealed."""

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
    failed attempt left behind — answers exactly like a missing one.
    """

    directory = resolve_experiment_dir(root, experiment_id)
    for reference in _ledger_result_refs(directory, read_ledger_records(directory)):
        if _result_name(reference) != name:
            continue
        path = (directory / str(reference)).resolve()
        if path.is_dir():
            path = path / "result.json"
        if path.is_relative_to(directory) and path.is_file():
            return path
    raise KeyError(f"unknown result: {name}")


def result_style(root: Path, experiment_id: str, name: str) -> dict[str, object]:
    """The canonical style sidecar of one ledger-named result.

    A name no ledger record carries raises ``KeyError`` and the route answers
    404. A named result that carries no usable sidecar — one recorded before
    style analysis existed, or written under an older schema — is an expected
    state, not a failure, so it answers ``available: false`` the way the
    sidecar's own sections report an unavailable figure. The console can then
    tell "nothing to show" from "no such result" without reading an error
    message. A sidecar that is there but cannot be read or parsed answers
    ``available: false`` with reason ``unreadable``, so a load failure is not
    shown as a result that never had one.
    """

    sidecar = ledger_result(root, experiment_id, name).parent / STYLE_ARTIFACT_NAME
    try:
        payload = read_json(sidecar)
    except (OSError, TypeError, ValueError):
        return {"available": False, "reason": "unreadable"}
    if payload.get("schema_version") != STYLE_SCHEMA_VERSION or payload.get("mode") not in _RESULT_MODES:
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
        payload = json.loads(path.read_text(encoding="utf-8"))
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
        "price", "commission", "stamp_duty", "realized_pnl", "reason",
    )
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(payload["rows"])
    return f"{experiment_id}__{name}.csv", stream.getvalue()
