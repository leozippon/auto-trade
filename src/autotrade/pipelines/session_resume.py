"""What an interrupted research session leaves behind, read back for its next attempt.

The trace is the source for the spend: every ``llm_call``/``tool_call``/
``session_end``/``session_error`` event of an attempt carries the cumulative
``budget_used`` block, and every ``context_compaction`` event carries the
summary that replaced the older history. An attempt whose trace hit its size
cap stopped recording before it ended, so what it last wrote is not its spend;
such a trace is refused rather than resumed from.

The validations an attempt recorded survive in the experiment's step tree,
whose nodes carry only opaque ids, plus a host-only sidecar per node holding
the raw revision id, the span, the summary and the result reference the
Pipeline needs to freeze it. A Step is durable from the moment it is recorded,
while the budget block only rides on the event that settles its tool call, so
an attempt that died inside a batch leaves Steps newer than its last block;
the resume names them by recorded time so their replay-years are added to the
block's (``ResearchSessionRequest.replay_years_spent``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from autotrade.environment.runtime import (
    agent_trace_path,
    redact_host_paths,
    write_json_atomic,
)
from autotrade.environment.step_tree import StepTree

from .config import BudgetUsed, EvaluationResult, SessionResume, StepResult
from .ledger import RESEARCH_STAGE

# Host-only, beside the run markers: never mounted and never Agent-visible.
STEP_SIDECAR_DIR = ".host/steps"


def record_step_sidecar(experiment_dir: str | Path, step: StepResult) -> Path:
    """Persist what the step tree cannot carry about one recorded Validation."""

    path = Path(experiment_dir) / STEP_SIDECAR_DIR / f"{step.step_id}.json"
    path.parent.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.mkdir(exist_ok=True, mode=0o700)
    write_json_atomic(
        path,
        {
            "step_id": step.step_id,
            "revision_id": step.revision_id,
            "span": step.span,
            "summary": dict(step.validation.summary),
            "result_ref": step.validation.result_ref,
        },
    )
    return path


def load_recorded_steps(experiment_dir: str | Path) -> tuple[StepResult, ...]:
    """The complete Validations the arm's research session recorded so far,
    in tree order, rebuilt from the published tree and the sidecars.

    A complete node without its sidecar is a validation the host cannot
    account for, so it fails the attempt instead of silently narrowing the
    trial pool the freeze gate deflates over.
    """

    directory = Path(experiment_dir)
    steps: list[StepResult] = []
    for node in _recorded_nodes(directory):
        node_id = str(node["node_id"])
        sidecar = directory / STEP_SIDECAR_DIR / f"{node_id}.json"
        if not sidecar.is_file():
            raise RuntimeError(
                f"step node {node_id} is recorded without its host sidecar; the "
                "session cannot resume with a validation it cannot account for"
            )
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        steps.append(
            StepResult(
                str(record["step_id"]),
                str(record["revision_id"]),
                EvaluationResult(dict(record["summary"]), str(record["result_ref"])),
                span=str(record["span"]),
            )
        )
    return tuple(steps)


def _recorded_nodes(experiment_dir: Path) -> list[Mapping[str, object]]:
    """The complete research Validations of the arm's published step tree."""

    tree_root = experiment_dir / "steps"
    if not (tree_root / "tree.json").is_file():
        return []
    return [
        node
        for node in StepTree(tree_root).nodes()
        if node.get("epoch_id") == RESEARCH_STAGE and node.get("complete_validation")
    ]


def resume_state(
    experiment_dir: str | Path, records: Sequence[Mapping[str, object]]
) -> SessionResume | None:
    """How the research session's earlier attempts ended, or None on a fresh one.

    Reads every failed research attempt's trace in ledger order: the last
    ``budget_used`` block is the cumulative spend (each attempt seeds its
    counters from the amounts before it), the last successful compaction's
    summary is the checkpoint, and the last event's time is where it stopped.
    The recorded Validations created after that last block are the ones its
    replay-years never counted.

    A trace that reached ``TRACE_MAX_BYTES`` stopped at its marker while the
    attempt kept spending, so seeding from it would let the arm run past its
    own budget; that is refused here instead.
    """

    attempts = [
        record
        for record in records
        if record.get("record_type") == "attempt_failed"
        and record.get("phase") == RESEARCH_STAGE
    ]
    if not attempts:
        return None
    budget = BudgetUsed()
    budget_at: datetime | None = None
    summary: str | None = None
    interrupted_at = ""
    transcripts: list[str] = []
    for attempt in attempts:
        run_id = str(attempt.get("run_id") or "")
        path = agent_trace_path(Path(experiment_dir) / "artifacts", run_id)
        run_ref = ""
        for event in _trace_events(path):
            if event.get("event_type") == "trace_limit_reached":
                raise RuntimeError(
                    f"attempt trace {run_id} reached its size cap; the spend it "
                    "records is not what the attempt actually used, so the "
                    "session cannot resume from it"
                )
            run_ref = str(event.get("run_id") or run_ref)
            used = event.get("budget_used")
            if isinstance(used, Mapping):
                budget = BudgetUsed.from_record(used)
                budget_at = _moment(event.get("ts"), f"a budget event of trace {run_id}")
            if (
                event.get("event_type") == "context_compaction"
                and event.get("status") == "ok"
                and isinstance(event.get("summary"), str)
                and event["summary"].strip()
            ):
                summary = str(event["summary"])
            interrupted_at = str(event.get("ts") or interrupted_at)
        if run_ref:
            transcripts.append(f"{run_ref}.txt")
    last = attempts[-1]
    return SessionResume(
        attempt=len(attempts) + 1,
        interrupted_at=interrupted_at or str(last.get("recorded_at") or ""),
        # The note this becomes is Agent-visible; the ledger keeps the raw text.
        error=redact_host_paths(str(last.get("error") or "")),
        compaction_summary=summary,
        budget_used=budget,
        unseen_step_ids=tuple(
            str(node["node_id"])
            for node in _recorded_nodes(Path(experiment_dir))
            if budget_at is None
            or _moment(node.get("created_at"), f"step node {node['node_id']}") > budget_at
        ),
        transcripts=tuple(transcripts),
    )


def _moment(value: object, what: str) -> datetime:
    """The aware time a trace event or a step node was written."""

    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        moment = None
    if moment is None or moment.tzinfo is None:
        raise RuntimeError(
            f"{what} carries no aware ISO time ({value!r}); the resumed replay-year "
            "counter cannot tell which recorded Validations it already counts"
        )
    return moment


def _trace_events(path: Path) -> list[dict[str, object]]:
    """The events of one attempt's trace; a run that never wrote one has none."""

    if not path.is_file():
        return []
    events: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # A torn last line of a killed run is not evidence of anything.
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


__all__ = [
    "STEP_SIDECAR_DIR",
    "load_recorded_steps",
    "record_step_sidecar",
    "resume_state",
]
