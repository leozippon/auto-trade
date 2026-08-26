"""Interactive (human-in-the-loop) session orchestration (docs/pipeline-design.md).

Drives the same ``run_meta`` / ``run_fold`` / ``run_heldout`` primitives as the
unattended pipeline, but with a researcher gate between sessions, per-session
directives, durable pause/stop, and ledger-based resume. The runner treats the
append-only ledger as the source of truth: completed sessions are skipped and
the parent artifact chain is reconstructed from their records.

All control state lives under ``experiments/<id>/hitl/`` as single-writer JSON
files (atomic replace, no locking needed):

  params.json    creation parameters (written once by the creator; rebuilt into
                 RollingExperimentConfig + backends deterministically on every start)
  control.json   written by the controller (web backend / researcher)
  status.json    written only by the worker (heartbeat, position, live trace)
  schedule.json  written by the worker at startup (planned sessions)

Pausing always lands at a session boundary: the worker finishes the session in
flight, then blocks at the next gate. ``mode="manual"`` additionally requires an
explicit per-session approval before each session starts.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from autotrade.agent.runner import AgentSessionDeadlineExceeded
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import AgentTraceWriter
from autotrade.environment.tools.base import SessionInterrupt

from .agent_inbox import expire_experiment_session_inbox
from .hitl_state import (
    DevelopmentSession,
    StatusReporter,
    consume_session_controls,
    consume_step_approval,
    consume_user_reply,
    read_control,
)
from .ledger import ExperimentLedger
from .meta_schedule import meta_learning_id


class ExperimentStopped(SessionInterrupt):
    """Raised at a gate when the controller requested a durable stop.

    Subclasses SessionInterrupt so a stop issued while a fold is held at a
    step gate re-raises through the Agent runner's tool dispatch instead of
    being swallowed into an error observation."""


SessionExecutor = Callable[[DevelopmentSession, dict[str, object]], dict[str, object] | None]


class InteractiveExperimentRunner:
    def __init__(
        self,
        *,
        experiment_id: str,
        sessions: tuple[DevelopmentSession, ...],
        execute_session: SessionExecutor,
        ledger: ExperimentLedger,
        control_path: str | Path,
        status_path: str | Path,
        ref_store: AgentRefStore | None = None,
        poll_seconds: float = 2.0,
        post_fold_hook: Callable[[dict[str, object]], None] | None = None,
        session_max_attempts: int = 3,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if session_max_attempts <= 0:
            raise ValueError("session_max_attempts must be positive")
        self.experiment_id = experiment_id
        self.sessions = sessions
        self.execute_session = execute_session
        self.post_fold_hook = post_fold_hook
        self.session_max_attempts = session_max_attempts
        self.ledger = ledger
        self.control_path = Path(control_path)
        self.status = StatusReporter(status_path)
        self.ref_store = ref_store
        self.poll_seconds = poll_seconds
        self._session_started_monotonic: float | None = None
        self._session_started_at: str | None = None
        self._researcher_wait_seconds = 0.0
        self._wait_started_monotonic: float | None = None
        self._wait_started_at: str | None = None

    def run(self) -> dict[str, object]:
        completed = self._completed_sessions()
        ran = 0
        reran: list[str] = []
        self.status.start()
        self.status.set(
            completed_sessions=len(completed),
            total_sessions=len(self.sessions) + 1,
        )
        try:
            for session in self.sessions:
                control = read_control(self.control_path)
                if control.test_revealed:
                    raise ExperimentStopped("the experiment is sealed after out-of-sample reveal")
                rerun_id = control.rerun_sessions.get(session.session_key)
                if session.session_key in completed:
                    if not self._needs_rerun(session, rerun_id):
                        continue
                    reran.append(session.session_key)
                self._gate(session)
                control = read_control(self.control_path)
                context = {
                    "directive": control.directives.get(session.session_key, ""),
                    "prompt_override": control.prompt_overrides.get(session.session_key, ""),
                    "resource_override": control.resource_overrides.get(session.session_key, {}),
                    "sandbox_gpu_count": control.gpu_counts.get(session.session_key),
                    "step_gate_hook": self.step_gate_hook(session.session_key),
                    "user_question_hook": self.user_question_hook(session.session_key),
                    "session_timing": self._session_timing,
                    "progress_hook": self.progress_hook(session),
                    # User-side step rollback: a control-plane override replaces
                    # the inherited frozen chain with a validated step-tree node.
                    "parent_override": control.parent_overrides.get(session.session_key, ""),
                    "rerun_id": rerun_id or "",
                    "session_key": session.session_key,
                }
                self._begin_session(session)
                record = self._execute_with_retries(session, context)
                if record is not None:
                    record = {
                        **record,
                        "session_key": session.session_key,
                    }
                    self.ledger.append(record)
                self._require_completed_record(session, rerun_id=rerun_id)
                if session.kind == "fold":
                    self._run_post_fold_hook(session)
                consume_session_controls(
                    self.control_path,
                    session.session_key,
                )
                latest = next(
                    (
                        row
                        for row in reversed(self.ledger.read())
                        if row.get("session_key") == session.session_key
                        and row.get("record_type") in ("fold", "meta_learning")
                    ),
                    None,
                )
                expire_experiment_session_inbox(
                    Path(self.control_path).resolve().parent.parent,
                    session.session_key,
                    expired_by=str((latest or {}).get("run_id") or session.session_key),
                )
                completed.add(session.session_key)
                ran += 1
                self.status.set(completed_sessions=len(completed))
                control = read_control(self.control_path)
                if control.request in ("pause", "stop"):
                    self.status.set(state="paused" if control.request == "pause" else "stopped")
                    return {"status": control.request, "sessions_run": ran, "reran_sessions": reran}
                if control.skip_to_heldout and session.kind == "fold":
                    break
            self.status.set(state="development_complete")
            return {"status": "complete", "sessions_run": ran, "reran_sessions": reran}
        except AgentSessionDeadlineExceeded:
            # Expected control flow: the session already closed gracefully at
            # its deadline (session_end{deadline_exceeded}) and the pipeline
            # layers above record a no-candidate outcome. Never mark the run
            # failed for it; only real errors take the failed state below.
            raise
        except Exception as exc:
            self.status.set(state="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.status.stop()

    def _execute_with_retries(
        self, session: DevelopmentSession, context: dict[str, object]
    ) -> dict[str, object] | None:
        last_error: Exception | None = None
        for attempt in range(1, self.session_max_attempts + 1):
            try:
                return self.execute_session(session, context)
            except (ExperimentStopped, AgentSessionDeadlineExceeded):
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= self.session_max_attempts:
                    break
                self.status.set(
                    environment_stage="session_retry",
                    error=(
                        f"{type(exc).__name__}: {exc} "
                        f"(attempt {attempt}/{self.session_max_attempts})"
                    ),
                )
        assert last_error is not None
        raise last_error

    def _run_post_fold_hook(self, session: DevelopmentSession) -> None:
        """Advisory post-fold strategy analysis: never fatal, always recorded.

        A failure here can never invalidate a completed Fold, so it is written
        to ``status.analysis_error`` for the console instead of raised."""
        if self.post_fold_hook is None:
            return
        self.status.set(environment_stage="analysis")
        record = next(
            (
                row
                for row in reversed(self.ledger.read("fold"))
                if row.get("session_key") == session.session_key
            ),
            None,
        )
        try:
            self.post_fold_hook(record or {})
            self.status.set(analysis_error=None)
        except Exception as exc:  # noqa: BLE001 - analysis is advisory, never fatal
            self.status.set(analysis_error=f"{type(exc).__name__}: {exc}")

    def step_gate_hook(self, session_key: str):
        def wait_for_step(step_index: int, summary: dict[str, object]) -> str:
            while True:
                control = read_control(self.control_path)
                enabled = control.step_gate.get(session_key, control.mode == "step")
                if not enabled:
                    self._resume_session(session_key)
                    return ""
                if control.step_go.get(session_key, 0) >= step_index:
                    self._resume_session(session_key)
                    approved, directive = consume_step_approval(
                        self.control_path,
                        session_key,
                        step_index,
                    )
                    if approved:
                        return directive
                if control.request == "stop":
                    raise ExperimentStopped("stop requested at Step gate")
                self._wait(
                    state="waiting_step_user",
                    session_key=session_key,
                    step_index=step_index,
                    step_summary=summary,
                    run_id=summary.get("run_id"),
                )
                time.sleep(self.poll_seconds)

        return wait_for_step

    def user_question_hook(self, session_key: str):
        question_index = 0

        def ask(question: str, summary: str = "") -> str:
            nonlocal question_index
            question_index += 1
            reply_key = f"{session_key}#q{question_index}"
            while True:
                control = read_control(self.control_path)
                if control.mode == "auto":
                    self._resume_session(session_key)
                    return ""
                if reply_key in control.user_replies:
                    self._resume_session(session_key)
                    replied, reply = consume_user_reply(self.control_path, reply_key)
                    if replied:
                        return reply
                if control.request == "stop":
                    raise ExperimentStopped("stop requested while waiting for a reply")
                self._wait(
                    state="waiting_user_reply",
                    session_key=session_key,
                    question_key=reply_key,
                    question=question,
                    question_summary=summary,
                )
                time.sleep(self.poll_seconds)

        return ask

    def _gate(self, session: DevelopmentSession) -> None:
        while True:
            control = read_control(self.control_path)
            if control.test_revealed:
                raise ExperimentStopped("the experiment is sealed")
            if control.request == "stop":
                raise ExperimentStopped("stop requested")
            if control.mode == "auto" or session.session_key in control.approved_sessions:
                return
            self.status.set(
                state="waiting_user",
                session_key=session.session_key,
                session_kind=session.kind,
                run_id=None,
                session_started_at=None,
                researcher_wait_seconds=0.0,
                wait_started_at=datetime.now(UTC).isoformat(),
                environment_stage=None,
                environment_progress=None,
            )
            time.sleep(self.poll_seconds)

    def _begin_session(self, session: DevelopmentSession) -> None:
        self._session_started_monotonic = time.monotonic()
        self._session_started_at = datetime.now(UTC).isoformat()
        self._researcher_wait_seconds = 0.0
        self._wait_started_monotonic = None
        self._wait_started_at = None
        self.status.set(
            state="running_session",
            session_key=session.session_key,
            session_kind=session.kind,
            session_started_at=self._session_started_at,
            researcher_wait_seconds=0.0,
            wait_started_at=None,
            environment_stage="preparing_session",
            environment_progress=None,
        )

    def progress_hook(self, session: DevelopmentSession):
        """Publish the current host-side phase without inventing a percentage."""

        def publish(stage: str, progress: dict[str, object] | None = None) -> None:
            values: dict[str, object] = {
                "session_key": session.session_key,
                "environment_stage": stage,
                "environment_progress": dict(progress) if progress is not None else None,
            }
            if progress is not None:
                run_id = progress.get("run_id")
                if isinstance(run_id, str) and run_id:
                    values["run_id"] = run_id
                    if self.ref_store is None:
                        raise RuntimeError("interactive trace publishing requires AgentRefStore")
                    raw_session_id = (
                        meta_learning_id(session.epoch_id, session.fold_index)
                        if session.kind == "meta"
                        else session.fold.fold_id
                        if session.fold is not None
                        else session.session_key
                    )
                    identity_namespace = "meta" if session.kind == "meta" else "fold"
                    trace = AgentTraceWriter(
                        self.status.path.parent.parent
                        / "artifacts/traces"
                        / f"{run_id}.jsonl",
                        ids={
                            "experiment_id": self.experiment_id,
                            "epoch_id": session.epoch_id,
                            "fold_id": self.ref_store.get_or_create(
                                identity_namespace, raw_session_id
                            ),
                            "run_id": self.ref_store.get_or_create("run", run_id),
                            "session_kind": "meta_learning" if session.kind == "meta" else session.kind,
                        },
                    )
                    trace.emit(
                        "environment_stage",
                        {
                            "stage": stage,
                            **{
                                key: value
                                for key, value in progress.items()
                                if key != "run_id"
                            },
                        },
                    )
            self.status.set(**values)

        return publish

    def _wait(self, **values: object) -> None:
        if self._wait_started_monotonic is None:
            self._wait_started_monotonic = time.monotonic()
            self._wait_started_at = datetime.now(UTC).isoformat()
        self.status.set(
            **values,
            session_started_at=self._session_started_at,
            researcher_wait_seconds=round(self._researcher_wait_seconds, 3),
            wait_started_at=self._wait_started_at,
        )

    def _resume_session(self, session_key: str) -> None:
        if self._wait_started_monotonic is not None:
            self._researcher_wait_seconds += max(
                0.0,
                time.monotonic() - self._wait_started_monotonic,
            )
        self._wait_started_monotonic = None
        self._wait_started_at = None
        self.status.set(
            state="running_session",
            session_key=session_key,
            session_started_at=self._session_started_at,
            researcher_wait_seconds=round(self._researcher_wait_seconds, 3),
            wait_started_at=None,
        )

    def _session_timing(self) -> dict[str, float]:
        if self._session_started_monotonic is None:
            return {"run_wall_seconds": 0.0, "researcher_wait_seconds": 0.0}
        active_wait = (
            max(0.0, time.monotonic() - self._wait_started_monotonic)
            if self._wait_started_monotonic is not None
            else 0.0
        )
        researcher_wait = self._researcher_wait_seconds + active_wait
        return {
            "run_wall_seconds": round(
                max(
                    0.0,
                    time.monotonic() - self._session_started_monotonic - researcher_wait,
                ),
                1,
            ),
            "researcher_wait_seconds": round(researcher_wait, 1),
        }

    def _needs_rerun(self, session: DevelopmentSession, rerun_id: str | None) -> bool:
        """A recorded fold session re-runs when a pending rerun token has not
        been absorbed by its latest ledger record yet."""
        if not rerun_id or session.kind != "fold":
            return False
        latest = next(
            (
                row
                for row in reversed(self.ledger.read("fold"))
                if row.get("session_key") == session.session_key
            ),
            None,
        )
        return latest is None or str(latest.get("rerun_id") or "") != rerun_id

    def _completed_sessions(self) -> set[str]:
        return {
            str(record["session_key"])
            for record in self.ledger.read()
            if record.get("record_type") in ("fold", "meta_learning") and record.get("session_key")
        }

    def _require_completed_record(
        self,
        session: DevelopmentSession,
        *,
        rerun_id: str | None = None,
    ) -> None:
        records = [
            row
            for row in self.ledger.read()
            if row.get("record_type") in ("fold", "meta_learning")
            and row.get("session_key") == session.session_key
        ]
        if not records:
            raise RuntimeError(
                f"session {session.session_key} returned without a durable success record"
            )
        if rerun_id and str(records[-1].get("rerun_id") or "") != rerun_id:
            raise RuntimeError(
                f"re-run of {session.session_key} did not record its rerun id; refusing to advance"
            )


__all__ = ["ExperimentStopped", "InteractiveExperimentRunner", "SessionExecutor"]
