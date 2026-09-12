"""Interactive (human-in-the-loop) session orchestration (docs/pipeline-design.md).

Drives the pipeline's ``run_meta_session`` / ``run_fold`` / ``run_heldout``
primitives with a researcher gate between sessions, per-session directives,
durable pause/stop, and ledger-based resume. This is the only orchestration
entry point; there is no unattended batch driver. The runner treats the
append-only ledger as the source of truth: completed sessions are skipped and
the parent artifact chain is reconstructed from their records.
Integrity-flagged Fold/Held-out rows are not successes: resume refuses them.

All control state lives under ``experiments/<id>/hitl/`` as single-writer JSON
files (atomic replace, no locking needed):

  params.json    creation parameters (written once by the creator; rebuilt into
                 RollingExperimentConfig + backends deterministically on every start)
  control.json   written by the controller (web backend / researcher)
  status.json    written only by the worker (heartbeat, position, live trace)
  schedule.json  written by the worker at startup (planned sessions)

Pausing always lands at a session boundary: the worker finishes the session in
flight, then blocks at the next gate. A deferred restart lands at the same
boundary: the run returns ``status="restart"`` so the worker entrypoint can
re-execute itself on the new code without discarding the session it was
running.
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

from .hitl_state import (
    DevelopmentSession,
    StatusReporter,
    consume_restart_request,
    consume_session_controls,
    read_control,
)
from .ledger import (
    ExperimentLedger,
    FrozenArtifactMutated,
    assert_no_frozen_artifact_mutation,
    is_durable_success_record,
    rerun_absorbed,
)
from .meta_schedule import meta_learning_id


class ExperimentStopped(SessionInterrupt):
    """Raised at a session gate: a durable stop, or a sealed experiment."""


SessionExecutor = Callable[[DevelopmentSession, dict[str, object]], None]

# The record types that complete a runner session, by kind.
_SESSION_RECORD_TYPES = ("fold", "meta_learning", "deployment_adjustment")


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
        after_reveal: bool = False,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if session_max_attempts <= 0:
            raise ValueError("session_max_attempts must be positive")
        self.experiment_id = experiment_id
        self.sessions = sessions
        # The deployment adjustment is by definition the one session that
        # runs after the Held-out sealed the experiment: the reveal stops do
        # not apply to it, everything else (gating, retries, the durable
        # record) does.
        self.after_reveal = after_reveal
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
            assert_no_frozen_artifact_mutation(self.ledger.read())
            for session in self.sessions:
                control = read_control(self.control_path)
                if control.test_revealed and not self.after_reveal:
                    raise ExperimentStopped("the experiment is sealed after out-of-sample reveal")
                rerun_id = control.rerun_sessions.get(session.session_key)
                if session.session_key in completed:
                    if not self._needs_rerun(session, rerun_id):
                        continue
                    reran.append(session.session_key)
                if self._gate():
                    return self._restart_result(ran, reran)
                control = read_control(self.control_path)
                context = {
                    "directive": control.directives.get(session.session_key, ""),
                    "resource_override": control.resource_overrides.get(session.session_key, {}),
                    "sandbox_gpu_count": control.gpu_counts.get(session.session_key),
                    "session_timing": self._session_timing,
                    "progress_hook": self.progress_hook(session),
                    "rerun_id": rerun_id or "",
                    "session_key": session.session_key,
                }
                self._execute_with_retries(session, context)
                # The session's own ledger append is what completes it, and the
                # same append expires its inbox (pipelines/experiment.py).
                self._require_completed_record(session, rerun_id=rerun_id)
                if session.kind == "fold":
                    self._run_post_fold_hook(session)
                consume_session_controls(
                    self.control_path,
                    session.session_key,
                )
                completed.add(session.session_key)
                ran += 1
                self.status.set(completed_sessions=len(completed))
                control = read_control(self.control_path)
                if control.request in ("pause", "stop"):
                    self.status.set(state="paused" if control.request == "pause" else "stopped")
                    return {"status": control.request, "sessions_run": ran, "reran_sessions": reran}
                if control.restart_pending and consume_restart_request(
                    self.control_path
                ):
                    return self._restart_result(ran, reran)
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
    ) -> None:
        last_error: Exception | None = None
        for attempt in range(1, self.session_max_attempts + 1):
            self._begin_session(session)
            try:
                self.execute_session(session, context)
            except (ExperimentStopped, AgentSessionDeadlineExceeded, FrozenArtifactMutated):
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
            else:
                # A retried failure stays visible while its retry is in
                # flight; only a successful attempt clears it, so status.json
                # never carries a stale "(attempt N/M)" after recovery.
                self.status.set(error=None)
                return
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
                and is_durable_success_record(row, record_types=("fold",))
            ),
            None,
        )
        try:
            self.post_fold_hook(record or {})
            self.status.set(analysis_error=None)
        except Exception as exc:  # noqa: BLE001 - analysis is advisory, never fatal
            self.status.set(analysis_error=f"{type(exc).__name__}: {exc}")

    def _restart_result(self, ran: int, reran: list[str]) -> dict[str, object]:
        """Hand a consumed session-boundary restart back to the entrypoint.

        ``launching`` is the state the console already writes for a worker that
        is coming up, and the pid survives the entrypoint's ``execv``, so the
        experiment keeps showing exactly one live worker across the swap."""

        self.status.set(state="launching")
        return {"status": "restart", "sessions_run": ran, "reran_sessions": reran}

    def _gate(self) -> bool:
        """Check the pre-session controls; True asks for a restart instead."""

        control = read_control(self.control_path)
        if control.test_revealed and not self.after_reveal:
            raise ExperimentStopped("the experiment is sealed")
        if control.request == "stop":
            raise ExperimentStopped("stop requested")
        # Before starting the next session, not only after finishing one: a
        # worker that has just finished one must swap code before it spends
        # hours running the old one.
        return control.restart_pending and consume_restart_request(self.control_path)

    def _begin_session(self, session: DevelopmentSession) -> None:
        self._session_started_monotonic = time.monotonic()
        self._session_started_at = datetime.now(UTC).isoformat()
        self.status.set(
            state="running_session",
            session_key=session.session_key,
            session_kind=session.kind,
            session_started_at=self._session_started_at,
            run_id=None,
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

    def _session_timing(self) -> dict[str, float]:
        if self._session_started_monotonic is None:
            return {"run_wall_seconds": 0.0}
        return {
            "run_wall_seconds": round(
                max(0.0, time.monotonic() - self._session_started_monotonic), 1
            )
        }

    def _needs_rerun(self, session: DevelopmentSession, rerun_id: str | None) -> bool:
        """A recorded fold session re-runs when a pending rerun token has not
        been absorbed by its latest ledger record yet."""
        if not rerun_id or session.kind != "fold":
            return False
        return not rerun_absorbed(
            self.ledger.read("fold"), session.session_key, rerun_id
        )

    def _completed_sessions(self) -> set[str]:
        return {
            str(record["session_key"])
            for record in self.ledger.read()
            if is_durable_success_record(record, record_types=_SESSION_RECORD_TYPES)
            and record.get("session_key")
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
            if is_durable_success_record(row, record_types=_SESSION_RECORD_TYPES)
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
