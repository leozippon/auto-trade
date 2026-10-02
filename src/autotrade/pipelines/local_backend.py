"""Runnable local daily backend and the deterministic baseline developer.

The production research session, the LLM Agent's, lives in
:mod:`autotrade.pipelines.research_session`.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.replay.stats import (
    PhaseTimer,
    attach_cost_sensitivity,
    attach_sub_window_benchmark,
    finalize_summary_timing,
)
from autotrade.environment.replay.style import (
    benchmark_summary_block,
    replay_style_analysis,
    write_style_rollup,
)
from autotrade.environment.runtime import utc_now_iso
from autotrade.environment.sandbox import SandboxConfig
from autotrade.environment.strategy_loader import validate_strategy_package

from .calendar import replay_window, yyyymmdd
from .config import (
    ArtifactRevision,
    EvaluationBackend,
    EvaluationRequest,
    EvaluationResult,
    ResearchSessionRequest,
    ResearchSessionResult,
    SnapshotBundle,
    StepResult,
    StrategyExperimentConfig,
)
from .experiment import DailyStrategyPipeline
from .skills import _assert_skills_absent_from_formal


class LocalDailySnapshotProvider:
    """Bind every phase to one immutable path selected at worker startup."""

    def __init__(self, daily_path: str | Path) -> None:
        self.daily_path = Path(daily_path).resolve(strict=True)
        if not self.daily_path.is_file():
            raise ValueError("daily_path must be a local Parquet file")

    def prepare(
        self,
        *,
        phase: str,
        start: str,
        end: str,
        decision_time: datetime,
    ) -> SnapshotBundle:
        del decision_time
        if phase not in {"valid", "heldout"}:
            raise ValueError(f"unsupported local snapshot phase: {phase}")
        return SnapshotBundle(
            snapshot_id=f"local_daily_{phase}_{start}_{end}",
            decision_ref=str(self.daily_path),
            replay_ref=str(self.daily_path),
            generation_id="local_daily",
        )

    def prepare_decision(self, *, decision_time: datetime) -> SnapshotBundle:
        return SnapshotBundle(
            snapshot_id=f"local_daily_decision_{decision_time:%Y%m%d}",
            decision_ref=str(self.daily_path),
            replay_ref="",
            generation_id="local_daily",
        )


class LocalDailyEvaluationBackend:
    """Evaluate an immutable strategy revision through DailyStrategyPipeline."""

    def __init__(
        self,
        daily_path: str | Path,
        results_root: str | Path,
        *,
        execution_mode: str,
        benchmark_index: str,
        sandbox: SandboxConfig | None = None,
        executor_factory=None,
    ) -> None:
        if execution_mode not in {"sandbox", "trusted"}:
            raise ValueError("execution_mode must be sandbox or trusted")
        # The arm's benchmark, which the style sidecar records; a plain daily
        # file carries no index series, so nothing is regressed against it.
        self.benchmark_index = benchmark_index
        self.daily_path = Path(daily_path).resolve(strict=True)
        self.results_root = Path(results_root).resolve()
        self.execution_mode = execution_mode
        self.sandbox = sandbox or SandboxConfig()
        self.executor_factory = executor_factory
        self._daily = pd.read_parquet(self.daily_path)
        if "trade_date" not in self._daily.columns:
            raise ValueError("daily Parquet must contain trade_date")
        self._daily = self._daily.copy()
        self._daily["trade_date"] = self._daily["trade_date"].map(yyyymmdd)

    @property
    def trading_days(self) -> list[str]:
        return sorted(set(self._daily["trade_date"].tolist()))

    def frame_between(self, start: str, end: str) -> pd.DataFrame:
        return self._daily[
            (self._daily["trade_date"] >= yyyymmdd(start))
            & (self._daily["trade_date"] <= yyyymmdd(end))
        ].copy()

    def evaluate(
        self,
        request: EvaluationRequest,
        *,
        max_days: int | None = None,
        start_day: str | None = None,
    ) -> EvaluationResult:
        """``start_day``/``max_days`` truncate the replay (``calendar.replay_window``).

        Same contract as the PIT backend: an unofficial smoke run gets the real
        replay path over a short window, never a short window dressed up as a
        full Validation.
        """
        if max_days is not None and max_days <= 0:
            raise ValueError("max_days must be a positive integer")
        if request.mode not in {"valid", "heldout"}:
            raise ValueError(f"unsupported local evaluation mode: {request.mode}")
        strategy_path = Path(request.revision.output_path) / "main.py"
        if not strategy_path.is_file():
            raise FileNotFoundError(
                f"strategy revision has no main.py: {strategy_path}"
            )
        validate_strategy_package(strategy_path)
        started_at = utc_now_iso()
        timer = PhaseTimer()
        with timer.phase("replay_frames"):
            frame = self.frame_between(request.start, request.end)
            replay_start = str(request.start)
            replay_end = str(request.end)
            if max_days is not None or start_day is not None:
                kept = replay_window(
                    set(frame["trade_date"]), start=start_day, max_days=max_days
                )
                frame = frame[frame["trade_date"].isin(set(kept))].copy()
                # A truncated replay must not claim it covered the whole span.
                if kept:
                    replay_start, replay_end = kept[0], kept[-1]
        if frame.empty:
            raise ValueError(
                f"daily replay is empty for {request.start}..{request.end}"
            )
        config = StrategyExperimentConfig(
            strategy_path=strategy_path,
            schedule=request.schedule,
            broker_profile=request.broker_profile,
            execution_mode=self.execution_mode,  # type: ignore[arg-type]
            sandbox=self.sandbox,
        )
        replay = DailyStrategyPipeline(
            config,
            executor_factory=self.executor_factory,
        ).run(frame)
        record = replay.to_record(start=replay_start, end=replay_end)
        with timer.phase("style_analysis"):
            style = replay_style_analysis(
                replay,
                frame,
                replay_dir=None,
                universes=(),
                mode=request.mode,
                benchmark_index=self.benchmark_index,
            )
        summary = record.get("stats")
        if not isinstance(summary, dict):
            raise TypeError("daily replay omitted stats")
        benchmark = benchmark_summary_block(style)
        if benchmark is not None:
            summary["benchmark"] = benchmark
        attach_sub_window_benchmark(summary, style)
        attach_cost_sensitivity(summary, request.broker_profile.slippage_bps)
        finalize_summary_timing(
            summary, started_at=started_at, setup_phases=timer.to_record()
        )
        result_id = f"{request.mode}_{uuid.uuid4().hex}"
        target = self.results_root / result_id / "result.json"
        target.parent.mkdir(parents=True, exist_ok=False)
        target.write_text(
            json.dumps(
                record, ensure_ascii=False, allow_nan=False, default=str, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        write_style_rollup(target.parent, style)
        return EvaluationResult(dict(summary), str(target))


class DeterministicBaselineDeveloper:
    """Replay the template unchanged on the research span and nominate it.

    Nothing is modified and nothing is judged: the arm's one session validates
    the template twice on the full span (the freeze gate counts full-span
    validations, the nominee included, and needs two) and nominates the
    second replay for the gate.
    """

    def __init__(
        self,
        *,
        baseline_strategy: str | Path,
        artifact_store: FilesystemArtifactStore,
        evaluator: EvaluationBackend,
        schedule,
        broker_profile,
        ref_store: AgentRefStore,
    ) -> None:
        self.baseline_strategy = Path(baseline_strategy).resolve(strict=True)
        self.ref_store = ref_store
        self.artifact_store = artifact_store
        self.evaluator = evaluator
        self.schedule = schedule
        self.broker_profile = broker_profile
        if not self.baseline_strategy.is_file():
            raise ValueError("baseline strategy must be a file")
        validate_strategy_package(self.baseline_strategy)
        self.baseline_root = self.artifact_store.root / "baseline_source"
        self.baseline_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.baseline_strategy, self.baseline_root / "main.py")

    def __call__(self, request: ResearchSessionRequest) -> ResearchSessionResult:
        source = self.baseline_root
        _assert_skills_absent_from_formal(source, None)
        # Step ids reach the Agent-facing projections, so they carry the same
        # opaque refs every agent-visible surface uses.
        prefix = (
            f"baseline_{self.ref_store.get_or_create('session', request.session_key)}__"
            f"{self.ref_store.get_or_create('run', request.run_id)}"
        )
        steps: list[StepResult] = []
        for index in (1, 2):
            revision = self.artifact_store.create_revision(source, models_path=None)
            typed_revision = ArtifactRevision(
                str(revision.revision_id),
                Path(revision.output_path),
                Path(revision.models_path) if revision.models_path is not None else None,
            )
            validation = self.evaluator.evaluate(
                request.validation.request(
                    typed_revision, schedule=self.schedule, broker_profile=self.broker_profile
                )
            )
            steps.append(
                StepResult(
                    f"{prefix}__valid_{index:03d}",
                    typed_revision.revision_id,
                    validation,
                    span=request.validation.label,
                    # The host's own deterministic replay: nothing was screened.
                    offline_trials=0,
                )
            )
        return ResearchSessionResult(
            f"deterministic_baseline_{request.run_id}",
            tuple(steps),
            "freeze",
            node_id=steps[-1].step_id,
            finish_reason="deterministic_baseline_replay_no_agent_improvement",
        )


__all__ = [
    "DeterministicBaselineDeveloper",
    "FilesystemArtifactStore",
    "LocalDailyEvaluationBackend",
    "LocalDailySnapshotProvider",
]
