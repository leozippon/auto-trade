"""What both evaluation backends share: the strategy replay itself, the
checks a request passes before it replays, the summary a finished replay
returns, and the seed of every random baseline.

The backends (``local_backend``, ``pit_backend``) build on this module and
never on the experiment orchestration (``experiment``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd

from autotrade.environment.delisting import DelistingTable
from autotrade.environment.executor import (
    DockerStrategyExecutor,
    StrategyExecutor,
    TrustedStrategyExecutor,
    attach_strategy_resources,
    discard_strategy_state,
    strategy_resource_usage,
)
from autotrade.environment.replay import (
    ContextDataProvider,
    ExecutionPriceProvider,
    ReplayResult,
    run_daily_replay,
)
from autotrade.environment.replay.stats import (
    PhaseTimer,
    attach_cost_sensitivity,
    attach_sub_window_benchmark,
    finalize_summary_timing,
)
from autotrade.environment.replay.style import benchmark_summary_block
from autotrade.environment.strategy_loader import validate_strategy_package

from .config import (
    FORWARD_PHASE,
    RESEARCH_PHASE,
    EvaluationRequest,
    StrategyExperimentConfig,
)


class DailyStrategyPipeline:
    """One strategy replay over a daily frame, in an executor of its own."""

    def __init__(
        self,
        config: StrategyExperimentConfig,
        *,
        context_data: ContextDataProvider | None = None,
        execution_price: ExecutionPriceProvider | None = None,
        executor_factory: Callable[[StrategyExperimentConfig], StrategyExecutor]
        | None = None,
    ) -> None:
        self.config = config
        self.context_data = context_data
        self.execution_price = execution_price
        self.executor_factory = executor_factory

    def run(
        self,
        daily: pd.DataFrame | str | Path,
        corporate_actions: pd.DataFrame | None = None,
        delistings: DelistingTable | None = None,
    ) -> ReplayResult:
        """``corporate_actions`` is the replay slot's ex-date table; the Broker
        credits its cash dividends on the ex-date (the share leg comes from the
        daily frame's ``pre_close``). The PIT backend always passes it; None is
        only for the local ``daily`` development backend and unit tests.
        ``delistings`` is the release's delisting table, which a profile with
        the delisting rules requires (``broker.BrokerProfile``)."""

        frame = pd.read_parquet(daily) if isinstance(daily, (str, Path)) else daily
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("daily must be a pandas DataFrame or parquet path")
        # Every replay starts from an empty state directory: it is created here
        # and discarded with the run, so nothing an earlier run wrote reaches
        # it. World-writable for the sandbox's non-root workers.
        state_dir = Path(mkdtemp(prefix="strategy_state_"))
        state_dir.chmod(0o777)
        try:
            executor = self._create_executor(state_dir)
            # The executor's own telemetry has to be read while its container
            # still exists, so it is closed before the usage is taken — on the
            # failing path too, where the peak and the clock that ran out are
            # exactly what the failure needs to report.
            try:
                try:
                    replay = run_daily_replay(
                        daily=frame,
                        strategy=executor,
                        schedule=self.config.schedule,
                        profile=self.config.broker_profile,
                        context_data=self.context_data,
                        execution_price=self.execution_price,
                        corporate_actions=corporate_actions,
                        delistings=delistings,
                    )
                finally:
                    executor.close()
            except BaseException as exc:
                attach_strategy_resources(exc, executor)
                raise
            return replace(replay, resources=strategy_resource_usage(executor))
        finally:
            discard_strategy_state(state_dir, self.config.sandbox.spec)

    def _create_executor(self, state_dir: Path) -> StrategyExecutor:
        if self.executor_factory is not None:
            executor = self.executor_factory(self.config)
            if not isinstance(executor, StrategyExecutor):
                raise TypeError("executor_factory must return a StrategyExecutor")
            return executor
        if self.config.execution_mode == "trusted":
            return TrustedStrategyExecutor.from_path(
                self.config.strategy_path,
                state_dir=state_dir,
                models_dir=self.config.models_dir,
            )
        return DockerStrategyExecutor(
            self.config.strategy_path,
            self.config.sandbox,
            models_dir=self.config.models_dir,
            state_dir=state_dir,
        )


def checked_strategy(request: EvaluationRequest, max_days: int | None) -> Path:
    """The request's strategy entry point, once the request is one a backend
    replays: a positive ``max_days``, a research or forward mode, and a
    revision whose ``main.py`` loads as a strategy package."""

    if max_days is not None and max_days <= 0:
        raise ValueError("max_days must be a positive integer")
    if request.mode not in {RESEARCH_PHASE, FORWARD_PHASE}:
        raise ValueError(f"unsupported evaluation mode: {request.mode}")
    strategy_path = Path(request.revision.output_path) / "main.py"
    if not strategy_path.is_file():
        raise FileNotFoundError(f"strategy revision has no main.py: {strategy_path}")
    validate_strategy_package(strategy_path)
    return strategy_path


def finished_summary(
    record: Mapping[str, object],
    style: Mapping[str, object],
    request: EvaluationRequest,
    *,
    started_at: str,
    timer: PhaseTimer,
) -> dict[str, object]:
    """A replay record's stats completed into the result's summary: the
    benchmark block and per-window benchmark readings from the style
    analysis, the cost sensitivity at the request's slippage, and the
    evaluation's timing."""

    summary = record.get("stats")
    if not isinstance(summary, dict):
        raise TypeError("daily replay omitted stats")
    benchmark = benchmark_summary_block(style)
    if benchmark is not None:
        summary["benchmark"] = benchmark
    attach_sub_window_benchmark(summary, style)
    attach_cost_sensitivity(summary, request.broker_profile.slippage_bps)
    finalize_summary_timing(summary, started_at=started_at, setup_phases=timer.to_record())
    return summary


def null_control_seed(key: str, role: str) -> int:
    """A stable 32-bit seed per session (or artifact) and role.

    Stable across processes and runs (``hash`` is not), so a re-run draws the
    same null and its percentile can be compared with the one the ledger
    already holds.
    """

    digest = hashlib.blake2b(f"{key}:{role}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


__all__ = [
    "DailyStrategyPipeline",
    "checked_strategy",
    "finished_summary",
    "null_control_seed",
]
