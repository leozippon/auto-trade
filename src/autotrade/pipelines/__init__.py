"""Research-arm pipeline package (docs/pipeline-design.md).

The facade re-exports only what callers outside the package actually import
through it; everything else is reached from its own module (``from
autotrade.pipelines import worker``). Keeping the list to its consumers also
keeps ``import autotrade.pipelines`` from pulling the worker's provider
gateways and sandbox-image machinery into every reader of a ledger.
"""

from .config import (
    ArtifactRevision,
    EvaluationResult,
    FrozenArtifact,
    RollingExperimentConfig,
    StepResult,
    StrategyExperimentConfig,
)
from .experiment import (
    DailyStrategyPipeline,
    RollingExperimentPipeline,
)
from .ledger import ExperimentLedger
from .pit_backend import (
    ResearchPITSnapshotProvider,
)

__all__ = [
    "ArtifactRevision",
    "DailyStrategyPipeline",
    "EvaluationResult",
    "ExperimentLedger",
    "FrozenArtifact",
    "ResearchPITSnapshotProvider",
    "RollingExperimentConfig",
    "RollingExperimentPipeline",
    "StepResult",
    "StrategyExperimentConfig",
]
