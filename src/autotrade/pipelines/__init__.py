"""Research pipeline package (docs/PIPELINE_DESIGN.md).

The facade re-exports only what callers outside the package actually import
through it; everything else is reached from its own module (``from
autotrade.pipelines import worker``). Keeping the list to its consumers also
keeps ``import autotrade.pipelines`` from pulling the worker's provider
gateways and sandbox-image machinery into every reader of a ledger.
"""

from autotrade.environment.artifacts import ArtifactRevision, RegisteredArtifact

from .config import (
    EvaluationResult,
    ExperimentConfig,
    StepResult,
    StrategyExperimentConfig,
)
from .experiment import (
    DailyStrategyPipeline,
    ExperimentPipeline,
)
from .ledger import ExperimentLedger
from .pit_backend import (
    ResearchPITSnapshotProvider,
)

__all__ = [
    "ArtifactRevision",
    "DailyStrategyPipeline",
    "EvaluationResult",
    "ExperimentConfig",
    "ExperimentLedger",
    "ExperimentPipeline",
    "RegisteredArtifact",
    "ResearchPITSnapshotProvider",
    "StepResult",
    "StrategyExperimentConfig",
]
