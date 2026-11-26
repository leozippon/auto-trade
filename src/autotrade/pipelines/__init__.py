"""Research pipeline package (docs/PIPELINE_DESIGN.md).

The facade re-exports only the names the command-line entry points import
through it; everything else is reached from its own module (``from
autotrade.pipelines import worker``).
"""

from .config import StrategyExperimentConfig
from .ledger import ExperimentLedger
from .replay_backend import DailyStrategyPipeline

__all__ = [
    "DailyStrategyPipeline",
    "ExperimentLedger",
    "StrategyExperimentConfig",
]
