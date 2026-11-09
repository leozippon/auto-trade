"""Daily replay core: the host engine and its supporting market/result/PIT-view
modules.

- ``engine``: host-side replay orchestrator (``DailyReplayEngine``,
  ``run_daily_replay``) driving the scheduled inference clock and the Broker.
- ``market``: daily market data with point-in-time visibility.
- ``stats``: ``ReplayResult`` container and return-statistics reducer.
- ``timeview``: rolling as-of PIT view over snapshot + replay parts.
- ``style``: Barra-lite benchmark/style attribution over frozen replay outputs.
- ``null_control``: the random-portfolio null control of one result and its
  random-baseline composite.
- ``curve``: one result as the daily curve the console and Paper draw.
"""

from .engine import (
    BacktestError,
    ContextDataProvider,
    DailyOrderInbox,
    DailyReplayEngine,
    ExecutionPriceProvider,
    StrategyDataView,
    resolve_execution_price,
    run_daily_replay,
)
from .market import DailyMarketData
from .null_control import run_null_control
from .stats import (
    PhaseTimer,
    ReplayResult,
    compute_return_stats,
    finalize_summary_timing,
)

__all__ = [
    "BacktestError",
    "ContextDataProvider",
    "DailyMarketData",
    "DailyOrderInbox",
    "DailyReplayEngine",
    "ExecutionPriceProvider",
    "PhaseTimer",
    "ReplayResult",
    "StrategyDataView",
    "compute_return_stats",
    "finalize_summary_timing",
    "resolve_execution_price",
    "run_daily_replay",
    "run_null_control",
]
