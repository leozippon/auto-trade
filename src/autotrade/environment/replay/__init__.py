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
    ContextDataProvider,
    ExecutionPriceProvider,
    run_daily_replay,
)
from .stats import ReplayResult

__all__ = [
    "ContextDataProvider",
    "ExecutionPriceProvider",
    "ReplayResult",
    "run_daily_replay",
]
