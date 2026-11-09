#!/usr/bin/env python
"""The 2026-11-07e round: round 20261107d's experiment on the local console, CPU-only.

`open_200k_10y_astra_local_20261107e` replaces
`open_200k_10y_astra_local_20261107b`, which the owner terminated on
2026-11-09 at 03:29Z, two days in, at its twelfth validated node: it was too
slow, because its strategies rebuilt a 550-day feature window on every
rebalance day and reloaded their model on every call. This round is the local
counterpart of round 20261107d and sends exactly its overrides: a week of
research time (10,080 minutes), an hour per `generate_orders` call, two hours
per `fit`, a 64 GiB strategy container (the session's included), the open pack
`open_200k_20261107` and round 20261107's directive. Like round 20261107d it
runs after the strategy state directory became a read-write working store
that persists across a replay's calls (the contract README's "Caching across
calls"), and like it pins no compaction threshold.

It differs from round 20261107d only in where it runs: this host's eight L20
cards are all held by other workloads, so the experiment asks for no card and
`validate` replays two candidates at a time on the CPU. Its predecessor's
strategies never used a card either.

The rest is round 20261107's: ten August-July research years
2015-08-01..2025-07-31, of which `research_years` 7 are the session's and the
validation period 2022-08..2025-07 is tested in twelve quarterly steps; the
ten-year profile and seed (`_profiles.FUND_TITLE_TEN_YEAR_CSI1000` on
`FUND_TITLE_TEN_YEAR_PIT_VIEWS_SEED`, CSI 1000, the 0.55 / 0.30 drawdown limits,
alpha 0.05); a CNY 200k account on the main board and ChiNext; gpt-6-astra at
reasoning effort xhigh with sub-agents and compaction on DeepSeek-V4.1-Flash;
and the budgets of 19,200 model calls, USD 150 and 36 null controls.

It is created on the local console, whose running slots the fill mode checks
first:

    ~/miniconda3/envs/quant/bin/python scripts/experiments/create_round_20261107e.py 38888 --fill

`--dry-run` with the port contacts no console and creates nothing: it runs the
console's create-time pre-flight, prints what the experiment sends, and, unless
a smoke record already covers the contract template on this seed, replays the
template from every research-year start and over the full span.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Appended, not prepended: the repository root carries directories named `data`,
# `logs` and `results`, which must never shadow an installed package. The
# launcher pins this repo's `src/` itself.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from scripts.experiments._profiles import (
    FUND_TITLE_TEN_YEAR_CSI1000,
    FUND_TITLE_TEN_YEAR_PIT_VIEWS_SEED,
)
from scripts.experiments._round import Round

PACK = "configs/workspace_refs/open_200k_20261107"
ASTRA = "gpt-6-astra"
FLASH = "DeepSeek-V4.1-Flash"

# Round 20261107's directive, as round 20261107d sends it.
DIRECTIVE = (
    "读 `refs/README.md`。"
    "在研究年上找一个能经得住验证期逐步 `test` 的策略；方向、方法与资源使用由会话自主决定。"
    "数据边界、账户与成本见参考包。"
)


def _experiment(model: str, *, gpus: int, replays: int) -> dict[str, object]:
    """One experiment on ``model`` with ``gpus`` cards and ``replays`` concurrent replays in `validate`."""

    return {
        "model": model,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "workspace_reference": PACK,
        "research_directive": DIRECTIVE,
        "gpu_count": gpus,
        "batch_validate_max_concurrency": replays,
    }


EXPERIMENTS: dict[str, dict[str, object]] = {
    # No card: every one of this host's is held by other workloads.
    "open_200k_10y_astra_local_20261107e": _experiment(ASTRA, gpus=0, replays=2),
}

ROUND = Round(
    experiments=EXPERIMENTS,
    pit_views_seed=FUND_TITLE_TEN_YEAR_PIT_VIEWS_SEED,
    overrides={
        **FUND_TITLE_TEN_YEAR_CSI1000,
        "research_years": 7,
        "initial_cash": 200_000,
        "permitted_boards": ["main", "gem"],
        # Round 20261107d's budgets: round 20261107's, but a week of research time.
        "max_api_spend_usd": 150.0,
        "max_llm_calls": 19_200,
        "max_research_minutes": 10_080,
        "max_null_controls": 36,
        # Round 20261107d's strategy limits, which the prompt states as the
        # compute budget; the session container takes the same memory.
        "strategy_inference_timeout_seconds": 3_600,
        "strategy_fit_timeout_seconds": 7_200,
        "strategy_memory": "64g",
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
