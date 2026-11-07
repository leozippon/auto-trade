#!/usr/bin/env python
"""The 2026-11-07c round: round 20261107's experiment again on the ZGCA A100, under the harness changes of 2026-11-07.

`open_200k_10y_astra_zgca_20261107c` is the paired contrast with
`open_200k_10y_astra_zgca_20261107`, which mounted the open pack
`open_200k_20261107` on seven research years and registered a strategy within
about three hours of a 7,200-minute budget. It runs the same model, pack,
directive, geometry, budgets and card, and differs only by the harness changes
of 2026-11-07: a candidate that defines `fit` registers without a seed
replicate, `validate` describes its candidates and controls in neutral terms,
the curated skills on parameter grids and artifact hygiene are gone, `shell`
allows detached jobs, and the experiment gets 900 s per `generate_orders`,
7,200 s per `fit` and a 64 GiB container (the session's included). Round
20261107b runs the new condition on the local console as a second replicate
and a model contrast.

The rest is round 20261107's: ten August-July research years
2015-08-01..2025-07-31, of which `research_years` 7 are the session's and the
validation period 2022-08..2025-07 is tested in twelve quarterly steps; the
ten-year profile and seed (`_profiles.FUND_TITLE_TEN_YEAR_CSI1000` on
`FUND_TITLE_TEN_YEAR_PIT_VIEWS_SEED`, CSI 1000, the 0.55 / 0.30 drawdown limits,
alpha 0.05); a CNY 200k account on the main board and ChiNext; gpt-6-astra at
reasoning effort xhigh with sub-agents and compaction on DeepSeek-V4.1-Flash,
compacting at 221,184 tokens, below the relay's 272,000-token prompt tier; the
instance's one A100 with one replay at a time in `validate`; and the budgets of
7,200 minutes, 19,200 model calls, USD 150 and 36 null controls.

It runs on the instance, whose fill loop creates it once the card is free:

    ~/miniconda3/envs/quant/bin/python ops/zgca/zgca.py run scripts/experiments/create_round_20261107c.py --fill

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
# Below the relay's 272,000-token prompt tier for gpt-6-astra (see the docstring).
COMPACT_TOKEN_THRESHOLD = 221_184

# Round 20261107's directive, unchanged so the harness is the only difference.
DIRECTIVE = (
    "读 `refs/README.md`。"
    "在研究年上找一个能经得住验证期逐步 `test` 的策略；方向、方法与资源使用由会话自主决定。"
    "数据边界、账户与成本见参考包。"
)


def _experiment(model: str, *, gpus: int) -> dict[str, object]:
    """One experiment on ``model`` with ``gpus`` cards and one replay per card in `validate`."""

    return {
        "model": model,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "compact_token_threshold": COMPACT_TOKEN_THRESHOLD,
        "workspace_reference": PACK,
        "research_directive": DIRECTIVE,
        "gpu_count": gpus,
        "batch_validate_max_concurrency": gpus,
    }


EXPERIMENTS: dict[str, dict[str, object]] = {
    # The instance's one A100, claimed by the console when it starts the worker.
    "open_200k_10y_astra_zgca_20261107c": _experiment(ASTRA, gpus=1),
}

ROUND = Round(
    experiments=EXPERIMENTS,
    pit_views_seed=FUND_TITLE_TEN_YEAR_PIT_VIEWS_SEED,
    overrides={
        **FUND_TITLE_TEN_YEAR_CSI1000,
        "research_years": 7,
        "initial_cash": 200_000,
        "permitted_boards": ["main", "gem"],
        # Round 20261107's budgets.
        "max_api_spend_usd": 150.0,
        "max_llm_calls": 19_200,
        "max_research_minutes": 7_200,
        "max_null_controls": 36,
        # This round's strategy limits; the session container takes the same memory.
        "strategy_inference_timeout_seconds": 900,
        "strategy_fit_timeout_seconds": 7_200,
        "strategy_memory": "64g",
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
