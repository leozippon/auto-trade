#!/usr/bin/env python
"""The 2026-11-07b round: round 20261107's open pack and geometry under the harness changes of 2026-11-07, gpt-6-astra and claude-opus-5-5 on the local console.

It asks whether the harness, not the model, made round 20261107's session
settle early. That round's one experiment, `open_200k_10y_astra_zgca_20261107`,
mounted the open pack `open_200k_20261107` on seven research years and
registered a strategy within about three hours of a 7,200-minute budget. This
round runs the same pack, directive and geometry after the harness changes of
2026-11-07: a candidate that defines `fit` registers without a seed replicate,
`validate` describes its candidates and controls in neutral terms, the curated
skills on parameter grids and artifact hygiene are gone, `shell` allows
detached jobs, and every experiment here gets 900 s per `generate_orders`,
7,200 s per `fit` and a 64 GiB container (the session's included).

`open_200k_10y_astra_local_20261107b` replicates the new condition on two
local L20 cards with two replays at a time in `validate`, beside round
20261107c, which runs it on the ZGCA A100 as the paired contrast with round
20261107's experiment; `open_200k_10y_opus_local_20261107b` is the same local
experiment on claude-opus-5-5, a model contrast (the relay's Claude ids take no
`reasoning_effort`). Every session's sub-agents and compaction run
DeepSeek-V4.1-Flash, compacting at 221,184 tokens, below the relay's
272,000-token prompt tier for gpt-6-astra and the same for every experiment so
the contrasts share it.

The rest is round 20261107's: ten August-July research years
2015-08-01..2025-07-31, of which `research_years` 7 are the session's and the
validation period 2022-08..2025-07 is tested in twelve quarterly steps; the
ten-year profile and seed (`_profiles.FUND_TITLE_TEN_YEAR_CSI1000` on
`FUND_TITLE_TEN_YEAR_PIT_VIEWS_SEED`, CSI 1000, the 0.55 / 0.30 drawdown limits,
alpha 0.05); a CNY 200k account on the main board and ChiNext; and the budgets
of 7,200 minutes, 19,200 model calls, USD 150 and 36 null controls.

Both experiments are created on the local console by the research fill cron
(`ops/cron/research_fill.cron`, `ROUNDS=scripts/experiments/create_round_20261107b.py`),
which runs this file with `--fill`; each asks for two cards, so it stays
pending until two are free. `--dry-run` with the port contacts no console and
creates nothing: it runs the console's create-time pre-flight on the shared
parameters and on every experiment, prints what each sends, and, unless a
smoke record already covers the contract template on this seed, replays the
template from every research-year start and over the full span:

    ~/miniconda3/envs/quant/bin/python scripts/experiments/create_round_20261107b.py 38888 --dry-run
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
OPUS = "claude-opus-5-5"
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
    "open_200k_10y_astra_local_20261107b": _experiment(ASTRA, gpus=2),
    "open_200k_10y_opus_local_20261107b": _experiment(OPUS, gpus=2),
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
