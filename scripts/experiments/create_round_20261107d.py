#!/usr/bin/env python
"""The 2026-11-07d round: round 20261107c's experiment again on the ZGCA A100, with the prompt that names the card.

`open_200k_10y_astra_zgca_20261107d` follows `open_200k_10y_astra_zgca_20261107c`,
which ran the 2026-11-07 harness on the instance and spent its research years
on LightGBM-and-ridge ensembles trained on the CPU while the A100 idled: the
system prompt said nothing about the card, so the session only met it through
`nvidia-smi`. This round differs from 20261107c in the harness, in when it compacts,
in how many replays `validate` runs at once, in its per-decision limit and in its
research time, a week (10,080 minutes) instead of 7,200 minutes. The run
facts state the sandbox GPU (count and device name) and, when the experiment has a
card, the research-orientation block adds one sentence asking the session to make
full use of the Sandbox's GPU compute. One `generate_orders` call may take an hour
instead of 20261107c's 15 minutes, and the orientation states that hour and the
two-hour `fit` limit as the strategy's compute budget, so the per-call seconds
the replay results report do not read as a cost to cut. No compaction threshold
is pinned, so every conversation compacts at its derived bound: 959,040 tokens for gpt-6-astra with
DeepSeek-V4.1-Flash compaction (the 1,000,000-token window less the
32,768-token output budget and the 8,192-token safety margin), the sub-agents
at the same figure, and the Agent is asked to compact itself at 90% of it.
`validate` replays three candidates at a time instead of one: 20261107c's
replays were CPU-bound LightGBM fits that left the card and most of the
instance's cores idle, so a six-candidate batch took about seven hours; the
strategy containers of concurrent replays share the experiment's one card and
both strategy deadlines scale with the batch's width. Model, pack, directive,
geometry, the other budgets and card are 20261107c's.

The rest is round 20261107's: ten August-July research years
2015-08-01..2025-07-31, of which `research_years` 7 are the session's and the
validation period 2022-08..2025-07 is tested in twelve quarterly steps; the
ten-year profile and seed (`_profiles.FUND_TITLE_TEN_YEAR_CSI1000` on
`FUND_TITLE_TEN_YEAR_PIT_VIEWS_SEED`, CSI 1000, the 0.55 / 0.30 drawdown limits,
alpha 0.05); a CNY 200k account on the main board and ChiNext; gpt-6-astra at
reasoning effort xhigh with sub-agents and compaction on DeepSeek-V4.1-Flash;
the instance's one A100 with three replays at a time in `validate`; and the budgets of
19,200 model calls, USD 150 and 36 null controls.

It runs on the instance, whose fill loop creates it once the card is free:

    ~/miniconda3/envs/quant/bin/python ops/zgca/zgca.py run scripts/experiments/create_round_20261107d.py --fill

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

# Round 20261107's directive, unchanged so the harness is the only difference.
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
    # The instance's one A100, claimed by the console when it starts the worker.
    "open_200k_10y_astra_zgca_20261107d": _experiment(ASTRA, gpus=1, replays=3),
}

ROUND = Round(
    experiments=EXPERIMENTS,
    pit_views_seed=FUND_TITLE_TEN_YEAR_PIT_VIEWS_SEED,
    overrides={
        **FUND_TITLE_TEN_YEAR_CSI1000,
        "research_years": 7,
        "initial_cash": 200_000,
        "permitted_boards": ["main", "gem"],
        # Round 20261107's budgets, but a week of research time.
        "max_api_spend_usd": 150.0,
        "max_llm_calls": 19_200,
        "max_research_minutes": 10_080,
        "max_null_controls": 36,
        # This round's strategy limits, which the prompt states as the compute
        # budget; the session container takes the same memory.
        "strategy_inference_timeout_seconds": 3_600,
        "strategy_fit_timeout_seconds": 7_200,
        "strategy_memory": "64g",
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
