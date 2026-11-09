#!/usr/bin/env python
"""The 2026-11-09b round: round 20261107d's experiment again on the ZGCA A100, after the walk-forward review.

`open_200k_10y_flash_zgca_20261109b` replaces `open_200k_10y_astra_zgca_20261107d`,
which the owner wrapped up on 2026-11-09 at 14:52Z at its twelfth validated node,
before its first `test`. The mid-run review of that round and of its local twin
(`.runtime/analysis/round_{d,e}_midpoint_review_20261109.md`) found both on
course to end the way 20261107c did: no plan for research between test steps,
and API spend, not inference time, as the budget that would stop them. This
round runs the code that followed: the prompt that tells the session research
goes on through the validation period, new strategies included, and that each
revealed interval is out-of-distribution evidence; `validates_per_step` 3
instead of 6; notices and a wrap-up on spend and model calls as well as on
time; honest `read_file` paging; seed replicates replayed beside a tested
candidate; and the fundamental event layer rebuilt without the
`disclosure_date.actual_date` look-ahead, on the seed `_profiles` names.

It is round 20261107d on the same card: the instance's one A100, with
`validate` replaying three candidates at a time, whose strategy containers
share the card. The
budgets are round 20261107d's except where this file says otherwise. Model,
pack, directive and geometry are round 20261107's: ten August-July research
years 2015-08-01..2025-07-31, of which `research_years` 7 are the session's
and the validation period 2022-08..2025-07 is tested in twelve quarterly steps;
the ten-year profile and seed (`_profiles.FUND_TITLE_TEN_YEAR_CSI1000` on
`FUND_TITLE_TEN_YEAR_PIT_VIEWS_SEED`, CSI 1000, the 0.55 / 0.30 drawdown
limits, alpha 0.05); a CNY 200k account on the main board and ChiNext;
DeepSeek-V4.1-Flash as the main model as well as for sub-agents and
compaction (the owner's choice for these rounds until the workflow is stable;
gpt-6-astra, the earlier rounds' main model, bills a whole request at twice
the prompt price past 272k tokens and would have spent its $150 cap around the
third test step), no compaction threshold pinned.

It runs on the instance, whose fill loop creates it once the card is free:

    ~/miniconda3/envs/quant/bin/python ops/zgca/zgca.py run scripts/experiments/create_round_20261109b.py --fill

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
    "open_200k_10y_flash_zgca_20261109b": _experiment(FLASH, gpus=1, replays=3),
}

ROUND = Round(
    experiments=EXPERIMENTS,
    pit_views_seed=FUND_TITLE_TEN_YEAR_PIT_VIEWS_SEED,
    overrides={
        **FUND_TITLE_TEN_YEAR_CSI1000,
        "research_years": 7,
        "initial_cash": 200_000,
        "permitted_boards": ["main", "gem"],
        # Round 20261107d's budgets: a week of research time, 19,200 model
        # calls, 36 null controls and USD 150, which Flash's prices leave ample.
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
