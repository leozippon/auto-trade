#!/usr/bin/env python
"""The 2026-11-15b round: round 20261115's self-enhancement arm alone, on two L20 cards.

Round 20261115 started its five continual-learning arms side by side on this
host, one card each. Local cards are too few for that, so the owner moved the
round to the ZGCA A100, run one experiment after another (round 20261115c),
and kept one experiment here on two cards. That one is the self-enhancement
arm, the heaviest user of card memory, since it keeps a population of model
copies: `selfenh_200k_10y_astra_local_20261115b` mounts round 20261115's
`selfenh` pack unchanged and asks whether self-enhancing copies, kept by their
realised results, let a model that carries its learned state from decision to
decision beat the same model refit cold each quarter. Every value but the card
count is round 20261115's.

It is created on the local console, whose running slots and free cards the
fill mode reads first:

    ~/miniconda3/envs/quant/bin/python scripts/experiments/create_round_20261115b.py 38888 --fill

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
from scripts.experiments._round import PACKS_DIR, Round

ASTRA = "gpt-6-astra"
FLASH = "DeepSeek-V4.1-Flash"

# The pack states the direction; the directive does not repeat it.
MECHANISM_DIRECTIVE = (
    "读 `refs/README.md`：研究方向、所有者的提案、让结果读得出来的对照、数据边界、账户与成本都在参考包里；"
    "方法与资源使用由会话自主决定。"
)

EXPERIMENTS: dict[str, dict[str, str]] = {
    "selfenh_200k_10y_astra_local_20261115b": {
        "workspace_reference": f"{PACKS_DIR}/selfenh_200k_20261115",
        "research_directive": MECHANISM_DIRECTIVE,
    },
}

ROUND = Round(
    experiments=EXPERIMENTS,
    pit_views_seed=FUND_TITLE_TEN_YEAR_PIT_VIEWS_SEED,
    overrides={
        **FUND_TITLE_TEN_YEAR_CSI1000,
        "research_years": 7,
        "initial_cash": 200_000,
        "permitted_boards": ["main", "gem"],
        # gpt-6-astra for the main conversation, Flash for the sub-agents and
        # compaction, as in every arm of round 20261115.
        "model": ASTRA,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        # Two cards, which the session container and every strategy container
        # attach, for the population of copies; three concurrent replays in
        # `validate`.
        "gpu_count": 2,
        "batch_validate_max_concurrency": 3,
        # A week of research time, 19,200 model calls, 36 null controls and
        # USD 150, the binding budget at gpt-6-astra's prices.
        "max_api_spend_usd": 150.0,
        "max_llm_calls": 19_200,
        "max_research_minutes": 10_080,
        "max_null_controls": 36,
        # Under the gpt-6 ids' long-context price tier.
        "compact_token_threshold": 221_184,
        # The strategy limits the prompt states as the compute budget; the
        # session container takes the same memory.
        "strategy_inference_timeout_seconds": 3_600,
        "strategy_fit_timeout_seconds": 7_200,
        "strategy_memory": "64g",
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
