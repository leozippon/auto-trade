#!/usr/bin/env python
"""The 2026-11-17 round: a continually learning language model on announcement titles, on DeepSeek-V4.1-Flash and two L20 cards.

One of two rounds under the owner's instructions of 2026-11-16, the other being
round 20261117b on the ZGCA A100:

- DeepSeek-V4.1-Flash as the main model, as round 20261113b's Flash arm ran it,
  with sub-agents and compaction on Flash as before;
- packs that require every tested candidate to run a model that needs the card,
  one of them a language model;
- the prompt and skill corrections made with them.

`llmadapt_200k_10y_flash_local_20261117` mounts pack `llmadapt_200k_20261117`:
a language model from the pretrained store reads the visible titles in the
replay and keeps learning as outcomes mature. It asks whether the text adds to
price and financial features, read against the same strategy without its text
branch, and whether continual learning adds to the frozen language model, read
against the same model as a frozen encoder, both validated in the same batch.
Everything else is round 20261115b's.

It is created on the local console once two cards are free; the fill mode reads
its running slots and free cards first:

    ~/miniconda3/envs/quant/bin/python scripts/experiments/create_round_20261117.py 38888 --fill

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

FLASH = "DeepSeek-V4.1-Flash"

# The pack states the direction and the owner's requirement on the card; the
# directive does not repeat them.
MECHANISM_DIRECTIVE = (
    "读 `refs/README.md`：研究方向、所有者的提案与算力要求、让结果读得出来的对照、数据边界、账户与成本都在参考包里；"
    "在这些之内，方法与资源使用由会话自主决定。"
)

EXPERIMENTS: dict[str, dict[str, str]] = {
    "llmadapt_200k_10y_flash_local_20261117": {
        # Flash for every conversation role; a main model other than the
        # console default is the experiment's own declaration.
        "model": FLASH,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "workspace_reference": f"{PACKS_DIR}/llmadapt_200k_20261117",
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
        # Two cards, which the session container and every strategy container
        # attach; three concurrent replays in `validate`.
        "gpu_count": 2,
        "batch_validate_max_concurrency": 3,
        # A week of research time, 19,200 model calls, 36 null controls and
        # USD 150, ample at Flash's prices.
        "max_api_spend_usd": 150.0,
        "max_llm_calls": 19_200,
        "max_research_minutes": 10_080,
        "max_null_controls": 36,
        # The threshold round 20261113b's Flash arm ran with.
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
