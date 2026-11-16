#!/usr/bin/env python
"""The 2026-11-17b round: an attention-only continual model and the open direction, on DeepSeek-V4.1-Flash, one after another on the ZGCA A100.

One of two rounds under the owner's instructions of 2026-11-16, the other being
round 20261117 on this host:

- DeepSeek-V4.1-Flash as the main model, as round 20261113b's Flash arm ran it,
  with sub-agents and compaction on Flash as before;
- packs that require every tested candidate to run a model that needs the card,
  one of them a language model;
- the prompt and skill corrections made with them.

`attnonly_200k_10y_flash_zgca_20261117b` mounts pack `attnonly_200k_20261117`:
an attention-only model, without feed-forward blocks, attending across each
decision day's cross-section and across each stock's daily series, trained on
the card and carrying its learned state from decision to decision. It is read
against the same architecture with the feed-forward blocks back and against
itself refit cold each quarter, both validated in the same batch.
`open_200k_10y_flash_zgca_20261117b` mounts pack `open_200k_20261117`: the
direction is open, but every tested candidate runs a model that needs the card.
Everything else is round 20261115c's.

It runs on the instance, whose fill loop creates the first experiment once the
card is free and the second once the first has ended:

    ~/miniconda3/envs/quant/bin/python ops/zgca/zgca.py run scripts/experiments/create_round_20261117b.py --fill

`--dry-run` with the port contacts no console and creates nothing: it runs the
console's create-time pre-flight, prints what each experiment sends, and, unless
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

# Each pack states the direction and the owner's requirement on the card; the
# directive does not repeat them.
MECHANISM_DIRECTIVE = (
    "读 `refs/README.md`：研究方向、所有者的提案与算力要求、让结果读得出来的对照、数据边界、账户与成本都在参考包里；"
    "在这些之内，方法与资源使用由会话自主决定。"
)
OPEN_DIRECTIVE = (
    "读 `refs/README.md`：研究方向、所有者的算力要求、让结果读得出来的对照、数据边界、账户与成本都在参考包里；"
    "在这些之内，方法与资源使用由会话自主决定。"
)


def _experiment(pack: str, directive: str) -> dict[str, str]:
    """One experiment on Flash for every conversation role; a main model other
    than the console default is the experiment's own declaration."""

    return {
        "model": FLASH,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "workspace_reference": f"{PACKS_DIR}/{pack}",
        "research_directive": directive,
    }


# File order is the fill order: the instance has one card, so they run one
# after another in this order.
EXPERIMENTS: dict[str, dict[str, str]] = {
    "attnonly_200k_10y_flash_zgca_20261117b": _experiment("attnonly_200k_20261117", MECHANISM_DIRECTIVE),
    "open_200k_10y_flash_zgca_20261117b": _experiment("open_200k_20261117", OPEN_DIRECTIVE),
}

ROUND = Round(
    experiments=EXPERIMENTS,
    pit_views_seed=FUND_TITLE_TEN_YEAR_PIT_VIEWS_SEED,
    overrides={
        **FUND_TITLE_TEN_YEAR_CSI1000,
        "research_years": 7,
        "initial_cash": 200_000,
        "permitted_boards": ["main", "gem"],
        # The instance's one A100, claimed by the console when it starts the
        # worker; three concurrent replays in `validate`.
        "gpu_count": 1,
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
