#!/usr/bin/env python
"""The 2026-11-11b round: the research-depth changes on the ZGCA A100.

`open_200k_10y_flash_zgca_20261111b` is round 20261111's experiment on the
instance's one A100, a card no other tenant shares, with `validate` replaying
three candidates at a time whose strategy containers share the card. It
repeats `open_200k_10y_flash_zgca_20261109b` on the code that followed the
2026-11-10 trace review; `create_round_20261111.py` lists what changed and what
stayed. Every value here is that round's except the cards and two trims that
state each idea once: the prompt's budget bullet, run-section budget line and
first message no longer restate that research continues between tests, and
pack 20261111b drops the direction sentence the prompt already states.

It runs on the instance, whose fill loop creates it once the card is free:

    ~/miniconda3/envs/quant/bin/python ops/zgca/zgca.py run scripts/experiments/create_round_20261111b.py --fill

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

PACK = "configs/workspace_refs/open_200k_20261111b"
FLASH = "DeepSeek-V4.1-Flash"

# The pack states the direction; the directive does not repeat it.
DIRECTIVE = (
    "读 `refs/README.md`：研究方向、数据边界、账户与成本都在参考包里；"
    "方法与资源使用由会话自主决定。"
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
    "open_200k_10y_flash_zgca_20261111b": _experiment(FLASH, gpus=1, replays=3),
}

ROUND = Round(
    experiments=EXPERIMENTS,
    # A `zgca.py run --fill` pass created its experiment on 2026-11-11 at 04:08Z.
    closed=True,
    pit_views_seed=FUND_TITLE_TEN_YEAR_PIT_VIEWS_SEED,
    overrides={
        **FUND_TITLE_TEN_YEAR_CSI1000,
        "research_years": 7,
        "initial_cash": 200_000,
        "permitted_boards": ["main", "gem"],
        # A week of research time, 19,200 model calls, 36 null controls and
        # USD 150, which Flash's prices leave ample.
        "max_api_spend_usd": 150.0,
        "max_llm_calls": 19_200,
        "max_research_minutes": 10_080,
        "max_null_controls": 36,
        # The strategy limits the prompt states as the compute budget; the
        # session container takes the same memory.
        "strategy_inference_timeout_seconds": 3_600,
        "strategy_fit_timeout_seconds": 7_200,
        "strategy_memory": "64g",
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
