#!/usr/bin/env python
"""The 2026-11-11 round: the research-depth changes on this host, on two L20 cards.

`open_200k_10y_flash_local_20261111` repeats `open_200k_10y_flash_local_20261109`
on the code and pack that followed the 2026-11-10 trace review (logbook), with
everything else held still, so the two experiments differ only in what the
session is shown and allowed. That experiment used 177 of its 19,200 model
calls and USD 1 of 150, ran two `validate` calls and tested its last eight
steps back to back; in the rounds before it the sessions ruled pretrained
models out on the pack's memory warning and stepped back from cards other
tenants were using. What changed since:

- the prompt states the session budgets and their share per test interval,
  and every `test` result reports the share used and the step reached;
- each interval between two tests is a research period, and the first message
  says so;
- a bounded `sleep` may wait for a detached job, so `test` is no longer the
  only call that waits;
- a `validate` call counts against its allowance only when one of its
  candidates completes its replay;
- pack `open_200k_20261111` asks for research through the whole budget,
  permits pretrained models and embeddings as an accepted limitation with a
  control, and gives the assigned cards to the experiment in full.

The directive only points at the pack, which states the direction once.

Unchanged from round 20261109: DeepSeek-V4.1-Flash as the main model and for
sub-agents and compaction; ten August-July research years 2015-08-01..2025-07-31
of which `research_years` 7 are the session's, the validation period
2022-08..2025-07 tested in twelve quarterly steps; the ten-year profile and
seed (`_profiles.FUND_TITLE_TEN_YEAR_CSI1000` on
`FUND_TITLE_TEN_YEAR_PIT_VIEWS_SEED`: CSI 1000, the 0.55 / 0.30 drawdown
limits); a CNY 200k account on the main board and ChiNext; two L20 cards with
`validate` replaying three candidates at a time; the budgets and strategy
limits below.

It is created on the local console, whose running slots the fill mode checks
first:

    ~/miniconda3/envs/quant/bin/python scripts/experiments/create_round_20261111.py 38888 --fill

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

PACK = "configs/workspace_refs/open_200k_20261111"
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
    # Two of this host's L20 cards, claimed by the console when it starts the worker.
    "open_200k_10y_flash_local_20261111": _experiment(FLASH, gpus=2, replays=3),
}

ROUND = Round(
    experiments=EXPERIMENTS,
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
