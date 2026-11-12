#!/usr/bin/env python
"""The 2026-11-12 round: `validate` on revealed steps, on two main models.

Two experiments created together that differ only in the main model, each on
one L20 card: `open_200k_10y_astra_local_20261112` on gpt-6-astra and
`open_200k_10y_flash_local_20261112` on DeepSeek-V4.1-Flash. Sub-agents and
compaction stay on Flash in both.

They are the first sessions in which `validate` can replay any candidate over
the validation steps already tested (`span: "revealed"`). Through round
20261111 a tested step returned scores for the tested candidate only, and every
session, on either model and under every prompt, finished its research before
the first `test` and then took the steps with a call or two each
(RESEARCH_LESSONS, 2026-11-09 and 2026-11-12). The pair asks two things:
whether sessions now research between tests, and whether the narrowness of the
Flash sessions (one model family on the CPU, no delegation, a method dropped
after one error) is the main model's. `create_round_20261112b.py`, started
hours earlier on the ZGCA A100 with gpt-6.1-sol on the code before this rule,
is the baseline for it.

Everything else is round 20261111b's — pack `open_200k_20261111b`, the ten-year
profile and seed, a CNY 200k account on the main board and ChiNext, seven
research years, twelve quarterly steps, `validate` replaying three candidates
at a time, the budgets and strategy limits below — except:

- one card per experiment instead of two, so the pair runs side by side;
- `compact_token_threshold` 221,184 in every arm, round 20261107c's value: the
  gpt-6 ids bill a request past 272,000 prompt tokens at twice the price, and a
  model comparison holds the threshold equal;
- the code since: the prompt no longer asks the session to use most of its
  budget and `test` results carry no budget line (neither changed a session,
  and budget use is a quantity to observe, not to hand over as a target); a
  provider's transient errors are retried until the call's deadline; traces
  record the model's reasoning text; a failed replay over the validation
  period reports its class, not the strategy's error text.

It is created on the local console, whose running slots the fill mode checks
first:

    ~/miniconda3/envs/quant/bin/python scripts/experiments/create_round_20261112.py 38888 --fill

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
from scripts.experiments._round import Round

PACK = "configs/workspace_refs/open_200k_20261111b"
ASTRA = "gpt-6-astra"
FLASH = "DeepSeek-V4.1-Flash"

# The pack states the direction; the directive does not repeat it.
DIRECTIVE = (
    "读 `refs/README.md`：研究方向、数据边界、账户与成本都在参考包里；"
    "方法与资源使用由会话自主决定。"
)


def _experiment(model: str) -> dict[str, object]:
    """One experiment on ``model``, on one card, with three concurrent replays in `validate`."""

    return {
        "model": model,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "workspace_reference": PACK,
        "research_directive": DIRECTIVE,
        "gpu_count": 1,
        "batch_validate_max_concurrency": 3,
    }


EXPERIMENTS: dict[str, dict[str, object]] = {
    "open_200k_10y_astra_local_20261112": _experiment(ASTRA),
    "open_200k_10y_flash_local_20261112": _experiment(FLASH),
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
        # USD 150: ample at Flash's prices, the binding budget at gpt-6-astra's.
        "max_api_spend_usd": 150.0,
        "max_llm_calls": 19_200,
        "max_research_minutes": 10_080,
        "max_null_controls": 36,
        # Under the gpt-6 ids' long-context price tier, equal across the arms.
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
