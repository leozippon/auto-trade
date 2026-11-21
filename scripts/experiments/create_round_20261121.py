#!/usr/bin/env python
"""The 2026-11-21 round: three paper Packs on DeepSeek-V4.1-Flash, one after another on the ZGCA A100.

The round follows the owner's instruction of 2026-11-21, verbatim:

    For subsequent experiments, please proceed as follows: Search for papers
    published at ICLR, ICML, and NeurIPS over the past three years on finance,
    time-series data, and other methods that could potentially be applied to
    financial markets. For each selected paper, provide the paper itself and its
    open-source repository (if available) in a research Pack. Let the Agent
    reproduce and fine-tune the proposed methods, then evaluate whether they have
    the potential to generate profitable trading strategies.

and keeps the earlier ones: DeepSeek-V4.1-Flash for every conversation role,
every experiment on the ZGCA instance. Each experiment mounts one paper Pack,
whose payload (the paper, its text, the authors' repository at one commit, any
small dataset or wheel) the pre-flight holds to the Pack's PAYLOAD.sha256; each
Pack requires the tested candidate to be the paper's method.

`tabm_200k_10y_flash_zgca_20261121` mounts pack `tabm_200k_20261121`: TabM
(ICLR 2025), one MLP that trains k weight-sharing submodels together and
averages them, read against the same network with k = 1, against k
independently trained MLPs averaged (TabM_packed) and against a LightGBM
ranker on the same inputs.
`alphaqcm_200k_10y_flash_zgca_20261121` mounts pack `alphaqcm_200k_20261121`:
AlphaQCM (ICML 2025), distributional reinforcement learning that mines a pool
of formulaic alphas, read against the same miner without its QCM variance
bonus, against a random formula pool of the same size and against a LightGBM
ranker on the same price-volume fields.
`timesfm_200k_10y_flash_zgca_20261121` mounts pack `timesfm_200k_20261121`:
TimesFM (ICML 2024), a decoder-only time-series foundation model, used
through the store's TimesFM 2.5 checkpoint and read against the same
architecture randomly initialised, against the frozen checkpoint when the
candidate is fine-tuned and against a LightGBM ranker on features of the same
series.

The instance's fill loop creates the first experiment once the card is free and
each next one once the one before it has ended:

    ~/miniconda3/envs/quant/bin/python ops/zgca/zgca.py run scripts/experiments/create_round_20261121.py --fill

`--dry-run` with the port contacts no console and creates nothing: it runs the
console's create-time pre-flight, the Packs' payload check included, prints
what each experiment sends, and, unless a smoke record already covers the
contract template on this seed, replays the template from every research-year
start and over the full span.
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

# The pack states the direction, the owner's requirement and the controls; the
# directive only sends the session there.
DIRECTIVE = "研究方向与所有者的要求在 `refs/README.md`；在它之内，方法与资源使用由会话自主决定。"


def _experiment(pack: str) -> dict[str, str]:
    """One experiment on Flash for every conversation role; a main model other
    than the console default is the experiment's own declaration."""

    return {
        "model": FLASH,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "workspace_reference": f"{PACKS_DIR}/{pack}",
        "research_directive": DIRECTIVE,
    }


# File order is the fill order: the instance has one card, so they run one
# after another in this order.
EXPERIMENTS: dict[str, dict[str, str]] = {
    "tabm_200k_10y_flash_zgca_20261121": _experiment("tabm_200k_20261121"),
    "alphaqcm_200k_10y_flash_zgca_20261121": _experiment("alphaqcm_200k_20261121"),
    "timesfm_200k_10y_flash_zgca_20261121": _experiment("timesfm_200k_20261121"),
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
