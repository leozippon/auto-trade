#!/usr/bin/env python
"""The 2026-11-27 round: three paper Packs on DeepSeek-V4.1-Flash, one after another on the ZGCA A100.

The round follows the owner's instruction of 2026-11-21, which each Pack's
references/proposal.md quotes verbatim, and keeps the earlier ones:
DeepSeek-V4.1-Flash for every conversation role, every experiment on the ZGCA
instance. None of the three papers is from the three venues the instruction
names; each proposal says why it was chosen. Each experiment mounts one paper
Pack, whose payload (the paper, its text, the authors' repository at one
commit, any small dataset) the pre-flight holds to the Pack's PAYLOAD.sha256;
each Pack requires the tested candidate to be the paper's method.

`kronos_200k_10y_flash_zgca_20261127` mounts pack `kronos_200k_20261127`:
Kronos (AAAI 2026), a decoder-only Transformer pretrained on tokenised
candlesticks, used through the store's checkpoints on daily and thirty-minute
bars and read against the same architecture randomly initialised, against
daily bars in place of the intraday ones, against the frozen checkpoint when
the candidate is fine-tuned and against a LightGBM ranker on the same bars.
`cograsp_200k_10y_flash_zgca_20261127` mounts pack `cograsp_200k_20261127`:
COGRASP (IJCAI 2025), a graph convolution over a stock co-occurrence graph
feeding a multi-window attention LSTM, its graph rebuilt point in time from
the announcement titles, research reports and news instead of the paper's
forum posts, read against industry, return-correlation, identity and
degree-preserving shuffled graphs and against a LightGBM ranker.
`chronos2_200k_10y_flash_zgca_20261127` mounts pack `chronos2_200k_20261127`:
Chronos-2 (Amazon technical report, 2025), an encoder-only time-series
foundation model whose group attention reads related series and covariates
together, used through the store's checkpoint and read against each series in
its own group, against no covariates, against the frozen checkpoint when the
candidate is fine-tuned, against the architecture randomly initialised and
against a LightGBM ranker on features of the same series.

The three share one seed, whose selection adds the thirty-minute bars, the
research reports and the news to the default one (`_profiles.py`). The
instance's fill loop holds every experiment until that seed's prebuild has
finished, then creates the first once the card is free and each next one once
the one before it has ended:

    ~/miniconda3/envs/quant/bin/python ops/zgca/zgca.py run scripts/experiments/create_round_20261127.py --fill

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
    BARS30_TEXT_TEN_YEAR_PIT_VIEWS_SEED,
    BARS30_TEXT_TEN_YEAR_SELECTION,
    FUND_TITLE_TEN_YEAR_CSI1000,
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
    "kronos_200k_10y_flash_zgca_20261127": _experiment("kronos_200k_20261127"),
    "cograsp_200k_10y_flash_zgca_20261127": _experiment("cograsp_200k_20261127"),
    "chronos2_200k_10y_flash_zgca_20261127": _experiment("chronos2_200k_20261127"),
}

ROUND = Round(
    experiments=EXPERIMENTS,
    pit_views_seed=BARS30_TEXT_TEN_YEAR_PIT_VIEWS_SEED,
    overrides={
        **FUND_TITLE_TEN_YEAR_CSI1000,
        **BARS30_TEXT_TEN_YEAR_SELECTION,
        "research_years": 7,
        "initial_cash": 200_000,
        "permitted_boards": ["main", "gem"],
        # The instance's one A100, claimed by the console when it starts the
        # worker; six concurrent replays in `validate`, so the batch of six
        # candidates each Pack asks for replays at once.
        "gpu_count": 1,
        "batch_validate_max_concurrency": 6,
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
