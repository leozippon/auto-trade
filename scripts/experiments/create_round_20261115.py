#!/usr/bin/env python
"""The 2026-11-15 round: four continual-learning mechanisms and an open control, on gpt-6-astra.

Five experiments on this host, each on one L20 card, that differ only in the
reference workspace they mount. Each of four packs asks whether one mechanism
lets a model that carries its learned state from decision to decision beat the
same model refit cold each quarter, read against its own mechanism-off version
validated in the same session:

- `kvedit`: key-value memory writes that leave unrelated keys unchanged;
- `pathway`: the inference pathways of a decision reinforced or suppressed by its reward;
- `selfenh`: a population of self-enhancing copies, kept by their realised results;
- `attnonly`: an attention-only model, without feed-forward blocks, updated as a whole.

`open_200k_10y_astra_local_20261115` mounts round 20261113b's pack unchanged and
is the open-direction control. All five are read on the yardstick merged on
2026-11-15: random baselines matched to the account's capital and drawn a
hundred times, a low-turnover leg in the residual regression, and the full-span
residual reading over the last two research years.

It is created on the local console, whose running slots and free cards the fill
mode reads first; the research cron (`ops/cron/research_fill.cron`) creates the
rest as cards free up:

    ~/miniconda3/envs/quant/bin/python scripts/experiments/create_round_20261115.py 38888 --fill

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

ASTRA = "gpt-6-astra"
FLASH = "DeepSeek-V4.1-Flash"

# The pack states the direction; the directive does not repeat it.
OPEN_DIRECTIVE = (
    "读 `refs/README.md`：研究方向、数据边界、账户与成本都在参考包里；"
    "方法与资源使用由会话自主决定。"
)
MECHANISM_DIRECTIVE = (
    "读 `refs/README.md`：研究方向、所有者的提案、让结果读得出来的对照、数据边界、账户与成本都在参考包里；"
    "方法与资源使用由会话自主决定。"
)


def _experiment(pack: str, directive: str) -> dict[str, str]:
    return {"workspace_reference": f"{PACKS_DIR}/{pack}", "research_directive": directive}


# File order is the fill order: with fewer free cards than experiments, the
# control and the key-value arm are created first.
EXPERIMENTS: dict[str, dict[str, str]] = {
    "open_200k_10y_astra_local_20261115": _experiment("open_200k_20261111b", OPEN_DIRECTIVE),
    "kvedit_200k_10y_astra_local_20261115": _experiment("kvedit_200k_20261115", MECHANISM_DIRECTIVE),
    "pathway_200k_10y_astra_local_20261115": _experiment("pathway_200k_20261115", MECHANISM_DIRECTIVE),
    "selfenh_200k_10y_astra_local_20261115": _experiment("selfenh_200k_20261115", MECHANISM_DIRECTIVE),
    "attnonly_200k_10y_astra_local_20261115": _experiment("attnonly_200k_20261115", MECHANISM_DIRECTIVE),
}

ROUND = Round(
    experiments=EXPERIMENTS,
    pit_views_seed=FUND_TITLE_TEN_YEAR_PIT_VIEWS_SEED,
    overrides={
        **FUND_TITLE_TEN_YEAR_CSI1000,
        "research_years": 7,
        "initial_cash": 200_000,
        "permitted_boards": ["main", "gem"],
        # Every arm on one model triple: gpt-6-astra for the main conversation,
        # Flash for the sub-agents and compaction.
        "model": ASTRA,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        # One card per experiment, so the five run side by side, with three
        # concurrent replays in `validate`.
        "gpu_count": 1,
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
