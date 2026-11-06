#!/usr/bin/env python
"""The 2026-11-03 round: the first under the sealed gate, an open pack at a CNY 200k account, gpt-6-astra on the local console.

It asks (logs/research_audit_20261102/round_20261103_design.md): under a gate
the session cannot compute before it registers, how does it research, spend and
generalize, and what sealed residual reading does it reach?
`open_200k_10y_astra_local_20261103a` and `…b`, identical but for their ids,
mount `open_200k_sealed_20261103`, which asks for a strategy that generalizes
to the sealed years and frames no mechanism. The round's two memory experiments,
which mounted the memory hypothesis pack, left this file with the memory
machinery on 2026-11-05; their ids are retired (`_round.RETIRED_IDS`). Every
experiment was created stating `ablation_required` (false), a rule retired with the
registration's ablation on 2026-11-05; the stamped value stays in their
`params.json`, where readers accept and ignore it. So do `sealed_years` 5 and
`max_replay_years` 144, which they were also created with and which were
retired with the sealed gate and the replay-year budget; this file no longer
sends them, since creation refuses both.

The pack states the direction and the host's sealed-gate paragraph states the
rule; this file copies neither. Research runs ten August-July years,
2015-08-01..2025-07-31, of which `sealed_years` 5 are sealed: sessions see
2015-08..2020-07, and the host replays the registered node once over
2020-08..2025-07, the window the null calibration measured, at the program's
one alpha, 0.05 (pinned by `_round.BASE_EXPECTED_DEFAULTS`). The forward
twelve months end 2026-07-31 and Held-out reaches 2026-10-31, clipped to the
release's last trading day. The selection, geometry and seed are rounds
20261031's and 20261102's (`_profiles.FUND_TITLE_TEN_YEAR_CSI1000`: daily, the
ten default fundamentals, the four macro datasets and the announcement titles,
graded against CSI 1000 with the 0.55 / 0.30 drawdown limits), on a CNY 200k
account restricted to the main board and ChiNext, with no holding count and no
single-name weight. Creation requires the seed's finished prebuild and its
published release, read from `provider.json` at pre-flight.

Each experiment's main session runs gpt-6-astra at reasoning effort xhigh;
sub-agents and compaction stay on DeepSeek-V4.1-Flash with the compaction
threshold at 221,184, and `llm_key` alternate starts every relay role on
`LIGHTING_API_KEY_ALT`. No remote replicate, and nothing queues this round: the
fill cron lists no round file.

The budgets are this round's create defaults: 144 replay-years (a full span is
now five years, so the same number of full spans 288 bought before), 36 null
controls, USD 150, 19,200 model calls, 7,200 minutes, 10,800 s per `fit` and
1,080 s per `generate_orders`, a strategy container of 8 CPUs and 64 GiB, no
GPU (the base default) and three replays at a time in `batch_validate`. An
experiment's own entry overrides any of them.

Create through the existing console round entry, not a separate launcher:

    ~/miniconda3/envs/quant/bin/python scripts/experiments/create_round_20261103.py <port> [--dry-run] [--fill]

`--dry-run` contacts no console and creates nothing. It runs the console's
create-time pre-flight on the shared parameters and then on every experiment, reading
the seed contract and the release every experiment pins, prints the shared request,
each experiment's own parameters, its acceptance rules and its directive, and, unless
a smoke record already covers this template, replays the contract template
from every research-year start, the sealed years included, and over the full
span. The template reads nothing and places no orders, so that smoke shows only
that the seed's views build for every decision, the strategy container starts
and the contract holds; it exercises no `fit`, no data read by a strategy and
no order or fill.
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

OPEN_PACK = "configs/workspace_refs/open_200k_sealed_20261103"
ASTRA = "gpt-6-astra"
FLASH = "DeepSeek-V4.1-Flash"
COMPACT_TOKEN_THRESHOLD = 221_184

# Point at the pack. Do not restate it or the host's sealed-gate paragraph,
# name a post-research reading, ask for novelty, or send the session to a
# repository document it cannot mount.
OPEN_DIRECTIVE = (
    "读 `refs/README.md`。"
    "找一个能推广到封存研究年的策略；方向、方法与资源使用由会话自主决定。"
    "数据边界、账户与成本见参考包，验收只看「环境」里的「封存检验」。"
)


def _experiment(pack: str, directive: str) -> dict[str, object]:
    return {
        "model": ASTRA,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "compact_token_threshold": COMPACT_TOKEN_THRESHOLD,
        "llm_key": "alternate",
        "workspace_reference": pack,
        "research_directive": directive,
    }


EXPERIMENTS: dict[str, dict[str, object]] = {
    "open_200k_10y_astra_local_20261103a": _experiment(OPEN_PACK, OPEN_DIRECTIVE),
    "open_200k_10y_astra_local_20261103b": _experiment(OPEN_PACK, OPEN_DIRECTIVE),
}

ROUND = Round(
    # Both experiments were created on 2026-11-03; the file is the record of
    # what was run.
    closed=True,
    experiments=EXPERIMENTS,
    pit_views_seed=FUND_TITLE_TEN_YEAR_PIT_VIEWS_SEED,
    overrides={
        **FUND_TITLE_TEN_YEAR_CSI1000,
        "initial_cash": 200_000,
        "permitted_boards": ["main", "gem"],
        # This round's defaults. An experiment entry may override them at create.
        "batch_validate_max_concurrency": 3,
        "max_null_controls": 36,
        "max_api_spend_usd": 150.0,
        "max_llm_calls": 19_200,
        "max_research_minutes": 7_200,
        "strategy_fit_timeout_seconds": 10_800,
        "strategy_inference_timeout_seconds": 1_080,
        "strategy_cpus": 8,
        "strategy_memory": "64g",
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
