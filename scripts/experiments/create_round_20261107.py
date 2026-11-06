#!/usr/bin/env python
"""The 2026-11-07 round: an open pack at a CNY 200k account on seven research years, gpt-6-astra on the ZGCA A100 instance.

It asks: with seven research years in view and three validation years revealed
only step by step through `test`, how does a session research, spend and choose
what to test, and what validation record and readout does its registered
strategy reach? `open_200k_10y_astra_zgca_20261107` mounts `open_200k_20261107`,
which asks for a strategy that holds up through the validation period's tests
and frames no mechanism.

The pack states the direction and the system prompt states the rules; this
file copies neither. Research runs ten August-July years, 2015-08-01..2025-07-31,
of which `research_years` 7 are the session's: it sees 2015-08..2022-07, and the
validation period 2022-08..2025-07 is tested in twelve quarterly steps
(`test_step_months` 3), each recorded once, by the session's `test` or by the
host's completion after it, and read at the program's one alpha, 0.05 (pinned by
`_round.BASE_EXPECTED_DEFAULTS`). The forward twelve months end 2026-07-31 and
Held-out reaches 2026-10-31, clipped to the release's last trading day. The
selection, geometry and seed are the ten-year profile's
(`_profiles.FUND_TITLE_TEN_YEAR_CSI1000` on `FUND_TITLE_TEN_YEAR_PIT_VIEWS_SEED`,
prebuilt for research years 5, 6 and 7: daily, the ten default fundamentals,
the four macro datasets and the announcement titles, graded against CSI 1000
with the 0.55 / 0.30 drawdown limits), on a CNY 200k account restricted to the
main board and ChiNext, with no holding count and no single-name weight.
Creation requires the seed's finished prebuild and its published release, read
from `provider.json` at pre-flight.

The main session runs gpt-6-astra at reasoning effort xhigh; sub-agents and
compaction run DeepSeek-V4.1-Flash. The experiment takes the instance's one
A100 (`gpu_count` 1), claimed by the console when it starts the worker. The
budgets are the launcher's and the console's defaults, and the experiment entry
overrides none of them: 2,400 minutes and 6,400 model calls
(`_round.BASE_OVERRIDES`), USD 50, 12 null controls, six `validate` calls
between two tests, 3,600 s per `fit` and 360 s per `generate_orders`, a
strategy container of 8 CPUs and 32 GiB, and three replays at a time in
`validate`.

The console refuses to create a GPU experiment while too few cards are free
(`ExperimentManager._claimable_gpus`), so queue it on the instance with
`--fill`, which creates it once the card is free:

    ~/miniconda3/envs/quant/bin/python ops/zgca/zgca.py run scripts/experiments/create_round_20261107.py --fill

`--dry-run` contacts no console and creates nothing:

    ~/miniconda3/envs/quant/bin/python scripts/experiments/create_round_20261107.py <port> --dry-run

It runs the console's create-time pre-flight on the shared parameters and then on
the experiment, reading the seed contract and the release it pins, prints the
shared request, the experiment's own parameters, its acceptance rules and its
directive, and, unless a smoke record already covers this template, replays the
contract template from every research-year start, the validation years
included, and over the full span. The template reads nothing and places no
orders, so that smoke shows only that the seed's views build for every
decision, the strategy container starts and the contract holds.
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

PACK = "configs/workspace_refs/open_200k_20261107"
ASTRA = "gpt-6-astra"
FLASH = "DeepSeek-V4.1-Flash"

# Point at the pack. Do not restate it or the system prompt's rules, name a
# reading after the research years, or send the session to a repository document
# it cannot mount.
DIRECTIVE = (
    "读 `refs/README.md`。"
    "在研究年上找一个能经得住验证期逐步 `test` 的策略；方向、方法与资源使用由会话自主决定。"
    "数据边界、账户与成本见参考包。"
)

EXPERIMENTS: dict[str, dict[str, object]] = {
    "open_200k_10y_astra_zgca_20261107": {
        "model": ASTRA,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "workspace_reference": PACK,
        "research_directive": DIRECTIVE,
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
        # The instance's one A100, claimed by the console when it starts the
        # worker; the session and every replay's strategy container share it.
        "gpu_count": 1,
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
