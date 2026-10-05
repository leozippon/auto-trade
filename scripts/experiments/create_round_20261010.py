#!/usr/bin/env python
"""The 2026-10-10 round: the event pair, now that the Broker taxes dividends.

Round 20261007 defined a second pair of event-time sessions on the
fundamentals pack and withdrew it within minutes: the Broker credited cash
dividends gross and charged no holding-period tax, so any book that bought
into an ex-date was flattered. The one event book that had frozen, a dividend
implementation capture, turned out to be exactly that: re-run with the tax its
research active IR went from 1.706 to -1.80 and its graduation was voided
(docs/research-lessons.md §3).

The Broker now charges the tax for new arms and the freeze gate also requires
the nominee's own equity to beat the benchmark at stressed costs. This round
queues the withdrawn pair under those rules, on the taxed twin of the
fundamentals seed, with the lineage round 20261007 gave it (the two event arms
and the two open arms of round 20261005; the voided arm's trials still count).

Text-evidence scoring stays on the local model in both arms.

Usage: create_round_20261010.py <port> [--dry-run] [--fill] [experiment_id ...]
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

from scripts.experiments._profiles import FUND_EIGHT_YEAR_100K, MIMO, TAXED_FUND_PIT_VIEWS_SEED
from scripts.experiments._round import Round
from scripts.experiments.create_round_20261007 import EVENT2_ARM, EVENT2_DIRECTIVE

EVENT3_DIRECTIVE = (
    EVENT2_DIRECTIVE
    + "Broker 按持有期扣红利税：买入后不满一个月卖出，按持有期间到账现金分红的 20% 在卖出时扣（满一个月不满一年 10%），"
    "跨除权日的书把它算进每个回合；上一场冻结的那本分红实施书计税后整期主动 IR 为负，不要以任何形式重走。"
    "提名的行还要过运行事实里的原始超额条件：成本压力下自己的权益要赢过基准。"
)

EVENT3_ARM: dict[str, object] = {**EVENT2_ARM, "research_directive": EVENT3_DIRECTIVE}

ARMS: dict[str, dict[str, object]] = {
    "fund_event3_100k_8y_qwen_20261010": dict(EVENT3_ARM),
    "fund_event3_100k_8y_mimo_20261010": {**MIMO, **EVENT3_ARM},
}

# The geometry and every round-level parameter are round 20261007's; the seed is
# the taxed twin.
ROUND = Round(
    arms=ARMS,
    pit_views_seed=TAXED_FUND_PIT_VIEWS_SEED,
    overrides=FUND_EIGHT_YEAR_100K,
    closed=True,
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
