#!/usr/bin/env python
"""The 2026-10-03 round: the sequence bag on a 100k account, and a second open pair.

The 2026-10-01 round ended with one arm clearing a gate that already carried
its family's earlier search: the rank-average bag of four sequence heads on the
whole-pool recipe (1m, 30 seats, weekly) read an active IR of 1.696 against an
inherited bar of 1.406, eight positive years of eight, with a seed replicate at
1.498 (logs/notes/round_20261003/DESIGN.md). The recipe's single heads had not
cleared their gates on a 100k account, and nobody had run the bag there.

So the first pair rebuilds the bag as a book a 100k account can hold: seats
with a one-lot affordability filter, swaps per review and cadence are the open
axes, the label's beta is corrected to adjusted prices, and the residualise
switch is decided on the active drawdown the gate grades. It inherits the ten
recipe arms and the 1m bag.

The second pair repeats the open 100k pack of the 2026-10-02 round unchanged.
With that round's pair it gives two sessions per model on identical work, which
separates what a model does from what one run happened to do; the rubric the
sessions are scored on was fixed before these arms ran (DESIGN.md §3).

Every pack is run twice at the same time, by the local model and by the hosted
`mimo-v2.6-flash`, with the same budgets, lineage and compaction threshold;
text-evidence scoring stays local. The queue, in fill order (ids end in
`_20261003`): `seqbag_alla_100k_8y_qwen` / `_mimo`, then
`open_research_100k_8y_qwen_r2` / `_mimo_r2`.

Usage: create_round_20261003.py <port> [--dry-run] [--fill] [experiment_id ...]
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

# Every arm that recorded a non-control trial on the whole-pool sequence recipe
# over these eight years, the 1m bag included: SEQBAG_LINEAGE, shared with the
# rounds after it.
from scripts.experiments._profiles import SEQBAG_LINEAGE
from scripts.experiments._round import Round
from scripts.experiments.create_round_20260927 import (
    CSI1000,
    DIVIDEND_PIT_VIEWS_SEED,
    EIGHT_YEAR,
    GATES,
)
from scripts.experiments.create_round_20261002 import MIMO, OPEN_DIRECTIVE, OPEN_PACK

SEQBAG_PACK = "configs/workspace_refs/seqbag_alla_100k_8y_20261003"

SEQBAG_DIRECTIVE = (
    "本臂是 [FEAT-14] 配方上四个序列头各两个种子的秩平均袋（b4s2）放进 10 万账户的书："
    "全 A 去北交所、科创板与 ST，基准 000852.SH。"
    "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX 一致，否则停；"
    "gpu_count 为 1，没有 CUDA 就停，不要改成 CPU。"
    "模型照包不改（β 已改用复权价），只开席位 {12, 20, 30}（一手可买的池过滤）、每次换名 {2, 4}、"
    "复核 {每周, 每半月} 与对树残差化，残差化只按验证行的 benchmark.active_max_drawdown 定。"
    "c_single 与 c_lgbm 与候选同书必跑；每个书变体都是一个试验，先在 Y1..Y4 少探、再取一两本上整期；"
    "门槛读每行的 information_ratio_bar，不自己算。"
    "第一批 offline_trials = 0；杀死线只结束候选，至多四批；提名前照 README 换种子基数做一次复现；"
    "失败后剩下的轴与收尾 reason 要写的读数照 refs/README.md。"
)

ARMS: dict[str, dict[str, object]] = {
    "seqbag_alla_100k_8y_qwen_20261003": {
        "workspace_reference": SEQBAG_PACK,
        "gpu_count": 1,
        "lineage_arms": SEQBAG_LINEAGE,
        "research_directive": SEQBAG_DIRECTIVE,
    },
    "seqbag_alla_100k_8y_mimo_20261003": {
        **MIMO,
        "workspace_reference": SEQBAG_PACK,
        "gpu_count": 1,
        "lineage_arms": SEQBAG_LINEAGE,
        "research_directive": SEQBAG_DIRECTIVE,
    },
    "open_research_100k_8y_qwen_r2_20261003": {
        "workspace_reference": OPEN_PACK,
        "research_directive": OPEN_DIRECTIVE,
    },
    "open_research_100k_8y_mimo_r2_20261003": {
        **MIMO,
        "workspace_reference": OPEN_PACK,
        "research_directive": OPEN_DIRECTIVE,
    },
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=DIVIDEND_PIT_VIEWS_SEED,
    overrides={
        **EIGHT_YEAR,
        **GATES,
        "max_replay_years": 96,
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
    },
    closed=True,
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
