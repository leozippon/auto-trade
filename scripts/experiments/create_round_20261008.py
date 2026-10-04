#!/usr/bin/env python
"""The 2026-10-08 round: the frozen 100k sequence bag's book, made for a holder.

The rank-average bag of four sequence heads froze at 100k on 12 seats, at most
4 swaps a weekly review and no residualisation, and held up after freezing
(`seqbag_alla_100k_8y_qwen_20261003`; the DeepSeek session froze the same
bytes). The owner wants it optimised for live use with profitability as the
objective. Three properties a holder trading it by hand would meet were never
tested (logs/notes/round_20261008/DESIGN.md): the same book read active IR
1.618 and 1.213 on two seed bases and a holder gets one draw; it turns over
about 44 times a year and paid about 31,000 yuan of commission and stamp duty
over eight years on a 100k start; and nobody measured how much of its edge
sits in illiquid micro-caps.

One pack serves three lanes and the directive picks the lane; the model, its
inputs and its label stay the baseline's, and `c_base` (the frozen book on the
same seed base) runs in every batch:

- `seqbag2_bag`: more seeds per head (4, 6). The claim is a narrower spread
  across seed bases with the worst one above the bar, not a higher reading.
- `seqbag2_cost`: swaps per review, half-monthly review, a wider keep band,
  score smoothing across reviews. The claim is on what the holder keeps: lower
  turnover and fees with raw return after costs and the cost-stress reading
  not lower, and active IR within a tolerance of `c_base`'s.
- `seqbag2_pool`: a liquidity floor, a market-value floor, index constituents
  only. The question is how much of the active reading survives.

Every claim is read on at least two seed bases; a row that clears the gate
without its lane's claim is not nominated, because the baseline itself is
already frozen. All six arms run on the fundamentals seed with the parameters
of round 20261006, so `c_base` rows are comparable across the two rounds, on
one GPU, by the local model and by `mimo-v2.6-flash`. Their lineage is round
20261006's; that round's four arms on the same baseline are still running and
have no recorded trial to inherit, so a freeze here is read against the pooled
trials of those arms and of this round's siblings before it is treated as a
result. Fill order (ids end in `_20261008`): `seqbag2_bag_100k_8y_qwen` /
`_mimo`, `seqbag2_cost_100k_8y_qwen` / `_mimo`, `seqbag2_pool_100k_8y_qwen` /
`_mimo`.

Usage: create_round_20261008.py <port> [--dry-run] [--fill] [experiment_id ...]
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

from scripts.experiments._round import Round
from scripts.experiments.create_round_20261002 import MIMO
from scripts.experiments.create_round_20261006 import ROUND as SEQAXES_ROUND
from scripts.experiments.create_round_20261006 import SEQAXES_LINEAGE

SEQBOOK_PACK = "configs/workspace_refs/seqbook_alla_100k_8y_20261008"

COMMON = (
    "c_base 是已冻结的 10 万序列袋（12 席、每次至多换 4 只、每周复核、不残差化），模型、输入与标签不动。"
    "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX 一致，否则停；"
    "gpu_count 为 1，没有 CUDA 就停，不要改成 CPU。"
    "每一批都带同 span、同种子基数的 c_base；主张至少在两个种子基数上读，单一种子基数上的胜负不算。"
)
CLOSING = (
    "主张不成立的行即使过了门也不提名、以 no_edge 收尾——c_base 已经冻结。"
    "一个候选失败就测同一条线里下一个登记的轴，它们不是参数邻域，本臂不换家族。"
    "每一行都报权益口径的扣费年化收益、年换手、佣金、印花税与每月完成回合。"
    "门槛读每行的 information_ratio_bar，不自己算；第一批 offline_trials = 0；杀死线只结束候选，至多四批；"
    "批次计划与决赛者规则照 refs/README.md。"
)

BAG_DIRECTIVE = (
    "本臂走袋线：只开每个头的种子数——4 与 6（c_base 是 2），每个袋大小在每个种子基数上的那一行都是一个试验。"
    + COMMON
    + "主张不是读数更高，而是更窄：每个袋大小与 c_base 在同样的两个种子基数上整期跑，回放年够时加第三个；"
    "它最低的那一行要过最新的 information_ratio_bar，种子基数之间的最大差不超过 0.20（c_base 是 0.40）；"
    "两三个种子基数估出的散布只是粗估，照实写。"
    + CLOSING
)

COST_DIRECTIVE = (
    "本臂走成本线：只开书跟分数的方式——每次换名 3 或 2、每半月复核、保留带 3 倍席位、分数按两周半衰期平滑，"
    "每条腿在每个种子基数上都是一个试验，同线内的组合也是。"
    + COMMON
    + "主张在持有人到手的东西上：两个种子基数上各与同种子的 c_base 比，年换手与佣金加印花税都更低，"
    "权益口径的扣费年化收益与该行的成本压力读数都不低，主动 IR 不比 c_base 低 0.25 以上。"
    "先在 Y1..Y4 上探五条单轴腿，只从它们与前两名的组合里按 refs/README.md 的规则选两个决赛者，"
    "第一次整期就带两个种子基数与 c_base。"
    + CLOSING
)

POOL_DIRECTIVE = (
    "本臂走池线：只开书能买哪些名字——去掉近 20 日成交额中位数最低的 40 %、去掉流通市值最小的 30 %、"
    "只买中证 1000 成分、只买中证 1000 或中证 500 成分，每条腿在每个种子基数上都是一个试验；"
    "过滤只用决策时可见的数据、只管新买，头照旧在全池上训练。"
    + COMMON
    + "要回答的是边还剩多少：每个限制在两个种子基数上与同种子的 c_base 比主动读数留下几成，"
    "连同该行的零技能面板读数与规模载荷一起报；两个种子基数都过门槛的限制书可以提名，"
    "边离开小盘就消失也是要写进 reason 的结果。"
    + CLOSING
)

BAG_ARM: dict[str, object] = {
    "workspace_reference": SEQBOOK_PACK,
    "gpu_count": 1,
    "lineage_arms": SEQAXES_LINEAGE,
    "research_directive": BAG_DIRECTIVE,
}
COST_ARM: dict[str, object] = {**BAG_ARM, "research_directive": COST_DIRECTIVE}
POOL_ARM: dict[str, object] = {**BAG_ARM, "research_directive": POOL_DIRECTIVE}

ARMS: dict[str, dict[str, object]] = {
    "seqbag2_bag_100k_8y_qwen_20261008": dict(BAG_ARM),
    "seqbag2_bag_100k_8y_mimo_20261008": {**MIMO, **BAG_ARM},
    "seqbag2_cost_100k_8y_qwen_20261008": dict(COST_ARM),
    "seqbag2_cost_100k_8y_mimo_20261008": {**MIMO, **COST_ARM},
    "seqbag2_pool_100k_8y_qwen_20261008": dict(POOL_ARM),
    "seqbag2_pool_100k_8y_mimo_20261008": {**MIMO, **POOL_ARM},
}

# The seed, the geometry and every round-level parameter are round 20261006's.
ROUND = Round(
    arms=ARMS,
    pit_views_seed=SEQAXES_ROUND.pit_views_seed,
    overrides=dict(SEQAXES_ROUND.overrides),
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
