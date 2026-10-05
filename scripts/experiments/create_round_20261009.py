#!/usr/bin/env python
"""The 2026-10-09 round: how a holder holds the frozen 100k sequence bag, and the industry cycle.

The rank-average bag of four sequence heads froze at 100k on 12 seats, at most
4 swaps a weekly review and no residualisation, and held up after freezing
(`seqbag_alla_100k_8y_qwen_20261003`). Its active reading is the platform's
one confirmed edge; what a person trading it by hand keeps also depends on two
things the graded active series cannot see (logs/notes/round_20261009/
DESIGN_new_lanes.md). The zero-skill panel copies the book's trade instants,
so the clock a swap executes on cancels in the active series and stays in the
account: the book sells at the open and buys at the close, leaving the
swapped money idle through the session, and in the research period A shares
earned positive returns intraday and negative ones overnight in every year.
And the neutralisation removes beta, while the book's beta to CSI 1000 is
about 0.67, so it keeps about seven tenths of a rally.

One pack serves two lanes and the directive picks the lane; the model, its
inputs, its label and its book stay the baseline's, and `c_base` (the frozen
book on the same seed base) runs in every batch:

- `seqhold_clock`: the review sells at the close and the next trading day
  buys the empty seats at the open. The claim is the holder's: on both seed
  bases, raw equity return after costs and the cost-stressed raw excess over
  the index higher than the same-seed `c_base`'s, turnover and order count
  unchanged. The panel adopts the same clock, so the row is expected to read
  below its bar; then the arm ends `no_edge` with the paired raw differences
  per year as an execution rule for the operator, which the pack states as a
  legitimate ending. One arm, on the local model: its legs are deterministic
  bytes a second model would only repeat.
- `seqhold_beta`: the book may buy only names whose ex-ante beta to the index
  (the label's estimator) is at least a floor, 1.0 or 1.1, estimated over 120
  or 60 days, against a placebo of the same floor on a date-seeded random
  score. The claim: on both seed bases the row's beta within 0.9-1.1 and the
  active reading within a tolerance of `c_base`'s, the placebo not earning
  credit by itself. The local model and `mimo-v2.6-flash` each run it.

A row that clears the gate without its lane's claim is not nominated, because
the baseline itself is already frozen. These three arms run on the
fundamentals seed with the parameters of round 20261008, so `c_base` rows are
comparable across rounds 20261006, 20261008 and this one, on one GPU. The
clock arm was created with round 20261006's lineage; the beta pair is created
after the clock arm and that round's four arms closed and inherits them
(`SEQBOOK_LINEAGE`). A freeze here is still read by hand against the pooled
trials of siblings that ran at the same time before it is treated as a result.

A second pack tests one new selection family on the same seed and parameters,
CPU only: `indcycle` ranks Shenwan L1 industries by the median acceleration of
their members' reported single-quarter revenue growth and the median change
of their single-quarter core return on equity (`G`; `GI` also subtracts the
change in capex intensity), and holds four large affordable names of each of
the three first industries, against the same book on randomly ordered
industries (`c_shuf`) and on industry price momentum (`c_px`). The
census's twelve price-based industry-level arms were archived and cannot be
inherited, so the reopening is priced by an `offline_trials` of 12 in the
first batch, stated in the directive. The local model and `mimo-v2.6-flash`
each run it.

Created (ids end in `_20261009`): `seqhold_clock_100k_8y_qwen`,
`indcycle_100k_8y_qwen`, `indcycle_100k_8y_mimo`. The beta pair
(`seqhold_beta_100k_8y_qwen` / `_mimo`) was withdrawn on 2026-10-05 before it
recorded a validation: it refines the two-seed bag, which round 20261008's
bag pair found short of the bar on a seed mean, so a lane on a baseline that
passes is a new round. Its ids are retired and the round is closed.

Usage: create_round_20261009.py <port> [--dry-run] [--fill] [experiment_id ...]
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
    FUND_EIGHT_YEAR_100K,
    MIMO,
    SEED_REPLICATES,
    SEQAXES_LINEAGE,
    SEQBOOK_LINEAGE,
    TAXED_FUND_PIT_VIEWS_SEED,
)
from scripts.experiments._round import Round

SEQHOLD_PACK = "configs/workspace_refs/seqhold_alla_100k_8y_20261009"
INDCYCLE_PACK = "configs/workspace_refs/indcycle_alla_100k_8y_20261009"

COMMON = (
    "c_base 是已冻结的 10 万序列袋（12 席、每次至多换 4 只、每周复核、不残差化），模型、输入、标签与书都不动。"
    "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX 一致，否则停；"
    "gpu_count 为 1，没有 CUDA 就停，不要改成 CPU。"
    "每一批都带同 span、同种子基数的 c_base；主张在两个种子基数上读，单一种子基数上的胜负不算。"
)
CLOSING = (
    "主张不成立的行即使过了门也不提名，c_base 已经冻结、永不提名。"
    "一个候选失败就测同一条线里下一个登记的值，它们不是参数邻域，本臂不换家族。"
    "每一行都报该行的原始读数、权益口径的扣费年化收益、权益回撤、年换手、订单数、佣金、印花税与每月完成回合。"
    "门槛读每行的 information_ratio_bar，不自己算；杀死线只结束候选，至多四批；批次计划与提名规则照 refs/README.md。"
)

CLOCK_DIRECTIVE = (
    "本臂走时钟线：只开换仓的成交时点——复核日 15:00 卖、下一个交易日 09:30 用这笔钱买入空出的席位"
    "（CLOCK 取 close_open），与同种子基数的 c_base（当日 09:30 卖、15:00 买）比。"
    + COMMON
    + "主张在持有人到手的钱上：两个种子基数上，时钟腿的权益口径扣费原始收益与该行成本压力下的原始超额"
    "都高于同种子的 c_base，年换手与订单数不变。"
    "零技能面板按同样的时点换名，门评的主动序列看不到时钟，包作者预期时钟腿的主动 IR 比 c_base 低 0.1 到 0.25、"
    "过不了门槛：主张成立而这一行不过门，就以 no_edge 收尾，把两个种子基数上逐年的原始收益差、"
    "面板自己的原始收益差与开盘竞价深度写进 reason，作为给运营者的一条执行规则。"
    "这是本线正当的结局，不是失败；不要为过门另找办法，不加登记之外的腿。"
    + CLOSING
    + "第一批 offline_trials = 1（包作者量过而没有登记的另一种时钟）。"
)

BETA_DIRECTIVE = (
    "本臂走 β 线：只开书能买哪些名字——事前 β（标签自己的估计量）不低于地板 1.0 或 1.1 的名字才买，"
    "估计窗 120 或 60 个交易日，持仓不因 β 卖出；同地板、同窗口在按决策日取种子的随机分数上的书 c_betashuf 是安慰剂对照。"
    + COMMON
    + "主张：两个种子基数上，该行对中证 1000 的 β 在 0.9 到 1.1 之间，主动 IR 不比同种子的 c_base 低 0.20 以上，"
    "正年不少于 6/8、主动回撤不超过 0.30，安慰剂的主动 IR 不到 +0.5。"
    "每一行报 β、上行捕获、下行捕获与权益回撤。"
    "研究期里中证 1000 整段在跌，把 β 拉到 1 的好处在研究期的原始读数里看不出来："
    "更低的原始收益与更深的权益回撤不是反对本线的证据，结论按 refs/README.md 的三种读法写。"
    + CLOSING
    + "第一批 offline_trials = 0。"
)

INDCYCLE_DIRECTIVE = (
    "本臂测行业层面的报表周期：申万一级行业按成员公司最新报表的单季营收增长加速度与单季扣非 ROE 变化的中位数排（G），"
    "或再减去资本开支强度变化的一半（GI），书在排前三的行业里各持四只买得起的最大流通市值名字，每月复核；"
    "同书的 c_shuf（行业块的先后按决策所在季度随机重排）与 c_px（行业按近 120 个交易日的价格动量排）是对照——"
    "零技能面板不控制行业，c_shuf 是这本书自己的运气带。"
    "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX 一致，否则停；只用 CPU，没有 fit。"
    "这是对普查「行业聚合分数」族的重开：那 12 条臂已经归档、不能作血缘，第一批申报 offline_trials = 12 为它付价。"
    "先在 Y1..Y4 上探两个候选与两个对照，比 c_shuf 高不到 0.30 或不高于 c_px 的候选不进整期；"
    "整期赢过两个对照之后才开登记的轴（16 席、行业内按营收加速度挑）。先验是低的，以 no_edge 收尾是正当的结局。"
    "每一行都报该行的原始读数、权益口径的扣费年化收益、权益回撤、年换手、订单数、佣金、印花税与每月完成回合。"
    "门槛读每行的 information_ratio_bar，不自己算；杀死线只结束候选，至多四批；批次计划与提名规则照 refs/README.md。"
)

ARM: dict[str, object] = {
    "workspace_reference": SEQHOLD_PACK,
    "gpu_count": 1,
    "lineage_arms": SEQAXES_LINEAGE,
}
CLOCK_ARM: dict[str, object] = {**ARM, "research_directive": CLOCK_DIRECTIVE}
BETA_ARM: dict[str, object] = {
    **ARM,
    "lineage_arms": SEQBOOK_LINEAGE,
    "research_directive": BETA_DIRECTIVE + SEED_REPLICATES,
}
INDCYCLE_ARM: dict[str, object] = {
    "workspace_reference": INDCYCLE_PACK,
    "gpu_count": 0,
    "research_directive": INDCYCLE_DIRECTIVE,
}

# The beta pair (`seqhold_beta_100k_8y_qwen` / `_mimo`) was withdrawn and its
# ids retired; `BETA_ARM` stays as the record of what was queued.
ARMS: dict[str, dict[str, object]] = {
    "seqhold_clock_100k_8y_qwen_20261009": dict(CLOCK_ARM),
    "indcycle_100k_8y_qwen_20261009": dict(INDCYCLE_ARM),
    "indcycle_100k_8y_mimo_20261009": {**MIMO, **INDCYCLE_ARM},
}

# The seed, the geometry and every round-level parameter are round 20261008's,
# for both packs.
ROUND = Round(
    arms=ARMS,
    pit_views_seed=TAXED_FUND_PIT_VIEWS_SEED,
    overrides=FUND_EIGHT_YEAR_100K,
    closed=True,
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
